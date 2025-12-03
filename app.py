"""
SCHEDULR — all-Python scheduling marketplace MVP
------------------------------------------------
Features:
- Auth (JWT cookie), Roles: CUSTOMER / PRO
- Professional profiles: bio, tags, location
- Services: duration, price, buffer, waiver text required
- Weekly availability rules + exceptions
- Customer discovery/search
- Booking with collision checks
- Mock payments (toggle paid)
- Waiver acceptance (timestamped)
- File uploads for appointment documents (local storage)
- Comments on profiles
- Reviews (1–5 stars; only after appointment marked "COMPLETED")
- Messaging inbox + thread + WebSocket live chat

Run:
  pip install fastapi uvicorn sqlmodel sqlalchemy passlib[bcrypt] python-jose[cryptography] python-multipart pydantic
  uvicorn app:app --reload

Default DB: SQLite file ./schedulr.db
Uploads: ./uploads

NOTE: This is an MVP to get you moving. For production:
- switch DB to Postgres
- add Alembic migrations
- add Stripe (PaymentIntents + webhooks)
- add S3 signed URLs for file uploads
- harden rate limits, CSRF, audit logs, moderation tooling, etc.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import Field, Session, SQLModel, create_engine, select
from zoneinfo import ZoneInfo


# ----------------------------
# Config
# ----------------------------

APP_NAME = "Schedulr"
JWT_SECRET = os.environ.get("SCHEDULR_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
COOKIE_NAME = "schedulr_token"
TOKEN_EXPIRE_DAYS = 7

DB_URL = os.environ.get("SCHEDULR_DB_URL", "sqlite:///./schedulr.db")
engine = create_engine(DB_URL, echo=False)

UPLOAD_DIR = Path(os.environ.get("SCHEDULR_UPLOAD_DIR", "./uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ----------------------------
# Models
# ----------------------------

class Role(str, Enum):
    CUSTOMER = "CUSTOMER"
    PRO = "PRO"
    ADMIN = "ADMIN"


class AppointmentStatus(str, Enum):
    PENDING = "PENDING"         # created, not paid or not confirmed
    CONFIRMED = "CONFIRMED"     # paid + waiver accepted if required
    CANCELED = "CANCELED"
    COMPLETED = "COMPLETED"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    password_hash: str
    role: Role = Field(default=Role.CUSTOMER)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProfessionalProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    display_name: str
    bio: str = ""
    tags: str = ""  # comma-separated
    location: str = ""
    timezone: str = "America/Chicago"
    avatar_url: str = ""  # optional URL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Service(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    name: str
    description: str = ""
    duration_min: int = 60
    buffer_min: int = 0
    price_cents: int = 5000
    require_waiver: bool = True
    waiver_text: str = "I acknowledge the risks and consent to the service."
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AvailabilityRule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    weekday: int = Field(index=True)      # 0=Mon .. 6=Sun
    start_hhmm: str                       # "09:00"
    end_hhmm: str                         # "17:00"


class AvailabilityException(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    on_date: date = Field(index=True)
    start_hhmm: str
    end_hhmm: str
    available: bool = False               # False => block time; True => add extra availability


class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    service_id: int = Field(foreign_key="service.id", index=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    customer_id: int = Field(foreign_key="user.id", index=True)
    start_at_utc: datetime = Field(index=True)
    end_at_utc: datetime
    status: AppointmentStatus = Field(default=AppointmentStatus.PENDING)
    paid: bool = False
    waiver_accepted_at_utc: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appointment_id: int = Field(foreign_key="appointment.id", index=True)
    amount_cents: int
    status: str = "mock_created"  # mock_created, succeeded, failed, refunded
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AppointmentDocument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appointment_id: int = Field(foreign_key="appointment.id", index=True)
    uploaded_by_user_id: int = Field(foreign_key="user.id", index=True)
    filename: str
    path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProfileComment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(foreign_key="professionalprofile.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appointment_id: int = Field(foreign_key="appointment.id", index=True, unique=True)
    profile_id: int = Field(foreign_key="professionalprofile.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    rating: int = 5
    body: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Thread(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    customer_id: int = Field(foreign_key="user.id", index=True)
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: int = Field(foreign_key="thread.id", index=True)
    sender_id: int = Field(foreign_key="user.id", index=True)
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ----------------------------
# DB / startup
# ----------------------------

def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


# ----------------------------
# Auth helpers
# ----------------------------

def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    return pwd_context.verify(pw, pw_hash)


def create_token(user_id: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)


def parse_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return int(payload.get("sub"))
    except Exception:
        return None


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
) -> Optional[User]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    user_id = parse_token(token)
    if not user_id:
        return None
    return session.get(User, user_id)


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Please log in.")
    return user


def require_pro(user: User = Depends(require_user)) -> User:
    if user.role != Role.PRO and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Professional access required.")
    return user


# ----------------------------
# UI helpers (all-Python HTML)
# ----------------------------

def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_money(cents: int) -> str:
    return f"${cents/100:,.2f}"


def html_page(title: str, user: Optional[User], body: str) -> HTMLResponse:
    auth_block = (
        f"""
        <div class="flex items-center gap-2">
          <a class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10" href="/inbox">Inbox</a>
          <a class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10" href="/me">Me</a>
          <form method="post" action="/logout">
            <button class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10">Logout</button>
          </form>
        </div>
        """
        if user
        else """
        <div class="flex items-center gap-2">
          <a class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10" href="/login">Login</a>
          <a class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10" href="/signup">Sign up</a>
        </div>
        """
    )

    pro_cta = (
        """
        <a class="px-3 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 font-semibold" href="/pro/dashboard">
          Pro Dashboard
        </a>
        """
        if user and user.role == Role.PRO
        else """
        <a class="px-3 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 font-semibold" href="/signup?role=PRO">
          Become a Pro
        </a>
        """
    )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <title>{esc(title)} • {APP_NAME}</title>
</head>
<body class="min-h-screen bg-slate-950 text-slate-100">
  <div class="sticky top-0 z-50 backdrop-blur bg-slate-950/70 border-b border-white/10">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-3">
      <a href="/" class="flex items-center gap-2">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-fuchsia-500"></div>
        <div class="font-bold tracking-tight text-lg">{APP_NAME}</div>
      </a>
      <div class="flex items-center gap-2">
        {pro_cta}
        {auth_block}
      </div>
    </div>
  </div>

  <main class="max-w-6xl mx-auto px-4 py-8">
    {body}
  </main>

  <footer class="max-w-6xl mx-auto px-4 pb-10 text-sm text-slate-400">
    <div class="border-t border-white/10 pt-6">
      MVP demo • Mock payments • Local uploads • Upgrade to Postgres + Stripe for production.
    </div>
  </footer>
</body>
</html>
"""
    return HTMLResponse(html)


def pill(text: str) -> str:
    return f'<span class="px-2 py-1 rounded-full bg-white/10 text-xs">{esc(text)}</span>'


def star_row(rating: float) -> str:
    # Simple star display
    full = int(round(rating))
    full = max(0, min(5, full))
    stars = "★" * full + "☆" * (5 - full)
    return f'<span class="text-amber-300 tracking-wide">{stars}</span>'


# ----------------------------
# Scheduling logic
# ----------------------------

HHMM_RE = re.compile(r"^\d{2}:\d{2}$")


def parse_hhmm(s: str) -> time:
    if not HHMM_RE.match(s):
        raise ValueError("Time must be HH:MM")
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def local_dt(d: date, t: time, tz: ZoneInfo) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=tz)


def to_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc)


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def generate_slots_for_day(
    *,
    prof_tz: ZoneInfo,
    day: date,
    rules: List[AvailabilityRule],
    exceptions: List[AvailabilityException],
    service_duration_min: int,
    buffer_min: int,
    existing_appointments: List[Tuple[datetime, datetime]],
    slot_step_min: int = 15,
) -> List[datetime]:
    """
    Returns list of slot start times (UTC datetimes) that are bookable for the given day.
    Ignores capacity >1 (MVP).
    """
    weekday = (day.weekday())  # 0=Mon
    day_rules = [r for r in rules if r.weekday == weekday]

    # Build base availability windows in local time
    windows_local: List[Tuple[datetime, datetime]] = []
    for r in day_rules:
        st = local_dt(day, parse_hhmm(r.start_hhmm), prof_tz)
        en = local_dt(day, parse_hhmm(r.end_hhmm), prof_tz)
        if en > st:
            windows_local.append((st, en))

    # Apply exceptions: block or add windows
    for ex in exceptions:
        if ex.on_date != day:
            continue
        st = local_dt(day, parse_hhmm(ex.start_hhmm), prof_tz)
        en = local_dt(day, parse_hhmm(ex.end_hhmm), prof_tz)
        if en <= st:
            continue
        if ex.available:
            windows_local.append((st, en))
        else:
            # subtract blocked window from each window (simple split)
            new_windows = []
            for wst, wen in windows_local:
                if en <= wst or st >= wen:
                    new_windows.append((wst, wen))
                else:
                    if st > wst:
                        new_windows.append((wst, st))
                    if en < wen:
                        new_windows.append((en, wen))
            windows_local = new_windows

    # normalize windows (merge overlaps)
    windows_local.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    for w in windows_local:
        if not merged:
            merged.append(w)
            continue
        last_st, last_en = merged[-1]
        st, en = w
        if st <= last_en:
            merged[-1] = (last_st, max(last_en, en))
        else:
            merged.append(w)

    dur = timedelta(minutes=service_duration_min)
    buf = timedelta(minutes=buffer_min)
    step = timedelta(minutes=slot_step_min)

    # Existing appointments are stored in UTC; compare in UTC
    existing = [(a, b) for (a, b) in existing_appointments]

    slots_utc: List[datetime] = []
    for wst_local, wen_local in merged:
        cursor = wst_local
        while cursor + dur <= wen_local:
            start_utc = to_utc(cursor)
            end_utc = to_utc(cursor + dur + buf)
            # Must not overlap with existing (consider buffer as part of occupied time)
            ok = True
            for ast, aen in existing:
                if overlaps(start_utc, end_utc, ast, aen):
                    ok = False
                    break
            if ok:
                slots_utc.append(start_utc)
            cursor = cursor + step

    return slots_utc


# ----------------------------
# Real-time chat manager
# ----------------------------

@dataclass
class WSConn:
    ws: WebSocket
    user_id: int


class ConnectionManager:
    def __init__(self) -> None:
        self.rooms: Dict[int, List[WSConn]] = {}

    async def connect(self, thread_id: int, ws: WebSocket, user_id: int) -> None:
        await ws.accept()
        self.rooms.setdefault(thread_id, []).append(WSConn(ws=ws, user_id=user_id))

    def disconnect(self, thread_id: int, ws: WebSocket) -> None:
        conns = self.rooms.get(thread_id, [])
        self.rooms[thread_id] = [c for c in conns if c.ws != ws]

    async def broadcast(self, thread_id: int, payload: Dict[str, Any]) -> None:
        conns = self.rooms.get(thread_id, [])
        dead: List[WebSocket] = []
        for c in conns:
            try:
                await c.ws.send_json(payload)
            except Exception:
                dead.append(c.ws)
        for ws in dead:
            self.disconnect(thread_id, ws)


manager = ConnectionManager()


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title=APP_NAME)
init_db()

# Optional: serve files under /uploads for demo convenience (in prod, do signed URLs + auth checks)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ----------------------------
# Routes: Public / Auth
# ----------------------------

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    q: str = "",
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    qlike = f"%{q.strip()}%"
    stmt = select(ProfessionalProfile).order_by(ProfessionalProfile.created_at.desc())
    if q.strip():
        stmt = (
            select(ProfessionalProfile)
            .where(
                (ProfessionalProfile.display_name.ilike(qlike))
                | (ProfessionalProfile.tags.ilike(qlike))
                | (ProfessionalProfile.location.ilike(qlike))
            )
            .order_by(ProfessionalProfile.created_at.desc())
        )
    pros = session.exec(stmt).all()

    # Ratings summary
    pro_cards = []
    for p in pros:
        avg, count = profile_rating(session, p.id)
        tags = [t.strip() for t in (p.tags or "").split(",") if t.strip()]
        pro_cards.append(
            f"""
            <a href="/p/{p.id}" class="group block rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 p-5 transition">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="text-lg font-semibold group-hover:text-white">{esc(p.display_name)}</div>
                  <div class="text-sm text-slate-400">{esc(p.location or "Remote / Not listed")} • {esc(p.timezone)}</div>
                </div>
                <div class="text-right">
                  <div>{star_row(avg)} <span class="text-sm text-slate-400">({count})</span></div>
                  <div class="text-xs text-slate-500 mt-1">Tap to view</div>
                </div>
              </div>
              <div class="mt-3 flex flex-wrap gap-2">
                {''.join(pill(t) for t in tags[:8])}
              </div>
              <div class="mt-3 text-sm text-slate-300 line-clamp-3">{esc(p.bio or "")}</div>
            </a>
            """
        )

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-8">
        <div class="rounded-3xl border border-white/10 bg-gradient-to-br from-white/10 to-white/5 p-6">
          <div class="text-3xl font-bold tracking-tight">Book professionals like it’s social.</div>
          <div class="mt-2 text-slate-300">
            Find trainers, therapists, coaches, and more. View real availability, book, pay, sign waivers, message, review.
          </div>
          <form class="mt-5 flex gap-2" method="get" action="/">
            <input name="q" value="{esc(q)}" placeholder="Search by name, tag, location…"
              class="w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
            <button class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Search</button>
          </form>
        </div>

        <div class="mt-6 grid sm:grid-cols-2 gap-4">
          {''.join(pro_cards) if pro_cards else '<div class="text-slate-400">No professionals found.</div>'}
        </div>
      </div>

      <div class="lg:col-span-4">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Quick Start</div>
          <ol class="mt-3 text-sm text-slate-300 list-decimal pl-5 space-y-2">
            <li>Sign up as a Customer (book + message + review).</li>
            <li>Or sign up as a Pro to create services and availability.</li>
            <li>Bookings use mock payments in this demo.</li>
          </ol>
          <div class="mt-5 flex gap-2">
            <a href="/signup" class="flex-1 text-center rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Sign up</a>
            <a href="/signup?role=PRO" class="flex-1 text-center rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Become a Pro</a>
          </div>
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Safety & Trust</div>
          <div class="mt-2 text-sm text-slate-300">
            Waivers + required docs before sessions. Verified reviews only after completed appointments.
          </div>
        </div>
      </div>
    </div>
    """
    return html_page("Discover", user, body)


@app.get("/signup", response_class=HTMLResponse)
def signup_form(
    request: Request,
    role: str = "CUSTOMER",
    user: Optional[User] = Depends(get_current_user),
):
    if user:
        return RedirectResponse("/me", status_code=303)
    role = role if role in ["CUSTOMER", "PRO"] else "CUSTOMER"
    body = f"""
    <div class="max-w-xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Create account</div>
      <div class="text-sm text-slate-400 mt-1">Role: <span class="font-semibold">{esc(role)}</span></div>

      <form method="post" action="/signup" class="mt-6 space-y-4">
        <input type="hidden" name="role" value="{esc(role)}"/>
        <div>
          <label class="text-sm text-slate-300">Email</label>
          <input name="email" type="email" required class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Password</label>
          <input name="password" type="password" required class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
        </div>

        <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Sign up</button>
      </form>

      <div class="mt-4 text-sm text-slate-400">
        Already have an account? <a class="text-indigo-300 hover:underline" href="/login">Log in</a>
      </div>
    </div>
    """
    return html_page("Sign up", None, body)


@app.post("/signup")
def signup(
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("CUSTOMER"),
    session: Session = Depends(get_session),
):
    email = email.strip().lower()
    if role not in ["CUSTOMER", "PRO"]:
        role = "CUSTOMER"

    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    user = User(email=email, password_hash=hash_password(password), role=Role(role))
    session.add(user)
    session.commit()
    session.refresh(user)

    # If PRO, create a starter profile
    if user.role == Role.PRO:
        prof = ProfessionalProfile(user_id=user.id, display_name=email.split("@")[0].title(), bio="", tags="trainer,coach", location="", timezone="America/Chicago")
        session.add(prof)
        session.commit()

    token = create_token(user.id)
    resp = RedirectResponse("/me", status_code=303)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=TOKEN_EXPIRE_DAYS * 86400)
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_form(
    request: Request,
    user: Optional[User] = Depends(get_current_user),
):
    if user:
        return RedirectResponse("/me", status_code=303)
    body = """
    <div class="max-w-xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Welcome back</div>
      <div class="text-sm text-slate-400 mt-1">Log in to book, message, and manage your schedule.</div>

      <form method="post" action="/login" class="mt-6 space-y-4">
        <div>
          <label class="text-sm text-slate-300">Email</label>
          <input name="email" type="email" required class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Password</label>
          <input name="password" type="password" required class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
        </div>

        <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Log in</button>
      </form>

      <div class="mt-4 text-sm text-slate-400">
        Need an account? <a class="text-indigo-300 hover:underline" href="/signup">Sign up</a>
      </div>
    </div>
    """
    return html_page("Log in", None, body)


@app.post("/login")
def login(
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    email = email.strip().lower()
    user = session.exec(select(User).where(User.email == email)).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_token(user.id)
    resp = RedirectResponse("/me", status_code=303)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=TOKEN_EXPIRE_DAYS * 86400)
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ----------------------------
# Routes: Me / Pro dashboard
# ----------------------------

@app.get("/me", response_class=HTMLResponse)
def me(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    pro_profile = None
    if user.role == Role.PRO:
        pro_profile = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()

    appts = session.exec(
        select(Appointment).where(Appointment.customer_id == user.id).order_by(Appointment.start_at_utc.desc())
    ).all()

    appt_rows = []
    for a in appts[:25]:
        s = session.get(Service, a.service_id)
        p = session.get(ProfessionalProfile, a.professional_id)
        tz = ZoneInfo(p.timezone if p else "UTC")
        st_local = a.start_at_utc.astimezone(tz)
        appt_rows.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="font-semibold">{esc(s.name if s else "Service")}</div>
                  <div class="text-sm text-slate-400">{esc(p.display_name if p else "Pro")} • {st_local.strftime('%b %d, %Y %I:%M %p')} ({esc(p.timezone if p else "UTC")})</div>
                  <div class="text-sm mt-2">
                    <span class="px-2 py-1 rounded-full bg-white/10 text-xs">{esc(a.status)}</span>
                    <span class="ml-2 text-slate-300">{'Paid' if a.paid else 'Unpaid'}</span>
                  </div>
                </div>
                <div class="text-right">
                  <a class="text-indigo-300 hover:underline text-sm" href="/appointment/{a.id}">View</a>
                </div>
              </div>
            </div>
            """
        )

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-2xl font-bold">Account</div>
          <div class="mt-2 text-slate-300 text-sm">
            <div><span class="text-slate-400">Email:</span> {esc(user.email)}</div>
            <div class="mt-1"><span class="text-slate-400">Role:</span> {esc(user.role)}</div>
          </div>
          {"<div class='mt-4'><a class='inline-block rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold' href='/pro/dashboard'>Open Pro Dashboard</a></div>" if user.role==Role.PRO else ""}
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold">Tips</div>
          <div class="text-sm text-slate-300 mt-2">
            Pros: set weekly availability, then add services. Customers: book a time, accept waiver, upload any docs.
          </div>
        </div>
      </div>

      <div class="lg:col-span-8">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="flex items-center justify-between">
            <div class="text-xl font-bold">Your bookings</div>
            <a href="/" class="text-indigo-300 hover:underline text-sm">Discover pros</a>
          </div>
          <div class="mt-4 space-y-3">
            {''.join(appt_rows) if appt_rows else '<div class="text-slate-400">No bookings yet.</div>'}
          </div>
        </div>
      </div>
    </div>
    """
    return html_page("Me", user, body)


@app.get("/pro/dashboard", response_class=HTMLResponse)
def pro_dashboard(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    services = session.exec(select(Service).where(Service.professional_id == prof.id)).all()
    rules = session.exec(select(AvailabilityRule).where(AvailabilityRule.professional_id == prof.id)).all()

    # Upcoming appointments
    upcoming = session.exec(
        select(Appointment)
        .where(Appointment.professional_id == prof.id)
        .where(Appointment.start_at_utc >= datetime.now(timezone.utc) - timedelta(hours=2))
        .order_by(Appointment.start_at_utc.asc())
    ).all()

    tz = ZoneInfo(prof.timezone)

    svc_cards = []
    for s in services:
        svc_cards.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div class="font-semibold">{esc(s.name)}</div>
              <div class="text-sm text-slate-400 mt-1">{esc(s.description)}</div>
              <div class="text-sm mt-2 text-slate-300">
                {fmt_money(s.price_cents)} • {s.duration_min}m • buffer {s.buffer_min}m • waiver {"on" if s.require_waiver else "off"}
              </div>
            </div>
            """
        )

    rule_rows = []
    for r in sorted(rules, key=lambda x: (x.weekday, x.start_hhmm)):
        rule_rows.append(
            f"""
            <div class="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <span class="font-semibold">{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][r.weekday]}</span>
                <span class="text-slate-300"> {esc(r.start_hhmm)}–{esc(r.end_hhmm)}</span>
              </div>
              <form method="post" action="/pro/rule/{r.id}/delete">
                <button class="text-sm text-rose-300 hover:underline">Delete</button>
              </form>
            </div>
            """
        )

    appt_rows = []
    for a in upcoming[:25]:
        s = session.get(Service, a.service_id)
        cust = session.get(User, a.customer_id)
        st_local = a.start_at_utc.astimezone(tz)
        appt_rows.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="font-semibold">{esc(s.name if s else "Service")}</div>
                  <div class="text-sm text-slate-400">{st_local.strftime('%b %d, %Y %I:%M %p')} ({esc(prof.timezone)})</div>
                  <div class="text-sm text-slate-300 mt-1">Customer: {esc(cust.email if cust else "Unknown")}</div>
                  <div class="text-sm mt-2">
                    <span class="px-2 py-1 rounded-full bg-white/10 text-xs">{esc(a.status)}</span>
                    <span class="ml-2 text-slate-300">{'Paid' if a.paid else 'Unpaid'}</span>
                  </div>
                </div>
                <div class="text-right space-y-2">
                  <a class="text-indigo-300 hover:underline text-sm" href="/appointment/{a.id}">View</a>
                  <form method="post" action="/pro/appointment/{a.id}/complete">
                    <button class="text-sm rounded-lg bg-white/10 hover:bg-white/15 px-3 py-2">Mark Complete</button>
                  </form>
                </div>
              </div>
            </div>
            """
        )

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-5">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-2xl font-bold">Pro Dashboard</div>
          <div class="text-sm text-slate-400 mt-1">Profile: <a class="text-indigo-300 hover:underline" href="/p/{prof.id}">{esc(prof.display_name)}</a></div>

          <div class="mt-5 grid gap-3">
            <a href="/pro/profile" class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold text-center">Edit Profile</a>
            <a href="/pro/service/new" class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold text-center">Add Service</a>
            <a href="/pro/availability" class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold text-center">Set Availability</a>
          </div>
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Services</div>
          <div class="mt-3 space-y-3">
            {''.join(svc_cards) if svc_cards else '<div class="text-slate-400 text-sm">No services yet.</div>'}
          </div>
        </div>
      </div>

      <div class="lg:col-span-7">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Weekly availability</div>
          <div class="text-sm text-slate-400 mt-1">Timezone: {esc(prof.timezone)}</div>

          <form method="post" action="/pro/rule/new" class="mt-4 grid md:grid-cols-4 gap-2 items-end">
            <div>
              <label class="text-sm text-slate-300">Weekday</label>
              <select name="weekday" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3">
                <option value="0">Mon</option><option value="1">Tue</option><option value="2">Wed</option>
                <option value="3">Thu</option><option value="4">Fri</option><option value="5">Sat</option><option value="6">Sun</option>
              </select>
            </div>
            <div>
              <label class="text-sm text-slate-300">Start</label>
              <input name="start_hhmm" placeholder="09:00" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
            </div>
            <div>
              <label class="text-sm text-slate-300">End</label>
              <input name="end_hhmm" placeholder="17:00" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
            </div>
            <button class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Add</button>
          </form>

          <div class="mt-4 space-y-2">
            {''.join(rule_rows) if rule_rows else '<div class="text-slate-400 text-sm">No availability rules yet.</div>'}
          </div>
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Upcoming appointments</div>
          <div class="mt-3 space-y-3">
            {''.join(appt_rows) if appt_rows else '<div class="text-slate-400 text-sm">No upcoming appointments.</div>'}
          </div>
        </div>
      </div>
    </div>
    """
    return html_page("Pro Dashboard", pro_user, body)


@app.get("/pro/profile", response_class=HTMLResponse)
def pro_profile_edit(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    body = f"""
    <div class="max-w-2xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Edit profile</div>
      <div class="text-sm text-slate-400 mt-1">This is what customers will see.</div>

      <form method="post" action="/pro/profile" class="mt-6 space-y-4">
        <div>
          <label class="text-sm text-slate-300">Display name</label>
          <input name="display_name" value="{esc(prof.display_name)}" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Location</label>
          <input name="location" value="{esc(prof.location)}" placeholder="Chelsea, OK or Remote" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Tags (comma separated)</label>
          <input name="tags" value="{esc(prof.tags)}" placeholder="trainer,therapist,coach" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Timezone</label>
          <input name="timezone" value="{esc(prof.timezone)}" placeholder="America/Chicago" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
          <div class="text-xs text-slate-500 mt-1">Use IANA TZ like America/Chicago, America/Los_Angeles, etc.</div>
        </div>
        <div>
          <label class="text-sm text-slate-300">Bio</label>
          <textarea name="bio" rows="5" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3">{esc(prof.bio)}</textarea>
        </div>

        <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Save</button>
      </form>
    </div>
    """
    return html_page("Edit Profile", pro_user, body)


@app.post("/pro/profile")
def pro_profile_save(
    display_name: str = Form(...),
    bio: str = Form(""),
    tags: str = Form(""),
    location: str = Form(""),
    timezone_str: str = Form("America/Chicago", alias="timezone"),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    # Validate timezone
    try:
        ZoneInfo(timezone_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timezone string (IANA format).")

    prof.display_name = display_name.strip()[:80]
    prof.bio = bio.strip()[:2000]
    prof.tags = tags.strip()[:200]
    prof.location = location.strip()[:120]
    prof.timezone = timezone_str.strip()[:64]

    session.add(prof)
    session.commit()
    return RedirectResponse("/pro/dashboard", status_code=303)


@app.get("/pro/service/new", response_class=HTMLResponse)
def pro_service_new_form(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    body = """
    <div class="max-w-2xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Add a service</div>
      <div class="text-sm text-slate-400 mt-1">Duration, pricing, and waiver settings.</div>

      <form method="post" action="/pro/service/new" class="mt-6 space-y-4">
        <div>
          <label class="text-sm text-slate-300">Service name</label>
          <input name="name" required placeholder="1:1 Training Session" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Description</label>
          <textarea name="description" rows="4" placeholder="What’s included, who it’s for, etc."
            class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"></textarea>
        </div>
        <div class="grid md:grid-cols-4 gap-3">
          <div>
            <label class="text-sm text-slate-300">Duration (min)</label>
            <input name="duration_min" value="60" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
          </div>
          <div>
            <label class="text-sm text-slate-300">Buffer (min)</label>
            <input name="buffer_min" value="0" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
          </div>
          <div class="md:col-span-2">
            <label class="text-sm text-slate-300">Price (USD)</label>
            <input name="price_usd" value="50" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3"/>
          </div>
        </div>

        <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div class="flex items-center gap-2">
            <input id="rw" name="require_waiver" type="checkbox" checked class="w-4 h-4"/>
            <label for="rw" class="text-sm text-slate-200 font-semibold">Require waiver acceptance</label>
          </div>
          <label class="text-sm text-slate-300 block mt-3">Waiver text</label>
          <textarea name="waiver_text" rows="5" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3">I acknowledge the risks and consent to the service.</textarea>
        </div>

        <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Create service</button>
      </form>
    </div>
    """
    return html_page("New Service", pro_user, body)


@app.post("/pro/service/new")
def pro_service_new(
    name: str = Form(...),
    description: str = Form(""),
    duration_min: int = Form(60),
    buffer_min: int = Form(0),
    price_usd: float = Form(50.0),
    require_waiver: Optional[str] = Form(None),
    waiver_text: str = Form("I acknowledge the risks and consent to the service."),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    if duration_min < 15 or duration_min > 240:
        raise HTTPException(status_code=400, detail="Duration must be between 15 and 240 minutes.")
    if buffer_min < 0 or buffer_min > 60:
        raise HTTPException(status_code=400, detail="Buffer must be between 0 and 60 minutes.")
    if price_usd < 0 or price_usd > 5000:
        raise HTTPException(status_code=400, detail="Price looks invalid.")

    svc = Service(
        professional_id=prof.id,
        name=name.strip()[:80],
        description=description.strip()[:2000],
        duration_min=int(duration_min),
        buffer_min=int(buffer_min),
        price_cents=int(round(price_usd * 100)),
        require_waiver=bool(require_waiver),
        waiver_text=waiver_text.strip()[:4000],
    )
    session.add(svc)
    session.commit()
    return RedirectResponse("/pro/dashboard", status_code=303)


@app.get("/pro/availability", response_class=HTMLResponse)
def pro_availability(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    body = f"""
    <div class="max-w-3xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Availability exceptions</div>
      <div class="text-sm text-slate-400 mt-1">Block time off or add extra availability for a specific date. (Timezone: {esc(prof.timezone)})</div>

      <form method="post" action="/pro/exception/new" class="mt-6 grid md:grid-cols-5 gap-2 items-end">
        <div>
          <label class="text-sm text-slate-300">Date</label>
          <input name="on_date" type="date" required class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">Start</label>
          <input name="start_hhmm" placeholder="12:00" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
        </div>
        <div>
          <label class="text-sm text-slate-300">End</label>
          <input name="end_hhmm" placeholder="14:00" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
        </div>
        <div class="md:col-span-1">
          <label class="text-sm text-slate-300">Type</label>
          <select name="available" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3">
            <option value="false">Block</option>
            <option value="true">Add</option>
          </select>
        </div>
        <button class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Save</button>
      </form>

      <div class="mt-6">
        <div class="font-semibold">Existing exceptions</div>
        <div class="mt-3 space-y-2">
          {render_exceptions(session, prof.id)}
        </div>
      </div>

      <div class="mt-6">
        <a class="text-indigo-300 hover:underline" href="/pro/dashboard">← Back to dashboard</a>
      </div>
    </div>
    """
    return html_page("Availability", pro_user, body)


@app.post("/pro/rule/new")
def pro_rule_new(
    weekday: int = Form(...),
    start_hhmm: str = Form(...),
    end_hhmm: str = Form(...),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")
    try:
        parse_hhmm(start_hhmm.strip())
        parse_hhmm(end_hhmm.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Times must be HH:MM.")

    if weekday < 0 or weekday > 6:
        raise HTTPException(status_code=400, detail="Invalid weekday.")

    rule = AvailabilityRule(
        professional_id=prof.id,
        weekday=int(weekday),
        start_hhmm=start_hhmm.strip(),
        end_hhmm=end_hhmm.strip(),
    )
    session.add(rule)
    session.commit()
    return RedirectResponse("/pro/dashboard", status_code=303)


@app.post("/pro/rule/{rule_id}/delete")
def pro_rule_delete(
    rule_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    rule = session.get(AvailabilityRule, rule_id)
    if not prof or not rule or rule.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Rule not found.")
    session.delete(rule)
    session.commit()
    return RedirectResponse("/pro/dashboard", status_code=303)


@app.post("/pro/exception/new")
def pro_exception_new(
    on_date: str = Form(...),
    start_hhmm: str = Form(...),
    end_hhmm: str = Form(...),
    available: str = Form("false"),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    try:
        d = date.fromisoformat(on_date)
        parse_hhmm(start_hhmm.strip())
        parse_hhmm(end_hhmm.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date/time format.")

    ex = AvailabilityException(
        professional_id=prof.id,
        on_date=d,
        start_hhmm=start_hhmm.strip(),
        end_hhmm=end_hhmm.strip(),
        available=(available == "true"),
    )
    session.add(ex)
    session.commit()
    return RedirectResponse("/pro/availability", status_code=303)


@app.post("/pro/appointment/{appointment_id}/complete")
def pro_mark_complete(
    appointment_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    appt = session.get(Appointment, appointment_id)
    if not prof or not appt or appt.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    appt.status = AppointmentStatus.COMPLETED
    session.add(appt)
    session.commit()
    return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)


# ----------------------------
# Routes: Profile + booking
# ----------------------------

def profile_rating(session: Session, profile_id: int) -> Tuple[float, int]:
    reviews = session.exec(select(Review).where(Review.profile_id == profile_id)).all()
    if not reviews:
        return (0.0, 0)
    avg = sum(r.rating for r in reviews) / len(reviews)
    return (avg, len(reviews))


def render_exceptions(session: Session, prof_id: int) -> str:
    exs = session.exec(
        select(AvailabilityException).where(AvailabilityException.professional_id == prof_id).order_by(AvailabilityException.on_date.desc())
    ).all()
    if not exs:
        return '<div class="text-sm text-slate-400">No exceptions.</div>'
    rows = []
    for ex in exs[:50]:
        rows.append(
            f"""
            <div class="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <span class="font-semibold">{ex.on_date.isoformat()}</span>
                <span class="text-slate-300"> {esc(ex.start_hhmm)}–{esc(ex.end_hhmm)}</span>
                <span class="ml-2 px-2 py-1 rounded-full bg-white/10 text-xs">{'ADD' if ex.available else 'BLOCK'}</span>
              </div>
              <form method="post" action="/pro/exception/{ex.id}/delete">
                <button class="text-sm text-rose-300 hover:underline">Delete</button>
              </form>
            </div>
            """
        )
    return "".join(rows)


@app.post("/pro/exception/{ex_id}/delete")
def pro_exception_delete(
    ex_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == pro_user.id)).first()
    ex = session.get(AvailabilityException, ex_id)
    if not prof or not ex or ex.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Exception not found.")
    session.delete(ex)
    session.commit()
    return RedirectResponse("/pro/availability", status_code=303)


@app.get("/p/{profile_id}", response_class=HTMLResponse)
def profile_page(
    request: Request,
    profile_id: int,
    day: Optional[str] = None,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    prof = session.get(ProfessionalProfile, profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found.")

    services = session.exec(select(Service).where(Service.professional_id == prof.id)).all()
    rules = session.exec(select(AvailabilityRule).where(AvailabilityRule.professional_id == prof.id)).all()

    # Day selection
    tz = ZoneInfo(prof.timezone)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    try:
        chosen_day = date.fromisoformat(day) if day else today_local
    except Exception:
        chosen_day = today_local

    # Exceptions on/around day
    exceptions = session.exec(
        select(AvailabilityException).where(AvailabilityException.professional_id == prof.id).where(AvailabilityException.on_date == chosen_day)
    ).all()

    # Comments and reviews
    comments = session.exec(
        select(ProfileComment).where(ProfileComment.profile_id == prof.id).order_by(ProfileComment.created_at.desc())
    ).all()
    reviews = session.exec(
        select(Review).where(Review.profile_id == prof.id).order_by(Review.created_at.desc())
    ).all()
    avg, count = profile_rating(session, prof.id)

    # Build a booking module per service (show slots for selected day)
    service_blocks = []
    for svc in services:
        # existing appointments for prof around selected day in UTC
        day_start_local = local_dt(chosen_day, time(0, 0), tz)
        day_end_local = day_start_local + timedelta(days=1)
        day_start_utc = to_utc(day_start_local)
        day_end_utc = to_utc(day_end_local)

        appts = session.exec(
            select(Appointment)
            .where(Appointment.professional_id == prof.id)
            .where(Appointment.status != AppointmentStatus.CANCELED)
            .where(Appointment.start_at_utc < day_end_utc)
            .where(Appointment.end_at_utc > day_start_utc)
        ).all()
        occupied = [(a.start_at_utc, a.end_at_utc) for a in appts]

        slots = generate_slots_for_day(
            prof_tz=tz,
            day=chosen_day,
            rules=rules,
            exceptions=exceptions,
            service_duration_min=svc.duration_min,
            buffer_min=svc.buffer_min,
            existing_appointments=occupied,
        )

        # Render slots (in local time)
        if slots:
            slot_buttons = []
            for st_utc in slots[:40]:
                st_local = st_utc.astimezone(tz)
                label = st_local.strftime("%I:%M %p").lstrip("0")
                slot_buttons.append(
                    f"""
                    <a href="/book/{svc.id}?day={chosen_day.isoformat()}&start={st_utc.isoformat()}"
                       class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-sm">
                      {label}
                    </a>
                    """
                )
            slot_html = f'<div class="mt-3 flex flex-wrap gap-2">{"".join(slot_buttons)}</div>'
        else:
            slot_html = '<div class="mt-3 text-sm text-slate-400">No slots available this day.</div>'

        service_blocks.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="text-lg font-semibold">{esc(svc.name)}</div>
                  <div class="text-sm text-slate-400 mt-1">{esc(svc.description)}</div>
                  <div class="text-sm text-slate-300 mt-2">{fmt_money(svc.price_cents)} • {svc.duration_min}m • buffer {svc.buffer_min}m</div>
                </div>
                <div class="text-right text-sm">
                  <div>{'Waiver required' if svc.require_waiver else 'No waiver'}</div>
                </div>
              </div>
              {slot_html}
            </div>
            """
        )

    # Render comments
    comment_cards = []
    for c in comments[:40]:
        u = session.get(User, c.user_id)
        comment_cards.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div class="text-sm text-slate-400">{esc(u.email if u else 'User')} • {c.created_at.astimezone(timezone.utc).strftime('%b %d, %Y')}</div>
              <div class="mt-2 text-slate-200">{esc(c.body)}</div>
            </div>
            """
        )

    # Render reviews
    review_cards = []
    for r in reviews[:40]:
        u = session.get(User, r.user_id)
        review_cards.append(
            f"""
            <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-center justify-between">
                <div class="text-sm text-slate-400">{esc(u.email if u else 'User')} • {r.created_at.astimezone(timezone.utc).strftime('%b %d, %Y')}</div>
                <div>{star_row(r.rating)}</div>
              </div>
              <div class="mt-2 text-slate-200">{esc(r.body)}</div>
            </div>
            """
        )

    tags = [t.strip() for t in (prof.tags or "").split(",") if t.strip()]
    message_cta = ""
    if user and user.id != prof.user_id:
        message_cta = f"""
        <form method="post" action="/thread/start">
          <input type="hidden" name="profile_id" value="{prof.id}"/>
          <button class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Message</button>
        </form>
        """

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-2xl font-bold">{esc(prof.display_name)}</div>
          <div class="mt-2 text-sm text-slate-400">{esc(prof.location or "Remote / Not listed")} • {esc(prof.timezone)}</div>

          <div class="mt-3">{star_row(avg)} <span class="text-sm text-slate-400">({count} reviews)</span></div>

          <div class="mt-4 flex flex-wrap gap-2">
            {''.join(pill(t) for t in tags[:10])}
          </div>

          <div class="mt-4 text-slate-200 text-sm whitespace-pre-wrap">{esc(prof.bio or "")}</div>

          <div class="mt-5 flex gap-2">
            {message_cta if user else '<a href="/login" class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Log in to message</a>'}
            <a href="/" class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Back</a>
          </div>
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold text-lg">Pick a day</div>
          <form method="get" action="/p/{prof.id}" class="mt-3 flex gap-2 items-end">
            <input type="date" name="day" value="{chosen_day.isoformat()}" class="w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
            <button class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Go</button>
          </form>
          <div class="text-xs text-slate-500 mt-2">Slots shown in {esc(prof.timezone)}.</div>
        </div>
      </div>

      <div class="lg:col-span-8 space-y-4">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-xl font-bold">Services & availability</div>
          <div class="mt-4 space-y-4">
            {''.join(service_blocks) if service_blocks else '<div class="text-slate-400">No services yet.</div>'}
          </div>
        </div>

        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-xl font-bold">Reviews</div>
          <div class="mt-4 space-y-3">
            {''.join(review_cards) if review_cards else '<div class="text-slate-400">No reviews yet.</div>'}
          </div>
        </div>

        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="flex items-center justify-between">
            <div class="text-xl font-bold">Comments</div>
          </div>

          {"<form method='post' action='/comment' class='mt-4 flex gap-2'><input type='hidden' name='profile_id' value='"+str(prof.id)+"'/><input name='body' placeholder='Leave a comment…' class='flex-1 rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3'/><button class='rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold'>Post</button></form>" if user else "<div class='mt-3 text-sm text-slate-400'>Log in to comment.</div>"}

          <div class="mt-4 space-y-3">
            {''.join(comment_cards) if comment_cards else '<div class="text-slate-400">No comments yet.</div>'}
          </div>
        </div>
      </div>
    </div>
    """
    return html_page(prof.display_name, user, body)


@app.post("/comment")
def post_comment(
    profile_id: int = Form(...),
    body: str = Form(...),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    prof = session.get(ProfessionalProfile, profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found.")
    text = body.strip()
    if len(text) < 2 or len(text) > 500:
        raise HTTPException(status_code=400, detail="Comment must be 2–500 chars.")
    c = ProfileComment(profile_id=profile_id, user_id=user.id, body=text)
    session.add(c)
    session.commit()
    return RedirectResponse(f"/p/{profile_id}", status_code=303)


@app.get("/book/{service_id}", response_class=HTMLResponse)
def book_form(
    request: Request,
    service_id: int,
    day: str,
    start: str,  # ISO datetime in UTC
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    svc = session.get(Service, service_id)
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found.")
    prof = session.get(ProfessionalProfile, svc.professional_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Pro not found.")

    if user.id == prof.user_id:
        raise HTTPException(status_code=400, detail="You cannot book yourself.")

    try:
        start_utc = datetime.fromisoformat(start)
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        start_utc = start_utc.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start time.")

    end_utc = start_utc + timedelta(minutes=svc.duration_min)

    tz = ZoneInfo(prof.timezone)
    start_local = start_utc.astimezone(tz)

    waiver_block = ""
    if svc.require_waiver:
        waiver_block = f"""
        <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div class="font-semibold">Waiver (required)</div>
          <div class="text-sm text-slate-300 mt-2 whitespace-pre-wrap">{esc(svc.waiver_text)}</div>
          <div class="mt-3 flex items-center gap-2">
            <input id="wa" name="accept_waiver" type="checkbox" class="w-4 h-4" required/>
            <label for="wa" class="text-sm text-slate-200">I accept the waiver terms.</label>
          </div>
        </div>
        """

    body = f"""
    <div class="max-w-2xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Confirm booking</div>
      <div class="text-sm text-slate-400 mt-1">{esc(prof.display_name)} • {esc(svc.name)}</div>

      <div class="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
        <div class="text-slate-300 text-sm">When</div>
        <div class="text-lg font-semibold mt-1">
          {start_local.strftime('%A, %b %d, %Y • %I:%M %p').lstrip('0')} ({esc(prof.timezone)})
        </div>
        <div class="mt-2 text-slate-300 text-sm">Price: <span class="font-semibold">{fmt_money(svc.price_cents)}</span></div>
      </div>

      <form method="post" action="/book/{svc.id}" class="mt-4 space-y-4">
        <input type="hidden" name="start_utc" value="{esc(start_utc.isoformat())}"/>
        <input type="hidden" name="service_id" value="{svc.id}"/>
        {waiver_block}
        <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
          <label class="text-sm text-slate-300">Notes (optional)</label>
          <textarea name="notes" rows="3" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3" placeholder="Any context you want the pro to know..."></textarea>
        </div>

        <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Create booking</button>
        <a class="block text-center text-sm text-slate-400 hover:underline" href="/p/{prof.id}?day={esc(day)}">Cancel</a>
      </form>
    </div>
    """
    return html_page("Confirm booking", user, body)


@app.post("/book/{service_id}")
def book_create(
    service_id: int,
    start_utc: str = Form(...),
    accept_waiver: Optional[str] = Form(None),
    notes: str = Form(""),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    svc = session.get(Service, service_id)
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found.")
    prof = session.get(ProfessionalProfile, svc.professional_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Pro not found.")

    try:
        st = datetime.fromisoformat(start_utc)
        if st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)
        st = st.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start time.")

    en = st + timedelta(minutes=svc.duration_min + svc.buffer_min)

    # Collision check (include buffer)
    existing = session.exec(
        select(Appointment)
        .where(Appointment.professional_id == prof.id)
        .where(Appointment.status != AppointmentStatus.CANCELED)
        .where(Appointment.start_at_utc < en)
        .where(Appointment.end_at_utc > st)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="That time just got booked. Pick another slot.")

    waiver_ts = None
    if svc.require_waiver:
        if not accept_waiver:
            raise HTTPException(status_code=400, detail="Waiver acceptance is required.")
        waiver_ts = datetime.now(timezone.utc)

    appt = Appointment(
        service_id=svc.id,
        professional_id=prof.id,
        customer_id=user.id,
        start_at_utc=st,
        end_at_utc=(st + timedelta(minutes=svc.duration_min + svc.buffer_min)),
        status=AppointmentStatus.PENDING,
        paid=False,
        waiver_accepted_at_utc=waiver_ts,
    )
    session.add(appt)
    session.commit()
    session.refresh(appt)

    pay = Payment(appointment_id=appt.id, amount_cents=svc.price_cents, status="mock_created")
    session.add(pay)
    session.commit()

    return RedirectResponse(f"/pay/{appt.id}", status_code=303)


@app.get("/pay/{appointment_id}", response_class=HTMLResponse)
def pay_page(
    request: Request,
    appointment_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt or appt.customer_id != user.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    svc = session.get(Service, appt.service_id)
    prof = session.get(ProfessionalProfile, appt.professional_id)
    pay = session.exec(select(Payment).where(Payment.appointment_id == appt.id)).first()

    tz = ZoneInfo(prof.timezone if prof else "UTC")
    st_local = appt.start_at_utc.astimezone(tz)

    status_line = f"<div class='text-slate-300 text-sm'>Payment status: <span class='font-semibold'>{esc(pay.status if pay else 'none')}</span></div>"

    pay_btn = ""
    if not appt.paid:
        pay_btn = """
        <form method="post" action="/pay/mock" class="mt-4">
          <input type="hidden" name="appointment_id" value="{appt_id}"/>
          <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Pay now (mock)</button>
        </form>
        """.replace("{appt_id}", str(appt.id))
    else:
        pay_btn = f"""
        <div class="mt-4 rounded-2xl border border-white/10 bg-emerald-500/10 p-4">
          <div class="font-semibold text-emerald-200">Paid ✓</div>
          <div class="text-sm text-slate-300 mt-1">You’re confirmed (assuming waiver is satisfied).</div>
        </div>
        """

    body = f"""
    <div class="max-w-2xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Payment</div>
      <div class="text-sm text-slate-400 mt-1">{esc(prof.display_name if prof else 'Pro')} • {esc(svc.name if svc else 'Service')}</div>

      <div class="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4">
        <div class="text-slate-300 text-sm">When</div>
        <div class="text-lg font-semibold mt-1">{st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} ({esc(prof.timezone if prof else 'UTC')})</div>
        <div class="mt-2 text-slate-300 text-sm">Amount: <span class="font-semibold">{fmt_money((svc.price_cents if svc else 0))}</span></div>
        <div class="mt-2">{status_line}</div>
      </div>

      {pay_btn}

      <div class="mt-4 flex gap-2">
        <a href="/appointment/{appt.id}" class="flex-1 text-center rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">View appointment</a>
        <a href="/p/{prof.id}" class="flex-1 text-center rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Back to profile</a>
      </div>
    </div>
    """
    return html_page("Payment", user, body)


@app.post("/pay/mock")
def mock_pay(
    appointment_id: int = Form(...),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt or appt.customer_id != user.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    svc = session.get(Service, appt.service_id)
    pay = session.exec(select(Payment).where(Payment.appointment_id == appt.id)).first()
    if not pay:
        pay = Payment(appointment_id=appt.id, amount_cents=(svc.price_cents if svc else 0), status="mock_created")

    appt.paid = True
    pay.status = "succeeded"

    # If waiver required, confirm only if accepted
    if svc and svc.require_waiver and not appt.waiver_accepted_at_utc:
        appt.status = AppointmentStatus.PENDING
    else:
        appt.status = AppointmentStatus.CONFIRMED

    session.add(appt)
    session.add(pay)
    session.commit()
    return RedirectResponse(f"/appointment/{appt.id}", status_code=303)


@app.get("/appointment/{appointment_id}", response_class=HTMLResponse)
def appointment_page(
    request: Request,
    appointment_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    svc = session.get(Service, appt.service_id)
    prof = session.get(ProfessionalProfile, appt.professional_id)
    cust = session.get(User, appt.customer_id)
    pay = session.exec(select(Payment).where(Payment.appointment_id == appt.id)).first()

    # Access: customer or professional owner
    is_pro_owner = False
    if user.role == Role.PRO:
        my_prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()
        is_pro_owner = bool(my_prof and prof and my_prof.id == prof.id)

    if user.id != appt.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    tz = ZoneInfo(prof.timezone if prof else "UTC")
    st_local = appt.start_at_utc.astimezone(tz)
    en_local = appt.end_at_utc.astimezone(tz)

    docs = session.exec(select(AppointmentDocument).where(AppointmentDocument.appointment_id == appt.id)).all()
    doc_list = []
    for d in docs:
        uploader = session.get(User, d.uploaded_by_user_id)
        doc_list.append(
            f"""
            <div class="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <div class="font-semibold">{esc(d.filename)}</div>
                <div class="text-slate-400 text-xs">Uploaded by {esc(uploader.email if uploader else 'User')} • {d.created_at.strftime('%b %d, %Y')}</div>
              </div>
              <a class="text-sm text-indigo-300 hover:underline" href="/uploads/{esc(Path(d.path).name)}" target="_blank">Open</a>
            </div>
            """
        )

    cancel_btn = ""
    if user.id == appt.customer_id and appt.status in [AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED]:
        cancel_btn = f"""
        <form method="post" action="/appointment/{appt.id}/cancel">
          <button class="rounded-xl bg-rose-500/90 hover:bg-rose-400 px-4 py-3 font-semibold">Cancel</button>
        </form>
        """

    review_block = ""
    # Review allowed only after completion and only by customer and only once
    existing_review = session.exec(select(Review).where(Review.appointment_id == appt.id)).first()
    if user.id == appt.customer_id and appt.status == AppointmentStatus.COMPLETED and not existing_review:
        review_block = f"""
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6 mt-4">
          <div class="text-xl font-bold">Leave a review</div>
          <form method="post" action="/review" class="mt-4 space-y-3">
            <input type="hidden" name="appointment_id" value="{appt.id}"/>
            <div>
              <label class="text-sm text-slate-300">Rating</label>
              <select name="rating" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3">
                <option value="5">5 - Excellent</option>
                <option value="4">4 - Great</option>
                <option value="3">3 - OK</option>
                <option value="2">2 - Not great</option>
                <option value="1">1 - Bad</option>
              </select>
            </div>
            <div>
              <label class="text-sm text-slate-300">Review</label>
              <textarea name="body" rows="4" class="mt-1 w-full rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3" placeholder="What was your experience like?"></textarea>
            </div>
            <button class="w-full rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Post review</button>
          </form>
        </div>
        """
    elif existing_review:
        review_block = f"""
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6 mt-4">
          <div class="text-xl font-bold">Your review</div>
          <div class="mt-3">{star_row(existing_review.rating)}</div>
          <div class="mt-2 text-slate-200">{esc(existing_review.body)}</div>
        </div>
        """

    upload_block = f"""
    <div class="rounded-3xl border border-white/10 bg-white/5 p-6 mt-4">
      <div class="text-xl font-bold">Documents</div>
      <div class="text-sm text-slate-400 mt-1">Upload any files needed before the appointment.</div>

      <form method="post" action="/appointment/{appt.id}/upload" enctype="multipart/form-data" class="mt-4 flex gap-2 items-end">
        <input type="file" name="file" required class="flex-1 rounded-xl bg-slate-900/60 border border-white/10 px-3 py-3"/>
        <button class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold">Upload</button>
      </form>

      <div class="mt-4 space-y-2">
        {''.join(doc_list) if doc_list else '<div class="text-slate-400 text-sm">No documents uploaded.</div>'}
      </div>
    </div>
    """

    body = f"""
    <div class="max-w-3xl mx-auto">
      <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
        <div class="text-2xl font-bold">Appointment</div>
        <div class="text-sm text-slate-400 mt-1">{esc(svc.name if svc else 'Service')} • {esc(prof.display_name if prof else 'Pro')}</div>

        <div class="mt-4 grid md:grid-cols-2 gap-4">
          <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
            <div class="text-sm text-slate-400">Time</div>
            <div class="font-semibold mt-1">{st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} → {en_local.strftime('%I:%M %p').lstrip('0')}</div>
            <div class="text-sm text-slate-300 mt-1">{esc(prof.timezone if prof else 'UTC')}</div>
          </div>

          <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
            <div class="text-sm text-slate-400">Status</div>
            <div class="mt-1">
              <span class="px-2 py-1 rounded-full bg-white/10 text-xs">{esc(appt.status)}</span>
              <span class="ml-2 text-slate-300">{'Paid' if appt.paid else 'Unpaid'}</span>
            </div>
            <div class="text-sm text-slate-300 mt-2">Customer: {esc(cust.email if cust else 'Unknown')}</div>
            <div class="text-sm text-slate-300 mt-1">Payment: {esc(pay.status if pay else 'none')}</div>
          </div>
        </div>

        <div class="mt-5 flex gap-2">
          <a class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold" href="/p/{prof.id}">Pro profile</a>
          {'<a class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold" href="/pay/'+str(appt.id)+'">Pay</a>' if (user.id==appt.customer_id and not appt.paid) else ''}
          {cancel_btn}
        </div>
      </div>

      {upload_block}
      {review_block}
    </div>
    """
    return html_page("Appointment", user, body)


@app.post("/appointment/{appointment_id}/upload")
async def upload_doc(
    appointment_id: int,
    file: UploadFile = File(...),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Access: customer or pro owner
    is_pro_owner = False
    if user.role == Role.PRO:
        my_prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()
        is_pro_owner = bool(my_prof and my_prof.id == appt.professional_id)
    if user.id != appt.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    # Save file locally (demo)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", file.filename or "upload.bin")[:120]
    token = secrets.token_hex(8)
    out_path = UPLOAD_DIR / f"{appointment_id}_{token}_{safe_name}"
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB).")
    out_path.write_bytes(data)

    doc = AppointmentDocument(
        appointment_id=appointment_id,
        uploaded_by_user_id=user.id,
        filename=safe_name,
        path=str(out_path),
    )
    session.add(doc)
    session.commit()
    return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)


@app.post("/appointment/{appointment_id}/cancel")
def cancel_appointment(
    appointment_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt or appt.customer_id != user.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    if appt.status in [AppointmentStatus.CANCELED, AppointmentStatus.COMPLETED]:
        return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)
    appt.status = AppointmentStatus.CANCELED
    session.add(appt)
    session.commit()
    return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)


@app.post("/review")
def post_review(
    appointment_id: int = Form(...),
    rating: int = Form(...),
    body: str = Form(""),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt or appt.customer_id != user.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    if appt.status != AppointmentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="You can only review after completion.")
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be 1–5.")
    existing = session.exec(select(Review).where(Review.appointment_id == appt.id)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Review already exists for this appointment.")

    review = Review(
        appointment_id=appt.id,
        profile_id=appt.professional_id,
        user_id=user.id,
        rating=int(rating),
        body=body.strip()[:1200],
    )
    session.add(review)
    session.commit()
    return RedirectResponse(f"/p/{appt.professional_id}", status_code=303)


# ----------------------------
# Routes: Messaging
# ----------------------------

def get_or_create_thread(session: Session, profile_id: int, customer_id: int) -> Thread:
    existing = session.exec(
        select(Thread).where(Thread.professional_id == profile_id).where(Thread.customer_id == customer_id)
    ).first()
    if existing:
        return existing
    t = Thread(professional_id=profile_id, customer_id=customer_id)
    session.add(t)
    session.commit()
    session.refresh(t)
    return t


@app.post("/thread/start")
def thread_start(
    profile_id: int = Form(...),
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    prof = session.get(ProfessionalProfile, profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found.")
    if user.id == prof.user_id:
        raise HTTPException(status_code=400, detail="You cannot message yourself.")
    thread = get_or_create_thread(session, profile_id=profile_id, customer_id=user.id)
    return RedirectResponse(f"/thread/{thread.id}", status_code=303)


@app.get("/inbox", response_class=HTMLResponse)
def inbox(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    threads: List[Thread] = []
    if user.role == Role.PRO:
        my_prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()
        if my_prof:
            threads = session.exec(
                select(Thread).where(Thread.professional_id == my_prof.id).order_by(Thread.last_message_at.desc())
            ).all()
    else:
        threads = session.exec(
            select(Thread).where(Thread.customer_id == user.id).order_by(Thread.last_message_at.desc())
        ).all()

    cards = []
    for t in threads[:50]:
        prof = session.get(ProfessionalProfile, t.professional_id)
        cust = session.get(User, t.customer_id)
        counterpart = prof.display_name if (user.role != Role.PRO) else (cust.email if cust else "Customer")
        subtitle = (cust.email if cust else "") if user.role != Role.PRO else (prof.display_name if prof else "Pro")
        cards.append(
            f"""
            <a href="/thread/{t.id}" class="block rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 p-4">
              <div class="font-semibold">{esc(counterpart)}</div>
              <div class="text-sm text-slate-400 mt-1">Last message: {t.last_message_at.strftime('%b %d, %Y')}</div>
            </a>
            """
        )

    body = f"""
    <div class="max-w-3xl mx-auto rounded-3xl border border-white/10 bg-white/5 p-6">
      <div class="text-2xl font-bold">Inbox</div>
      <div class="mt-4 space-y-3">
        {''.join(cards) if cards else '<div class="text-slate-400">No conversations yet.</div>'}
      </div>
    </div>
    """
    return html_page("Inbox", user, body)


@app.get("/thread/{thread_id}", response_class=HTMLResponse)
def thread_page(
    request: Request,
    thread_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    thread = session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")

    prof = session.get(ProfessionalProfile, thread.professional_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Pro not found.")

    # Access rules
    is_pro_owner = False
    if user.role == Role.PRO:
        my_prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()
        is_pro_owner = bool(my_prof and my_prof.id == prof.id)
    if user.id != thread.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    messages = session.exec(
        select(Message).where(Message.thread_id == thread.id).order_by(Message.created_at.asc())
    ).all()

    bubbles = []
    for m in messages[-100:]:
        mine = (m.sender_id == user.id)
        align = "justify-end" if mine else "justify-start"
        bg = "bg-indigo-500/30 border-indigo-400/30" if mine else "bg-white/5 border-white/10"
        bubbles.append(
            f"""
            <div class="flex {align}">
              <div class="max-w-[80%] rounded-2xl border {bg} px-4 py-3">
                <div class="text-sm text-slate-200 whitespace-pre-wrap">{esc(m.body)}</div>
                <div class="text-xs text-slate-400 mt-1">{m.created_at.strftime('%I:%M %p').lstrip('0')}</div>
              </div>
            </div>
            """
        )

    title = prof.display_name if user.role != Role.PRO else (session.get(User, thread.customer_id).email if session.get(User, thread.customer_id) else "Customer")

    body = f"""
    <div class="max-w-4xl mx-auto grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="text-xl font-bold">Chat</div>
          <div class="text-sm text-slate-400 mt-1">With: {esc(title)}</div>
          <div class="mt-4 flex gap-2">
            <a class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold" href="/inbox">Back</a>
            <a class="rounded-xl bg-white/10 hover:bg-white/15 px-4 py-3 font-semibold" href="/p/{prof.id}">Profile</a>
          </div>
        </div>

        <div class="mt-4 rounded-3xl border border-white/10 bg-white/5 p-6">
          <div class="font-semibold">Live updates</div>
          <div class="text-sm text-slate-400 mt-1">This thread uses WebSockets for instant messages.</div>
        </div>
      </div>

      <div class="lg:col-span-8">
        <div class="rounded-3xl border border-white/10 bg-white/5 p-6">
          <div id="messages" class="space-y-3 max-h-[60vh] overflow-auto pr-2">
            {''.join(bubbles) if bubbles else '<div class="text-slate-400">Say hi 👋</div>'}
          </div>

          <form id="sendForm" class="mt-4 flex gap-2">
            <input id="msgInput" placeholder="Message..." class="flex-1 rounded-xl bg-slate-900/60 border border-white/10 px-4 py-3 outline-none focus:border-indigo-400"/>
            <button class="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-3 font-semibold">Send</button>
          </form>

          <script>
            const threadId = {thread.id};
            const wsProto = (location.protocol === "https:") ? "wss" : "ws";
            const ws = new WebSocket(`${{wsProto}}://${{location.host}}/ws/thread/${{threadId}}`);

            const messagesEl = document.getElementById("messages");
            const form = document.getElementById("sendForm");
            const input = document.getElementById("msgInput");

            function appendBubble({{mine, body, ts}}) {{
              const wrap = document.createElement("div");
              wrap.className = `flex ${{mine ? "justify-end" : "justify-start"}}`;
              wrap.innerHTML = `
                <div class="max-w-[80%] rounded-2xl border ${{mine ? "bg-indigo-500/30 border-indigo-400/30" : "bg-white/5 border-white/10"}} px-4 py-3">
                  <div class="text-sm text-slate-200 whitespace-pre-wrap"></div>
                  <div class="text-xs text-slate-400 mt-1"></div>
                </div>
              `;
              wrap.querySelector("div div").textContent = body;
              wrap.querySelector("div div + div").textContent = ts;
              messagesEl.appendChild(wrap);
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }}

            ws.onmessage = (ev) => {{
              const data = JSON.parse(ev.data);
              if (data.type === "msg") {{
                appendBubble({{
                  mine: data.sender_id === {user.id},
                  body: data.body,
                  ts: data.ts
                }});
              }}
            }};

            form.addEventListener("submit", (e) => {{
              e.preventDefault();
              const text = (input.value || "").trim();
              if (!text) return;
              ws.send(JSON.stringify({{ body: text }}));
              input.value = "";
            }});

            messagesEl.scrollTop = messagesEl.scrollHeight;
          </script>
        </div>
      </div>
    </div>
    """
    return html_page("Thread", user, body)


@app.websocket("/ws/thread/{thread_id}")
async def thread_ws(
    websocket: WebSocket,
    thread_id: int,
):
    # Cookie auth
    token = websocket.cookies.get(COOKIE_NAME)
    user_id = parse_token(token) if token else None
    if not user_id:
        await websocket.close(code=4401)
        return

    # Validate access
    with Session(engine) as session:
        thread = session.get(Thread, thread_id)
        if not thread:
            await websocket.close(code=4404)
            return

        user = session.get(User, user_id)
        if not user:
            await websocket.close(code=4401)
            return

        prof = session.get(ProfessionalProfile, thread.professional_id)
        if not prof:
            await websocket.close(code=4404)
            return

        is_pro_owner = False
        if user.role == Role.PRO:
            my_prof = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()
            is_pro_owner = bool(my_prof and my_prof.id == prof.id)

        if user.id != thread.customer_id and not is_pro_owner and user.role != Role.ADMIN:
            await websocket.close(code=4403)
            return

    await manager.connect(thread_id, websocket, user_id)

    try:
        while True:
            data = await websocket.receive_json()
            body = (data.get("body") or "").strip()
            if not body:
                continue
            if len(body) > 2000:
                body = body[:2000]

            with Session(engine) as session:
                thread = session.get(Thread, thread_id)
                if not thread:
                    continue

                msg = Message(thread_id=thread_id, sender_id=user_id, body=body)
                thread.last_message_at = datetime.now(timezone.utc)
                session.add(msg)
                session.add(thread)
                session.commit()
                session.refresh(msg)

                ts = msg.created_at.astimezone(timezone.utc).strftime("%I:%M %p").lstrip("0")

            await manager.broadcast(thread_id, {
                "type": "msg",
                "thread_id": thread_id,
                "sender_id": user_id,
                "body": body,
                "ts": ts,
            })

    except WebSocketDisconnect:
        manager.disconnect(thread_id, websocket)


# ----------------------------
# Tiny “seed” helper (optional)
# ----------------------------

@app.get("/dev/seed", response_class=HTMLResponse)
def dev_seed(session: Session = Depends(get_session)):
    """
    Creates 2 demo pros (with services) if they don't exist.
    """
    def ensure_user(email: str, role: Role) -> User:
        u = session.exec(select(User).where(User.email == email)).first()
        if u:
            return u
        u = User(email=email, password_hash=hash_password("password123"), role=role)
        session.add(u)
        session.commit()
        session.refresh(u)
        return u

    pro1 = ensure_user("trainer@example.com", Role.PRO)
    pro2 = ensure_user("therapist@example.com", Role.PRO)

    def ensure_profile(u: User, name: str, tags: str) -> ProfessionalProfile:
        p = session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == u.id)).first()
        if p:
            return p
        p = ProfessionalProfile(user_id=u.id, display_name=name, bio="Demo profile — edit me in Pro Dashboard.", tags=tags, location="Remote", timezone="America/Chicago")
        session.add(p)
        session.commit()
        session.refresh(p)
        return p

    p1 = ensure_profile(pro1, "Alex Trainer", "trainer,fitness,strength")
    p2 = ensure_profile(pro2, "Morgan Therapist", "therapist,anxiety,coaching")

    def ensure_service(p: ProfessionalProfile, name: str, mins: int, price: int):
        s = session.exec(select(Service).where(Service.professional_id == p.id).where(Service.name == name)).first()
        if s:
            return
        s = Service(professional_id=p.id, name=name, description="Demo service", duration_min=mins, buffer_min=10, price_cents=price, require_waiver=True, waiver_text="Demo waiver: you agree.")
        session.add(s)
        session.commit()

    ensure_service(p1, "1:1 Training", 60, 7500)
    ensure_service(p2, "Therapy Session", 50, 12000)

    def ensure_rules(p: ProfessionalProfile):
        existing = session.exec(select(AvailabilityRule).where(AvailabilityRule.professional_id == p.id)).first()
        if existing:
            return
        for wd in range(0, 5):
            session.add(AvailabilityRule(professional_id=p.id, weekday=wd, start_hhmm="09:00", end_hhmm="12:00"))
            session.add(AvailabilityRule(professional_id=p.id, weekday=wd, start_hhmm="13:00", end_hhmm="17:00"))
        session.commit()

    ensure_rules(p1)
    ensure_rules(p2)

    return HTMLResponse("""
        <meta http-equiv="refresh" content="0; url=/" />
        Seeded. Go back home.
    """)


# ----------------------------
# Basic health
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME, "utc": datetime.now(timezone.utc).isoformat()}
