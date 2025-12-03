"""
Schedge — Scheduling marketplace with social layer (All-Python MVP)
------------------------------------------------------------------
Upgrades in this version:
- Brand rename: Schedge
- Black/Gold UI theme
- Root "/" is a Login/Signup portal (Customer or Pro)
- External calendar: ICS + Google + Outlook links
- Pro calendar controls: weekly rules + date exceptions + reschedule/cancel appointments
- Billing model scaffolding:
    - Pro subscription: $10/month (MOCK activate, DB-backed)
    - Platform fee: $1 per successful transaction (recorded as platform_fee_cents)

Run:
  pip install fastapi uvicorn sqlmodel sqlalchemy passlib[bcrypt] python-jose[cryptography] python-multipart pydantic
  uvicorn app:app --reload

Notes:
- SQLite by default; for persistence on hosts, set:
    SCHEDGE_DB_URL=sqlite:////var/data/schedge.db
- Uploads stored locally in UPLOAD_DIR (demo). Use S3 for production.
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
from urllib.parse import quote

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
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt
from passlib.context import CryptContext
from sqlmodel import Field, Session, SQLModel, create_engine, select
from zoneinfo import ZoneInfo


# ----------------------------
# Config / Branding
# ----------------------------

APP_NAME = "Schedge"

JWT_SECRET = os.environ.get("SCHEDGE_JWT_SECRET", os.environ.get("SCHEDULR_JWT_SECRET", "dev-secret-change-me"))
JWT_ALG = "HS256"
COOKIE_NAME = "schedge_token"
TOKEN_EXPIRE_DAYS = 7

DB_URL = os.environ.get("SCHEDGE_DB_URL", os.environ.get("SCHEDULR_DB_URL", "sqlite:///./schedge.db"))
engine = create_engine(DB_URL, echo=False)

UPLOAD_DIR = Path(os.environ.get("SCHEDGE_UPLOAD_DIR", os.environ.get("SCHEDULR_UPLOAD_DIR", "./uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Billing model
PRO_SUBSCRIPTION_PRICE_CENTS = 1000  # $10/mo
PLATFORM_TX_FEE_CENTS = 100          # $1 per successful transaction

# UI constants
GOLD = "#D4AF37"  # classic gold


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
    discoverable: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProSubscription(SQLModel, table=True):
    """
    MVP: DB-backed subscription status. In production:
      replace with Stripe subscription IDs + webhook-driven status.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True, unique=True)
    status: str = "inactive"  # inactive|active|past_due|canceled
    started_at_utc: Optional[datetime] = None
    current_period_end_utc: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Service(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    professional_id: int = Field(foreign_key="professionalprofile.id", index=True)
    name: str
    description: str = ""
    duration_min: int = 60
    buffer_min: int = 0
    slot_step_min: int = 15  # how granular slot starts are
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

    customer_notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appointment_id: int = Field(foreign_key="appointment.id", index=True, unique=True)
    amount_cents: int
    platform_fee_cents: int = PLATFORM_TX_FEE_CENTS
    pro_amount_cents: int = 0
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
    if user.role not in (Role.PRO, Role.ADMIN):
        raise HTTPException(status_code=403, detail="Professional access required.")
    return user


# ----------------------------
# UI helpers (black + gold)
# ----------------------------

def esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_money(cents: int) -> str:
    return f"${cents/100:,.2f}"


def pill(text: str) -> str:
    return f'<span class="px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-xs">{esc(text)}</span>'


def star_row(rating: float) -> str:
    full = int(round(rating))
    full = max(0, min(5, full))
    stars = "★" * full + "☆" * (5 - full)
    return f'<span class="tracking-wide" style="color:{GOLD}">{stars}</span>'


def button_primary(label: str, href: Optional[str] = None) -> str:
    if href:
        return f'<a href="{esc(href)}" class="btn btn-primary">{esc(label)}</a>'
    return f'<button class="btn btn-primary">{esc(label)}</button>'


def button_ghost(label: str, href: Optional[str] = None) -> str:
    if href:
        return f'<a href="{esc(href)}" class="btn btn-ghost">{esc(label)}</a>'
    return f'<button class="btn btn-ghost">{esc(label)}</button>'


def html_page(title: str, user: Optional[User], body: str) -> HTMLResponse:
    auth_block = (
        f"""
        <div class="flex items-center gap-2">
          <a class="navlink" href="/discover">Discover</a>
          <a class="navlink" href="/inbox">Inbox</a>
          <a class="navlink" href="/me">Me</a>
          <form method="post" action="/logout">
            <button class="navlink">Logout</button>
          </form>
        </div>
        """
        if user
        else """
        <div class="flex items-center gap-2">
          <a class="navlink" href="/">Login / Signup</a>
        </div>
        """
    )

    pro_cta = (
        '<a class="btn btn-primary" href="/pro/dashboard">Pro Dashboard</a>'
        if user and user.role == Role.PRO
        else '<a class="btn btn-primary" href="/?tab=signup&role=PRO">Become a Pro</a>'
    )

    css = f"""
:root {{
  --gold: {GOLD};
}}
html, body {{
  background: #050607;
  color: #e7e7e7;
}}
.gold {{
  color: var(--gold);
}}
.card {{
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 24px;
}}
.navlink {{
  padding: 0.55rem 0.85rem;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.02);
}}
.navlink:hover {{
  background: rgba(255,255,255,0.06);
}}
.btn {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: .5rem;
  padding: .75rem 1rem;
  border-radius: 16px;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.10);
  transition: transform .05s ease, background .15s ease, border-color .15s ease;
}}
.btn:active {{ transform: translateY(1px); }}
.btn-primary {{
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(212,175,55,0.25));
  color: #0b0b0b;
  border-color: rgba(212,175,55,0.45);
}}
.btn-primary:hover {{
  background: linear-gradient(135deg, rgba(212,175,55,1.0), rgba(212,175,55,0.30));
}}
.btn-ghost {{
  background: rgba(255,255,255,0.02);
}}
.btn-ghost:hover {{
  background: rgba(255,255,255,0.06);
}}
.input {{
  width: 100%;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  padding: .85rem 1rem;
  outline: none;
}}
.input:focus {{
  border-color: rgba(212,175,55,0.45);
  box-shadow: 0 0 0 4px rgba(212,175,55,0.12);
}}
.mono {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}}
.badge {{
  font-size: 12px;
  padding: .25rem .5rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
}}
.badge-gold {{
  border-color: rgba(212,175,55,0.35);
  background: rgba(212,175,55,0.12);
  color: rgba(212,175,55,1.0);
}}
"""

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <title>{esc(title)} • {APP_NAME}</title>
  <style>{css}</style>
</head>
<body class="min-h-screen">
  <div class="sticky top-0 z-50 backdrop-blur border-b border-white/10" style="background: rgba(5,6,7,0.72);">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-3">
      <a href="/" class="flex items-center gap-2">
        <div class="w-9 h-9 rounded-2xl" style="background: radial-gradient(circle at 35% 35%, rgba(212,175,55,.95), rgba(212,175,55,.20), rgba(255,255,255,.03)); border:1px solid rgba(212,175,55,.25);"></div>
        <div class="font-black tracking-tight text-lg">{APP_NAME}</div>
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

  <footer class="max-w-6xl mx-auto px-4 pb-10 text-sm text-white/60">
    <div class="border-t border-white/10 pt-6">
      Schedge MVP • Calendar export supported • Billing scaffolding: $10/mo + $1/tx.
    </div>
  </footer>
</body>
</html>
"""
    return HTMLResponse(html)


# ----------------------------
# Scheduling utilities
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


def merge_windows(windows: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    windows = [w for w in windows if w[1] > w[0]]
    windows.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    for st, en in windows:
        if not merged:
            merged.append((st, en))
            continue
        lst, len_ = merged[-1]
        if st <= len_:
            merged[-1] = (lst, max(len_, en))
        else:
            merged.append((st, en))
    return merged


def subtract_window(base: List[Tuple[datetime, datetime]], block: Tuple[datetime, datetime]) -> List[Tuple[datetime, datetime]]:
    st, en = block
    out: List[Tuple[datetime, datetime]] = []
    for bst, ben in base:
        if en <= bst or st >= ben:
            out.append((bst, ben))
        else:
            if st > bst:
                out.append((bst, st))
            if en < ben:
                out.append((en, ben))
    return out


def generate_slots_for_day(
    *,
    prof_tz: ZoneInfo,
    day: date,
    rules: List[AvailabilityRule],
    exceptions: List[AvailabilityException],
    service_duration_min: int,
    buffer_min: int,
    slot_step_min: int,
    existing_appointments_utc: List[Tuple[datetime, datetime]],
) -> List[datetime]:
    weekday = day.weekday()
    day_rules = [r for r in rules if r.weekday == weekday]

    windows: List[Tuple[datetime, datetime]] = []
    for r in day_rules:
        st = local_dt(day, parse_hhmm(r.start_hhmm), prof_tz)
        en = local_dt(day, parse_hhmm(r.end_hhmm), prof_tz)
        windows.append((st, en))

    windows = merge_windows(windows)

    # apply exceptions
    for ex in exceptions:
        if ex.on_date != day:
            continue
        st = local_dt(day, parse_hhmm(ex.start_hhmm), prof_tz)
        en = local_dt(day, parse_hhmm(ex.end_hhmm), prof_tz)
        if ex.available:
            windows = merge_windows(windows + [(st, en)])
        else:
            windows = merge_windows(subtract_window(windows, (st, en)))

    dur = timedelta(minutes=service_duration_min)
    buf = timedelta(minutes=buffer_min)
    step = timedelta(minutes=max(5, min(60, slot_step_min)))

    slots: List[datetime] = []
    for wst_local, wen_local in windows:
        cursor = wst_local
        while cursor + dur <= wen_local:
            st_utc = to_utc(cursor)
            en_utc = to_utc(cursor + dur + buf)  # occupy buffer too
            ok = True
            for a_st, a_en in existing_appointments_utc:
                if overlaps(st_utc, en_utc, a_st, a_en):
                    ok = False
                    break
            if ok:
                slots.append(st_utc)
            cursor += step

    return slots


# ----------------------------
# Calendar exports (ICS + links)
# ----------------------------

def dt_to_ics(dt_utc: datetime) -> str:
    # Always store/emit UTC for ICS
    dt_utc = dt_utc.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%dT%H%M%SZ")


def build_ics(
    *,
    uid: str,
    summary: str,
    description: str,
    location: str,
    start_utc: datetime,
    end_utc: datetime,
) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Schedge//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dt_to_ics(now)}",
        f"DTSTART:{dt_to_ics(start_utc)}",
        f"DTEND:{dt_to_ics(end_utc)}",
        f"SUMMARY:{summary.replace('\\n',' ')}",
        f"DESCRIPTION:{description.replace('\\n',' ')}",
        f"LOCATION:{location.replace('\\n',' ')}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def google_calendar_link(summary: str, details: str, location: str, start_utc: datetime, end_utc: datetime) -> str:
    dates = f"{dt_to_ics(start_utc)}/{dt_to_ics(end_utc)}"
    return (
        "https://calendar.google.com/calendar/render?action=TEMPLATE"
        f"&text={quote(summary)}"
        f"&details={quote(details)}"
        f"&location={quote(location)}"
        f"&dates={quote(dates)}"
    )


def outlook_web_link(summary: str, details: str, location: str, start_utc: datetime, end_utc: datetime) -> str:
    # Outlook web uses ISO-ish timestamps; Z is okay.
    s = start_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    e = end_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "https://outlook.live.com/calendar/0/deeplink/compose"
        f"?subject={quote(summary)}"
        f"&body={quote(details)}"
        f"&location={quote(location)}"
        f"&startdt={quote(s)}"
        f"&enddt={quote(e)}"
    )


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
# App init
# ----------------------------

app = FastAPI(title=APP_NAME)
init_db()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ----------------------------
# Helpers: pro/profile/billing
# ----------------------------

def get_my_prof(session: Session, user: User) -> Optional[ProfessionalProfile]:
    if user.role != Role.PRO:
        return None
    return session.exec(select(ProfessionalProfile).where(ProfessionalProfile.user_id == user.id)).first()


def profile_rating(session: Session, profile_id: int) -> Tuple[float, int]:
    reviews = session.exec(select(Review).where(Review.profile_id == profile_id)).all()
    if not reviews:
        return (0.0, 0)
    avg = sum(r.rating for r in reviews) / len(reviews)
    return (avg, len(reviews))


def get_or_create_subscription(session: Session, prof_id: int) -> ProSubscription:
    sub = session.exec(select(ProSubscription).where(ProSubscription.professional_id == prof_id)).first()
    if sub:
        return sub
    sub = ProSubscription(professional_id=prof_id, status="inactive")
    session.add(sub)
    session.commit()
    session.refresh(sub)
    return sub


def pro_can_accept_bookings(session: Session, prof: ProfessionalProfile) -> bool:
    if not prof.discoverable:
        return False
    sub = get_or_create_subscription(session, prof.id)
    return sub.status == "active"


# ----------------------------
# Auth + Portal
# ----------------------------

@app.get("/", response_class=HTMLResponse)
def portal(
    request: Request,
    tab: str = "login",
    role: str = "CUSTOMER",
    user: Optional[User] = Depends(get_current_user),
):
    if user:
        return RedirectResponse("/discover", status_code=303)

    tab = tab if tab in ("login", "signup") else "login"
    role = role if role in ("CUSTOMER", "PRO") else "CUSTOMER"

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6 items-start">
      <div class="lg:col-span-7">
        <div class="card p-7">
          <div class="text-4xl font-black tracking-tight leading-tight">
            <span class="gold">Schedge</span> — schedule pros like it&apos;s social.
          </div>
          <div class="mt-3 text-white/70">
            Customers discover, book, pay, sign waivers, upload docs, message, comment, and leave verified reviews.
          </div>

          <div class="mt-6 flex gap-2">
            <a class="btn {'btn-primary' if tab=='login' else 'btn-ghost'}" href="/?tab=login">Log in</a>
            <a class="btn {'btn-primary' if tab=='signup' else 'btn-ghost'}" href="/?tab=signup&role=CUSTOMER">Sign up</a>
          </div>

          <div class="mt-6 grid md:grid-cols-2 gap-4">
            <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
              <div class="font-bold">For Customers</div>
              <div class="text-sm text-white/70 mt-2">Book sessions, save to calendar, upload docs, and review after completion.</div>
            </div>
            <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
              <div class="font-bold">For Pros</div>
              <div class="text-sm text-white/70 mt-2">Manage availability, accept bookings, message clients, and run your schedule clean.</div>
              <div class="text-sm mt-2">
                <span class="badge badge-gold">$10/mo</span>
                <span class="badge ml-2">$1/transaction</span>
              </div>
            </div>
          </div>

        </div>
      </div>

      <div class="lg:col-span-5">
        <div class="card p-7">
          <div class="flex items-center justify-between">
            <div class="text-2xl font-black">{'Log in' if tab=='login' else 'Create account'}</div>
            {('<a class="text-sm gold hover:underline" href="/?tab=signup&role=PRO">Sign up as Pro</a>' if tab=='signup' and role!='PRO' else '')}
          </div>
          <div class="text-sm text-white/60 mt-1">
            {'Welcome back — pick up where you left off.' if tab=='login' else 'Choose Customer or Pro.'}
          </div>

          {""
            if tab=="login"
            else f"""
            <div class="mt-5 flex gap-2">
              <a class="btn {'btn-primary' if role=='CUSTOMER' else 'btn-ghost'} flex-1" href="/?tab=signup&role=CUSTOMER">Customer</a>
              <a class="btn {'btn-primary' if role=='PRO' else 'btn-ghost'} flex-1" href="/?tab=signup&role=PRO">Pro</a>
            </div>
          """}

          <form method="post" action="/login" class="mt-6 space-y-4" style="display:{'block' if tab=='login' else 'none'};">
            <div>
              <label class="text-sm text-white/70">Email</label>
              <input name="email" type="email" required class="input mt-1" />
            </div>
            <div>
              <label class="text-sm text-white/70">Password</label>
              <input name="password" type="password" required class="input mt-1" />
            </div>
            <button class="btn btn-primary w-full">Log in</button>
            <div class="text-sm text-white/60 text-center">
              No account? <a class="gold hover:underline" href="/?tab=signup&role=CUSTOMER">Sign up</a>
            </div>
          </form>

          <form method="post" action="/signup" class="mt-6 space-y-4" style="display:{'block' if tab=='signup' else 'none'};">
            <input type="hidden" name="role" value="{esc(role)}"/>
            <div>
              <label class="text-sm text-white/70">Email</label>
              <input name="email" type="email" required class="input mt-1" />
            </div>
            <div>
              <label class="text-sm text-white/70">Password (min 8 chars)</label>
              <input name="password" type="password" required class="input mt-1" />
            </div>
            <button class="btn btn-primary w-full">Create {esc(role.title())} account</button>
            <div class="text-sm text-white/60 text-center">
              Already have an account? <a class="gold hover:underline" href="/?tab=login">Log in</a>
            </div>
          </form>
        </div>
      </div>
    </div>
    """
    return html_page("Portal", None, body)


@app.post("/signup")
def signup(
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("CUSTOMER"),
    session: Session = Depends(get_session),
):
    email = email.strip().lower()
    role = role if role in ("CUSTOMER", "PRO") else "CUSTOMER"

    existing = session.exec(select(User).where(User.email == email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    user = User(email=email, password_hash=hash_password(password), role=Role(role))
    session.add(user)
    session.commit()
    session.refresh(user)

    # Pro starter profile + subscription record
    if user.role == Role.PRO:
        prof = ProfessionalProfile(
            user_id=user.id,
            display_name=email.split("@")[0].title(),
            bio="",
            tags="trainer,coach",
            location="",
            timezone="America/Chicago",
            discoverable=True,
        )
        session.add(prof)
        session.commit()
        session.refresh(prof)
        get_or_create_subscription(session, prof.id)

    token = create_token(user.id)
    resp = RedirectResponse("/discover", status_code=303)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=TOKEN_EXPIRE_DAYS * 86400)
    return resp


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
    resp = RedirectResponse("/discover", status_code=303)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=TOKEN_EXPIRE_DAYS * 86400)
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


# ----------------------------
# Discover
# ----------------------------

@app.get("/discover", response_class=HTMLResponse)
def discover(
    request: Request,
    q: str = "",
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    q = q.strip()
    qlike = f"%{q}%"

    stmt = select(ProfessionalProfile).where(ProfessionalProfile.discoverable == True).order_by(ProfessionalProfile.created_at.desc())
    if q:
        stmt = (
            select(ProfessionalProfile)
            .where(ProfessionalProfile.discoverable == True)
            .where(
                (ProfessionalProfile.display_name.ilike(qlike))
                | (ProfessionalProfile.tags.ilike(qlike))
                | (ProfessionalProfile.location.ilike(qlike))
            )
            .order_by(ProfessionalProfile.created_at.desc())
        )
    pros = session.exec(stmt).all()

    cards = []
    for p in pros:
        avg, count = profile_rating(session, p.id)
        tags = [t.strip() for t in (p.tags or "").split(",") if t.strip()]

        # show subscription status subtly
        sub = get_or_create_subscription(session, p.id)
        sub_badge = '<span class="badge badge-gold">Pro Active</span>' if sub.status == "active" else '<span class="badge">Pro Inactive</span>'

        cards.append(
            f"""
            <a href="/p/{p.id}" class="block rounded-3xl border border-white/10 bg-white/5 hover:bg-white/10 p-5 transition">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="text-lg font-black">{esc(p.display_name)}</div>
                  <div class="text-sm text-white/60 mt-1">{esc(p.location or "Remote")} • {esc(p.timezone)}</div>
                </div>
                <div class="text-right">
                  <div>{star_row(avg)} <span class="text-sm text-white/60">({count})</span></div>
                  <div class="mt-2">{sub_badge}</div>
                </div>
              </div>
              <div class="mt-3 flex flex-wrap gap-2">
                {''.join(pill(t) for t in tags[:8])}
              </div>
              <div class="mt-3 text-sm text-white/75 line-clamp-3">{esc(p.bio or "")}</div>
            </a>
            """
        )

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-8">
        <div class="card p-6">
          <div class="flex items-center justify-between gap-3">
            <div>
              <div class="text-3xl font-black tracking-tight">Discover Pros</div>
              <div class="text-white/60 mt-1">Search by name, tag, or location.</div>
            </div>
            <div class="text-right text-sm text-white/60">
              Logged in as <span class="gold font-bold">{esc(user.email)}</span>
            </div>
          </div>
          <form class="mt-5 flex gap-2" method="get" action="/discover">
            <input name="q" value="{esc(q)}" placeholder="Search..."
              class="input" />
            <button class="btn btn-primary">Search</button>
          </form>
        </div>

        <div class="mt-6 grid sm:grid-cols-2 gap-4">
          {''.join(cards) if cards else '<div class="text-white/60">No pros found.</div>'}
        </div>
      </div>

      <div class="lg:col-span-4">
        <div class="card p-6">
          <div class="text-xl font-black">Quick actions</div>
          <div class="mt-4 flex flex-col gap-2">
            <a class="btn btn-ghost" href="/me">Your account</a>
            <a class="btn btn-ghost" href="/inbox">Inbox</a>
            {('<a class="btn btn-primary" href="/pro/dashboard">Pro Dashboard</a>' if user.role==Role.PRO else '<a class="btn btn-primary" href="/?tab=signup&role=PRO">Become a Pro</a>')}
          </div>
        </div>

        <div class="card p-6 mt-4">
          <div class="text-xl font-black">Calendar</div>
          <div class="text-white/60 mt-2 text-sm">
            Every appointment includes an .ics file and direct Google/Outlook “add” links.
          </div>
        </div>
      </div>
    </div>
    """
    return html_page("Discover", user, body)


# ----------------------------
# Pro dashboard, profile, billing, services, availability
# ----------------------------

@app.get("/pro/dashboard", response_class=HTMLResponse)
def pro_dashboard(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")
    sub = get_or_create_subscription(session, prof.id)

    services = session.exec(select(Service).where(Service.professional_id == prof.id)).all()
    rules = session.exec(select(AvailabilityRule).where(AvailabilityRule.professional_id == prof.id)).all()

    upcoming = session.exec(
        select(Appointment)
        .where(Appointment.professional_id == prof.id)
        .where(Appointment.status != AppointmentStatus.CANCELED)
        .where(Appointment.start_at_utc >= datetime.now(timezone.utc) - timedelta(hours=2))
        .order_by(Appointment.start_at_utc.asc())
    ).all()

    tz = ZoneInfo(prof.timezone)

    svc_cards = []
    for s in services:
        svc_cards.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-4">
              <div class="font-black">{esc(s.name)}</div>
              <div class="text-sm text-white/60 mt-1">{esc(s.description)}</div>
              <div class="text-sm mt-2 text-white/80">
                {fmt_money(s.price_cents)} • {s.duration_min}m • buffer {s.buffer_min}m • step {s.slot_step_min}m
              </div>
              <div class="text-xs text-white/60 mt-2">Waiver: {"required" if s.require_waiver else "off"}</div>
            </div>
            """
        )

    rule_rows = []
    for r in sorted(rules, key=lambda x: (x.weekday, x.start_hhmm)):
        rule_rows.append(
            f"""
            <div class="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <span class="font-black">{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][r.weekday]}</span>
                <span class="text-white/70"> {esc(r.start_hhmm)}–{esc(r.end_hhmm)}</span>
              </div>
              <form method="post" action="/pro/rule/{r.id}/delete">
                <button class="text-sm gold hover:underline">Delete</button>
              </form>
            </div>
            """
        )

    appt_rows = []
    for a in upcoming[:30]:
        s = session.get(Service, a.service_id)
        cust = session.get(User, a.customer_id)
        st_local = a.start_at_utc.astimezone(tz)
        appt_rows.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="font-black">{esc(s.name if s else "Service")}</div>
                  <div class="text-sm text-white/60 mt-1">{st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} ({esc(prof.timezone)})</div>
                  <div class="text-sm text-white/80 mt-1">Customer: {esc(cust.email if cust else "Unknown")}</div>
                  <div class="text-sm mt-2 flex items-center gap-2">
                    <span class="badge">{esc(a.status)}</span>
                    <span class="badge {'badge-gold' if a.paid else ''}">{'Paid' if a.paid else 'Unpaid'}</span>
                  </div>
                </div>
                <div class="text-right flex flex-col gap-2">
                  <a class="gold hover:underline text-sm" href="/appointment/{a.id}">View</a>
                  <a class="text-sm gold hover:underline" href="/pro/appointment/{a.id}/reschedule">Reschedule</a>
                  <form method="post" action="/pro/appointment/{a.id}/cancel">
                    <button class="text-sm gold hover:underline">Cancel</button>
                  </form>
                  <form method="post" action="/pro/appointment/{a.id}/complete">
                    <button class="btn btn-ghost text-sm">Mark Complete</button>
                  </form>
                </div>
              </div>
            </div>
            """
        )

    sub_line = (
        f'<span class="badge badge-gold">Subscription: ACTIVE</span>'
        if sub.status == "active"
        else f'<span class="badge">Subscription: {esc(sub.status.upper())}</span>'
    )

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-5 space-y-4">
        <div class="card p-6">
          <div class="text-2xl font-black">Pro Dashboard</div>
          <div class="text-white/60 mt-1">Profile: <a class="gold hover:underline" href="/p/{prof.id}">{esc(prof.display_name)}</a></div>
          <div class="mt-4 flex items-center gap-2">{sub_line} <span class="badge">$10/mo</span> <span class="badge">$1/tx</span></div>

          <div class="mt-6 grid gap-2">
            <a class="btn btn-primary" href="/pro/billing">Billing & Subscription</a>
            <a class="btn btn-ghost" href="/pro/profile">Edit Profile</a>
            <a class="btn btn-ghost" href="/pro/service/new">Add Service</a>
            <a class="btn btn-ghost" href="/pro/availability">Availability & Exceptions</a>
          </div>

          <div class="mt-5 text-sm text-white/60">
            Tip: bookings are only allowed when your subscription is <span class="gold font-bold">ACTIVE</span>.
          </div>
        </div>

        <div class="card p-6">
          <div class="text-xl font-black">Services</div>
          <div class="mt-4 space-y-3">
            {''.join(svc_cards) if svc_cards else '<div class="text-white/60 text-sm">No services yet.</div>'}
          </div>
        </div>
      </div>

      <div class="lg:col-span-7 space-y-4">
        <div class="card p-6">
          <div class="text-xl font-black">Weekly availability</div>
          <div class="text-sm text-white/60 mt-1">Timezone: {esc(prof.timezone)}</div>

          <form method="post" action="/pro/rule/new" class="mt-5 grid md:grid-cols-4 gap-2 items-end">
            <div>
              <label class="text-sm text-white/60">Weekday</label>
              <select name="weekday" class="input mt-1">
                <option value="0">Mon</option><option value="1">Tue</option><option value="2">Wed</option>
                <option value="3">Thu</option><option value="4">Fri</option><option value="5">Sat</option><option value="6">Sun</option>
              </select>
            </div>
            <div>
              <label class="text-sm text-white/60">Start</label>
              <input name="start_hhmm" placeholder="09:00" class="input mt-1"/>
            </div>
            <div>
              <label class="text-sm text-white/60">End</label>
              <input name="end_hhmm" placeholder="17:00" class="input mt-1"/>
            </div>
            <button class="btn btn-primary">Add</button>
          </form>

          <div class="mt-4 space-y-2">
            {''.join(rule_rows) if rule_rows else '<div class="text-white/60 text-sm">No availability rules yet.</div>'}
          </div>
        </div>

        <div class="card p-6">
          <div class="text-xl font-black">Upcoming appointments</div>
          <div class="mt-4 space-y-3">
            {''.join(appt_rows) if appt_rows else '<div class="text-white/60 text-sm">No upcoming appointments.</div>'}
          </div>
        </div>
      </div>
    </div>
    """
    return html_page("Pro Dashboard", pro_user, body)


@app.get("/pro/billing", response_class=HTMLResponse)
def pro_billing(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")
    sub = get_or_create_subscription(session, prof.id)

    status_badge = (
        '<span class="badge badge-gold">ACTIVE</span>'
        if sub.status == "active"
        else f'<span class="badge">{esc(sub.status.upper())}</span>'
    )

    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Billing & Subscription</div>
      <div class="text-white/60 mt-1">Plan: <span class="gold font-bold">$10/month</span> + <span class="gold font-bold">$1 per transaction</span></div>

      <div class="mt-6 rounded-3xl border border-white/10 bg-white/5 p-5">
        <div class="flex items-center justify-between">
          <div>
            <div class="font-black">Subscription status</div>
            <div class="text-sm text-white/60 mt-1">You must be ACTIVE to accept bookings.</div>
          </div>
          <div>{status_badge}</div>
        </div>

        <div class="mt-4 text-sm text-white/70">
          Started: <span class="mono">{esc(sub.started_at_utc.isoformat() if sub.started_at_utc else "—")}</span><br/>
          Period end: <span class="mono">{esc(sub.current_period_end_utc.isoformat() if sub.current_period_end_utc else "—")}</span>
        </div>

        <div class="mt-5 flex gap-2">
          <form method="post" action="/pro/billing/mock_activate" class="flex-1">
            <button class="btn btn-primary w-full">Activate (mock)</button>
          </form>
          <form method="post" action="/pro/billing/mock_cancel" class="flex-1">
            <button class="btn btn-ghost w-full">Cancel (mock)</button>
          </form>
        </div>

        <div class="mt-4 text-sm text-white/60">
          Production upgrade: replace mock controls with Stripe subscription + webhooks.
        </div>
      </div>

      <div class="mt-6">
        <a class="gold hover:underline" href="/pro/dashboard">← Back</a>
      </div>
    </div>
    """
    return html_page("Billing", pro_user, body)


@app.post("/pro/billing/mock_activate")
def pro_billing_mock_activate(
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    sub = get_or_create_subscription(session, prof.id)
    sub.status = "active"
    sub.started_at_utc = datetime.now(timezone.utc)
    sub.current_period_end_utc = datetime.now(timezone.utc) + timedelta(days=30)
    session.add(sub)
    session.commit()
    return RedirectResponse("/pro/billing", status_code=303)


@app.post("/pro/billing/mock_cancel")
def pro_billing_mock_cancel(
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    sub = get_or_create_subscription(session, prof.id)
    sub.status = "canceled"
    session.add(sub)
    session.commit()
    return RedirectResponse("/pro/billing", status_code=303)


@app.get("/pro/profile", response_class=HTMLResponse)
def pro_profile_edit(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Edit Profile</div>
      <div class="text-white/60 mt-1">This is what customers see.</div>

      <form method="post" action="/pro/profile" class="mt-6 space-y-4">
        <div>
          <label class="text-sm text-white/60">Display name</label>
          <input name="display_name" value="{esc(prof.display_name)}" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Location</label>
          <input name="location" value="{esc(prof.location)}" placeholder="Remote / City" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Tags (comma separated)</label>
          <input name="tags" value="{esc(prof.tags)}" placeholder="trainer,therapist,coach" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Timezone</label>
          <input name="timezone" value="{esc(prof.timezone)}" placeholder="America/Chicago" class="input mt-1"/>
          <div class="text-xs text-white/50 mt-1">Use IANA tz like America/Chicago, America/Los_Angeles</div>
        </div>
        <div>
          <label class="text-sm text-white/60">Discoverable</label>
          <select name="discoverable" class="input mt-1">
            <option value="true" {"selected" if prof.discoverable else ""}>Yes</option>
            <option value="false" {"selected" if not prof.discoverable else ""}>No</option>
          </select>
        </div>
        <div>
          <label class="text-sm text-white/60">Bio</label>
          <textarea name="bio" rows="6" class="input mt-1">{esc(prof.bio)}</textarea>
        </div>
        <button class="btn btn-primary w-full">Save</button>
      </form>

      <div class="mt-6">
        <a class="gold hover:underline" href="/pro/dashboard">← Back</a>
      </div>
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
    discoverable: str = Form("true"),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")
    try:
        ZoneInfo(timezone_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timezone string (IANA).")

    prof.display_name = display_name.strip()[:80]
    prof.bio = bio.strip()[:4000]
    prof.tags = tags.strip()[:400]
    prof.location = location.strip()[:120]
    prof.timezone = timezone_str.strip()[:64]
    prof.discoverable = (discoverable == "true")

    session.add(prof)
    session.commit()
    return RedirectResponse("/pro/dashboard", status_code=303)


@app.get("/pro/service/new", response_class=HTMLResponse)
def pro_service_new_form(
    request: Request,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Add Service</div>
      <div class="text-white/60 mt-1">Duration, slot granularity, pricing, and waiver settings.</div>

      <form method="post" action="/pro/service/new" class="mt-6 space-y-4">
        <div>
          <label class="text-sm text-white/60">Service name</label>
          <input name="name" required placeholder="1:1 Training Session" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Description</label>
          <textarea name="description" rows="4" class="input mt-1" placeholder="What’s included, who it’s for..."></textarea>
        </div>

        <div class="grid md:grid-cols-4 gap-3">
          <div>
            <label class="text-sm text-white/60">Duration (min)</label>
            <input name="duration_min" value="60" class="input mt-1"/>
          </div>
          <div>
            <label class="text-sm text-white/60">Buffer (min)</label>
            <input name="buffer_min" value="0" class="input mt-1"/>
          </div>
          <div>
            <label class="text-sm text-white/60">Slot step (min)</label>
            <input name="slot_step_min" value="15" class="input mt-1"/>
          </div>
          <div>
            <label class="text-sm text-white/60">Price (USD)</label>
            <input name="price_usd" value="50" class="input mt-1"/>
          </div>
        </div>

        <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
          <div class="flex items-center gap-2">
            <input id="rw" name="require_waiver" type="checkbox" checked class="w-4 h-4"/>
            <label for="rw" class="text-sm text-white/80 font-bold">Require waiver acceptance</label>
          </div>
          <label class="text-sm text-white/60 block mt-3">Waiver text</label>
          <textarea name="waiver_text" rows="5" class="input mt-1">I acknowledge the risks and consent to the service.</textarea>
        </div>

        <button class="btn btn-primary w-full">Create service</button>
      </form>

      <div class="mt-6">
        <a class="gold hover:underline" href="/pro/dashboard">← Back</a>
      </div>
    </div>
    """
    return html_page("New Service", pro_user, body)


@app.post("/pro/service/new")
def pro_service_new(
    name: str = Form(...),
    description: str = Form(""),
    duration_min: int = Form(60),
    buffer_min: int = Form(0),
    slot_step_min: int = Form(15),
    price_usd: float = Form(50.0),
    require_waiver: Optional[str] = Form(None),
    waiver_text: str = Form("I acknowledge the risks and consent to the service."),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    if duration_min < 15 or duration_min > 240:
        raise HTTPException(status_code=400, detail="Duration must be between 15 and 240.")
    if buffer_min < 0 or buffer_min > 60:
        raise HTTPException(status_code=400, detail="Buffer must be between 0 and 60.")
    if slot_step_min < 5 or slot_step_min > 60:
        raise HTTPException(status_code=400, detail="Slot step must be between 5 and 60.")
    if price_usd < 0 or price_usd > 5000:
        raise HTTPException(status_code=400, detail="Price looks invalid.")

    svc = Service(
        professional_id=prof.id,
        name=name.strip()[:80],
        description=description.strip()[:3000],
        duration_min=int(duration_min),
        buffer_min=int(buffer_min),
        slot_step_min=int(slot_step_min),
        price_cents=int(round(price_usd * 100)),
        require_waiver=bool(require_waiver),
        waiver_text=waiver_text.strip()[:6000],
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
    prof = get_my_prof(session, pro_user)
    if not prof:
        raise HTTPException(status_code=400, detail="No professional profile found.")

    exs = session.exec(
        select(AvailabilityException)
        .where(AvailabilityException.professional_id == prof.id)
        .order_by(AvailabilityException.on_date.desc())
    ).all()

    rows = []
    for ex in exs[:60]:
        rows.append(
            f"""
            <div class="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <span class="font-black">{ex.on_date.isoformat()}</span>
                <span class="text-white/70"> {esc(ex.start_hhmm)}–{esc(ex.end_hhmm)}</span>
                <span class="badge {'badge-gold' if ex.available else ''} ml-2">{'ADD' if ex.available else 'BLOCK'}</span>
              </div>
              <form method="post" action="/pro/exception/{ex.id}/delete">
                <button class="text-sm gold hover:underline">Delete</button>
              </form>
            </div>
            """
        )

    body = f"""
    <div class="max-w-4xl mx-auto card p-7">
      <div class="text-2xl font-black">Availability Exceptions</div>
      <div class="text-white/60 mt-1">Quickly block time off or add extra availability (Timezone: {esc(prof.timezone)})</div>

      <form method="post" action="/pro/exception/new" class="mt-6 grid md:grid-cols-5 gap-2 items-end">
        <div>
          <label class="text-sm text-white/60">Date</label>
          <input name="on_date" type="date" required class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Start</label>
          <input name="start_hhmm" placeholder="12:00" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">End</label>
          <input name="end_hhmm" placeholder="14:00" class="input mt-1"/>
        </div>
        <div>
          <label class="text-sm text-white/60">Type</label>
          <select name="available" class="input mt-1">
            <option value="false">Block</option>
            <option value="true">Add</option>
          </select>
        </div>
        <button class="btn btn-primary">Save</button>
      </form>

      <div class="mt-6">
        <div class="text-xl font-black">Existing exceptions</div>
        <div class="mt-3 space-y-2">
          {''.join(rows) if rows else '<div class="text-white/60 text-sm">No exceptions yet.</div>'}
        </div>
      </div>

      <div class="mt-6">
        <a class="gold hover:underline" href="/pro/dashboard">← Back</a>
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
    prof = get_my_prof(session, pro_user)
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
    prof = get_my_prof(session, pro_user)
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
    prof = get_my_prof(session, pro_user)
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


@app.post("/pro/exception/{ex_id}/delete")
def pro_exception_delete(
    ex_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    ex = session.get(AvailabilityException, ex_id)
    if not prof or not ex or ex.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Exception not found.")
    session.delete(ex)
    session.commit()
    return RedirectResponse("/pro/availability", status_code=303)


@app.post("/pro/appointment/{appointment_id}/complete")
def pro_mark_complete(
    appointment_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    appt = session.get(Appointment, appointment_id)
    if not prof or not appt or appt.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    appt.status = AppointmentStatus.COMPLETED
    session.add(appt)
    session.commit()
    return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)


@app.post("/pro/appointment/{appointment_id}/cancel")
def pro_cancel_appointment(
    appointment_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    appt = session.get(Appointment, appointment_id)
    if not prof or not appt or appt.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    if appt.status in (AppointmentStatus.CANCELED, AppointmentStatus.COMPLETED):
        return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)
    appt.status = AppointmentStatus.CANCELED
    session.add(appt)
    session.commit()
    return RedirectResponse(f"/appointment/{appointment_id}", status_code=303)


@app.get("/pro/appointment/{appointment_id}/reschedule", response_class=HTMLResponse)
def pro_reschedule_form(
    request: Request,
    appointment_id: int,
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    appt = session.get(Appointment, appointment_id)
    if not prof or not appt or appt.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    svc = session.get(Service, appt.service_id)
    tz = ZoneInfo(prof.timezone)
    start_local = appt.start_at_utc.astimezone(tz)

    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Reschedule appointment</div>
      <div class="text-white/60 mt-1">{esc(svc.name if svc else "Service")} • current: {start_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} ({esc(prof.timezone)})</div>

      <form method="post" action="/pro/appointment/{appt.id}/reschedule" class="mt-6 space-y-4">
        <div class="grid md:grid-cols-2 gap-3">
          <div>
            <label class="text-sm text-white/60">New date</label>
            <input name="new_date" type="date" required class="input mt-1"/>
          </div>
          <div>
            <label class="text-sm text-white/60">New start time (HH:MM)</label>
            <input name="new_time" placeholder="14:30" required class="input mt-1"/>
          </div>
        </div>
        <div class="text-sm text-white/60">Times are interpreted in your timezone: <span class="gold font-bold">{esc(prof.timezone)}</span></div>
        <button class="btn btn-primary w-full">Reschedule</button>
      </form>

      <div class="mt-6">
        <a class="gold hover:underline" href="/appointment/{appt.id}">← Back</a>
      </div>
    </div>
    """
    return html_page("Reschedule", pro_user, body)


@app.post("/pro/appointment/{appointment_id}/reschedule")
def pro_reschedule_save(
    appointment_id: int,
    new_date: str = Form(...),
    new_time: str = Form(...),
    pro_user: User = Depends(require_pro),
    session: Session = Depends(get_session),
):
    prof = get_my_prof(session, pro_user)
    appt = session.get(Appointment, appointment_id)
    if not prof or not appt or appt.professional_id != prof.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    if appt.status == AppointmentStatus.CANCELED:
        raise HTTPException(status_code=400, detail="Cannot reschedule a canceled appointment.")

    svc = session.get(Service, appt.service_id)
    if not svc:
        raise HTTPException(status_code=400, detail="Service missing.")

    try:
        d = date.fromisoformat(new_date)
        t = parse_hhmm(new_time.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date/time.")

    tz = ZoneInfo(prof.timezone)
    st_local = local_dt(d, t, tz)
    st_utc = to_utc(st_local)
    en_utc = st_utc + timedelta(minutes=svc.duration_min + svc.buffer_min)

    # collision check
    existing = session.exec(
        select(Appointment)
        .where(Appointment.professional_id == prof.id)
        .where(Appointment.id != appt.id)
        .where(Appointment.status != AppointmentStatus.CANCELED)
        .where(Appointment.start_at_utc < en_utc)
        .where(Appointment.end_at_utc > st_utc)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="That new time overlaps an existing booking.")

    appt.start_at_utc = st_utc
    appt.end_at_utc = en_utc
    session.add(appt)
    session.commit()
    return RedirectResponse(f"/appointment/{appt.id}", status_code=303)


# ----------------------------
# Profile + booking
# ----------------------------

@app.get("/p/{profile_id}", response_class=HTMLResponse)
def profile_page(
    request: Request,
    profile_id: int,
    day: Optional[str] = None,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    prof = session.get(ProfessionalProfile, profile_id)
    if not prof or not prof.discoverable:
        raise HTTPException(status_code=404, detail="Profile not found.")

    services = session.exec(select(Service).where(Service.professional_id == prof.id)).all()
    rules = session.exec(select(AvailabilityRule).where(AvailabilityRule.professional_id == prof.id)).all()

    tz = ZoneInfo(prof.timezone)
    today_local = datetime.now(timezone.utc).astimezone(tz).date()
    try:
        chosen_day = date.fromisoformat(day) if day else today_local
    except Exception:
        chosen_day = today_local

    exceptions = session.exec(
        select(AvailabilityException)
        .where(AvailabilityException.professional_id == prof.id)
        .where(AvailabilityException.on_date == chosen_day)
    ).all()

    comments = session.exec(
        select(ProfileComment).where(ProfileComment.profile_id == prof.id).order_by(ProfileComment.created_at.desc())
    ).all()
    reviews = session.exec(
        select(Review).where(Review.profile_id == prof.id).order_by(Review.created_at.desc())
    ).all()
    avg, count = profile_rating(session, prof.id)

    active = pro_can_accept_bookings(session, prof)
    book_gate = (
        '<div class="rounded-3xl border border-white/10 bg-amber-500/10 p-4 text-sm text-white/75">'
        '<span class="gold font-black">Bookings paused:</span> this Pro is not currently active on the subscription.</div>'
        if not active else ""
    )

    service_blocks = []
    for svc in services:
        # existing appts around selected day
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
            slot_step_min=svc.slot_step_min,
            existing_appointments_utc=occupied,
        )

        if slots and active:
            slot_buttons = []
            for st_utc in slots[:40]:
                st_local = st_utc.astimezone(tz)
                label = st_local.strftime("%I:%M %p").lstrip("0")
                slot_buttons.append(
                    f"""
                    <a href="/book/{svc.id}?day={chosen_day.isoformat()}&start={st_utc.isoformat()}"
                       class="px-3 py-2 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 text-sm">
                      {label}
                    </a>
                    """
                )
            slot_html = f'<div class="mt-4 flex flex-wrap gap-2">{"".join(slot_buttons)}</div>'
        else:
            slot_html = '<div class="mt-4 text-sm text-white/60">No slots available this day.</div>' if active else '<div class="mt-4 text-sm text-white/60">Bookings are paused.</div>'

        service_blocks.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="text-lg font-black">{esc(svc.name)}</div>
                  <div class="text-sm text-white/60 mt-1">{esc(svc.description)}</div>
                  <div class="text-sm text-white/80 mt-2">{fmt_money(svc.price_cents)} • {svc.duration_min}m • step {svc.slot_step_min}m</div>
                </div>
                <div class="text-right text-sm text-white/70">
                  <div>{'Waiver required' if svc.require_waiver else 'No waiver'}</div>
                </div>
              </div>
              {slot_html}
            </div>
            """
        )

    comment_cards = []
    for c in comments[:40]:
        u = session.get(User, c.user_id)
        comment_cards.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-4">
              <div class="text-sm text-white/60">{esc(u.email if u else 'User')} • {c.created_at.strftime('%b %d, %Y')}</div>
              <div class="mt-2 text-white/85">{esc(c.body)}</div>
            </div>
            """
        )

    review_cards = []
    for r in reviews[:40]:
        u = session.get(User, r.user_id)
        review_cards.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-center justify-between">
                <div class="text-sm text-white/60">{esc(u.email if u else 'User')} • {r.created_at.strftime('%b %d, %Y')}</div>
                <div>{star_row(r.rating)}</div>
              </div>
              <div class="mt-2 text-white/85">{esc(r.body)}</div>
            </div>
            """
        )

    tags = [t.strip() for t in (prof.tags or "").split(",") if t.strip()]
    message_cta = ""
    if user.id != prof.user_id:
        message_cta = f"""
        <form method="post" action="/thread/start">
          <input type="hidden" name="profile_id" value="{prof.id}"/>
          <button class="btn btn-primary">Message</button>
        </form>
        """

    body = f"""
    <div class="grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4 space-y-4">
        <div class="card p-6">
          <div class="text-2xl font-black">{esc(prof.display_name)}</div>
          <div class="mt-2 text-sm text-white/60">{esc(prof.location or "Remote")} • {esc(prof.timezone)}</div>

          <div class="mt-3">{star_row(avg)} <span class="text-sm text-white/60">({count} reviews)</span></div>

          <div class="mt-4 flex flex-wrap gap-2">
            {''.join(pill(t) for t in tags[:10])}
          </div>

          <div class="mt-4 text-white/80 text-sm whitespace-pre-wrap">{esc(prof.bio or "")}</div>

          <div class="mt-5 flex gap-2">
            {message_cta}
            <a class="btn btn-ghost" href="/discover">Back</a>
          </div>
        </div>

        <div class="card p-6">
          <div class="text-xl font-black">Pick a day</div>
          <form method="get" action="/p/{prof.id}" class="mt-3 flex gap-2 items-end">
            <input type="date" name="day" value="{chosen_day.isoformat()}" class="input"/>
            <button class="btn btn-primary">Go</button>
          </form>
          <div class="text-xs text-white/50 mt-2">Slots shown in {esc(prof.timezone)}.</div>
        </div>

        {book_gate}
      </div>

      <div class="lg:col-span-8 space-y-4">
        <div class="card p-6">
          <div class="text-xl font-black">Services & availability</div>
          <div class="mt-4 space-y-4">
            {''.join(service_blocks) if service_blocks else '<div class="text-white/60">No services yet.</div>'}
          </div>
        </div>

        <div class="card p-6">
          <div class="text-xl font-black">Reviews</div>
          <div class="mt-4 space-y-3">
            {''.join(review_cards) if review_cards else '<div class="text-white/60">No reviews yet.</div>'}
          </div>
        </div>

        <div class="card p-6">
          <div class="text-xl font-black">Comments</div>

          <form method="post" action="/comment" class="mt-4 flex gap-2">
            <input type="hidden" name="profile_id" value="{prof.id}"/>
            <input name="body" placeholder="Leave a comment…" class="input"/>
            <button class="btn btn-primary">Post</button>
          </form>

          <div class="mt-4 space-y-3">
            {''.join(comment_cards) if comment_cards else '<div class="text-white/60">No comments yet.</div>'}
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
    start: str,  # ISO UTC
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    svc = session.get(Service, service_id)
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found.")
    prof = session.get(ProfessionalProfile, svc.professional_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Pro not found.")

    if not pro_can_accept_bookings(session, prof):
        raise HTTPException(status_code=403, detail="This pro is not accepting bookings right now.")

    if user.id == prof.user_id:
        raise HTTPException(status_code=400, detail="You cannot book yourself.")

    try:
        start_utc = datetime.fromisoformat(start)
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        start_utc = start_utc.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start time.")

    tz = ZoneInfo(prof.timezone)
    start_local = start_utc.astimezone(tz)

    waiver_block = ""
    if svc.require_waiver:
        waiver_block = f"""
        <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
          <div class="font-black">Waiver (required)</div>
          <div class="text-sm text-white/70 mt-2 whitespace-pre-wrap">{esc(svc.waiver_text)}</div>
          <div class="mt-3 flex items-center gap-2">
            <input id="wa" name="accept_waiver" type="checkbox" class="w-4 h-4" required/>
            <label for="wa" class="text-sm text-white/85">I accept the waiver terms.</label>
          </div>
        </div>
        """

    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Confirm booking</div>
      <div class="text-white/60 mt-1">{esc(prof.display_name)} • {esc(svc.name)}</div>

      <div class="mt-5 rounded-3xl border border-white/10 bg-white/5 p-5">
        <div class="text-white/60 text-sm">When</div>
        <div class="text-xl font-black mt-1">{start_local.strftime('%A, %b %d, %Y • %I:%M %p').lstrip('0')} <span class="text-white/60 text-sm">({esc(prof.timezone)})</span></div>
        <div class="mt-2 text-white/80 text-sm">Price: <span class="gold font-black">{fmt_money(svc.price_cents)}</span></div>
      </div>

      <form method="post" action="/book/{svc.id}" class="mt-4 space-y-4">
        <input type="hidden" name="start_utc" value="{esc(start_utc.isoformat())}"/>
        <input type="hidden" name="service_id" value="{svc.id}"/>
        {waiver_block}
        <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
          <label class="text-sm text-white/60">Notes (optional)</label>
          <textarea name="notes" rows="3" class="input mt-1" placeholder="Any context you want the pro to know..."></textarea>
        </div>

        <button class="btn btn-primary w-full">Create booking</button>
        <a class="block text-center text-sm text-white/60 hover:underline" href="/p/{prof.id}?day={esc(day)}">Cancel</a>
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
    if not pro_can_accept_bookings(session, prof):
        raise HTTPException(status_code=403, detail="This pro is not accepting bookings right now.")

    try:
        st = datetime.fromisoformat(start_utc)
        if st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)
        st = st.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start time.")

    en = st + timedelta(minutes=svc.duration_min + svc.buffer_min)

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
        end_at_utc=en,
        status=AppointmentStatus.PENDING,
        paid=False,
        waiver_accepted_at_utc=waiver_ts,
        customer_notes=notes.strip()[:2000],
    )
    session.add(appt)
    session.commit()
    session.refresh(appt)

    pay = Payment(
        appointment_id=appt.id,
        amount_cents=svc.price_cents,
        platform_fee_cents=PLATFORM_TX_FEE_CENTS,
        pro_amount_cents=max(0, svc.price_cents - PLATFORM_TX_FEE_CENTS),
        status="mock_created",
    )
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
    pay = session.get(Payment, 1)  # placeholder not used

    pay = session.exec(select(Payment).where(Payment.appointment_id == appt.id)).first()

    tz = ZoneInfo(prof.timezone if prof else "UTC")
    st_local = appt.start_at_utc.astimezone(tz)

    status_line = f"<div class='text-white/70 text-sm'>Payment status: <span class='gold font-black'>{esc(pay.status if pay else 'none')}</span></div>"

    pay_btn = ""
    if not appt.paid:
        pay_btn = f"""
        <form method="post" action="/pay/mock" class="mt-4">
          <input type="hidden" name="appointment_id" value="{appt.id}"/>
          <button class="btn btn-primary w-full">Pay now (mock)</button>
        </form>
        """
    else:
        pay_btn = f"""
        <div class="mt-4 rounded-3xl border border-white/10 bg-emerald-500/10 p-5">
          <div class="font-black text-emerald-200">Paid ✓</div>
          <div class="text-sm text-white/70 mt-1">Confirmed (assuming waiver is satisfied).</div>
        </div>
        """

    body = f"""
    <div class="max-w-3xl mx-auto card p-7">
      <div class="text-2xl font-black">Payment</div>
      <div class="text-white/60 mt-1">{esc(prof.display_name if prof else 'Pro')} • {esc(svc.name if svc else 'Service')}</div>

      <div class="mt-5 rounded-3xl border border-white/10 bg-white/5 p-5">
        <div class="text-white/60 text-sm">When</div>
        <div class="text-xl font-black mt-1">{st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} <span class="text-white/60 text-sm">({esc(prof.timezone if prof else 'UTC')})</span></div>

        <div class="mt-2 text-white/80 text-sm">
          Amount: <span class="gold font-black">{fmt_money((pay.amount_cents if pay else (svc.price_cents if svc else 0)))}</span>
        </div>

        <div class="mt-1 text-white/70 text-sm">
          Platform fee: <span class="gold font-black">{fmt_money((pay.platform_fee_cents if pay else PLATFORM_TX_FEE_CENTS))}</span>
          • Pro payout: <span class="gold font-black">{fmt_money((pay.pro_amount_cents if pay else 0))}</span>
        </div>

        <div class="mt-2">{status_line}</div>
      </div>

      {pay_btn}

      <div class="mt-4 flex gap-2">
        <a class="btn btn-ghost flex-1" href="/appointment/{appt.id}">View appointment</a>
        <a class="btn btn-ghost flex-1" href="/p/{prof.id}">Back to profile</a>
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
        pay = Payment(
            appointment_id=appt.id,
            amount_cents=(svc.price_cents if svc else 0),
            platform_fee_cents=PLATFORM_TX_FEE_CENTS,
            pro_amount_cents=max(0, (svc.price_cents if svc else 0) - PLATFORM_TX_FEE_CENTS),
            status="mock_created",
        )

    appt.paid = True
    pay.status = "succeeded"

    if svc and svc.require_waiver and not appt.waiver_accepted_at_utc:
        appt.status = AppointmentStatus.PENDING
    else:
        appt.status = AppointmentStatus.CONFIRMED

    session.add(appt)
    session.add(pay)
    session.commit()
    return RedirectResponse(f"/appointment/{appt.id}", status_code=303)


# ----------------------------
# Appointment page + calendar export
# ----------------------------

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

    my_prof = get_my_prof(session, user)
    is_pro_owner = bool(my_prof and prof and my_prof.id == prof.id)

    if user.id != appt.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    tz = ZoneInfo(prof.timezone if prof else "UTC")
    st_local = appt.start_at_utc.astimezone(tz)
    en_local = appt.end_at_utc.astimezone(tz)

    # Documents
    docs = session.exec(select(AppointmentDocument).where(AppointmentDocument.appointment_id == appt.id)).all()
    doc_list = []
    for d in docs:
        uploader = session.get(User, d.uploaded_by_user_id)
        doc_list.append(
            f"""
            <div class="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-3">
              <div class="text-sm">
                <div class="font-black">{esc(d.filename)}</div>
                <div class="text-white/60 text-xs">Uploaded by {esc(uploader.email if uploader else 'User')} • {d.created_at.strftime('%b %d, %Y')}</div>
              </div>
              <a class="gold hover:underline text-sm" href="/uploads/{esc(Path(d.path).name)}" target="_blank">Open</a>
            </div>
            """
        )

    # Review allowed after completion & only customer & once
    existing_review = session.exec(select(Review).where(Review.appointment_id == appt.id)).first()
    review_block = ""
    if user.id == appt.customer_id and appt.status == AppointmentStatus.COMPLETED and not existing_review:
        review_block = f"""
        <div class="card p-6 mt-4">
          <div class="text-xl font-black">Leave a review</div>
          <form method="post" action="/review" class="mt-4 space-y-3">
            <input type="hidden" name="appointment_id" value="{appt.id}"/>
            <div>
              <label class="text-sm text-white/60">Rating</label>
              <select name="rating" class="input mt-1">
                <option value="5">5 - Excellent</option>
                <option value="4">4 - Great</option>
                <option value="3">3 - OK</option>
                <option value="2">2 - Not great</option>
                <option value="1">1 - Bad</option>
              </select>
            </div>
            <div>
              <label class="text-sm text-white/60">Review</label>
              <textarea name="body" rows="4" class="input mt-1" placeholder="What was your experience like?"></textarea>
            </div>
            <button class="btn btn-primary w-full">Post review</button>
          </form>
        </div>
        """
    elif existing_review:
        review_block = f"""
        <div class="card p-6 mt-4">
          <div class="text-xl font-black">Review</div>
          <div class="mt-3">{star_row(existing_review.rating)}</div>
          <div class="mt-2 text-white/85">{esc(existing_review.body)}</div>
        </div>
        """

    # Calendar links
    summary = f"{APP_NAME}: {svc.name if svc else 'Appointment'}"
    details = f"Pro: {prof.display_name if prof else 'Pro'}\nCustomer: {cust.email if cust else 'Customer'}\nNotes: {appt.customer_notes}"
    location = prof.location if prof and prof.location else "Remote"
    gcal = google_calendar_link(summary, details, location, appt.start_at_utc, appt.end_at_utc)
    ocal = outlook_web_link(summary, details, location, appt.start_at_utc, appt.end_at_utc)

    cancel_btn = ""
    if user.id == appt.customer_id and appt.status in (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED):
        cancel_btn = f"""
        <form method="post" action="/appointment/{appt.id}/cancel">
          <button class="btn btn-ghost">Cancel</button>
        </form>
        """

    pro_tools = ""
    if is_pro_owner:
        pro_tools = f"""
        <a class="btn btn-ghost" href="/pro/appointment/{appt.id}/reschedule">Reschedule</a>
        <form method="post" action="/pro/appointment/{appt.id}/cancel">
          <button class="btn btn-ghost">Cancel (Pro)</button>
        </form>
        """

    body = f"""
    <div class="max-w-4xl mx-auto">
      <div class="card p-7">
        <div class="text-2xl font-black">Appointment</div>
        <div class="text-white/60 mt-1">{esc(svc.name if svc else 'Service')} • {esc(prof.display_name if prof else 'Pro')}</div>

        <div class="mt-5 grid md:grid-cols-2 gap-4">
          <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
            <div class="text-sm text-white/60">Time</div>
            <div class="font-black mt-1">{st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')} → {en_local.strftime('%I:%M %p').lstrip('0')}</div>
            <div class="text-sm text-white/70 mt-1">{esc(prof.timezone if prof else 'UTC')}</div>
          </div>

          <div class="rounded-3xl border border-white/10 bg-white/5 p-5">
            <div class="text-sm text-white/60">Status</div>
            <div class="mt-2 flex items-center gap-2">
              <span class="badge">{esc(appt.status)}</span>
              <span class="badge {'badge-gold' if appt.paid else ''}">{'Paid' if appt.paid else 'Unpaid'}</span>
            </div>
            <div class="text-sm text-white/80 mt-3">Customer: {esc(cust.email if cust else 'Unknown')}</div>
            <div class="text-sm text-white/70 mt-1">Payment: {esc(pay.status if pay else 'none')}</div>
          </div>
        </div>

        <div class="mt-5 flex flex-wrap gap-2">
          <a class="btn btn-ghost" href="/p/{prof.id}">Pro profile</a>
          {('<a class="btn btn-primary" href="/pay/'+str(appt.id)+'">Pay</a>' if (user.id==appt.customer_id and not appt.paid) else '')}
          {cancel_btn}
          {pro_tools}
        </div>

        <div class="mt-5 rounded-3xl border border-white/10 bg-white/5 p-5">
          <div class="text-xl font-black">Add to calendar</div>
          <div class="mt-3 flex flex-wrap gap-2">
            <a class="btn btn-primary" href="/appointment/{appt.id}/ics">Download .ics</a>
            <a class="btn btn-ghost" href="{esc(gcal)}" target="_blank">Google Calendar</a>
            <a class="btn btn-ghost" href="{esc(ocal)}" target="_blank">Outlook Web</a>
          </div>
          <div class="text-xs text-white/50 mt-2">Tip: “Download .ics” works with Apple Calendar.</div>
        </div>
      </div>

      <div class="card p-6 mt-4">
        <div class="text-xl font-black">Documents</div>
        <div class="text-sm text-white/60 mt-1">Upload any files needed before the appointment.</div>

        <form method="post" action="/appointment/{appt.id}/upload" enctype="multipart/form-data" class="mt-4 flex gap-2 items-end">
          <input type="file" name="file" required class="input"/>
          <button class="btn btn-primary">Upload</button>
        </form>

        <div class="mt-4 space-y-2">
          {''.join(doc_list) if doc_list else '<div class="text-white/60 text-sm">No documents uploaded.</div>'}
        </div>
      </div>

      {review_block}
    </div>
    """
    return html_page("Appointment", user, body)


@app.get("/appointment/{appointment_id}/ics")
def appointment_ics(
    appointment_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    prof = session.get(ProfessionalProfile, appt.professional_id)
    svc = session.get(Service, appt.service_id)
    cust = session.get(User, appt.customer_id)

    my_prof = get_my_prof(session, user)
    is_pro_owner = bool(my_prof and my_prof.id == appt.professional_id)
    if user.id != appt.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    summary = f"{APP_NAME}: {svc.name if svc else 'Appointment'}"
    description = f"Pro: {prof.display_name if prof else 'Pro'}\nCustomer: {cust.email if cust else 'Customer'}\nNotes: {appt.customer_notes}"
    location = prof.location if prof and prof.location else "Remote"

    ics = build_ics(
        uid=f"schedge-appt-{appt.id}@schedge.local",
        summary=summary,
        description=description,
        location=location,
        start_utc=appt.start_at_utc,
        end_utc=appt.end_at_utc,
    )
    filename = f"schedge_appointment_{appt.id}.ics"
    return PlainTextResponse(
        ics,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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

    my_prof = get_my_prof(session, user)
    is_pro_owner = bool(my_prof and my_prof.id == appt.professional_id)
    if user.id != appt.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

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
def cancel_appointment_customer(
    appointment_id: int,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    appt = session.get(Appointment, appointment_id)
    if not appt or appt.customer_id != user.id:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    if appt.status in (AppointmentStatus.CANCELED, AppointmentStatus.COMPLETED):
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
# Account (Me)
# ----------------------------

@app.get("/me", response_class=HTMLResponse)
def me(
    request: Request,
    user: User = Depends(require_user),
    session: Session = Depends(get_session),
):
    my_prof = get_my_prof(session, user)
    pro_profile_block = ""
    if my_prof:
        sub = get_or_create_subscription(session, my_prof.id)
        pro_profile_block = f"""
        <div class="rounded-3xl border border-white/10 bg-white/5 p-5 mt-4">
          <div class="text-xl font-black">Pro</div>
          <div class="text-white/60 mt-1">Profile: <a class="gold hover:underline" href="/p/{my_prof.id}">{esc(my_prof.display_name)}</a></div>
          <div class="mt-3 flex items-center gap-2">
            <span class="badge {'badge-gold' if sub.status=='active' else ''}">Subscription: {esc(sub.status.upper())}</span>
            <a class="btn btn-ghost" href="/pro/billing">Billing</a>
            <a class="btn btn-primary" href="/pro/dashboard">Dashboard</a>
          </div>
        </div>
        """

    appts = session.exec(
        select(Appointment).where(Appointment.customer_id == user.id).order_by(Appointment.start_at_utc.desc())
    ).all()

    rows = []
    for a in appts[:25]:
        s = session.get(Service, a.service_id)
        p = session.get(ProfessionalProfile, a.professional_id)
        tz = ZoneInfo(p.timezone if p else "UTC")
        st_local = a.start_at_utc.astimezone(tz)
        rows.append(
            f"""
            <div class="rounded-3xl border border-white/10 bg-white/5 p-4">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="font-black">{esc(s.name if s else "Service")}</div>
                  <div class="text-sm text-white/60 mt-1">{esc(p.display_name if p else "Pro")} • {st_local.strftime('%b %d, %Y • %I:%M %p').lstrip('0')}</div>
                  <div class="mt-2 flex gap-2 items-center">
                    <span class="badge">{esc(a.status)}</span>
                    <span class="badge {'badge-gold' if a.paid else ''}">{'Paid' if a.paid else 'Unpaid'}</span>
                  </div>
                </div>
                <a class="gold hover:underline text-sm" href="/appointment/{a.id}">Open</a>
              </div>
            </div>
            """
        )

    body = f"""
    <div class="max-w-4xl mx-auto">
      <div class="card p-7">
        <div class="text-2xl font-black">Account</div>
        <div class="text-white/60 mt-2">Email: <span class="gold font-black">{esc(user.email)}</span></div>
        <div class="text-white/60 mt-1">Role: <span class="gold font-black">{esc(user.role)}</span></div>

        {pro_profile_block}
      </div>

      <div class="card p-6 mt-4">
        <div class="text-xl font-black">Your bookings</div>
        <div class="mt-4 space-y-3">
          {''.join(rows) if rows else '<div class="text-white/60 text-sm">No bookings yet.</div>'}
        </div>
      </div>
    </div>
    """
    return html_page("Me", user, body)


# ----------------------------
# Messaging
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
    my_prof = get_my_prof(session, user)
    if my_prof:
        threads = session.exec(
            select(Thread).where(Thread.professional_id == my_prof.id).order_by(Thread.last_message_at.desc())
        ).all()
    else:
        threads = session.exec(
            select(Thread).where(Thread.customer_id == user.id).order_by(Thread.last_message_at.desc())
        ).all()

    cards = []
    for t in threads[:60]:
        prof = session.get(ProfessionalProfile, t.professional_id)
        cust = session.get(User, t.customer_id)
        counterpart = prof.display_name if not my_prof else (cust.email if cust else "Customer")

        cards.append(
            f"""
            <a href="/thread/{t.id}" class="block rounded-3xl border border-white/10 bg-white/5 hover:bg-white/10 p-4">
              <div class="font-black">{esc(counterpart)}</div>
              <div class="text-sm text-white/60 mt-1">Last message: {t.last_message_at.strftime('%b %d, %Y')}</div>
            </a>
            """
        )

    body = f"""
    <div class="max-w-4xl mx-auto card p-7">
      <div class="text-2xl font-black">Inbox</div>
      <div class="mt-4 space-y-3">
        {''.join(cards) if cards else '<div class="text-white/60">No conversations yet.</div>'}
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

    my_prof = get_my_prof(session, user)
    is_pro_owner = bool(my_prof and my_prof.id == prof.id)
    if user.id != thread.customer_id and not is_pro_owner and user.role != Role.ADMIN:
        raise HTTPException(status_code=403, detail="Not allowed.")

    messages = session.exec(
        select(Message).where(Message.thread_id == thread.id).order_by(Message.created_at.asc())
    ).all()

    bubbles = []
    for m in messages[-150:]:
        mine = (m.sender_id == user.id)
        align = "justify-end" if mine else "justify-start"
        bg = "bg-amber-500/10 border-amber-500/20" if mine else "bg-white/5 border-white/10"
        bubbles.append(
            f"""
            <div class="flex {align}">
              <div class="max-w-[82%] rounded-3xl border {bg} px-4 py-3">
                <div class="text-sm text-white/90 whitespace-pre-wrap">{esc(m.body)}</div>
                <div class="text-xs text-white/50 mt-1">{m.created_at.strftime('%I:%M %p').lstrip('0')}</div>
              </div>
            </div>
            """
        )

    title = prof.display_name if not is_pro_owner else (session.get(User, thread.customer_id).email if session.get(User, thread.customer_id) else "Customer")

    body = f"""
    <div class="max-w-5xl mx-auto grid lg:grid-cols-12 gap-6">
      <div class="lg:col-span-4">
        <div class="card p-6">
          <div class="text-xl font-black">Chat</div>
          <div class="text-sm text-white/60 mt-1">With: <span class="gold font-black">{esc(title)}</span></div>

          <div class="mt-5 flex gap-2">
            <a class="btn btn-ghost flex-1" href="/inbox">Back</a>
            <a class="btn btn-primary flex-1" href="/p/{prof.id}">Profile</a>
          </div>

          <div class="mt-5 text-sm text-white/60">
            Live updates powered by WebSockets.
          </div>
        </div>
      </div>

      <div class="lg:col-span-8">
        <div class="card p-6">
          <div id="messages" class="space-y-3 max-h-[62vh] overflow-auto pr-2">
            {''.join(bubbles) if bubbles else '<div class="text-white/60">Say hi 👋</div>'}
          </div>

          <form id="sendForm" class="mt-4 flex gap-2">
            <input id="msgInput" placeholder="Message..." class="input"/>
            <button class="btn btn-primary">Send</button>
          </form>

          <script>
            const threadId = {thread.id};
            const wsProto = (location.protocol === "https:") ? "wss" : "ws";
            const ws = new WebSocket(`${{wsProto}}://${{location.host}}/ws/thread/${{threadId}}`);

            const messagesEl = document.getElementById("messages");
            const form = document.getElementById("sendForm");
            const input = document.getElementById("msgInput");

            function appendBubble(mine, body, ts) {{
              const wrap = document.createElement("div");
              wrap.className = `flex ${{mine ? "justify-end" : "justify-start"}}`;
              wrap.innerHTML = `
                <div class="max-w-[82%] rounded-3xl border ${{mine ? "bg-amber-500/10 border-amber-500/20" : "bg-white/5 border-white/10"}} px-4 py-3">
                  <div class="text-sm text-white/90 whitespace-pre-wrap"></div>
                  <div class="text-xs text-white/50 mt-1"></div>
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
                appendBubble(data.sender_id === {user.id}, data.body, data.ts);
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
async def thread_ws(websocket: WebSocket, thread_id: int):
    token = websocket.cookies.get(COOKIE_NAME)
    user_id = parse_token(token) if token else None
    if not user_id:
        await websocket.close(code=4401)
        return

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

        my_prof = get_my_prof(session, user)
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
# Health
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME, "utc": datetime.now(timezone.utc).isoformat()}
