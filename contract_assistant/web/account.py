"""The signed-in user's own account: their name, their role and what it lets them do, their
signed-in browsers and their recent activity.

Every signed-in user can open it. The password form is the sign-in area's (/account/password);
this page links to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from contract_assistant import accounts, audit
from contract_assistant.accounts import AccountError
from contract_assistant.audit import EventFilter
from contract_assistant.db.models import ADMIN, USER
from contract_assistant.web import activity
from contract_assistant.web.admin import openable
from contract_assistant.web.auth import COOKIE, PASSWORD_PATH
from contract_assistant.web.core import ROLE_NAMES, Tau, UserWorkspace, Ws, language, render
from contract_assistant.web.i18n import translate

router = APIRouter()

SEE_OTHER = 303
ACCOUNT_PATH = "/account"
# The user's own events shown on the page.
RECENT = 20

# What each role lets its holder do and what it makes them responsible for, from the rules
# of the product (see access.py).
CAN = {
    USER: (
        (
            "Add contracts to your own space and keep them in folders: rename, move, delete to "
            "the trash and restore them."
        ),
        "Share your folders with chosen people, to view or to edit, and stop sharing them.",
        (
            "In a folder shared with you to view: read the contracts, ask about them, have "
            "summaries written and download them."
        ),
        (
            "In a folder shared with you to edit: also confirm or reject passages, rename, add "
            "contracts and folders, and send a contract for approval."
        ),
        "Change your name and your password, and see the history of the contracts you can open.",
    ),
    ADMIN: (
        "Everything a user can do, in every person's space.",
        (
            "Create accounts for users and administrators, rename them, change their roles, "
            "deactivate and activate them, and reset passwords. Accounts are never deleted, so "
            "their history stays."
        ),
        (
            "Approve a contract sent for approval, send it back with a note, or reopen an "
            "approved one."
        ),
        "Read the whole activity log, filter it and download it as a CSV file.",
        "Delete folders and contracts in the trash for good.",
    ),
}
DUTIES = {
    USER: (
        "Check the passages of a contract while it is In review, then send it for approval.",
        (
            "Share a folder only with the people who need it, and stop sharing when they no "
            "longer do."
        ),
        (
            "Keep your password to yourself. If you forget it, an administrator can give you a "
            "temporary one."
        ),
    ),
    ADMIN: (
        "Approve or send back the contracts waiting for approval.",
        "Give each person the account they need, and deactivate the accounts no longer needed.",
        "Keep a second administrator, so that someone can always manage the accounts.",
    ),
}
NAME_SAVED = "Your name is saved."
ENDED = "Other browsers signed out: {count}."
UNKNOWN_BROWSER = "Unknown browser"
BROWSER_ON = "{browser} on {system}"
TRANSLATED = (
    *CAN[USER],
    *CAN[ADMIN],
    *DUTIES[USER],
    *DUTIES[ADMIN],
    NAME_SAVED,
    ENDED,
    UNKNOWN_BROWSER,
    BROWSER_ON,
)

# Names of browsers and systems as they appear in a user agent; the first that matches wins,
# so the ones that also name another (Edge and Opera name Chrome) come first.
BROWSERS = (
    ("Edge", re.compile(r"Edg(e|A|iOS)?/")),
    ("Opera", re.compile(r"OPR/|Opera")),
    ("Firefox", re.compile(r"Firefox/|FxiOS/")),
    ("Chrome", re.compile(r"Chrome/|CriOS/")),
    ("Safari", re.compile(r"Safari/")),
)
SYSTEMS = (
    ("iPhone", re.compile(r"iPhone")),
    ("iPad", re.compile(r"iPad")),
    ("Android", re.compile(r"Android")),
    ("Windows", re.compile(r"Windows")),
    ("macOS", re.compile(r"Mac OS X|Macintosh")),
    ("ChromeOS", re.compile(r"CrOS")),
    ("Linux", re.compile(r"Linux")),
)


@dataclass(frozen=True)
class Browser:
    """A signed-in browser of the user, as the page lists it."""

    browser: str
    created: str
    last_seen: str
    ip: str | None
    current: bool


def browser_name(user_agent: str | None, lang: str) -> str:
    """Return a browser as "Chrome on macOS", from its user agent, in the reader's language."""
    agent = user_agent or ""
    browser = next((name for name, pattern in BROWSERS if pattern.search(agent)), None)
    system = next((name for name, pattern in SYSTEMS if pattern.search(agent)), None)
    if browser is None:
        return translate(UNKNOWN_BROWSER, lang)
    if system is None:
        return browser
    return translate(BROWSER_ON, lang).format(browser=browser, system=system)


def _browsers(ws: UserWorkspace, token: str | None, lang: str) -> list[Browser]:
    current = accounts.token_digest(token) if token else None
    with ws.database.session() as session:
        rows = accounts.sessions_of(session, ws.actor.id)
        found = [
            Browser(
                browser=browser_name(row.user_agent, lang),
                created=activity.when(row.created_at, lang),
                last_seen=activity.when(row.last_seen_at, lang),
                ip=row.ip,
                current=row.token_hash == current,
            )
            for row in rows
        ]
    # This browser first, then the others as last seen.
    return sorted(found, key=lambda browser: not browser.current)


def _page(
    request: Request,
    ws: UserWorkspace,
    tau,
    *,
    status_code: int = 200,
    notice: str | None = None,
    error: str | None = None,
    name: str | None = None,
) -> HTMLResponse:
    lang = language(request)
    with ws.database.session() as session:
        user = accounts.get_user(session, ws.actor.id)
        events = audit.events(session, EventFilter(actor_id=ws.actor.id), limit=RECENT)
        links = openable(session, ws.actor)
    role = user.role
    return render(
        request,
        "account/account.html",
        status_code=status_code,
        tau=tau,
        cache=None,
        user=user,
        typed_name=user.name if name is None else name,
        role_name=ROLE_NAMES[role],
        can=CAN[role],
        duties=DUTIES[role],
        browsers=_browsers(ws, request.cookies.get(COOKIE), lang),
        rows=activity.rows(events, lang),
        openable=links,
        notice=notice,
        error=error,
        password_path=PASSWORD_PATH,
    )


@router.get(ACCOUNT_PATH, response_class=HTMLResponse)
def account_page(request: Request, ws: Ws, tau: Tau, done: str = "", ended: str = ""):
    """The user's own account."""
    lang = language(request)
    notice = None
    if done == "name":
        notice = translate(NAME_SAVED, lang)
    elif ended.isdigit():
        notice = translate(ENDED, lang).format(count=int(ended))
    return _page(request, ws, tau, notice=notice)


@router.post(ACCOUNT_PATH + "/name")
def rename(request: Request, ws: Ws, tau: Tau, name: Annotated[str, Form()] = ""):
    """Change the user's own display name."""
    try:
        with ws.database.session() as session:
            accounts.rename_self(session, ws.actor.id, name, ip=ws.ip)
    except AccountError as error:
        message = translate(error.message, language(request))
        return _page(request, ws, tau, status_code=error.status_code, error=message, name=name)
    return RedirectResponse(f"{ACCOUNT_PATH}?done=name", status_code=SEE_OTHER)


@router.post(ACCOUNT_PATH + "/sign-out-others")
def sign_out_others(request: Request, ws: Ws):
    """Sign out every other browser of the user, keeping this one."""
    token = request.cookies.get(COOKIE)
    with ws.database.session() as session:
        ended = accounts.end_sessions_of(session, ws.actor.id, keep_token=token)
    if ended:
        ws.log(audit.SESSIONS_ENDED, details={"count": ended})
    return RedirectResponse(f"{ACCOUNT_PATH}?ended={ended}", status_code=SEE_OTHER)
