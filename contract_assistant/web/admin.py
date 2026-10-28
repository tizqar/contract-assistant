"""The admin pages: accounts, the activity log and the contracts waiting for approval.

Each area needs its permission: managing accounts, reading the whole log, approving reviews.
Anyone without it gets a short page that says so (a form gets a plain 403). The pages are
whole pages with plain forms: a change is posted and the browser comes back to the page with
a line saying what was done. A temporary password is shown once, in the answer to the form
that made it, and never again.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import islice
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.orm import Session

from contract_assistant import accounts, audit, files
from contract_assistant.access import Actor, Permission, require_permission
from contract_assistant.accounts import AccountError
from contract_assistant.audit import EventFilter, LoggedEvent
from contract_assistant.db.models import ADMIN, USER, User
from contract_assistant.web import activity
from contract_assistant.web.core import (
    ROLE_NAMES,
    Tau,
    UserWorkspace,
    Ws,
    attachment,
    language,
    render,
)
from contract_assistant.web.i18n import translate

router = APIRouter()

SEE_OTHER = 303
FORBIDDEN = 403
NOT_FOUND = 404
USERS_PATH = "/admin/users"
ACTIVITY_PATH = "/admin/activity"
APPROVALS_PATH = "/admin/approvals"
PAGE_SIZE = 50
# Events shown on an account's page.
RECENT = 20
# Events read at a time while looking for the ones that match the filters.
SCAN_BATCH = 500
CSV_NAME = "activity.csv"
CSV_HEADER = (
    "time_utc",
    "actor_name",
    "actor_email",
    "action",
    "description",
    "contract_id",
    "contract_name",
    "address",
)
# A spreadsheet reads a cell starting with one of these as a formula; a quote keeps it text.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
# The groups of actions about a contract, whose details["name"] is the contract's name.
CONTRACT_GROUPS = frozenset({"contract", "clause", "review", "chat"})
# The roles in the order the forms offer them.
ROLE_CHOICES = (USER, ADMIN)

FOR_ADMINS = "This page is for administrators"
FOR_ADMINS_TEXT = (
    "Your account cannot open it. If something needs doing here, ask an administrator."
)
NO_ACCOUNT = "No such account"
NAME_SAVED = "The name is saved."
ROLE_SAVED = "The role is saved."
DEACTIVATED = "The account is deactivated and its browsers are signed out."
ACTIVATED = "The account is activated; the person can sign in again."
CREATED = "Account created for {name} ({email})."
PASSWORD_RESET = "The password is reset and the account's browsers are signed out."
TRANSLATED = (
    FOR_ADMINS,
    FOR_ADMINS_TEXT,
    NO_ACCOUNT,
    NAME_SAVED,
    ROLE_SAVED,
    DEACTIVATED,
    ACTIVATED,
    CREATED,
    PASSWORD_RESET,
)

# What the `done` parameter of an account's page says after a change.
DONE = {
    "name": NAME_SAVED,
    "role": ROLE_SAVED,
    "deactivated": DEACTIVATED,
    "activated": ACTIVATED,
}


@dataclass(frozen=True)
class AccountRow:
    """An account as the list and its page show it."""

    id: str
    name: str
    email: str
    role: str
    active: bool
    must_change_password: bool
    last_login: str | None
    created: str
    me: bool


@dataclass(frozen=True)
class Shown:
    """A temporary password to show once, and whose it is."""

    name: str
    email: str
    password: str
    message: str


@dataclass(frozen=True)
class Waiting:
    """A contract waiting for approval, as its row shows it."""

    id: str
    name: str
    owner_name: str
    sent_by: str | None
    sent_at: str | None


@dataclass(frozen=True)
class ActivityQuery:
    """The filters of the activity page, as typed in its form."""

    person: str = ""
    group: str = ""
    contract: str = ""
    since: str = ""
    until: str = ""

    def where(self, before_id: int | None = None) -> EventFilter:
        """Return the filter of the log for everything but the contract's name."""
        start, end = _day(self.since), _day(self.until)
        return EventFilter(
            actor_id=self.person or None,
            actions=activity.actions_of(self.group) if self.group else (),
            since=_midnight(start) if start else None,
            until=_midnight(end + timedelta(days=1)) if end else None,
            before_id=before_id,
        )

    def matches(self, event: LoggedEvent) -> bool:
        """Tell whether an event's contract name contains the text asked for, if any."""
        wanted = self.contract.strip().casefold()
        if not wanted:
            return True
        return any(wanted in name.casefold() for name in _contract_names(event))

    def params(self) -> dict[str, str]:
        """Return the filters that are set, as the page's query parameters."""
        values = {
            "person": self.person,
            "group": self.group,
            "contract": self.contract.strip(),
            "from": self.since,
            "to": self.until,
        }
        return {key: value for key, value in values.items() if value}


def _day(text: str) -> date | None:
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None


def _midnight(day: date) -> datetime:
    """Return the start of a day in this computer's time zone, where the pages show times."""
    return datetime.combine(day, time.min).astimezone()


def _contract_names(event: LoggedEvent) -> list[str]:
    """Return the names of the contract an event is about: now, then, and before a rename."""
    names = [event.contract_name] if event.contract_name else []
    group = event.action.split(".")[0]
    if event.contract_id is None and group not in CONTRACT_GROUPS:
        return names
    for key in ("name", "old_name"):
        if isinstance(event.details.get(key), str):
            names.append(event.details[key])
    return names


def _t(request: Request, text: str) -> str:
    return translate(text, language(request))


def _refused(request: Request, tau, actor: Actor, permission: Permission) -> Response | None:
    """Return the page that says the area is for administrators, unless the actor may see it."""
    if actor.can(permission):
        return None
    return _notice(request, tau, FOR_ADMINS, FOR_ADMINS_TEXT, FORBIDDEN, "/files")


def _notice(
    request: Request, tau, title: str, text: str, status_code: int, back: str
) -> HTMLResponse:
    return render(
        request,
        "admin/notice.html",
        status_code=status_code,
        tau=tau,
        cache=None,
        title=_t(request, title),
        text=_t(request, text),
        back=back,
    )


def _page(
    request: Request,
    ws: UserWorkspace,
    tau,
    template: str,
    tab: str,
    *,
    status_code: int = 200,
    waiting: int | None = None,
    **context,
) -> HTMLResponse:
    """Render an admin page with its tab bar and the count of contracts waiting."""
    if waiting is None and ws.actor.can(Permission.APPROVE_REVIEWS):
        with ws.database.session() as session:
            waiting = len(files.awaiting_approval(session, ws.actor))
    return render(
        request,
        template,
        status_code=status_code,
        tau=tau,
        cache=None,
        tab=tab,
        waiting=waiting,
        role_names=ROLE_NAMES,
        **context,
    )


def _no_store(response: Response) -> Response:
    """Keep a page with a password out of the browser's cache."""
    response.headers["Cache-Control"] = "no-store"
    return response


def _account_row(user: User, actor: Actor, lang: str) -> AccountRow:
    return AccountRow(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        active=user.active,
        must_change_password=user.must_change_password,
        last_login=activity.when(user.last_login_at, lang) if user.last_login_at else None,
        created=activity.when(user.created_at, lang),
        me=user.id == actor.id,
    )


def _roles() -> list[tuple[str, str]]:
    return [(role, ROLE_NAMES[role]) for role in ROLE_CHOICES]


def openable(session: Session, actor: Actor) -> set[str]:
    """Return the ids of the contracts an actor can open, to link the events about them."""
    return {contract.id for contract in files.visible_contracts(session, actor)}


@router.get("/admin")
def admin_home(request: Request, ws: Ws, tau: Tau):
    """Go to the first admin page."""
    refused = _refused(request, tau, ws.actor, Permission.MANAGE_USERS)
    if refused is not None:
        return refused
    return RedirectResponse(USERS_PATH, status_code=SEE_OTHER)


# Accounts


@router.get(USERS_PATH, response_class=HTMLResponse)
def users_page(request: Request, ws: Ws, tau: Tau):
    """Every account, and the form to create one."""
    refused = _refused(request, tau, ws.actor, Permission.MANAGE_USERS)
    if refused is not None:
        return refused
    return _users(request, ws, tau)


def _users(
    request: Request,
    ws: UserWorkspace,
    tau,
    *,
    status_code: int = 200,
    error: str | None = None,
    form: dict[str, str] | None = None,
    shown: Shown | None = None,
) -> HTMLResponse:
    lang = language(request)
    with ws.database.session() as session:
        rows = [_account_row(user, ws.actor, lang) for user in accounts.list_users(session)]
    return _page(
        request,
        ws,
        tau,
        "admin/users.html",
        "users",
        status_code=status_code,
        accounts=rows,
        error=error,
        form=form or {"name": "", "email": "", "role": USER},
        shown=shown,
        roles=_roles(),
    )


@router.post(USERS_PATH, response_class=HTMLResponse)
def create_account(
    request: Request,
    ws: Ws,
    tau: Tau,
    name: Annotated[str, Form()] = "",
    email: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = USER,
):
    """Create an account with a temporary password and show the password once."""
    require_permission(ws.actor, Permission.MANAGE_USERS)
    try:
        with ws.database.session() as session:
            user, password = accounts.create_user(session, ws.actor, email, name, role, ip=ws.ip)
            made = (user.name, user.email)
    except AccountError as error:
        form = {"name": name, "email": email, "role": role}
        message = _t(request, error.message)
        return _users(request, ws, tau, status_code=error.status_code, error=message, form=form)
    message = _t(request, CREATED).format(name=made[0], email=made[1])
    shown = Shown(name=made[0], email=made[1], password=password, message=message)
    return _no_store(_users(request, ws, tau, shown=shown))


@router.get(USERS_PATH + "/{user_id}", response_class=HTMLResponse)
def account_page(request: Request, user_id: str, ws: Ws, tau: Tau, done: str = ""):
    """One account: its name, role and access, a password reset, and its recent activity."""
    refused = _refused(request, tau, ws.actor, Permission.MANAGE_USERS)
    if refused is not None:
        return refused
    notice = DONE.get(done)
    return _account(request, ws, tau, user_id, notice=_t(request, notice) if notice else None)


def _account(
    request: Request,
    ws: UserWorkspace,
    tau,
    user_id: str,
    *,
    status_code: int = 200,
    notice: str | None = None,
    error: str | None = None,
    error_at: str | None = None,
    shown: Shown | None = None,
) -> HTMLResponse:
    lang = language(request)
    with ws.database.session() as session:
        user = accounts.get_user(session, user_id)
        if user is None:
            return _notice(
                request, tau, NO_ACCOUNT, accounts.NO_SUCH_USER, NOT_FOUND, USERS_PATH
            )
        row = _account_row(user, ws.actor, lang)
        events = audit.events(session, EventFilter(involving_user_id=user_id), limit=RECENT)
        links = openable(session, ws.actor)
    return _page(
        request,
        ws,
        tau,
        "admin/user.html",
        "users",
        status_code=status_code,
        account=row,
        rows=activity.rows(events, lang),
        openable=links,
        notice=notice,
        error=error,
        error_at=error_at,
        shown=shown,
        roles=_roles(),
    )


def _change(
    request: Request,
    ws: UserWorkspace,
    tau,
    user_id: str,
    form: str,
    change: Callable[[Session], object],
    done: str,
) -> Response:
    """Make a change to an account; show a refusal at its form, else go back to the page."""
    require_permission(ws.actor, Permission.MANAGE_USERS)
    try:
        with ws.database.session() as session:
            change(session)
    except AccountError as error:
        if error.status_code == NOT_FOUND:
            raise
        message = _t(request, error.message)
        return _account(
            request, ws, tau, user_id, status_code=error.status_code, error=message, error_at=form
        )
    return RedirectResponse(f"{USERS_PATH}/{user_id}?done={done}", status_code=SEE_OTHER)


@router.post(USERS_PATH + "/{user_id}/name")
def rename_account(
    request: Request, user_id: str, ws: Ws, tau: Tau, name: Annotated[str, Form()] = ""
):
    """Rename an account."""

    def change(session: Session) -> None:
        accounts.update_user(session, ws.actor, user_id, name=name, ip=ws.ip)

    return _change(request, ws, tau, user_id, "name", change, "name")


@router.post(USERS_PATH + "/{user_id}/role")
def change_role(
    request: Request, user_id: str, ws: Ws, tau: Tau, role: Annotated[str, Form()] = ""
):
    """Make an account an administrator or a user."""

    def change(session: Session) -> None:
        accounts.update_user(session, ws.actor, user_id, role=role, ip=ws.ip)

    return _change(request, ws, tau, user_id, "role", change, "role")


@router.post(USERS_PATH + "/{user_id}/active")
def set_active(
    request: Request, user_id: str, ws: Ws, tau: Tau, active: Annotated[str, Form()] = ""
):
    """Deactivate an account (active=0) or activate it again (active=1)."""
    on = active == "1"

    def change(session: Session) -> None:
        accounts.set_active(session, ws.actor, user_id, on, ip=ws.ip)

    return _change(
        request, ws, tau, user_id, "active", change, "activated" if on else "deactivated"
    )


@router.post(USERS_PATH + "/{user_id}/password", response_class=HTMLResponse)
def reset_password(request: Request, user_id: str, ws: Ws, tau: Tau):
    """Give an account a new temporary password and show it once."""
    require_permission(ws.actor, Permission.MANAGE_USERS)
    with ws.database.session() as session:
        password = accounts.reset_password(session, ws.actor, user_id, ip=ws.ip)
        user = accounts.get_user(session, user_id)
        name, email = user.name, user.email
    message = _t(request, PASSWORD_RESET)
    shown = Shown(name=name, email=email, password=password, message=message)
    return _no_store(_account(request, ws, tau, user_id, shown=shown))


# The activity log


def scan(
    session: Session, query: ActivityQuery, before_id: int | None = None
) -> Iterator[LoggedEvent]:
    """Yield the events that match a query, newest first, reading the log a batch at a time."""
    cursor = before_id
    while True:
        batch = audit.events(session, query.where(cursor), limit=SCAN_BATCH)
        yield from (event for event in batch if query.matches(event))
        if len(batch) < SCAN_BATCH:
            return
        cursor = batch[-1].id


def _query(person: str, group: str, contract: str, since: str, until: str) -> ActivityQuery:
    return ActivityQuery(
        person=person.strip(),
        group=group if group in activity.GROUPS else "",
        contract=contract,
        since=since if _day(since) else "",
        until=until if _day(until) else "",
    )


def _before(text: str) -> int | None:
    return int(text) if text.isdigit() else None


@router.get(ACTIVITY_PATH, response_class=HTMLResponse)
def activity_page(
    request: Request,
    ws: Ws,
    tau: Tau,
    person: str = "",
    group: str = "",
    contract: str = "",
    since: Annotated[str, Query(alias="from")] = "",
    until: Annotated[str, Query(alias="to")] = "",
    before: str = "",
):
    """The whole activity log, newest first, filtered and a page at a time."""
    refused = _refused(request, tau, ws.actor, Permission.READ_ALL_ACTIVITY)
    if refused is not None:
        return refused
    lang = language(request)
    query = _query(person, group, contract, since, until)
    before_id = _before(before)
    with ws.database.session() as session:
        found = list(islice(scan(session, query, before_id), PAGE_SIZE + 1))
        people = [(user.id, user.name, user.email) for user in accounts.list_users(session)]
        links = openable(session, ws.actor)
    shown, more = found[:PAGE_SIZE], len(found) > PAGE_SIZE
    params = query.params()
    older = None
    if more:
        older = f"{ACTIVITY_PATH}?{urlencode({**params, 'before': shown[-1].id})}"
    newest = f"{ACTIVITY_PATH}?{urlencode(params)}" if before_id is not None else None
    return _page(
        request,
        ws,
        tau,
        "admin/activity.html",
        "activity",
        query=query,
        rows=activity.rows(shown, lang),
        openable=links,
        people=people,
        groups=list(activity.GROUPS.items()),
        older=older,
        newest=newest,
        csv_url=f"{ACTIVITY_PATH}.csv" + (f"?{urlencode(params)}" if params else ""),
        filtered=bool(params),
    )


@router.get(ACTIVITY_PATH + ".csv")
def activity_csv(
    ws: Ws,
    person: str = "",
    group: str = "",
    contract: str = "",
    since: Annotated[str, Query(alias="from")] = "",
    until: Annotated[str, Query(alias="to")] = "",
):
    """Every event that matches the filters as a CSV file, in English; reading it logs nothing."""
    require_permission(ws.actor, Permission.READ_ALL_ACTIVITY)
    query = _query(person, group, contract, since, until)
    with ws.database.session() as session:
        text = to_csv(scan(session, query))
    return Response(text, media_type="text/csv; charset=utf-8", headers=attachment(CSV_NAME))


def to_csv(events: Iterator[LoggedEvent]) -> str:
    """Return events as CSV text with a header line: times in UTC, sentences in English."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_HEADER)
    for event in events:
        names = _contract_names(event)
        cells = (
            event.at.astimezone(UTC).isoformat(timespec="seconds"),
            event.actor_name or "",
            event.actor_email or "",
            event.action,
            activity.sentence(event, "en"),
            event.contract_id or "",
            names[0] if names else "",
            event.ip or "",
        )
        writer.writerow([_cell(cell) for cell in cells])
    return buffer.getvalue()


def _cell(value: str) -> str:
    return "'" + value if value.startswith(FORMULA_STARTS) else value


# Approvals


@router.get(APPROVALS_PATH, response_class=HTMLResponse)
def approvals_page(request: Request, ws: Ws, tau: Tau):
    """The contracts sent for approval, the longest waiting first, each to approve or send back."""
    refused = _refused(request, tau, ws.actor, Permission.APPROVE_REVIEWS)
    if refused is not None:
        return refused
    lang = language(request)
    with ws.database.session() as session:
        contracts = files.awaiting_approval(session, ws.actor)
    items = [
        Waiting(
            id=contract.id,
            name=contract.name,
            owner_name=contract.owner_name,
            sent_by=contract.review_changed_by_name,
            sent_at=activity.when(contract.review_changed_at, lang)
            if contract.review_changed_at
            else None,
        )
        for contract in contracts
    ]
    return _page(
        request,
        ws,
        tau,
        "admin/approvals.html",
        "approvals",
        waiting=len(items),
        items=items,
        next_path=APPROVALS_PATH,
    )
