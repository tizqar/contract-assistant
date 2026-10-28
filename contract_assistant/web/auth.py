"""Signing in and out, setting up the first admin, and changing one's password.

Every page needs a signed-in user except the ones here and the static files. `guard` is the
app's middleware: it reads the session cookie and puts the user on the request as
`request.state.actor`; it sends a browser that is not signed in to the sign-in page (and
answers 401 to a script or an htmx request), sends to the set-up page while there are no
accounts, sends a user who must change their password to that form, and refuses a form that
another site posted.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from contract_assistant import accounts
from contract_assistant.access import Actor
from contract_assistant.accounts import AccountError
from contract_assistant.web.core import SIGN_IN_FIRST, client_ip, language, render
from contract_assistant.web.i18n import translate

router = APIRouter()

COOKIE = "ca_session"
SEE_OTHER = 303
UNAUTHORIZED = 401
FORBIDDEN = 403
LOGIN_PATH = "/login"
SETUP_PATH = "/setup"
PASSWORD_PATH = "/account/password"
LOGOUT_PATH = "/logout"
# Reachable without signing in.
OPEN_PATHS = (LOGIN_PATH, SETUP_PATH, "/language", "/favicon.ico")
OPEN_PREFIXES = ("/static/",)
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
HTMX_HEADER = "HX-Request"
HTMX_REDIRECT = "HX-Redirect"
# What a browser accepts when it opens a page; a request for JSON or a stream gets 401.
PAGE_TYPES = ("text/html", "*/*")

OTHER_SITE = "This form was sent from another site, so it was refused."
PASSWORDS_DIFFER = "The two new passwords are not the same."
TRANSLATED = (OTHER_SITE, PASSWORDS_DIFFER)


async def guard(request: Request, call_next):
    """Let a request through only from a signed-in user, or to an open page."""
    if request.method in UNSAFE_METHODS and not _same_site(request):
        return PlainTextResponse(translate(OTHER_SITE, language(request)), FORBIDDEN)
    path = request.url.path
    if path in OPEN_PATHS or path.startswith(OPEN_PREFIXES):
        return await call_next(request)
    signed_in = await run_in_threadpool(_signed_in, request)
    if signed_in is None:
        if not await run_in_threadpool(has_accounts, request):
            return _send_to(request, SETUP_PATH)
        return _send_to(request, _login_url(request))
    actor, must_change = signed_in
    request.state.actor = actor
    if must_change and path not in (PASSWORD_PATH, LOGOUT_PATH):
        return _send_to(request, PASSWORD_PATH)
    return await call_next(request)


def _same_site(request: Request) -> bool:
    """Tell whether a form comes from this site: by its Origin, else its Referer.

    A request with neither comes from a script, not a browser, and carries no cookie of a
    signed-in browser it could abuse.
    """
    sent = request.headers.get("origin") or request.headers.get("referer")
    if not sent:
        return True
    return sent != "null" and urlsplit(sent).netloc == request.url.netloc


def _signed_in(request: Request) -> tuple[Actor, bool] | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    workspace = request.app.state.workspace
    with workspace.database.session() as session:
        user = accounts.session_user(session, token)
        if user is None:
            return None
        return accounts.actor_of(user), user.must_change_password


def has_accounts(request: Request) -> bool:
    """Tell whether any account exists; remembered once it does."""
    state = request.app.state
    if getattr(state, "has_accounts", False):
        return True
    with state.workspace.database.session() as session:
        state.has_accounts = accounts.count_users(session) > 0
    return state.has_accounts


def _login_url(request: Request) -> str:
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    if target == "/":
        return LOGIN_PATH
    return f"{LOGIN_PATH}?{urlencode({'next': target})}"


def _send_to(request: Request, url: str) -> Response:
    """Send the browser to a page: a redirect for a page, an htmx redirect, or 401 for a script."""
    if request.headers.get(HTMX_HEADER):
        return Response(status_code=UNAUTHORIZED, headers={HTMX_REDIRECT: url})
    accepts = request.headers.get("accept", "")
    if request.method == "GET" and (not accepts or any(kind in accepts for kind in PAGE_TYPES)):
        return RedirectResponse(url, status_code=SEE_OTHER)
    return PlainTextResponse(translate(SIGN_IN_FIRST, language(request)), UNAUTHORIZED)


def safe_next(target: str | None) -> str:
    """Return a page of this site to go to after signing in; anything else is the home page."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def _sign_in_response(request: Request, token: str, target: str) -> RedirectResponse:
    settings = request.app.state.workspace.settings
    response = RedirectResponse(target, status_code=SEE_OTHER)
    response.set_cookie(
        COOKIE,
        token,
        max_age=settings.session_days * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return response


def _error(request: Request, template: str, error: AccountError, **context) -> Response:
    message = translate(error.message, language(request))
    return render(request, template, status_code=error.status_code, error=message, **context)


@router.get(LOGIN_PATH)
def login_page(request: Request, next: str = "/"):
    """The sign-in form; the set-up page instead while there are no accounts."""
    if not has_accounts(request):
        return RedirectResponse(SETUP_PATH, status_code=SEE_OTHER)
    return render(request, "auth/login.html", next=safe_next(next), email="", error=None)


@router.post(LOGIN_PATH)
def login(
    request: Request,
    email: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "/",
):
    """Check the email and password, start a session, and go where the reader was going."""
    workspace = request.app.state.workspace
    ip = client_ip(request)
    with workspace.database.session() as session:
        result = accounts.authenticate(
            session, email, password, throttle=workspace.throttle, ip=ip
        )
        token = must_change = None
        if result.ok:
            token = accounts.start_session(
                session,
                result.user,
                days=workspace.settings.session_days,
                user_agent=request.headers.get("user-agent"),
                ip=ip,
            )
            must_change = result.user.must_change_password
    if not result.ok:
        return _error(
            request, "auth/login.html", result.error, next=safe_next(next), email=email
        )
    target = PASSWORD_PATH if must_change else safe_next(next)
    return _sign_in_response(request, token, target)


@router.post(LOGOUT_PATH)
def logout(request: Request):
    """End this browser's session and go to the sign-in page."""
    workspace = request.app.state.workspace
    with workspace.database.session() as session:
        accounts.end_session(session, request.cookies.get(COOKIE), ip=client_ip(request))
    response = RedirectResponse(LOGIN_PATH, status_code=SEE_OTHER)
    response.delete_cookie(COOKIE, path="/")
    return response


@router.get(SETUP_PATH)
def setup_page(request: Request):
    """The first-run page: create the first admin; the sign-in page once accounts exist."""
    if has_accounts(request):
        return RedirectResponse(LOGIN_PATH, status_code=SEE_OTHER)
    return render(request, "auth/setup.html", person="", email="", error=None)


@router.post(SETUP_PATH)
def setup(
    request: Request,
    name: Annotated[str, Form()] = "",
    email: Annotated[str, Form()] = "",
    password: Annotated[str, Form()] = "",
    confirm: Annotated[str, Form()] = "",
):
    """Create the first admin and sign them in."""
    workspace = request.app.state.workspace
    form = {"person": name, "email": email}
    if password != confirm:
        return _error(request, "auth/setup.html", AccountError(PASSWORDS_DIFFER), **form)
    ip = client_ip(request)
    try:
        with workspace.database.session() as session:
            user = accounts.set_up_first_admin(session, email, name, password, ip=ip)
            token = accounts.start_session(
                session,
                user,
                days=workspace.settings.session_days,
                user_agent=request.headers.get("user-agent"),
                ip=ip,
            )
    except AccountError as error:
        if has_accounts(request):
            return RedirectResponse(LOGIN_PATH, status_code=SEE_OTHER)
        return _error(request, "auth/setup.html", error, **form)
    request.app.state.has_accounts = True
    return _sign_in_response(request, token, "/")


@router.get(PASSWORD_PATH)
def password_page(request: Request):
    """The form to change one's password; required after an admin set a temporary one."""
    return render(request, "auth/password.html", error=None, forced=_must_change(request))


@router.post(PASSWORD_PATH)
def change_password(
    request: Request,
    current: Annotated[str, Form()] = "",
    new: Annotated[str, Form()] = "",
    confirm: Annotated[str, Form()] = "",
):
    """Change the password, end the user's other sessions, and go to the home page."""
    actor: Actor = request.state.actor
    forced = _must_change(request)
    if new != confirm:
        return _error(request, "auth/password.html", AccountError(PASSWORDS_DIFFER), forced=forced)
    workspace = request.app.state.workspace
    try:
        with workspace.database.session() as session:
            accounts.change_password(
                session,
                actor.id,
                current,
                new,
                ip=client_ip(request),
                keep_token=request.cookies.get(COOKIE),
            )
    except AccountError as error:
        return _error(request, "auth/password.html", error, forced=forced)
    return RedirectResponse("/", status_code=SEE_OTHER)


def _must_change(request: Request) -> bool:
    workspace = request.app.state.workspace
    with workspace.database.session() as session:
        user = accounts.get_user(session, request.state.actor.id)
        return bool(user and user.must_change_password)
