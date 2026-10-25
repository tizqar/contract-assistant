"""Accounts: passwords, the first admin, the accounts admins manage, sign-in and sessions.

There is no public sign-up. The first admin is set up on the first-run page while there are no
accounts, or from the command line; after that admins create the accounts. Accounts are
deactivated, never deleted, so their history stays readable. A signed-in browser holds a
random token in a cookie; only its hash is stored. Every change records its event in the
session that makes it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import cache

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from contract_assistant import audit
from contract_assistant.access import Actor, Permission, require_permission
from contract_assistant.db.models import (
    ADMIN,
    EMAIL_LENGTH,
    NAME_LENGTH,
    ROLES,
    LoginSession,
    User,
    utcnow,
)

# Passwords: scrypt from the standard library, stored as scrypt$<n>$<r>$<p>$<salt>$<hash>.
SCHEME = "scrypt"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16
# Bounds on the parameters a stored hash may ask for, so a damaged one cannot stall a sign-in.
MAX_SCRYPT_N = 2**17
MAX_SCRYPT_R = 32
MAX_SCRYPT_P = 16
MAX_SCRYPT_MEM = 256 * 1024 * 1024
MIN_PASSWORD_LENGTH = 10
# Long enough for any passphrase, short enough that hashing it costs nothing extra.
MAX_PASSWORD_LENGTH = 256
# Temporary passwords: lower-case letters and digits without the ones read alike (0 o 1 l i).
TEMPORARY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
TEMPORARY_GROUPS = 4
TEMPORARY_GROUP_LENGTH = 4

EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
TOKEN_BYTES = 32
IP_LENGTH = 64
SEEN_EVERY = timedelta(minutes=5)
MAX_FAILURES = 5
FAILURE_WINDOW_S = 900
# The throttle forgets emails whose failures are old once it tracks this many.
THROTTLE_SWEEP_AT = 10_000

# Why a sign-in failed, kept in the details of its LOGIN_FAILED event.
UNKNOWN_EMAIL = "unknown_email"
WRONG_PASSWORD = "wrong_password"
DEACTIVATED = "deactivated"
THROTTLED = "throttled"

BAD_REQUEST = 400
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404
CONFLICT = 409
TOO_MANY_REQUESTS = 429

PASSWORD_TOO_SHORT = f"The password must be at least {MIN_PASSWORD_LENGTH} characters long."
PASSWORD_TOO_LONG = f"The password can be at most {MAX_PASSWORD_LENGTH} characters long."
PASSWORD_IS_EMAIL = "The password cannot be the same as the email address."
SAME_PASSWORD = "The new password must be different from the current one."
WRONG_CURRENT_PASSWORD = "The current password is not right."
INVALID_EMAIL = "Enter a valid email address."
EMAIL_TAKEN = "An account with this email address already exists."
EMPTY_NAME = "Enter a name for the account."
NAME_TOO_LONG = f"The account name can be at most {NAME_LENGTH} characters long."
UNKNOWN_ROLE = "Choose a role: administrator or user."
NO_SUCH_USER = "That account does not exist."
ALREADY_SET_UP = "Accounts are already set up. Sign in instead."
OWN_ROLE = "You cannot change your own role."
DEACTIVATE_SELF = "You cannot deactivate your own account."
LAST_ADMIN = (
    "This is the only active administrator. Make another account an administrator first."
)
WRONG_CREDENTIALS = "The email address or the password is not right."
ACCOUNT_DEACTIVATED = "This account is deactivated. Ask an administrator to activate it."
TOO_MANY_ATTEMPTS = (
    "Too many failed sign-ins with this email address. "
    f"Try again in {FAILURE_WINDOW_S // 60} minutes."
)
TRANSLATED = (
    PASSWORD_TOO_SHORT,
    PASSWORD_TOO_LONG,
    PASSWORD_IS_EMAIL,
    SAME_PASSWORD,
    WRONG_CURRENT_PASSWORD,
    INVALID_EMAIL,
    EMAIL_TAKEN,
    EMPTY_NAME,
    NAME_TOO_LONG,
    UNKNOWN_ROLE,
    NO_SUCH_USER,
    ALREADY_SET_UP,
    OWN_ROLE,
    DEACTIVATE_SELF,
    LAST_ADMIN,
    WRONG_CREDENTIALS,
    ACCOUNT_DEACTIVATED,
    TOO_MANY_ATTEMPTS,
)


class AccountError(Exception):
    """A refusal about an account to tell the reader, with the HTTP status that fits it."""

    def __init__(self, message: str, status_code: int = BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Passwords


def hash_password(password: str) -> str:
    """Return the scrypt hash of a password with a new random salt, in the stored form."""
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_DKLEN)
    return "$".join(
        (SCHEME, str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), _b64(salt), _b64(digest))
    )


def verify_password(password: str, stored: str) -> bool:
    """Return whether a password matches a stored hash; False for "!" or a malformed hash."""
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        cost = int(n), int(r), int(p)
        expected = base64.b64decode(digest, validate=True)
        if scheme != SCHEME or not _sane(*cost, len(expected)):
            return False
        actual = _scrypt(password, base64.b64decode(salt, validate=True), *cost, len(expected))
    except (AttributeError, TypeError, ValueError, OverflowError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=MAX_SCRYPT_MEM
    )


def _sane(n: int, r: int, p: int, dklen: int) -> bool:
    return (
        1 < n <= MAX_SCRYPT_N
        and n & (n - 1) == 0
        and 1 <= r <= MAX_SCRYPT_R
        and 1 <= p <= MAX_SCRYPT_P
        and 16 <= dklen <= 64
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@cache
def _decoy_hash() -> str:
    """Return a hash to check unknown emails against, so they take as long as known ones."""
    return hash_password(secrets.token_urlsafe(16))


def check_password_rules(password: str, email: str = "") -> None:
    """Raise AccountError when a password is too short, too long or the account's email."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(PASSWORD_TOO_SHORT)
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AccountError(PASSWORD_TOO_LONG)
    if email and normalize_email(password) == normalize_email(email):
        raise AccountError(PASSWORD_IS_EMAIL)


def generate_password() -> str:
    """Return a random temporary password that reads easily, such as "k7mq-4hzt-p2wd-9xra"."""
    return "-".join(
        "".join(secrets.choice(TEMPORARY_ALPHABET) for _ in range(TEMPORARY_GROUP_LENGTH))
        for _ in range(TEMPORARY_GROUPS)
    )


# Users


def normalize_email(email: str) -> str:
    """Return an email as it is stored: trimmed and in lower case."""
    return email.strip().lower()


def count_users(session: Session) -> int:
    """Return how many accounts there are, active or not."""
    return session.scalar(select(func.count()).select_from(User)) or 0


def get_user(session: Session, user_id: str) -> User | None:
    """Return the account with an id, or None."""
    return session.get(User, user_id)


def find_by_email(session: Session, email: str) -> User | None:
    """Return the account with an email, whatever its case and surrounding spaces, or None."""
    return session.scalars(select(User).where(User.email == normalize_email(email))).first()


def list_users(session: Session) -> list[User]:
    """Return every account, by name."""
    return list(session.scalars(select(User).order_by(func.lower(User.name), User.email)))


def actor_of(user: User) -> Actor:
    """Return the account as the Actor the services act for."""
    return Actor(id=user.id, email=user.email, name=user.name, role=user.role)


def set_up_first_admin(
    session: Session, email: str, name: str, password: str, ip: str | None = None
) -> User:
    """Create the first account, an admin, while there are no accounts at all."""
    if count_users(session):
        raise AccountError(ALREADY_SET_UP, CONFLICT)
    email, name = _valid_email(email), _valid_name(name)
    check_password_rules(password, email)
    user = User(email=email, name=name, password_hash=hash_password(password), role=ADMIN)
    session.add(user)
    session.flush()
    audit.record(
        session,
        audit.USER_CREATED,
        actor_id=user.id,
        subject_user_id=user.id,
        details={"role": ADMIN, "email": email},
        ip=ip,
    )
    return user


def create_user(
    session: Session,
    by: Actor | None,
    email: str,
    name: str,
    role: str,
    password: str | None = None,
    ip: str | None = None,
) -> tuple[User, str]:
    """Create an account and return it with its password, generated when none is given.

    `by` None is the command line. The account must change its password at the first sign-in
    when an admin made it, whose password the admin knows, or when the password was generated.
    """
    _require_admin(by)
    email, name, role = _valid_email(email), _valid_name(name), _valid_role(role)
    generated = password is None
    password = generate_password() if password is None else password
    check_password_rules(password, email)
    if find_by_email(session, email) is not None:
        raise AccountError(EMAIL_TAKEN, CONFLICT)
    user = User(
        email=email,
        name=name,
        password_hash=hash_password(password),
        role=role,
        must_change_password=by is not None or generated,
    )
    session.add(user)
    session.flush()
    audit.record(
        session,
        audit.USER_CREATED,
        actor_id=_id(by),
        subject_user_id=user.id,
        details={"role": role, "email": email},
        ip=ip,
    )
    return user, password


def update_user(
    session: Session,
    by: Actor | None,
    user_id: str,
    *,
    name: str | None = None,
    role: str | None = None,
    ip: str | None = None,
) -> User:
    """Rename an account or change its role; `by` None is the command line.

    An admin cannot change their own role, and the last active admin stays an admin.
    """
    _require_admin(by)
    user = _user(session, user_id)
    new_name = None if name is None else _valid_name(name)
    new_role = None if role is None else _valid_role(role)
    if new_role is not None and new_role != user.role:
        if by is not None and by.id == user.id:
            raise AccountError(OWN_ROLE, CONFLICT)
        if _is_last_active_admin(session, user):
            raise AccountError(LAST_ADMIN, CONFLICT)
        old_role, user.role = user.role, new_role
        audit.record(
            session,
            audit.ROLE_CHANGED,
            actor_id=_id(by),
            subject_user_id=user.id,
            details={"role": new_role, "old_role": old_role},
            ip=ip,
        )
    if new_name is not None and new_name != user.name:
        _rename(session, user, new_name, _id(by), ip)
    return user


def set_active(
    session: Session, by: Actor | None, user_id: str, active: bool, ip: str | None = None
) -> User:
    """Deactivate or activate an account; deactivating it ends its sessions.

    An admin cannot deactivate themself, and the last active admin stays active.
    """
    _require_admin(by)
    user = _user(session, user_id)
    if user.active == active:
        return user
    if not active:
        if by is not None and by.id == user.id:
            raise AccountError(DEACTIVATE_SELF, CONFLICT)
        if _is_last_active_admin(session, user):
            raise AccountError(LAST_ADMIN, CONFLICT)
        end_sessions_of(session, user.id)
    user.active = active
    audit.record(
        session,
        audit.USER_ACTIVATED if active else audit.USER_DEACTIVATED,
        actor_id=_id(by),
        subject_user_id=user.id,
        ip=ip,
    )
    return user


def reset_password(
    session: Session, by: Actor | None, user_id: str, ip: str | None = None
) -> str:
    """Give an account a temporary password, end its sessions, and return the password."""
    _require_admin(by)
    user = _user(session, user_id)
    password = generate_password()
    user.password_hash = hash_password(password)
    user.must_change_password = True
    end_sessions_of(session, user.id)
    audit.record(
        session, audit.PASSWORD_RESET, actor_id=_id(by), subject_user_id=user.id, ip=ip
    )
    return password


def change_password(
    session: Session,
    user_id: str,
    current: str,
    new: str,
    ip: str | None = None,
    keep_token: str | None = None,
) -> User:
    """Change an account's own password and end its other sessions, keeping keep_token's."""
    user = _user(session, user_id)
    if not verify_password(current, user.password_hash):
        raise AccountError(WRONG_CURRENT_PASSWORD)
    check_password_rules(new, user.email)
    if new == current:
        raise AccountError(SAME_PASSWORD)
    user.password_hash = hash_password(new)
    user.must_change_password = False
    end_sessions_of(session, user.id, keep_token=keep_token)
    audit.record(
        session, audit.PASSWORD_CHANGED, actor_id=user.id, subject_user_id=user.id, ip=ip
    )
    return user


def rename_self(session: Session, user_id: str, name: str, ip: str | None = None) -> User:
    """Change an account's own display name."""
    user = _user(session, user_id)
    name = _valid_name(name)
    if name != user.name:
        _rename(session, user, name, user.id, ip)
    return user


def _rename(session: Session, user: User, name: str, actor_id: str | None, ip: str | None):
    old_name, user.name = user.name, name
    audit.record(
        session,
        audit.USER_RENAMED,
        actor_id=actor_id,
        subject_user_id=user.id,
        details={"old_name": old_name, "name": name},
        ip=ip,
    )


def _require_admin(by: Actor | None) -> None:
    if by is not None:
        require_permission(by, Permission.MANAGE_USERS)


def _id(by: Actor | None) -> str | None:
    return None if by is None else by.id


def _user(session: Session, user_id: str) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise AccountError(NO_SUCH_USER, NOT_FOUND)
    return user


def _is_last_active_admin(session: Session, user: User) -> bool:
    if user.role != ADMIN or not user.active:
        return False
    # Locking the active admins' rows (Postgres) makes a second admin removing the first, at the
    # same moment as the first removes the second, wait and then see one admin left.
    admins = select(User.id).where(User.role == ADMIN, User.active.is_(True))
    return set(session.scalars(admins.order_by(User.id).with_for_update())) <= {user.id}


def _valid_email(email: str) -> str:
    email = normalize_email(email)
    if len(email) > EMAIL_LENGTH or not EMAIL_PATTERN.fullmatch(email):
        raise AccountError(INVALID_EMAIL)
    return email


def _valid_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise AccountError(EMPTY_NAME)
    if len(name) > NAME_LENGTH:
        raise AccountError(NAME_TOO_LONG)
    return name


def _valid_role(role: str) -> str:
    if role not in ROLES:
        raise AccountError(UNKNOWN_ROLE)
    return role


# Sign-in


class LoginThrottle:
    """Failed sign-ins per email, in memory: too many within the window lock the email out.

    The lock lifts by itself once fewer than max_failures of them are younger than the window.
    """

    def __init__(
        self,
        max_failures: int = MAX_FAILURES,
        window_s: float = FAILURE_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_failures = max_failures
        self.window_s = window_s
        self.clock = clock
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def locked(self, email: str) -> bool:
        """Return whether sign-ins with an email are refused for now."""
        with self._lock:
            return len(self._recent(_key(email), self.clock())) >= self.max_failures

    def fail(self, email: str) -> None:
        """Count a failed sign-in with an email."""
        now = self.clock()
        with self._lock:
            key = _key(email)
            failures = self._recent(key, now)
            failures.append(now)
            self._failures[key] = failures
            if len(self._failures) > THROTTLE_SWEEP_AT:
                self._sweep(now)

    def reset(self, email: str) -> None:
        """Forget the failures of an email, after it signs in."""
        with self._lock:
            self._failures.pop(_key(email), None)

    def _recent(self, key: str, now: float) -> deque[float]:
        failures = self._failures.get(key, deque())
        while failures and failures[0] <= now - self.window_s:
            failures.popleft()
        if not failures:
            self._failures.pop(key, None)
        return failures

    def _sweep(self, now: float) -> None:
        for key in list(self._failures):
            self._recent(key, now)


def _key(email: str) -> str:
    return normalize_email(email)[:EMAIL_LENGTH]


@dataclass(frozen=True)
class SignInResult:
    """What a sign-in came to: the account, or the refusal to show (never raised)."""

    user: User | None = None
    error: AccountError | None = None

    @property
    def ok(self) -> bool:
        return self.user is not None


def authenticate(
    session: Session,
    email: str,
    password: str,
    *,
    throttle: LoginThrottle,
    ip: str | None = None,
) -> SignInResult:
    """Check an email and a password; return the account, or why it is refused.

    A refused sign-in is returned, not raised, because it records a LOGIN_FAILED event in the
    session: the caller lets the session commit (with `Database.session`, the block ends
    without raising) and then shows `result.error.message` with `result.error.status_code`.
    An unknown email and a wrong password get the same message; a deactivated account is told
    so only when the password is right. A success resets the throttle, sets last_login_at and
    records LOGIN; the caller then starts a session.
    """
    key = _key(email)
    user = find_by_email(session, key)
    if throttle.locked(key):
        return _refuse(session, TOO_MANY_ATTEMPTS, TOO_MANY_REQUESTS, key, user, THROTTLED, ip)
    matches = verify_password(password, user.password_hash if user else _decoy_hash())
    if user is None or not matches:
        throttle.fail(key)
        reason = WRONG_PASSWORD if user else UNKNOWN_EMAIL
        return _refuse(session, WRONG_CREDENTIALS, UNAUTHORIZED, key, user, reason, ip)
    if not user.active:
        return _refuse(session, ACCOUNT_DEACTIVATED, FORBIDDEN, key, user, DEACTIVATED, ip)
    throttle.reset(key)
    user.last_login_at = utcnow()
    audit.record(session, audit.LOGIN, actor_id=user.id, subject_user_id=user.id, ip=ip)
    return SignInResult(user=user)


def _refuse(
    session: Session,
    message: str,
    status_code: int,
    email: str,
    user: User | None,
    reason: str,
    ip: str | None,
) -> SignInResult:
    audit.record(
        session,
        audit.LOGIN_FAILED,
        subject_user_id=user.id if user else None,
        details={"email": email, "reason": reason},
        ip=ip,
    )
    return SignInResult(error=AccountError(message, status_code))


# Sessions


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def start_session(
    session: Session,
    user: User,
    *,
    days: int,
    user_agent: str | None = None,
    ip: str | None = None,
) -> str:
    """Start a signed-in session for an account and return its token, for the cookie."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    now = utcnow()
    session.add(
        LoginSession(
            token_hash=_digest(token),
            user_id=user.id,
            created_at=now,
            expires_at=now + timedelta(days=days),
            last_seen_at=now,
            user_agent=user_agent[:NAME_LENGTH] if user_agent else None,
            ip=ip[:IP_LENGTH] if ip else None,
        )
    )
    return token


def session_user(session: Session, token: str | None) -> User | None:
    """Return the account a token signs in, or None when unknown, expired or inactive."""
    if not token:
        return None
    row = session.get(LoginSession, _digest(token))
    now = utcnow()
    if row is None or row.expires_at <= now:
        return None
    user = session.get(User, row.user_id)
    if user is None or not user.active:
        return None
    if now - row.last_seen_at >= SEEN_EVERY:
        row.last_seen_at = now
    return user


def end_session(session: Session, token: str | None, ip: str | None = None) -> None:
    """Sign a browser out: end the session of its token and record LOGOUT."""
    row = session.get(LoginSession, _digest(token)) if token else None
    if row is None:
        return
    session.delete(row)
    user_id = row.user_id
    audit.record(session, audit.LOGOUT, actor_id=user_id, subject_user_id=user_id, ip=ip)


def end_sessions_of(session: Session, user_id: str, keep_token: str | None = None) -> int:
    """End every session of an account but keep_token's; return how many ended."""
    query = delete(LoginSession).where(LoginSession.user_id == user_id)
    if keep_token:
        query = query.where(LoginSession.token_hash != _digest(keep_token))
    return session.execute(query).rowcount


def purge_expired_sessions(session: Session) -> int:
    """Remove the sessions that have expired; return how many."""
    query = delete(LoginSession).where(LoginSession.expires_at <= utcnow())
    return session.execute(query).rowcount
