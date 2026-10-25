"""Signing in: the generic refusal, the throttle, and the sessions a browser holds."""

import hashlib
from datetime import timedelta

import pytest
from sqlalchemy import select

from contract_assistant import accounts, audit
from contract_assistant.accounts import LoginThrottle
from contract_assistant.audit import EventFilter
from contract_assistant.db.models import LoginSession, User, utcnow

PASSWORD = "correct horse battery"
WRONG = "not the password at all"
EMAIL = "ada@example.com"


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def ada(database):
    with database.session() as session:
        return accounts.set_up_first_admin(session, EMAIL, "Ada Admin", PASSWORD)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def throttle(clock):
    return LoginThrottle(clock=clock)


def attempt(database, throttle, email=EMAIL, password=PASSWORD, ip="10.0.0.9"):
    with database.session() as session:
        return accounts.authenticate(session, email, password, throttle=throttle, ip=ip)


def failures(database):
    with database.session() as session:
        return audit.events(session, EventFilter(actions=[audit.LOGIN_FAILED]), limit=100)


def sign_in(database, user_id, **kwargs):
    with database.session() as session:
        return accounts.start_session(session, session.get(User, user_id), days=14, **kwargs)


def signed_in_user(database, token):
    with database.session() as session:
        return accounts.session_user(session, token)


def row_of(database, token):
    digest = hashlib.sha256(token.encode()).hexdigest()
    with database.session() as session:
        return session.get(LoginSession, digest)


def move_row(database, token, **times):
    digest = hashlib.sha256(token.encode()).hexdigest()
    with database.session() as session:
        row = session.get(LoginSession, digest)
        for name, value in times.items():
            setattr(row, name, value)


# Signing in


def test_the_right_password_signs_in_and_records_it(database, ada, throttle):
    before = utcnow()
    result = attempt(database, throttle, email="  ADA@Example.com ")
    assert result.ok and result.error is None and result.user.id == ada.id
    with database.session() as session:
        user = session.get(User, ada.id)
        (login,) = audit.events(session, EventFilter(actions=[audit.LOGIN]))
    assert user.last_login_at >= before and user.last_login_at.tzinfo is not None
    assert (login.actor_id, login.subject_user_id, login.ip) == (ada.id, ada.id, "10.0.0.9")
    assert failures(database) == []


def test_an_unknown_email_and_a_wrong_password_get_the_same_message(database, ada, throttle):
    unknown = attempt(database, throttle, email="nobody@example.com")
    wrong = attempt(database, throttle, password=WRONG)
    for result in (unknown, wrong):
        assert not result.ok and result.user is None
        assert (result.error.message, result.error.status_code) == (
            accounts.WRONG_CREDENTIALS,
            401,
        )
    on_wrong, on_unknown = failures(database)
    assert on_unknown.actor_id is None and on_unknown.subject_user_id is None
    assert on_unknown.details == {"email": "nobody@example.com", "reason": "unknown_email"}
    assert on_wrong.actor_id is None and on_wrong.subject_user_id == ada.id
    assert on_wrong.details == {"email": EMAIL, "reason": "wrong_password"}
    assert on_wrong.ip == "10.0.0.9"
    assert WRONG not in str(on_wrong.details)
    assert audit.describe(on_unknown) == "A sign-in as nobody@example.com failed"


def test_the_failed_sign_in_is_kept_when_the_session_block_ends(database, ada, throttle):
    with database.session() as session:
        result = accounts.authenticate(session, EMAIL, WRONG, throttle=throttle)
    assert result.error.message == accounts.WRONG_CREDENTIALS
    with database.session() as session:
        (newest,) = audit.events(session, limit=1)
    assert newest.action == audit.LOGIN_FAILED


def test_a_deactivated_account_is_told_so_only_with_the_right_password(database, ada, throttle):
    with database.session() as session:
        bea, _ = accounts.create_user(
            session, None, "bea@example.com", "Bea", "user", password=PASSWORD
        )
        accounts.set_active(session, None, bea.id, False)
    wrong = attempt(database, throttle, email="bea@example.com", password=WRONG)
    assert wrong.error.message == accounts.WRONG_CREDENTIALS
    right = attempt(database, throttle, email="bea@example.com")
    assert not right.ok
    assert (right.error.message, right.error.status_code) == (accounts.ACCOUNT_DEACTIVATED, 403)
    newest = failures(database)[0]
    assert (newest.subject_user_id, newest.details["reason"]) == (bea.id, "deactivated")
    with database.session() as session:
        assert session.get(User, bea.id).last_login_at is None


def test_five_failures_lock_the_email_until_the_window_passes(database, ada, clock, throttle):
    for _ in range(accounts.MAX_FAILURES):
        assert attempt(database, throttle, password=WRONG).error.message == (
            accounts.WRONG_CREDENTIALS
        )
        clock.now += 10
    locked = attempt(database, throttle)
    assert not locked.ok, "even the right password is refused while locked"
    assert (locked.error.message, locked.error.status_code) == (accounts.TOO_MANY_ATTEMPTS, 429)
    assert failures(database)[0].details == {"email": EMAIL, "reason": "throttled"}
    assert attempt(database, throttle, email="ADA@example.com").error.message == (
        accounts.TOO_MANY_ATTEMPTS
    ), "the lock is on the email, whatever its case"
    clock.now += accounts.FAILURE_WINDOW_S - 60
    assert attempt(database, throttle).error.message == accounts.TOO_MANY_ATTEMPTS
    clock.now += 20
    assert attempt(database, throttle).ok, "the oldest failure has left the window"
    with database.session() as session:
        assert session.get(User, ada.id).last_login_at is not None


def test_signing_in_resets_the_throttle(database, ada, throttle):
    for _ in range(accounts.MAX_FAILURES - 1):
        attempt(database, throttle, password=WRONG)
    assert attempt(database, throttle).ok
    for _ in range(accounts.MAX_FAILURES - 1):
        attempt(database, throttle, password=WRONG)
    assert attempt(database, throttle).ok


def test_the_throttle_counts_each_email_on_its_own(clock):
    throttle = LoginThrottle(max_failures=2, window_s=60, clock=clock)
    throttle.fail("ana@example.com")
    throttle.fail(" ANA@example.com")
    throttle.fail("bea@example.com")
    assert throttle.locked("ana@example.com")
    assert not throttle.locked("bea@example.com")
    clock.now += 60
    assert not throttle.locked("ana@example.com")
    throttle.fail("bea@example.com")
    assert not throttle.locked("bea@example.com"), "its first failure is out of the window"
    throttle.fail("bea@example.com")
    throttle.reset("Bea@example.com")
    assert not throttle.locked("bea@example.com")


def test_the_throttle_forgets_old_failures_when_it_tracks_many(clock, monkeypatch):
    monkeypatch.setattr(accounts, "THROTTLE_SWEEP_AT", 3)
    throttle = LoginThrottle(clock=clock)
    for number in range(3):
        throttle.fail(f"user{number}@example.com")
    clock.now += accounts.FAILURE_WINDOW_S + 1
    throttle.fail("late@example.com")
    assert list(throttle._failures) == ["late@example.com"]


# Sessions


def test_a_session_finds_its_user_and_only_the_hash_of_its_token_is_kept(database, ada):
    token = sign_in(database, ada.id, user_agent="Safari " * 100, ip="10.0.0.9")
    assert len(token) >= 40
    assert signed_in_user(database, token).id == ada.id
    with database.session() as session:
        (row,) = session.scalars(select(LoginSession)).all()
    assert row.token_hash == hashlib.sha256(token.encode()).hexdigest() != token
    assert (row.user_id, row.ip, len(row.user_agent)) == (ada.id, "10.0.0.9", 255)
    assert row.expires_at - row.created_at == timedelta(days=14)
    assert sign_in(database, ada.id) != token


def test_an_unknown_or_expired_token_signs_in_no_one(database, ada):
    token = sign_in(database, ada.id)
    assert signed_in_user(database, None) is None
    assert signed_in_user(database, "") is None
    assert signed_in_user(database, "not a token") is None
    move_row(database, token, expires_at=utcnow() - timedelta(seconds=1))
    assert signed_in_user(database, token) is None


def test_a_session_of_an_inactive_account_signs_in_no_one(database, ada):
    token = sign_in(database, ada.id)
    with database.session() as session:
        session.get(User, ada.id).active = False
    assert signed_in_user(database, token) is None


def test_last_seen_moves_at_most_every_five_minutes(database, ada):
    token = sign_in(database, ada.id)
    recently = utcnow() - timedelta(minutes=4)
    move_row(database, token, last_seen_at=recently)
    assert signed_in_user(database, token).id == ada.id
    assert row_of(database, token).last_seen_at == recently
    move_row(database, token, last_seen_at=utcnow() - timedelta(minutes=6))
    before = utcnow()
    signed_in_user(database, token)
    assert row_of(database, token).last_seen_at >= before


def test_signing_out_ends_the_session_and_records_it(database, ada):
    token, other = sign_in(database, ada.id), sign_in(database, ada.id)
    with database.session() as session:
        accounts.end_session(session, token, ip="10.0.0.9")
    assert signed_in_user(database, token) is None
    assert signed_in_user(database, other).id == ada.id
    with database.session() as session:
        (logout,) = audit.events(session, EventFilter(actions=[audit.LOGOUT]))
        accounts.end_session(session, token)
        accounts.end_session(session, None)
    assert (logout.actor_id, logout.subject_user_id, logout.ip) == (ada.id, ada.id, "10.0.0.9")
    with database.session() as session:
        assert len(audit.events(session, EventFilter(actions=[audit.LOGOUT]))) == 1


def test_ending_every_session_of_an_account_but_one(database, ada):
    with database.session() as session:
        bea, _ = accounts.create_user(session, None, "bea@example.com", "Bea", "user")
    keep, drop = sign_in(database, ada.id), sign_in(database, ada.id)
    hers = sign_in(database, bea.id)
    with database.session() as session:
        assert accounts.end_sessions_of(session, ada.id, keep_token=keep) == 1
    assert signed_in_user(database, keep).id == ada.id
    assert signed_in_user(database, drop) is None
    assert signed_in_user(database, hers).id == bea.id
    with database.session() as session:
        assert accounts.end_sessions_of(session, ada.id) == 1
    assert signed_in_user(database, keep) is None


def test_purging_removes_only_the_expired_sessions(database, ada):
    old, fresh = sign_in(database, ada.id), sign_in(database, ada.id)
    move_row(database, old, expires_at=utcnow() - timedelta(days=1))
    with database.session() as session:
        assert accounts.purge_expired_sessions(session) == 1
    assert row_of(database, old) is None
    assert signed_in_user(database, fresh).id == ada.id
