"""Accounts on the real Postgres: the main flows, to catch what SQLite lets through."""

import threading
from datetime import timedelta

import pytest
from sqlalchemy import select

from contract_assistant import accounts, audit
from contract_assistant.accounts import AccountError, LoginThrottle
from contract_assistant.audit import EventFilter
from contract_assistant.db.models import ADMIN, USER, LoginSession, User, utcnow

PASSWORD = "correct horse battery"
OTHER_PASSWORD = "staple of the gods"

pytestmark = pytest.mark.pg


def test_accounts_are_found_and_kept_unique_whatever_the_case(pg_database):
    with pg_database.session() as session:
        ada = accounts.set_up_first_admin(session, " Ada@Example.COM", "Ada", PASSWORD)
        by = accounts.actor_of(ada)
        bea, temporary = accounts.create_user(session, by, "Bea@Example.com", "bea", USER)
    with pg_database.session() as session:
        assert accounts.find_by_email(session, "BEA@example.com ").id == bea.id
        assert session.get(User, bea.id).email == "bea@example.com"
    with pytest.raises(AccountError, match=accounts.EMAIL_TAKEN), pg_database.session() as s:
        accounts.create_user(s, by, "bea@EXAMPLE.com", "Other Bea", USER)
    with pytest.raises(AccountError, match=accounts.ALREADY_SET_UP), pg_database.session() as s:
        accounts.set_up_first_admin(s, "eve@example.com", "Eve", PASSWORD)
    with pg_database.session() as session:
        promoted = accounts.update_user(session, by, bea.id, name="Bea", role=ADMIN)
        accounts.update_user(session, accounts.actor_of(promoted), ada.id, role=USER)
        assert [user.name for user in accounts.list_users(session)] == ["Ada", "Bea"]
    with pytest.raises(AccountError, match=accounts.LAST_ADMIN), pg_database.session() as s:
        accounts.set_active(s, None, bea.id, False)
    with pg_database.session() as session:
        actions = [event.action for event in audit.events(session)]
    assert actions == [
        audit.ROLE_CHANGED,
        audit.USER_RENAMED,
        audit.ROLE_CHANGED,
        audit.USER_CREATED,
        audit.USER_CREATED,
    ]
    assert accounts.verify_password(temporary, bea.password_hash)


def test_signing_in_and_out_with_sessions(pg_database):
    throttle = LoginThrottle()
    with pg_database.session() as session:
        ada = accounts.set_up_first_admin(session, "ada@example.com", "Ada", PASSWORD)
    with pg_database.session() as session:
        refused = accounts.authenticate(session, "ADA@example.com", "wrong", throttle=throttle)
    assert refused.error.message == accounts.WRONG_CREDENTIALS
    with pg_database.session() as session:
        result = accounts.authenticate(session, " Ada@Example.com", PASSWORD, throttle=throttle)
        here = accounts.start_session(session, result.user, days=14, user_agent="Safari")
        elsewhere = accounts.start_session(session, result.user, days=14)
    with pg_database.session() as session:
        assert accounts.session_user(session, here).id == ada.id
        assert session.get(User, ada.id).last_login_at is not None
        accounts.change_password(session, ada.id, PASSWORD, OTHER_PASSWORD, keep_token=here)
    with pg_database.session() as session:
        assert accounts.session_user(session, elsewhere) is None
        assert accounts.session_user(session, here).id == ada.id
        accounts.end_session(session, here)
    with pg_database.session() as session:
        assert accounts.session_user(session, here) is None
        actions = [event.action for event in audit.events(session)]
        (failed,) = audit.events(session, EventFilter(actions=[audit.LOGIN_FAILED]))
    assert actions == [
        audit.LOGOUT,
        audit.PASSWORD_CHANGED,
        audit.LOGIN,
        audit.LOGIN_FAILED,
        audit.USER_CREATED,
    ]
    assert (failed.subject_user_id, failed.details["email"]) == (ada.id, "ada@example.com")


def test_expired_sessions_and_a_reset_on_postgres(pg_database):
    with pg_database.session() as session:
        ada = accounts.set_up_first_admin(session, "ada@example.com", "Ada", PASSWORD)
        bea, _ = accounts.create_user(session, None, "bea@example.com", "Bea", USER)
        old = accounts.start_session(session, ada, days=14)
        hers = accounts.start_session(session, bea, days=14)
    with pg_database.session() as session:
        for row in session.scalars(select(LoginSession).where(LoginSession.user_id == ada.id)):
            row.expires_at = utcnow() - timedelta(minutes=1)
    with pg_database.session() as session:
        assert accounts.session_user(session, old) is None
        assert accounts.purge_expired_sessions(session) == 1
        temporary = accounts.reset_password(session, accounts.actor_of(ada), bea.id)
    with pg_database.session() as session:
        assert accounts.session_user(session, hers) is None
        user = session.get(User, bea.id)
        assert user.must_change_password
        assert accounts.verify_password(temporary, user.password_hash)


def test_two_admins_removing_each_other_at_once_leave_one_admin(pg_database):
    with pg_database.session() as session:
        ada = accounts.set_up_first_admin(session, "ada@example.com", "Ada", PASSWORD)
        dan, _ = accounts.create_user(session, None, "dan@example.com", "Dan", ADMIN)
    as_ada, as_dan = accounts.actor_of(ada), accounts.actor_of(dan)
    outcome = {}

    def dan_demotes_ada():
        try:
            with pg_database.session() as session:
                accounts.update_user(session, as_dan, ada.id, role=USER)
        except AccountError as error:
            outcome["refused"] = error.message

    with pg_database.session() as first:
        accounts.set_active(first, as_ada, dan.id, False)
        first.flush()
        second = threading.Thread(target=dan_demotes_ada)
        second.start()
        second.join(timeout=0.5)
        assert second.is_alive(), "the second change waits for the first"
    second.join(timeout=10)
    assert outcome == {"refused": accounts.LAST_ADMIN}
    with pg_database.session() as session:
        assert (session.get(User, ada.id).role, session.get(User, ada.id).active) == (ADMIN, True)
