"""Accounts: password hashing and rules, the first admin, and what admins do to accounts."""

import re

import pytest
from sqlalchemy import select

from contract_assistant import accounts, audit
from contract_assistant.access import Forbidden
from contract_assistant.accounts import AccountError
from contract_assistant.audit import EventFilter
from contract_assistant.db.models import ADMIN, USER, LoginSession, User
from tests.support.db import add_user

PASSWORD = "correct horse battery"
OTHER_PASSWORD = "staple of the gods"


def first_admin(database, email="ada@example.com", name="Ada Admin"):
    with database.session() as session:
        return accounts.actor_of(accounts.set_up_first_admin(session, email, name, PASSWORD))


def new_user(database, by, email="bea@example.com", name="Bea User", role=USER, password=None):
    with database.session() as session:
        user, password = accounts.create_user(session, by, email, name, role, password=password)
        return user, password


def logged(database, action=None, **where):
    with database.session() as session:
        found = audit.events(session, EventFilter(**where), limit=500)
    return [event for event in found if action is None or event.action == action]


def user_row(database, user_id):
    with database.session() as session:
        return session.get(User, user_id)


def sessions_of(database, user_id):
    with database.session() as session:
        query = select(LoginSession).where(LoginSession.user_id == user_id)
        return session.scalars(query).all()


def refusal(raised):
    return raised.value.message, raised.value.status_code


def sign_in(database, user_id):
    with database.session() as session:
        return accounts.start_session(session, session.get(User, user_id), days=14)


def signed_in_user(database, token):
    with database.session() as session:
        return accounts.session_user(session, token)


# Passwords


def test_a_hash_verifies_its_password_and_no_other():
    stored = accounts.hash_password(PASSWORD)
    assert stored.startswith("scrypt$16384$8$1$")
    assert accounts.verify_password(PASSWORD, stored)
    assert not accounts.verify_password(OTHER_PASSWORD, stored)
    assert not accounts.verify_password(PASSWORD.upper(), stored)
    assert accounts.hash_password(PASSWORD) != stored, "every hash has its own salt"


@pytest.mark.parametrize(
    "stored",
    [
        "!",
        "",
        "scrypt",
        "scrypt$16384$8$1$c2FsdA==",
        "bcrypt$16384$8$1$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$x$8$1$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$16384$8$1$not base64!$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$1000$8$1$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$1099511627776$8$1$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$16384$8$100000$c2FsdHNhbHRzYWx0c2FsdA==$aGFzaGhhc2hoYXNoaGFzaA==",
        "scrypt$16384$8$1$c2FsdHNhbHRzYWx0c2FsdA==$",
    ],
)
def test_an_unusable_or_malformed_hash_verifies_nothing(stored):
    assert accounts.verify_password(PASSWORD, stored) is False


def test_verifying_never_raises_on_odd_input():
    stored = accounts.hash_password(PASSWORD)
    assert accounts.verify_password("\ud800 lone surrogate", stored) is False
    assert accounts.verify_password(PASSWORD, None) is False  # type: ignore[arg-type]


def test_the_password_rules():
    accounts.check_password_rules("x" * accounts.MIN_PASSWORD_LENGTH, "ana@example.com")
    with pytest.raises(AccountError) as short:
        accounts.check_password_rules("x" * (accounts.MIN_PASSWORD_LENGTH - 1))
    assert refusal(short) == (accounts.PASSWORD_TOO_SHORT, 400)
    with pytest.raises(AccountError, match=accounts.PASSWORD_TOO_LONG):
        accounts.check_password_rules("x" * (accounts.MAX_PASSWORD_LENGTH + 1))
    with pytest.raises(AccountError, match=accounts.PASSWORD_IS_EMAIL):
        accounts.check_password_rules(" Ana.Long@Example.com", "ana.long@example.com")


def test_generated_passwords_are_readable_random_and_pass_the_rules():
    passwords = {accounts.generate_password() for _ in range(50)}
    assert len(passwords) == 50
    for password in passwords:
        assert re.fullmatch(r"[a-z2-9]{4}(-[a-z2-9]{4}){3}", password)
        assert not set(password) & set("01ilo")
        accounts.check_password_rules(password, "ana@example.com")


# Users and the first admin


def test_the_first_admin_is_set_up_only_while_there_are_no_accounts(database):
    with database.session() as session:
        assert accounts.count_users(session) == 0
        admin = accounts.set_up_first_admin(
            session, "  Ada@Example.COM ", " Ada Admin ", PASSWORD, ip="10.0.0.1"
        )
    assert (admin.email, admin.name, admin.role) == ("ada@example.com", "Ada Admin", ADMIN)
    assert admin.active and not admin.must_change_password
    assert accounts.verify_password(PASSWORD, admin.password_hash)
    (event,) = logged(database, audit.USER_CREATED)
    assert (event.actor_id, event.subject_user_id, event.ip) == (admin.id, admin.id, "10.0.0.1")
    assert event.details == {"role": ADMIN, "email": "ada@example.com"}
    with pytest.raises(AccountError) as again, database.session() as session:
        accounts.set_up_first_admin(session, "eve@example.com", "Eve", PASSWORD)
    assert refusal(again) == (accounts.ALREADY_SET_UP, 409)
    with database.session() as session:
        assert accounts.count_users(session) == 1


def test_the_first_admin_follows_the_rules(database):
    for email, password, message in (
        ("ada@example.com", "short", accounts.PASSWORD_TOO_SHORT),
        ("ada", PASSWORD, accounts.INVALID_EMAIL),
    ):
        with pytest.raises(AccountError, match=message), database.session() as session:
            accounts.set_up_first_admin(session, email, "Ada", password)
    with database.session() as session:
        assert accounts.count_users(session) == 0
    assert logged(database) == []


def test_finding_listing_and_acting_as_accounts(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin, "Bea@Example.com", "bea")
    new_user(database, admin, "cid@example.com", "Cid")
    with database.session() as session:
        assert accounts.find_by_email(session, " BEA@example.COM ").id == bea.id
        assert accounts.find_by_email(session, "nobody@example.com") is None
        assert accounts.get_user(session, bea.id).email == "bea@example.com"
        assert accounts.get_user(session, "nobody") is None
        names = [user.name for user in accounts.list_users(session)]
        assert names == ["Ada Admin", "bea", "Cid"]
        actor = accounts.actor_of(accounts.get_user(session, bea.id))
    assert (actor.id, actor.email, actor.name, actor.role) == (bea.id, bea.email, "bea", USER)


# Creating accounts


def test_an_admin_creates_an_account_with_a_temporary_password(database):
    admin = first_admin(database)
    with database.session() as session:
        user, temporary = accounts.create_user(
            session, admin, " Bea@Example.com", "Bea User", USER, ip="10.0.0.2"
        )
    assert (user.email, user.role, user.must_change_password) == ("bea@example.com", USER, True)
    assert accounts.verify_password(temporary, user.password_hash)
    (event,) = logged(database, audit.USER_CREATED, subject_user_id=user.id)
    assert (event.actor_id, event.ip) == (admin.id, "10.0.0.2")
    assert event.details == {"role": USER, "email": "bea@example.com"}
    assert temporary not in str(event.details)


def test_who_must_change_the_password_at_the_first_sign_in(database):
    admin = first_admin(database)
    by_admin, given = new_user(database, admin, "bea@example.com", password=OTHER_PASSWORD)
    assert given == OTHER_PASSWORD and by_admin.must_change_password, "the admin knows it"
    by_cli, given = new_user(database, None, "cid@example.com", password=OTHER_PASSWORD)
    assert given == OTHER_PASSWORD and not by_cli.must_change_password
    generated, temporary = new_user(database, None, "dan@example.com", role=ADMIN)
    assert generated.must_change_password and generated.role == ADMIN
    assert accounts.verify_password(temporary, generated.password_hash)
    (event,) = logged(database, audit.USER_CREATED, subject_user_id=by_cli.id)
    assert event.actor_id is None, "the command line acts as the system"


def test_a_user_cannot_create_accounts(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin)
    with pytest.raises(Forbidden), database.session() as session:
        accounts.create_user(session, accounts.actor_of(bea), "eve@example.com", "Eve", USER)
    assert user_count(database) == 2
    assert len(logged(database, audit.USER_CREATED)) == 2


def test_an_email_taken_in_another_case_is_refused(database):
    admin = first_admin(database)
    new_user(database, admin, "Bea@Example.com")
    with pytest.raises(AccountError) as taken:
        new_user(database, admin, " bea@example.COM ", "Another Bea")
    assert refusal(taken) == (accounts.EMAIL_TAKEN, 409)
    assert user_count(database) == 2


@pytest.mark.parametrize(
    ("email", "name", "role", "password", "message"),
    [
        ("no-at-sign", "Bea", USER, None, accounts.INVALID_EMAIL),
        ("bea@localhost", "Bea", USER, None, accounts.INVALID_EMAIL),
        ("b ea@example.com", "Bea", USER, None, accounts.INVALID_EMAIL),
        ("bea@@example.com", "Bea", USER, None, accounts.INVALID_EMAIL),
        ("b" * 310 + "@example.com", "Bea", USER, None, accounts.INVALID_EMAIL),
        ("bea@example.com", "   ", USER, None, accounts.EMPTY_NAME),
        ("bea@example.com", "B" * 256, USER, None, accounts.NAME_TOO_LONG),
        ("bea@example.com", "Bea", "owner", None, accounts.UNKNOWN_ROLE),
        ("bea@example.com", "Bea", USER, "short", accounts.PASSWORD_TOO_SHORT),
        ("bea@example.com", "Bea", USER, "BEA@example.com", accounts.PASSWORD_IS_EMAIL),
    ],
)
def test_invalid_input_creates_nothing(database, email, name, role, password, message):
    admin = first_admin(database)
    with pytest.raises(AccountError) as refused:
        new_user(database, admin, email, name, role, password)
    assert refusal(refused) == (message, 400)
    assert user_count(database) == 1


def user_count(database):
    with database.session() as session:
        return accounts.count_users(session)


# Changing accounts


def test_renaming_and_changing_a_role_record_their_events(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin)
    with database.session() as session:
        user = accounts.update_user(
            session, admin, bea.id, name=" Beatrice ", role=ADMIN, ip="ip"
        )
    assert (user.name, user.role) == ("Beatrice", ADMIN)
    (renamed,) = logged(database, audit.USER_RENAMED)
    assert (renamed.actor_id, renamed.subject_user_id, renamed.ip) == (admin.id, bea.id, "ip")
    assert renamed.details == {"old_name": "Bea User", "name": "Beatrice"}
    (role,) = logged(database, audit.ROLE_CHANGED)
    assert (role.actor_id, role.subject_user_id) == (admin.id, bea.id)
    assert role.details == {"role": ADMIN, "old_role": USER}
    with database.session() as session:
        accounts.update_user(session, admin, bea.id, name="Beatrice", role=ADMIN)
    assert len(logged(database)) == 4, "no change, no event"


def test_an_admin_cannot_change_their_own_role(database):
    admin = first_admin(database)
    new_user(database, admin, "dan@example.com", role=ADMIN)
    with pytest.raises(AccountError) as own, database.session() as session:
        accounts.update_user(session, admin, admin.id, role=USER)
    assert refusal(own) == (accounts.OWN_ROLE, 409)
    assert user_row(database, admin.id).role == ADMIN
    assert logged(database, audit.ROLE_CHANGED) == []


def test_the_last_active_admin_cannot_be_demoted(database):
    admin = first_admin(database)
    with pytest.raises(AccountError, match=accounts.LAST_ADMIN), database.session() as session:
        accounts.update_user(session, None, admin.id, role=USER)
    dan, _ = new_user(database, admin, "dan@example.com", role=ADMIN)
    with database.session() as session:
        accounts.set_active(session, admin, dan.id, False)
    stale_dan = accounts.actor_of(dan)
    with pytest.raises(AccountError, match=accounts.LAST_ADMIN), database.session() as session:
        accounts.update_user(session, stale_dan, admin.id, role=USER)
    with database.session() as session:
        accounts.set_active(session, admin, dan.id, True)
        accounts.update_user(session, accounts.actor_of(dan), admin.id, role=USER)
    assert user_row(database, admin.id).role == USER
    (event,) = logged(database, audit.ROLE_CHANGED)
    assert (event.actor_id, event.subject_user_id) == (dan.id, admin.id)


def test_a_user_cannot_change_accounts(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin)
    as_bea = accounts.actor_of(bea)
    for change in (
        lambda s: accounts.update_user(s, as_bea, bea.id, name="Queen Bea"),
        lambda s: accounts.update_user(s, as_bea, bea.id, role=ADMIN),
        lambda s: accounts.set_active(s, as_bea, admin.id, False),
        lambda s: accounts.reset_password(s, as_bea, admin.id),
    ):
        with pytest.raises(Forbidden), database.session() as session:
            change(session)
    assert user_row(database, bea.id).name == "Bea User"
    assert len(logged(database)) == 2


def test_an_unknown_account_is_not_found(database):
    admin = first_admin(database)
    for change in (
        lambda s: accounts.update_user(s, admin, "nobody", name="X"),
        lambda s: accounts.set_active(s, admin, "nobody", False),
        lambda s: accounts.reset_password(s, admin, "nobody"),
        lambda s: accounts.change_password(s, "nobody", PASSWORD, OTHER_PASSWORD),
        lambda s: accounts.rename_self(s, "nobody", "X"),
    ):
        with pytest.raises(AccountError) as missing, database.session() as session:
            change(session)
        assert refusal(missing) == (accounts.NO_SUCH_USER, 404)


def test_deactivating_ends_the_sessions_and_activating_brings_the_account_back(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin)
    token = sign_in(database, bea.id)
    assert signed_in_user(database, token).id == bea.id
    with database.session() as session:
        user = accounts.set_active(session, admin, bea.id, False, ip="ip")
    assert not user.active
    assert sessions_of(database, bea.id) == []
    assert signed_in_user(database, token) is None
    (off,) = logged(database, audit.USER_DEACTIVATED)
    assert (off.actor_id, off.subject_user_id, off.ip) == (admin.id, bea.id, "ip")
    with database.session() as session:
        accounts.set_active(session, admin, bea.id, False)
        accounts.set_active(session, admin, bea.id, True)
        accounts.set_active(session, admin, bea.id, True)
    assert user_row(database, bea.id).active
    assert signed_in_user(database, token) is None, "an old session does not come back"
    (on,) = logged(database, audit.USER_ACTIVATED)
    assert (on.actor_id, on.subject_user_id) == (admin.id, bea.id)
    assert len(logged(database, audit.USER_DEACTIVATED)) == 1, "no change, no event"


def test_an_admin_cannot_deactivate_themself(database):
    admin = first_admin(database)
    new_user(database, admin, "dan@example.com", role=ADMIN)
    with pytest.raises(AccountError) as own, database.session() as session:
        accounts.set_active(session, admin, admin.id, False)
    assert refusal(own) == (accounts.DEACTIVATE_SELF, 409)
    assert user_row(database, admin.id).active


def test_the_last_active_admin_cannot_be_deactivated(database):
    admin = first_admin(database)
    token = sign_in(database, admin.id)
    with pytest.raises(AccountError, match=accounts.LAST_ADMIN), database.session() as session:
        accounts.set_active(session, None, admin.id, False)
    assert user_row(database, admin.id).active
    assert signed_in_user(database, token).id == admin.id
    assert logged(database, audit.USER_DEACTIVATED) == []


def test_a_reset_gives_a_temporary_password_and_ends_the_sessions(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin, password=PASSWORD)
    with database.session() as session:
        accounts.change_password(session, bea.id, PASSWORD, OTHER_PASSWORD)
    token = sign_in(database, bea.id)
    with database.session() as session:
        temporary = accounts.reset_password(session, admin, bea.id, ip="ip")
    user = user_row(database, bea.id)
    assert user.must_change_password
    assert not accounts.verify_password(OTHER_PASSWORD, user.password_hash)
    assert accounts.verify_password(temporary, user.password_hash)
    assert signed_in_user(database, token) is None
    assert sessions_of(database, bea.id) == []
    (event,) = logged(database, audit.PASSWORD_RESET)
    assert (event.actor_id, event.subject_user_id, event.ip) == (admin.id, bea.id, "ip")
    assert temporary not in str(event.details)


# The account's own changes


def test_a_wrong_current_password_changes_nothing(database):
    admin = first_admin(database)
    before = user_row(database, admin.id).password_hash
    with pytest.raises(AccountError) as wrong, database.session() as session:
        accounts.change_password(session, admin.id, OTHER_PASSWORD, "a brand new password")
    assert refusal(wrong) == (accounts.WRONG_CURRENT_PASSWORD, 400)
    assert user_row(database, admin.id).password_hash == before
    assert logged(database, audit.PASSWORD_CHANGED) == []


@pytest.mark.parametrize(
    ("new", "message"),
    [
        ("short", accounts.PASSWORD_TOO_SHORT),
        ("ADA@example.com", accounts.PASSWORD_IS_EMAIL),
        (PASSWORD, accounts.SAME_PASSWORD),
    ],
)
def test_a_new_password_follows_the_rules_and_differs(database, new, message):
    admin = first_admin(database)
    with pytest.raises(AccountError, match=message), database.session() as session:
        accounts.change_password(session, admin.id, PASSWORD, new)


def test_changing_the_password_ends_the_other_sessions_and_keeps_this_one(database):
    admin = first_admin(database)
    bea, temporary = new_user(database, admin)
    here, elsewhere = sign_in(database, bea.id), sign_in(database, bea.id)
    with database.session() as session:
        user = accounts.change_password(
            session, bea.id, temporary, OTHER_PASSWORD, ip="ip", keep_token=here
        )
    assert not user.must_change_password
    assert accounts.verify_password(OTHER_PASSWORD, user.password_hash)
    assert not accounts.verify_password(temporary, user.password_hash)
    assert signed_in_user(database, here).id == bea.id
    assert signed_in_user(database, elsewhere) is None
    (event,) = logged(database, audit.PASSWORD_CHANGED)
    assert (event.actor_id, event.subject_user_id, event.ip) == (bea.id, bea.id, "ip")
    assert OTHER_PASSWORD not in str(event.details)


def test_an_account_renames_itself(database):
    admin = first_admin(database)
    with database.session() as session:
        assert accounts.rename_self(session, admin.id, "  Ada Lovelace ", ip="ip").name == (
            "Ada Lovelace"
        )
        accounts.rename_self(session, admin.id, "Ada Lovelace")
    (event,) = logged(database, audit.USER_RENAMED)
    assert (event.actor_id, event.subject_user_id, event.ip) == (admin.id, admin.id, "ip")
    assert event.details == {"old_name": "Ada Admin", "name": "Ada Lovelace"}
    with pytest.raises(AccountError, match=accounts.EMPTY_NAME), database.session() as session:
        accounts.rename_self(session, admin.id, " ")


def test_events_of_accounts_read_as_sentences(database):
    admin = first_admin(database)
    bea, _ = new_user(database, admin)
    with database.session() as session:
        accounts.update_user(session, admin, bea.id, name="Beatrice")
    sentences = [audit.describe(event) for event in logged(database)]
    assert sentences == [
        "Ada Admin renamed Bea User to Beatrice",
        "Ada Admin created the account of Beatrice (user)",
        "Ada Admin created the account of Ada Admin (admin)",
    ]


def test_every_message_has_serbian():
    from contract_assistant.web.i18n import SERBIAN, missing

    assert missing(SERBIAN, list(accounts.TRANSLATED)) == []


def test_the_test_helper_accounts_cannot_sign_in(database):
    actor = add_user(database, "helper@example.com")
    with database.session() as session:
        assert not accounts.verify_password("", session.get(User, actor.id).password_hash)
