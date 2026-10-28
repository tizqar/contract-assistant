"""The admin pages: who may open them, the accounts, the activity log and its CSV, approvals."""

import csv
import io
from datetime import datetime, timedelta

import pytest
from markupsafe import escape
from sqlalchemy import func, select

from contract_assistant import accounts, audit, files
from contract_assistant.access import NOT_ALLOWED, ReviewStep
from contract_assistant.db.models import APPROVED, IN_REVIEW, Event, LoginSession, User
from contract_assistant.web import admin
from contract_assistant.web.i18n import translate
from tests.support.db import add_user
from tests.web.conftest import MEDIUM_DOC_ID, SHORT_DOC_ID

PAGES = ("/admin", "/admin/users", "/admin/users/reader", "/admin/activity", "/admin/approvals")
FORMS = (
    ("/admin/users", {"name": "Eve", "email": "eve@example.com", "role": "admin"}),
    ("/admin/users/reader/name", {"name": "Someone else"}),
    ("/admin/users/reader/role", {"role": "admin"}),
    ("/admin/users/reader/active", {"active": "0"}),
    ("/admin/users/reader/password", {}),
)


@pytest.fixture
def as_admin(make_client, admin):
    return make_client(as_actor=admin)


def user_row(database, user_id: str) -> User:
    with database.session() as session:
        return session.get(User, user_id)


def logged(database, action: str) -> list[audit.LoggedEvent]:
    with database.session() as session:
        return audit.events(session, audit.EventFilter(actions=[action]))


def sessions_of(database, user_id: str) -> int:
    with database.session() as session:
        query = select(func.count()).where(LoginSession.user_id == user_id)
        return session.scalar(query.select_from(LoginSession))


def count_events(database) -> int:
    with database.session() as session:
        return session.scalar(select(func.count()).select_from(Event))


# Who may open them


def test_a_user_is_told_in_plain_words_that_the_admin_pages_are_not_for_them(make_client):
    client = make_client()
    for url in PAGES:
        response = client.get(url, follow_redirects=False)
        assert response.status_code == 403, url
        assert admin.FOR_ADMINS in response.text and 'href="/files"' in response.text
        assert 'class="admin-tabs"' not in response.text


def test_a_user_cannot_post_an_admin_form_or_download_the_log(make_client, database):
    client = make_client()
    for url, data in FORMS:
        response = client.post(url, data=data, follow_redirects=False)
        assert response.status_code == 403 and response.text == NOT_ALLOWED, url
    assert client.get("/admin/activity.csv").status_code == 403
    assert user_row(database, "reader").name == "Reader"
    assert user_row(database, "reader").role == "user"


def test_the_admin_home_opens_the_users_with_the_tab_bar(as_admin):
    response = as_admin.get("/admin", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/admin/users"
    page = as_admin.get("/admin/users").text
    for href in ('href="/admin/users"', 'href="/admin/activity"', 'href="/admin/approvals"'):
        assert href in page
    assert 'aria-current="page">Users</a>' in page


# Accounts


def test_the_users_table_lists_every_account(as_admin, database, reader):
    page = as_admin.get("/admin/users").text
    assert "reader@example.com" in page and "admin@example.com" in page
    assert 'href="/admin/users/reader"' in page and "(you)" in page
    assert ">Never</td>" in page and ">Active</span>" in page


def test_creating_an_account_shows_its_temporary_password_once(as_admin, database, admin):
    form = {"name": "Neo Anderson", "email": " Neo@Example.com ", "role": "user"}
    response = as_admin.post("/admin/users", data=form)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    with database.session() as session:
        user = accounts.find_by_email(session, "neo@example.com")
    assert user.name == "Neo Anderson" and user.must_change_password and user.role == "user"
    password = response.text.split('id="temporary-password">', 1)[1].split("<", 1)[0]
    assert accounts.verify_password(password, user.password_hash)
    assert "Account created for Neo Anderson (neo@example.com)." in response.text
    assert "they must change it at the first sign-in" in response.text
    assert f'data-copy="{password}"' in response.text
    assert password not in as_admin.get("/admin/users").text
    [created] = logged(database, audit.USER_CREATED)
    assert created.actor_id == admin.id and created.subject_user_id == user.id
    assert created.details == {"role": "user", "email": "neo@example.com"}


def test_an_email_that_is_taken_is_refused_at_the_form(as_admin, database):
    form = {"name": "Second reader", "email": "READER@example.com", "role": "admin"}
    response = as_admin.post("/admin/users", data=form)
    assert response.status_code == 409 and accounts.EMAIL_TAKEN in response.text
    assert 'value="Second reader"' in response.text
    assert '<option value="admin" selected>' in response.text
    assert 'id="temporary-password"' not in response.text
    assert logged(database, audit.USER_CREATED) == []


def test_a_bad_email_is_refused_in_the_readers_language(as_admin):
    as_admin.cookies.set("lang", "sr")
    response = as_admin.post("/admin/users", data={"name": "X", "email": "nope", "role": "user"})
    assert response.status_code == 400
    assert translate(accounts.INVALID_EMAIL, "sr") in response.text


def test_an_account_page_shows_the_account_and_what_involved_it(as_admin, reader):
    page = as_admin.get("/admin/users/reader").text
    assert "<h1" in page and "Reader" in page and "reader@example.com" in page
    assert 'action="/admin/users/reader/name"' in page
    assert 'action="/admin/users/reader/role"' in page
    assert 'action="/admin/users/reader/password"' in page
    assert 'name="active" value="0"' in page
    # The reader added the demo contracts: those events link to the contracts.
    assert "Reader added short_supply_agreement.txt" in page
    assert f'href="/documents/{SHORT_DOC_ID}"' in page
    assert 'href="/admin/activity?person=reader"' in page


def test_an_unknown_account_is_a_plain_page(as_admin):
    response = as_admin.get("/admin/users/nobody")
    assert response.status_code == 404
    assert admin.NO_ACCOUNT in response.text and accounts.NO_SUCH_USER in response.text
    assert as_admin.post("/admin/users/nobody/name", data={"name": "X"}).status_code == 404


def test_renaming_an_account(as_admin, database):
    response = as_admin.post(
        "/admin/users/reader/name", data={"name": " Rita Reader "}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/users/reader?done=name"
    assert user_row(database, "reader").name == "Rita Reader"
    page = as_admin.get(response.headers["location"]).text
    assert admin.NAME_SAVED in page
    [renamed] = logged(database, audit.USER_RENAMED)
    assert renamed.details == {"old_name": "Reader", "name": "Rita Reader"}
    empty = as_admin.post("/admin/users/reader/name", data={"name": "  "})
    assert empty.status_code == 400 and accounts.EMPTY_NAME in empty.text


def test_changing_a_role(as_admin, database):
    response = as_admin.post(
        "/admin/users/reader/role", data={"role": "admin"}, follow_redirects=False
    )
    assert response.headers["location"] == "/admin/users/reader?done=role"
    assert user_row(database, "reader").role == "admin"
    [changed] = logged(database, audit.ROLE_CHANGED)
    assert changed.details == {"role": "admin", "old_role": "user"}
    unknown = as_admin.post("/admin/users/reader/role", data={"role": "owner"})
    assert unknown.status_code == 400 and accounts.UNKNOWN_ROLE in unknown.text


def test_an_admin_cannot_change_their_own_role_or_deactivate_themself(as_admin, database):
    own = as_admin.get("/admin/users/admin").text
    assert accounts.OWN_ROLE in own and accounts.DEACTIVATE_SELF in own
    assert 'action="/admin/users/admin/role"' not in own
    assert 'action="/admin/users/admin/password"' not in own
    assert 'href="/account/password"' in own
    as_admin.cookies.set("lang", "sr")
    role = as_admin.post("/admin/users/admin/role", data={"role": "user"})
    assert role.status_code == 409 and translate(accounts.OWN_ROLE, "sr") in role.text
    assert 'class="auth-error" role="alert"' in role.text
    active = as_admin.post("/admin/users/admin/active", data={"active": "0"})
    assert active.status_code == 409
    assert translate(accounts.DEACTIVATE_SELF, "sr") in active.text
    assert user_row(database, "admin").role == "admin" and user_row(database, "admin").active


def test_the_last_admin_guard_is_shown_at_its_form(as_admin, monkeypatch):
    def refuse(*args, **kwargs):
        raise accounts.AccountError(accounts.LAST_ADMIN, 409)

    monkeypatch.setattr(accounts, "update_user", refuse)
    response = as_admin.post("/admin/users/reader/role", data={"role": "admin"})
    assert response.status_code == 409 and accounts.LAST_ADMIN in response.text
    head, form = response.text.split('id="role"', 1)
    assert accounts.LAST_ADMIN not in head and accounts.LAST_ADMIN in form.split('id="active"')[0]


def test_deactivating_signs_the_account_out_and_activating_lets_it_back(
    as_admin, make_client, database, reader
):
    make_client()  # a signed-in browser of the reader
    assert sessions_of(database, "reader") == 1
    response = as_admin.post(
        "/admin/users/reader/active", data={"active": "0"}, follow_redirects=False
    )
    assert response.headers["location"] == "/admin/users/reader?done=deactivated"
    assert not user_row(database, "reader").active and sessions_of(database, "reader") == 0
    page = as_admin.get(response.headers["location"]).text
    assert admin.DEACTIVATED in page and 'name="active" value="1"' in page
    assert ">Deactivated</span>" in as_admin.get("/admin/users").text
    again = as_admin.post("/admin/users/reader/active", data={"active": "1"}, follow_redirects=False)
    assert again.headers["location"] == "/admin/users/reader?done=activated"
    assert user_row(database, "reader").active
    assert [event.action for event in logged(database, audit.USER_DEACTIVATED)] == [
        audit.USER_DEACTIVATED
    ]
    assert len(logged(database, audit.USER_ACTIVATED)) == 1


def test_resetting_a_password_shows_the_new_one_once(as_admin, make_client, database):
    make_client()
    response = as_admin.post("/admin/users/reader/password")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    password = response.text.split('id="temporary-password">', 1)[1].split("<", 1)[0]
    user = user_row(database, "reader")
    assert accounts.verify_password(password, user.password_hash) and user.must_change_password
    assert sessions_of(database, "reader") == 0
    assert escape(admin.PASSWORD_RESET) in response.text
    assert "Temporary password for Reader" in response.text
    assert password not in as_admin.get("/admin/users/reader").text
    [reset] = logged(database, audit.PASSWORD_RESET)
    assert reset.actor_id == "admin" and reset.subject_user_id == "reader"


# The activity log


def add_events(database, count: int, actor_id: str = "reader") -> None:
    with database.session() as session:
        for _ in range(count):
            audit.record(session, audit.LOGIN, actor_id=actor_id, subject_user_id=actor_id)


def sentences(page: str) -> list[str]:
    return [part.split("</span>", 1)[0] for part in page.split('class="activity-sentence">')[1:]]


def test_the_log_shows_the_newest_first_with_links_and_addresses(as_admin, database):
    with database.session() as session:
        audit.record(
            session,
            audit.CLAUSE_CONFIRMED,
            actor_id="reader",
            contract_id=SHORT_DOC_ID,
            details={"clause": "Governing Law", "name": "short_supply_agreement.txt"},
            ip="10.0.0.7",
        )
    page = as_admin.get("/admin/activity").text
    found = sentences(page)
    assert found[0] == "Reader confirmed the Governing Law passage of short_supply_agreement.txt"
    assert found[-1] == "Reader added long_supply_agreement.txt" or len(found) == 4
    assert f'href="/documents/{SHORT_DOC_ID}"' in page and "10.0.0.7" in page
    assert 'aria-current="page">Activity</a>' in page


def test_the_log_filters_by_person_kind_contract_and_dates(as_admin, database, reader):
    other = add_user(database, "other@example.com", name="Other", user_id="other")
    add_events(database, 2, actor_id=other.id)
    by_person = sentences(as_admin.get("/admin/activity?person=other").text)
    assert by_person == ["Other signed in", "Other signed in"]
    by_kind = sentences(as_admin.get("/admin/activity?group=contract").text)
    assert len(by_kind) == 3 and all(" added " in line for line in by_kind)
    by_name = sentences(as_admin.get("/admin/activity?contract=SPONSORSHIP").text)
    assert by_name == ["Reader added medium_sponsorship_agreement.txt"]
    today = datetime.now().astimezone().date()
    yesterday = (today - timedelta(days=1)).isoformat()
    assert len(sentences(as_admin.get(f"/admin/activity?from={today}&to={today}").text)) == 5
    page = as_admin.get(f"/admin/activity?to={yesterday}").text
    assert sentences(page) == [] and "No activity matches these filters." in page
    assert 'href="/admin/activity">Clear the filters</a>' in page
    nonsense = as_admin.get("/admin/activity?group=nothing&from=soon&before=x")
    assert nonsense.status_code == 200 and len(sentences(nonsense.text)) == 5


def test_a_renamed_contract_is_found_by_its_old_and_new_name(as_admin, database, reader):
    with database.session() as session:
        files.rename_contract(session, reader, MEDIUM_DOC_ID, "Sponsorship deal.txt")
    by_new = sentences(as_admin.get("/admin/activity?contract=DEAL").text)
    by_old = sentences(as_admin.get("/admin/activity?contract=medium_sponsorship").text)
    assert by_new == [
        "Reader renamed medium_sponsorship_agreement.txt to Sponsorship deal.txt",
        "Reader added Sponsorship deal.txt",
    ]
    assert by_old == by_new


def test_the_log_pages_fifty_at_a_time(as_admin, database):
    add_events(database, 60)
    first = as_admin.get("/admin/activity").text
    assert len(sentences(first)) == admin.PAGE_SIZE
    older = first.split('class="admin-pager">', 1)[1].split('href="', 1)[1].split('"', 1)[0]
    assert older.startswith("/admin/activity?before=")
    second = as_admin.get(older.replace("&amp;", "&")).text
    assert len(sentences(second)) == 13  # 60 sign-ins and the three contracts added
    assert 'class="admin-pager"' not in second and "Back to the newest" in second
    filtered = as_admin.get("/admin/activity?person=reader&group=user").text
    link = filtered.split('class="admin-pager">', 1)[1].split('href="', 1)[1].split('"', 1)[0]
    assert "person=reader" in link and "group=user" in link and "before=" in link


def test_the_csv_has_every_matching_event_in_english_and_logs_nothing(as_admin, database):
    add_events(database, 60)
    with database.session() as session:
        audit.record(
            session,
            audit.CONTRACT_EXPORTED,
            actor_id="reader",
            contract_id=SHORT_DOC_ID,
            details={"name": "short_supply_agreement.txt", "what": "the report"},
            ip="10.0.0.9",
        )
    before = count_events(database)
    as_admin.cookies.set("lang", "sr")
    response = as_admin.get("/admin/activity.csv?contract=short")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/csv")
    assert 'filename="activity.csv"' in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0] == list(admin.CSV_HEADER)
    assert len(rows) == 3
    exported = rows[1]
    assert exported[0].endswith("+00:00") and "T" in exported[0]
    assert exported[1:4] == ["Reader", "reader@example.com", audit.CONTRACT_EXPORTED]
    assert exported[4] == "Reader downloaded the report of short_supply_agreement.txt"
    assert exported[5:] == [SHORT_DOC_ID, "short_supply_agreement.txt", "10.0.0.9"]
    everything = list(csv.reader(io.StringIO(as_admin.get("/admin/activity.csv").text)))
    assert len(everything) == 1 + 60 + 3 + 1
    assert count_events(database) == before


def test_a_csv_cell_never_starts_a_formula(database):
    event = audit.LoggedEvent(
        id=1,
        at=datetime.now().astimezone(),
        action=audit.USER_RENAMED,
        actor_id="a",
        actor_name="=HYPERLINK(1)",
        actor_email="a@example.com",
        contract_id=None,
        contract_name=None,
        folder_id=None,
        folder_name=None,
        subject_user_id="a",
        subject_name="=HYPERLINK(1)",
        ip=None,
        details={"old_name": "A", "name": "=HYPERLINK(1)"},
    )
    [_, row] = list(csv.reader(io.StringIO(admin.to_csv(iter([event])))))
    assert row[1] == "'=HYPERLINK(1)" and row[4].startswith("'=HYPERLINK(1) renamed")


# Approvals


def submit(database, actor, contract_id: str) -> None:
    with database.session() as session:
        files.review(session, actor, contract_id, ReviewStep.SUBMIT)


def test_the_approvals_list_the_contracts_sent_with_their_forms(as_admin, database, reader):
    empty = as_admin.get("/admin/approvals").text
    assert "No contract is waiting for approval." in empty and 'class="admin-count"' not in empty
    submit(database, reader, SHORT_DOC_ID)
    page = as_admin.get("/admin/approvals").text
    assert 'aria-current="page">Approvals <span class="admin-count"' in page
    assert "Owner: Reader" in page and "Sent for approval by Reader, " in page
    assert page.count(f'href="/documents/{SHORT_DOC_ID}"') == 2
    approve = page.split(f'action="/documents/{SHORT_DOC_ID}/review/approve"', 1)[1]
    assert '<input type="hidden" name="next" value="/admin/approvals">' in approve.split("</form>")[0]
    send_back = page.split(f'action="/documents/{SHORT_DOC_ID}/review/return"', 1)[1]
    form = send_back.split("</form>")[0]
    assert '<input type="hidden" name="next" value="/admin/approvals">' in form
    assert 'name="note"' in form and "required" in form
    assert ">1</span>" in as_admin.get("/admin/users").text


def test_approving_and_sending_back_come_back_to_the_list(as_admin, database, reader):
    submit(database, reader, SHORT_DOC_ID)
    submit(database, reader, MEDIUM_DOC_ID)
    approved = as_admin.post(
        f"/documents/{SHORT_DOC_ID}/review/approve",
        data={"next": admin.APPROVALS_PATH},
        follow_redirects=False,
    )
    assert approved.status_code == 303 and approved.headers["location"] == admin.APPROVALS_PATH
    returned = as_admin.post(
        f"/documents/{MEDIUM_DOC_ID}/review/return",
        data={"note": "Check the notice period.", "next": admin.APPROVALS_PATH},
        follow_redirects=False,
    )
    assert returned.headers["location"] == admin.APPROVALS_PATH
    with database.session() as session:
        short = files.contract(session, reader, SHORT_DOC_ID)
        medium = files.contract(session, reader, MEDIUM_DOC_ID)
    assert short.review_status == APPROVED
    assert medium.review_status == IN_REVIEW and medium.review_note == "Check the notice period."
    assert "No contract is waiting for approval." in as_admin.get(admin.APPROVALS_PATH).text
