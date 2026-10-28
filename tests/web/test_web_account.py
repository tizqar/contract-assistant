"""The account page: name, role and what it means, the signed-in browsers, recent activity."""

from sqlalchemy import select

from contract_assistant import accounts, audit
from contract_assistant.db.models import LoginSession, User
from contract_assistant.web import account
from contract_assistant.web.auth import COOKIE
from contract_assistant.web.i18n import translate
from tests.web.conftest import SHORT_DOC_ID

CHROME_MAC = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
EDGE_WINDOWS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
)
SAFARI_IPHONE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
)


def other_browser(database, user_id: str, user_agent: str = CHROME_MAC) -> str:
    with database.session() as session:
        user = session.get(User, user_id)
        return accounts.start_session(session, user, days=1, user_agent=user_agent, ip="10.1.1.1")


def signed_in(database, token: str) -> bool:
    with database.session() as session:
        return accounts.session_user(session, token) is not None


def test_a_user_sees_their_details_and_what_their_role_lets_them_do(client):
    page = client.get("/account").text
    assert 'value="Reader"' in page and "reader@example.com" in page
    assert ">User</span>" in page and 'href="/account/password"' in page
    for line in (*account.CAN["user"], *account.DUTIES["user"]):
        assert line in page
    assert account.CAN["admin"][1] not in page
    assert "This browser" in page and "Only this browser is signed in." in page


def test_an_admin_sees_the_administrators_list(make_client, admin):
    page = make_client(as_actor=admin).get("/account").text
    assert ">Administrator</span>" in page
    for line in (*account.CAN["admin"], *account.DUTIES["admin"]):
        assert line.replace("'", "&#39;") in page
    assert account.DUTIES["user"][0] not in page


def test_the_page_lists_the_users_own_recent_activity(client, database, admin):
    page = client.get("/account").text
    assert "Reader added short_supply_agreement.txt" in page
    assert f'href="/documents/{SHORT_DOC_ID}"' in page
    with database.session() as session:
        audit.record(session, audit.LOGIN, actor_id=admin.id, subject_user_id=admin.id)
    assert "Admin signed in" not in client.get("/account").text


def test_renaming_oneself(client, database):
    response = client.post("/account/name", data={"name": " Rita "}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/account?done=name"
    page = client.get(response.headers["location"]).text
    assert account.NAME_SAVED in page and ">Rita</span></summary>" in page
    with database.session() as session:
        [renamed] = audit.events(session, audit.EventFilter(actions=[audit.USER_RENAMED]))
    assert renamed.actor_id == "reader" and renamed.details["old_name"] == "Reader"
    empty = client.post("/account/name", data={"name": " "})
    assert empty.status_code == 400 and accounts.EMPTY_NAME in empty.text
    assert 'value=" "' in empty.text


def test_signing_out_the_other_browsers_keeps_this_one(client, database):
    first = other_browser(database, "reader")
    second = other_browser(database, "reader", SAFARI_IPHONE)
    page = client.get("/account").text
    assert "Chrome on macOS" in page and "Safari on iPhone" in page and "10.1.1.1" in page
    assert 'action="/account/sign-out-others"' in page
    response = client.post("/account/sign-out-others", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/account?ended=2"
    assert not signed_in(database, first) and not signed_in(database, second)
    assert signed_in(database, client.cookies.get(COOKIE))
    after = client.get(response.headers["location"])
    assert after.status_code == 200 and "Other browsers signed out: 2." in after.text
    with database.session() as session:
        rows = session.scalars(select(LoginSession)).all()
        [logged] = audit.events(session, audit.EventFilter(actions=[audit.SESSIONS_ENDED]))
    assert len(rows) == 1
    assert logged.actor_id == "reader" and logged.details == {"count": 2}


def test_signing_out_the_others_leaves_other_peoples_browsers(client, database, admin):
    theirs = other_browser(database, admin.id)
    client.post("/account/sign-out-others")
    assert signed_in(database, theirs)


def test_the_page_speaks_serbian(client):
    client.cookies.set("lang", "sr")
    page = client.get("/account").text
    assert translate("What your role lets you do", "sr") in page
    assert translate(account.CAN["user"][0], "sr") in page
    response = client.post("/account/sign-out-others")
    assert "Odjavljeni drugi pregledači: 0." in response.text


def test_a_browser_is_named_in_plain_words():
    assert account.browser_name(CHROME_MAC, "en") == "Chrome on macOS"
    assert account.browser_name(EDGE_WINDOWS, "en") == "Edge on Windows"
    assert account.browser_name(SAFARI_IPHONE, "sr") == "Safari na sistemu iPhone"
    assert account.browser_name("testclient", "en") == account.UNKNOWN_BROWSER
    assert account.browser_name(None, "sr") == "Nepoznat pregledač"
