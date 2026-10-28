"""Signing in: the guard on every page, the first admin, passwords, sign-out, other sites."""

from fastapi.testclient import TestClient
from sqlalchemy import select

from contract_assistant import accounts, audit
from contract_assistant.db.models import User
from contract_assistant.web import auth
from contract_assistant.web.app import create_app
from tests.web.conftest import READER_EMAIL, SHORT_DOC_ID

PASSWORD = "correct horse battery"


def signed_out(make_services, database) -> TestClient:
    return TestClient(create_app(make_services(), database))


def with_password(database, email: str, password: str = PASSWORD, must_change=False) -> None:
    with database.session() as session:
        user = session.scalars(select(User).where(User.email == email)).one()
        user.password_hash = accounts.hash_password(password)
        user.must_change_password = must_change


def events(database, action: str):
    with database.session() as session:
        return audit.events(session, audit.EventFilter(actions=[action]))


def test_without_accounts_every_page_leads_to_the_set_up(make_services, database):
    client = signed_out(make_services, database)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == auth.SETUP_PATH
    assert client.get(auth.LOGIN_PATH, follow_redirects=False).headers["location"] == "/setup"
    page = client.get(auth.SETUP_PATH).text
    assert "Create the administrator account" in page and 'action="/setup"' in page


def test_the_set_up_makes_the_first_admin_and_signs_them_in(make_services, database):
    client = signed_out(make_services, database)
    form = {"name": "Ana", "email": "Ana@Example.com", "password": PASSWORD, "confirm": PASSWORD}
    response = client.post(auth.SETUP_PATH, data=form, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert auth.COOKIE in response.cookies
    home = client.get("/")
    assert "Read a contract with a local assistant" in home.text and ">Ana</summary>" in home.text
    assert ">Admin</a>" in home.text
    with database.session() as session:
        assert accounts.find_by_email(session, "ana@example.com").role == "admin"
    again = client.post(auth.SETUP_PATH, data=form, follow_redirects=False)
    assert again.headers["location"] == auth.LOGIN_PATH


def test_the_set_up_asks_for_the_same_password_twice(make_services, database):
    client = signed_out(make_services, database)
    form = {"name": "Ana", "email": "ana@example.com", "password": PASSWORD, "confirm": "other"}
    response = client.post(auth.SETUP_PATH, data=form)
    assert response.status_code == 400 and auth.PASSWORDS_DIFFER in response.text
    assert 'value="ana@example.com"' in response.text


def test_a_page_sends_a_browser_to_sign_in_and_back(make_services, database, reader):
    client = signed_out(make_services, database)
    response = client.get(f"/documents/{SHORT_DOC_ID}?low=0.1", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?next=%2Fdocuments%2Fsha256")
    assert client.get("/", follow_redirects=False).headers["location"] == auth.LOGIN_PATH


def test_htmx_and_scripts_get_401(make_services, database, reader):
    client = signed_out(make_services, database)
    htmx = client.get(f"/documents/{SHORT_DOC_ID}/rail", headers={"HX-Request": "true"})
    assert htmx.status_code == 401 and htmx.headers["HX-Redirect"].startswith("/login")
    posted = client.post("/upload", headers={"Accept": "application/x-ndjson"})
    assert posted.status_code == 401 and posted.text == "Sign in first."


def test_the_static_files_and_the_language_switch_need_no_account(make_services, database):
    client = signed_out(make_services, database)
    assert client.get("/static/app.css").status_code == 200
    assert client.post("/language", data={"lang": "sr", "next": "/"}, follow_redirects=False).status_code == 303


def test_signing_in_starts_a_session_and_goes_where_the_reader_was(
    make_services, database, reader
):
    with_password(database, READER_EMAIL)
    client = signed_out(make_services, database)
    form = {"email": "READER@example.com ", "password": PASSWORD, "next": "/portfolio"}
    response = client.post(auth.LOGIN_PATH, data=form, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/portfolio"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/portfolio").status_code == 200
    assert events(database, audit.LOGIN)[0].actor_id == reader.id


def test_a_wrong_password_is_refused_and_logged(make_services, database, reader):
    with_password(database, READER_EMAIL)
    client = signed_out(make_services, database)
    form = {"email": READER_EMAIL, "password": "not it at all", "next": "/"}
    response = client.post(auth.LOGIN_PATH, data=form)
    assert response.status_code == 401 and accounts.WRONG_CREDENTIALS in response.text
    assert auth.COOKIE not in response.cookies
    [failed] = events(database, audit.LOGIN_FAILED)
    assert failed.details["email"] == READER_EMAIL and failed.actor_id is None


def test_after_signing_in_only_pages_of_this_site_are_followed(make_services, database, reader):
    with_password(database, READER_EMAIL)
    client = signed_out(make_services, database)
    for target in ("https://example.com/", "//example.com/x", "/\\example.com"):
        form = {"email": READER_EMAIL, "password": PASSWORD, "next": target}
        response = client.post(auth.LOGIN_PATH, data=form, follow_redirects=False)
        assert response.headers["location"] == "/"


def test_a_temporary_password_must_be_changed_first(make_services, database, reader):
    with_password(database, READER_EMAIL, must_change=True)
    client = signed_out(make_services, database)
    form = {"email": READER_EMAIL, "password": PASSWORD, "next": "/portfolio"}
    response = client.post(auth.LOGIN_PATH, data=form, follow_redirects=False)
    assert response.headers["location"] == auth.PASSWORD_PATH
    elsewhere = client.get("/portfolio", follow_redirects=False)
    assert elsewhere.headers["location"] == auth.PASSWORD_PATH
    page = client.get(auth.PASSWORD_PATH).text
    assert "temporary password" in page and ">Cancel</a>" not in page
    new = "a brand new passphrase"
    changed = client.post(
        auth.PASSWORD_PATH,
        data={"current": PASSWORD, "new": new, "confirm": new},
        follow_redirects=False,
    )
    assert changed.status_code == 303 and changed.headers["location"] == "/"
    assert client.get("/portfolio").status_code == 200


def test_a_wrong_current_password_keeps_the_old_one(client, database):
    with_password(database, READER_EMAIL)
    form = {"current": "not the password", "new": "a brand new passphrase"}
    response = client.post(auth.PASSWORD_PATH, data={**form, "confirm": form["new"]})
    assert response.status_code == 400 and accounts.WRONG_CURRENT_PASSWORD in response.text
    with database.session() as session:
        user = accounts.find_by_email(session, READER_EMAIL)
        assert accounts.verify_password(PASSWORD, user.password_hash)


def test_signing_out_ends_the_session(client, database):
    response = client.post(auth.LOGOUT_PATH, follow_redirects=False)
    assert response.headers["location"] == auth.LOGIN_PATH
    assert client.get("/portfolio", follow_redirects=False).status_code == 303
    assert events(database, audit.LOGOUT)


def test_a_deactivated_account_is_signed_out(client, database, reader):
    with database.session() as session:
        session.get(User, reader.id).active = False
    assert client.get("/portfolio", follow_redirects=False).status_code == 303


def test_a_form_from_another_site_is_refused(client):
    url = f"/documents/{SHORT_DOC_ID}/chat/clear"
    refused = client.post(url, headers={"Origin": "https://evil.example"})
    assert refused.status_code == 403 and auth.OTHER_SITE in refused.text
    assert client.post(url, headers={"Origin": "null"}).status_code == 403
    assert client.post(url, headers={"Referer": "https://evil.example/page"}).status_code == 403
    assert client.post(url, headers={"Origin": "http://testserver"}).status_code == 200
    assert client.post(url).status_code == 200


def test_the_user_menu_names_the_reader_and_their_role(client):
    html = client.get(f"/documents/{SHORT_DOC_ID}").text
    assert ">Reader</summary>" in html and READER_EMAIL in html and ">User</span>" in html
    assert 'action="/logout"' in html and ">Admin</a>" not in html
