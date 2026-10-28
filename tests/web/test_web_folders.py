"""The file manager's pages: who sees which place, and that rows offer what is allowed."""

import json
import re

import pytest

from contract_assistant import access, files
from contract_assistant.db.models import EDIT, VIEW
from contract_assistant.extract.extractor import Extractor
from tests.support.db import add_user
from tests.support.fakes import FakeQA
from tests.web.conftest import LONG_DOC_ID, MEDIUM_DOC_ID, SHORT_DOC_ID, sign_in

# Every action URL a page offers: the dialogs it opens and the changes it posts.
ACTION = re.compile(
    r'hx-(?:get|post)="/files/(folders|contracts)/([^"/?]+)/'
    r"(rename|move|share|delete|restore|purge)"
)
EVERY_ACTION = {"rename", "move", "share", "delete"}


@pytest.fixture
def deals(database, settings, reader):
    """The reader's folder "Deals" with the short contract in it."""
    with database.session() as session:
        folder = files.create_folder(session, reader, None, "Deals")
    return folder


@pytest.fixture
def filled(make_client, database, reader, deals):
    """A client of the reader, the short contract moved into Deals."""
    client = make_client()
    with database.session() as session:
        files.move_contract(session, reader, SHORT_DOC_ID, deals.id)
    return client


@pytest.fixture
def colleague(database):
    return add_user(database, "colleague@example.com", name="Colleague", user_id="colleague")


def share(database, reader, folder, colleague, level: str) -> None:
    with database.session() as session:
        files.share_folder(session, reader, folder.id, colleague.email, level)


def as_user(make_client, database, actor, **kwargs):
    return sign_in(make_client(**kwargs), database, actor)


def offered(html: str) -> dict[str, set[str]]:
    """Return the actions a page offers, by the id of the folder or contract."""
    found: dict[str, set[str]] = {}
    for _, item_id, action in ACTION.findall(html):
        found.setdefault(item_id, set()).add(action)
    return found


def text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


# The places


def test_my_files_lists_the_folders_first_then_the_contracts(filled, deals):
    html = filled.get("/files").text
    assert html.index(f'href="/files/folders/{deals.id}"') < html.index(
        f'href="/documents/{LONG_DOC_ID}"'
    )
    assert f'href="/documents/{SHORT_DOC_ID}"' not in html
    entry = filled.ws.entry(MEDIUM_DOC_ID)
    shown = text(html)
    assert "Medium sponsorship agreement" in shown and "In review" in shown
    assert f"Text · {entry.size}" in shown and f"{entry.found} found" in shown
    assert 'class="badge files-review files-review-in_review"' in html
    assert "Owner:" not in html and 'aria-current="true">' in html
    assert 'hx-trigger="files-changed from:body"' in html and 'hx-get="/files"' in html
    # The rows' requests must not inherit the view's hx-select, or their dialogs come out empty.
    assert 'hx-select="#files-view" hx-swap="outerHTML" hx-disinherit="*"' in html
    assert 'id="files-dialog"' in html and "data-drawer-toggle" not in html


def test_a_folder_shows_where_it_is_and_uploads_into_itself(filled, deals):
    html = filled.get(f"/files/folders/{deals.id}").text
    assert '<a class="crumb" href="/files">My files</a>' in html
    assert '<span class="crumb crumb-current" aria-current="page">Deals</span>' in html
    assert f'href="/documents/{SHORT_DOC_ID}"' in html
    assert f'href="/documents/{MEDIUM_DOC_ID}"' not in html
    assert f'name="folder" value="{deals.id}"' in html and "It goes into Deals." in html
    assert f'hx-get="/files/folders/new?parent={deals.id}"' in html


def test_an_empty_folder_says_so(filled, database, reader):
    with database.session() as session:
        empty = files.create_folder(session, reader, None, "Empty")
    assert "This folder is empty." in filled.get(f"/files/folders/{empty.id}").text


def test_a_new_users_empty_space_shows_the_welcome(
    make_client, database, settings, reader, deals, colleague
):
    share(database, reader, deals, colleague, VIEW)
    extractor = Extractor(FakeQA(), [], settings, "fake")
    client = as_user(make_client, database, colleague, extractor=extractor)
    html = client.get("/files").text
    assert "Read a contract with a local assistant" in html and 'action="/upload"' in html
    assert 'name="folder"' not in html and "New folder" in html
    assert "Other people share folders with you." in html and 'href="/files/shared"' in html
    with database.session() as session:
        files.create_folder(session, colleague, None, "Mine")
    assert "Read a contract with a local assistant" not in client.get("/files").text


# Levels: the owner, a share to view, a share to edit, a stranger and an admin


def test_the_owner_is_offered_every_action(filled, deals):
    top = offered(filled.get("/files").text)
    assert top[deals.id] == EVERY_ACTION
    assert top[MEDIUM_DOC_ID] == {"rename", "move", "delete"}


def test_a_share_to_view_lists_the_folder_and_offers_nothing(
    make_client, database, reader, filled, deals, colleague
):
    share(database, reader, deals, colleague, VIEW)
    client = as_user(make_client, database, colleague)
    shared = client.get("/files/shared").text
    assert f'href="/files/folders/{deals.id}"' in shared
    assert "Owner: Reader" in shared and ">can view<" in shared
    html = client.get(f"/files/folders/{deals.id}").text
    assert '<a class="crumb" href="/files/shared">Shared with me</a>' in html
    assert "Owner: Reader" in html and "You can read the contracts here." in html
    assert f'href="/documents/{SHORT_DOC_ID}"' in html
    assert offered(html) == {} and "New folder" not in html and 'action="/upload"' not in html
    assert client.get(f"/files/contracts/{SHORT_DOC_ID}/rename").status_code == 403
    refused = client.post(f"/files/contracts/{SHORT_DOC_ID}/rename", data={"name": "x.txt"})
    assert refused.status_code == 403 and refused.text == access.NOT_ALLOWED
    assert client.post(f"/files/folders/{deals.id}/delete").status_code == 403
    assert client.get("/files/folders/new", params={"parent": deals.id}).status_code == 403


def test_a_share_to_edit_offers_rename_and_adding_only(
    make_client, database, reader, filled, deals, colleague
):
    share(database, reader, deals, colleague, EDIT)
    client = as_user(make_client, database, colleague)
    assert ">can edit<" in client.get("/files/shared").text
    html = client.get(f"/files/folders/{deals.id}").text
    assert offered(html) == {SHORT_DOC_ID: {"rename"}}
    assert "You can read and change the contracts here." in html
    assert "New folder" in html and f'name="folder" value="{deals.id}"' in html
    moved = client.post(f"/files/contracts/{SHORT_DOC_ID}/move", data={"target": ""})
    assert moved.status_code == 403
    renamed = client.post(f"/files/folders/{deals.id}/rename", data={"name": "Deals 2"})
    assert renamed.status_code == 204


def test_a_stranger_cannot_open_or_change_a_folder(
    make_client, database, filled, deals, colleague
):
    client = as_user(make_client, database, colleague)
    for url in (
        f"/files/folders/{deals.id}",
        f"/files/folders/{deals.id}/rename",
        f"/files/folders/{deals.id}/move",
        f"/files/folders/{deals.id}/share",
        f"/files/contracts/{SHORT_DOC_ID}/move",
        "/files/trash?owner=reader",
    ):
        response = client.get(url)
        assert response.status_code == 404 and response.text == access.NOT_FOUND, url
    for url in (
        f"/files/folders/{deals.id}/rename",
        f"/files/folders/{deals.id}/delete",
        f"/files/folders/{deals.id}/shares",
        f"/files/contracts/{MEDIUM_DOC_ID}/delete",
    ):
        assert client.post(url, data={"name": "x", "email": "a@b.c"}).status_code == 404, url
    assert client.get("/files/spaces").status_code == 403
    assert client.get("/files/spaces/reader").status_code == 403
    assert client.get("/files/shared").status_code == 200


def test_an_admin_sees_every_space_and_can_do_everything_there(
    make_client, database, filled, deals, admin
):
    client = as_user(make_client, database, admin)
    spaces = client.get("/files/spaces").text
    assert 'href="/files/spaces/reader"' in spaces and "3 contracts" in spaces
    assert 'href="/files/trash?owner=reader"' in spaces and 'href="/files"' in spaces
    html = client.get("/files/spaces/reader").text
    assert '<a class="crumb" href="/files/spaces">All spaces</a>' in html
    assert offered(html)[deals.id] == EVERY_ACTION
    assert offered(html)[LONG_DOC_ID] == {"rename", "move", "delete"}
    assert "Owner: Reader" in html and 'action="/upload"' not in html
    assert "/files/folders/new?parent=&amp;owner=reader" in html
    inside = client.get(f"/files/folders/{deals.id}").text
    assert f'name="folder" value="{deals.id}"' in inside
    own = client.get("/files/spaces/admin", follow_redirects=False)
    assert own.headers["location"] == "/files"
    assert ">All spaces</span>" in client.get("/files").text


def test_every_action_offered_is_allowed_and_every_other_is_refused(
    make_client, database, reader, filled, deals, colleague, admin
):
    share(database, reader, deals, colleague, EDIT)
    with database.session() as session:
        files.create_folder(session, reader, deals.id, "Inside")
    for actor in (reader, colleague, admin):
        client = as_user(make_client, database, actor)
        html = client.get(f"/files/folders/{deals.id}").text
        actions = offered(html)
        items = {item: "folders" for item in actions if item != SHORT_DOC_ID}
        items[SHORT_DOC_ID] = "contracts"
        for item_id, kind in items.items():
            shown = actions.get(item_id, set())
            dialogs = ("rename", "move", "share") if kind == "folders" else ("rename", "move")
            for action in dialogs:
                status = client.get(f"/files/{kind}/{item_id}/{action}").status_code
                assert (status == 200) == (action in shown), (actor.name, kind, action)
            if "delete" not in shown:
                assert client.post(f"/files/{kind}/{item_id}/delete").status_code == 403


# The trash and the language


def test_the_trash_offers_restore_and_only_an_admin_deleting_for_good(
    make_client, database, reader, filled, admin
):
    with database.session() as session:
        files.delete_contract(session, reader, MEDIUM_DOC_ID)
    html = filled.get("/files/trash").text
    assert offered(html) == {MEDIUM_DOC_ID: {"restore"}}
    assert "Only an administrator can delete them for good." in html
    assert 'aria-current="true"' in html and "Medium sponsorship agreement" in html
    admin_view = as_user(make_client, database, admin).get("/files/trash?owner=reader").text
    assert offered(admin_view) == {MEDIUM_DOC_ID: {"restore", "purge"}}
    assert "/purge?owner=reader" in admin_view and ">Reader</a>" in admin_view
    own_trash = as_user(make_client, database, admin).get("/files/trash").text
    assert "The trash is empty." in own_trash


def test_the_file_manager_is_in_serbian_with_the_cookie(filled, deals):
    filled.cookies.set("lang", "sr")
    html = filled.get("/files").text
    assert ">Moji fajlovi<" in html and ">Deljeno sa mnom<" in html and ">Otpad<" in html
    assert "Nova fascikla" in html and "U pregledu" in html and "Poslednja izmena" in html
    assert "My files" not in html and "Move to the trash" not in html
    moved = filled.post(f"/files/contracts/{LONG_DOC_ID}/delete")
    message = json.loads(moved.headers["hx-trigger"])["files-changed"]["message"]
    assert message == "Premešteno u otpad: Long outsourcing agreement."
