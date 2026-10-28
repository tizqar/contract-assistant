"""The file manager's changes (folders, moves, the trash, sharing), the drawer and uploads."""

import json
import re

import pytest

from contract_assistant import access, files
from contract_assistant.db.models import EDIT, VIEW
from contract_assistant.extract.extractor import Extractor
from contract_assistant.web import folders
from tests.support.db import add_user
from tests.support.fakes import FakeQA
from tests.web.conftest import LONG_DOC_ID, MEDIUM_DOC_ID, SHORT_DOC_ID, frames, sign_in

TEXT = "Preamble. This agreement is governed by the laws of Vanuatu. Final words."
# A destination of the move dialog: its value, then its name.
TARGET = re.compile(
    r'name="target" value="([^"]*)"[^>]*>\s*<svg.*?</svg><span[^>]*>([^<]+)<', re.DOTALL
)


@pytest.fixture
def deals(make_client, database, reader):
    """The reader's folder "Deals" with the short contract in it."""
    make_client()
    with database.session() as session:
        folder = files.create_folder(session, reader, None, "Deals")
        files.move_contract(session, reader, SHORT_DOC_ID, folder.id)
    return folder


@pytest.fixture
def colleague(database):
    return add_user(database, "colleague@example.com", name="Colleague", user_id="colleague")


def as_user(make_client, database, actor, **kwargs):
    return sign_in(make_client(**kwargs), database, actor)


def toast(response) -> dict:
    """Return what a change tells the page: its message and whether the dialog stays."""
    return json.loads(response.headers["hx-trigger"])[folders.CHANGED]


def listed(client, url: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", client.get(url).text).split())


# Folders


def test_a_new_folder_is_made_in_the_folder_shown(client, database, deals):
    dialog = client.get("/files/folders/new", params={"parent": deals.id}).text
    assert 'hx-post="/files/folders"' in dialog
    assert f'name="parent" value="{deals.id}"' in dialog
    made = client.post("/files/folders", data={"name": " Q3  renewals ", "parent": deals.id})
    assert made.status_code == 204
    assert toast(made) == {"message": "Folder created: Q3 renewals.", "keep": False}
    assert ">Q3 renewals</a>" in client.get(f"/files/folders/{deals.id}").text
    taken = client.post("/files/folders", data={"name": "q3 RENEWALS", "parent": deals.id})
    assert taken.status_code == 409 and taken.text == files.FOLDER_NAME_TAKEN
    empty = client.post("/files/folders", data={"name": "  ", "parent": deals.id})
    assert empty.status_code == 400 and empty.text == files.EMPTY_NAME


def test_an_admin_makes_a_folder_at_the_top_of_another_space(make_client, database, admin):
    client = as_user(make_client, database, admin)
    made = client.post("/files/folders", data={"name": "From admin", "owner": "reader"})
    assert made.status_code == 204
    assert ">From admin</a>" in client.get("/files/spaces/reader").text
    assert "From admin" not in client.get("/files").text


def test_renaming_a_contract_keeps_its_file_name_in_the_dialog(client, deals):
    dialog = client.get(f"/files/contracts/{SHORT_DOC_ID}/rename").text
    assert 'value="short_supply_agreement.txt"' in dialog
    assert "Rename Short supply agreement" in dialog
    renamed = client.post(
        f"/files/contracts/{SHORT_DOC_ID}/rename", data={"name": "acme_supply.txt"}
    )
    assert toast(renamed)["message"] == "Renamed: Acme supply."
    assert "Acme supply" in listed(client, f"/files/folders/{deals.id}")


# Moves stay in one space


def test_the_move_dialog_offers_the_folders_of_the_same_space_only(
    make_client, client, database, reader, deals, colleague
):
    with database.session() as session:
        inner = files.create_folder(session, reader, deals.id, "Inner")
        gone = files.create_folder(session, reader, None, "Old")
        files.delete_folder(session, reader, gone.id)
        files.create_folder(session, colleague, None, "Theirs")
    dialog = client.get(f"/files/contracts/{SHORT_DOC_ID}/move").text
    targets = TARGET.findall(dialog)
    assert targets == [("", "My files"), (deals.id, "Deals"), (inner.id, "Inner")]
    assert re.search(rf'value="{deals.id}" required disabled', dialog)
    assert "it is here now" in dialog and "Theirs" not in dialog and "Old" not in dialog
    folder_move = client.get(f"/files/folders/{deals.id}/move").text
    assert TARGET.findall(folder_move) == [("", "My files")]


def test_a_move_lands_in_the_folder_chosen_and_never_in_another_space(
    make_client, client, database, reader, deals, colleague
):
    moved = client.post(f"/files/contracts/{MEDIUM_DOC_ID}/move", data={"target": deals.id})
    assert moved.status_code == 204 and toast(moved)["message"] == "Moved to Deals."
    assert client.ws.contract(MEDIUM_DOC_ID).folder_id == deals.id
    back = client.post(f"/files/contracts/{MEDIUM_DOC_ID}/move", data={"target": ""})
    assert toast(back)["message"] == "Moved to My files."
    with database.session() as session:
        theirs = files.create_folder(session, colleague, None, "Theirs")
    away = client.post(f"/files/contracts/{MEDIUM_DOC_ID}/move", data={"target": theirs.id})
    assert away.status_code == 404 and client.ws.contract(MEDIUM_DOC_ID).folder_id is None
    with database.session() as session:
        inner = files.create_folder(session, reader, deals.id, "Inner")
    loop = client.post(f"/files/folders/{deals.id}/move", data={"target": inner.id})
    assert loop.status_code == 400 and loop.text == files.INTO_ITSELF


def test_an_admin_moving_in_another_space_sees_its_top_by_the_owners_name(
    make_client, database, deals, admin
):
    client = as_user(make_client, database, admin)
    dialog = client.get(f"/files/contracts/{LONG_DOC_ID}/move").text
    assert TARGET.findall(dialog)[0] == ("", "Reader (top level)")
    moved = client.post(f"/files/contracts/{SHORT_DOC_ID}/move", data={"target": ""})
    assert toast(moved)["message"] == "Moved to Reader (top level)."


# The trash


def test_a_deleted_folder_goes_to_the_trash_with_what_is_in_it_and_comes_back(client, deals):
    deleted = client.post(f"/files/folders/{deals.id}/delete")
    assert deleted.status_code == 204
    assert toast(deleted)["message"] == "Moved to the trash: Deals."
    assert f"/files/folders/{deals.id}" not in client.get("/files").text
    assert client.get(f"/documents/{SHORT_DOC_ID}").status_code == 404
    trash = client.get("/files/trash").text
    assert "Folder, with everything in it" in trash and "Short supply agreement" not in trash
    restored = client.post(f"/files/folders/{deals.id}/restore")
    assert toast(restored)["message"] == "Restored: Deals."
    assert "The trash is empty." in client.get("/files/trash").text
    assert client.get(f"/documents/{SHORT_DOC_ID}").status_code == 200


def test_restoring_something_not_in_the_trash_is_refused(client):
    refused = client.post(f"/files/contracts/{LONG_DOC_ID}/restore")
    assert refused.status_code == 409 and refused.text == files.NOT_IN_TRASH


def test_only_an_admin_deletes_for_good_and_only_after_confirming(
    make_client, client, database, reader, admin
):
    with database.session() as session:
        files.delete_contract(session, reader, LONG_DOC_ID)
    url = f"/files/contracts/{LONG_DOC_ID}/purge"
    assert client.get(url).status_code == 403
    mine = client.post(url, data={"confirm": "yes", "owner": "reader"})
    assert mine.status_code == 403 and mine.text == access.NOT_ALLOWED
    boss = as_user(make_client, database, admin)
    dialog = boss.get(url, params={"owner": "reader"}).text
    assert "Delete Long outsourcing agreement for good?" in dialog
    assert 'name="confirm" value="yes"' in dialog and 'name="owner" value="reader"' in dialog
    unconfirmed = boss.post(url, data={"owner": "reader"})
    assert unconfirmed.status_code == 400 and unconfirmed.text == folders.CONFIRM_FIRST
    assert "Long outsourcing agreement" in boss.get("/files/trash?owner=reader").text
    purged = boss.post(url, data={"confirm": "yes", "owner": "reader"})
    assert purged.status_code == 204
    assert toast(purged)["message"] == "Deleted for good: Long outsourcing agreement."
    assert "The trash is empty." in boss.get("/files/trash?owner=reader").text
    again = boss.post(url, data={"confirm": "yes", "owner": "reader"})
    assert again.status_code == 404
    in_use = boss.get(f"/files/contracts/{MEDIUM_DOC_ID}/purge", params={"owner": "reader"})
    assert in_use.status_code == 404


# Sharing


def test_sharing_by_email_adds_changes_and_removes_access(client, database, deals, colleague):
    dialog = client.get(f"/files/folders/{deals.id}/share").text
    assert "Only you and the administrators can open it now." in dialog
    added = client.post(
        f"/files/folders/{deals.id}/shares",
        data={"email": " Colleague@Example.com ", "level": VIEW},
    )
    assert added.status_code == 200
    assert toast(added) == {"message": "Colleague now has access to the folder.", "keep": True}
    assert "colleague@example.com" in added.text and 'value="view" selected' in added.text
    assert ">Shared</span>" in client.get("/files").text
    changed = client.post(
        f"/files/folders/{deals.id}/shares",
        data={"email": "colleague@example.com", "level": EDIT},
    )
    assert toast(changed)["message"] == "Access changed: Colleague."
    assert 'value="edit" selected' in changed.text
    removed = client.post(f"/files/folders/{deals.id}/shares/colleague/delete")
    assert toast(removed)["message"] == "Colleague no longer has access to the folder."
    assert "colleague@example.com" not in removed.text
    assert ">Shared</span>" not in client.get("/files").text
    again = client.post(f"/files/folders/{deals.id}/shares/colleague/delete")
    assert again.status_code == 200 and "hx-trigger" not in again.headers


def test_sharing_with_an_unknown_email_or_with_oneself_is_refused(client, deals):
    unknown = client.post(
        f"/files/folders/{deals.id}/shares", data={"email": "nobody@example.com", "level": VIEW}
    )
    assert unknown.status_code == 404 and unknown.text == files.NO_SUCH_USER
    own = client.post(
        f"/files/folders/{deals.id}/shares", data={"email": "reader@example.com", "level": VIEW}
    )
    assert own.status_code == 400 and own.text == files.OWNER_ALREADY
    level = client.post(
        f"/files/folders/{deals.id}/shares", data={"email": "x@example.com", "level": "own"}
    )
    assert level.status_code == 400 and level.text == files.UNKNOWN_LEVEL


def test_only_the_owner_or_an_admin_can_share(make_client, database, reader, deals, colleague):
    with database.session() as session:
        files.share_folder(session, reader, deals.id, colleague.email, EDIT)
    client = as_user(make_client, database, colleague)
    assert client.get(f"/files/folders/{deals.id}/share").status_code == 403
    refused = client.post(
        f"/files/folders/{deals.id}/shares", data={"email": "admin@example.com", "level": VIEW}
    )
    assert refused.status_code == 403


# The drawer


def test_the_drawer_shows_the_folder_of_the_contract_and_what_is_beside_it(client, deals):
    rail = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    assert '<a class="crumb" href="/files">My files</a>' in rail
    assert f'<a class="crumb" href="/files/folders/{deals.id}">Deals</a>' in rail
    assert 'href="/files">All files</a>' in rail
    assert f'href="/documents/{SHORT_DOC_ID}"' in rail
    assert f'href="/documents/{MEDIUM_DOC_ID}"' not in rail
    assert f'name="folder" value="{deals.id}"' in rail and "It goes into Deals." in rail
    top = client.get(f"/documents/{MEDIUM_DOC_ID}/rail").text
    assert f'href="/documents/{LONG_DOC_ID}"' in top
    assert f'href="/documents/{SHORT_DOC_ID}"' not in top
    assert 'name="folder"' not in top and "It goes into" not in top


@pytest.mark.parametrize(("level", "into_folder"), [(VIEW, False), (EDIT, True)])
def test_the_drawer_of_a_shared_contract_adds_where_the_reader_may(
    make_client, database, reader, deals, colleague, level, into_folder
):
    with database.session() as session:
        files.share_folder(session, reader, deals.id, colleague.email, level)
    client = as_user(make_client, database, colleague)
    rail = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    assert '<a class="crumb" href="/files/shared">Shared with me</a>' in rail
    assert (f'name="folder" value="{deals.id}"' in rail) is into_folder
    assert ("It goes into My files." in rail) is not into_folder


def test_an_admin_sees_the_owner_of_the_contract_in_the_drawer(make_client, database, admin):
    client = as_user(make_client, database, admin)
    rail = client.get(f"/documents/{LONG_DOC_ID}/rail").text
    assert '<a class="crumb" href="/files/spaces/reader">Reader</a>' in rail
    assert "It goes into My files." in rail and 'name="folder"' not in rail


# Uploads and samples from a folder


def test_an_upload_from_a_folder_lands_in_it(make_client, database, settings, deals):
    client = make_client(extractor=Extractor(FakeQA(), [], settings, "fake"))
    page = client.get(f"/files/folders/{deals.id}").text
    form = re.search(r'action="/upload".*?name="folder" value="([^"]+)"', page, re.DOTALL)
    folder = form.group(1)
    sent = frames(
        client, "POST", "/upload", files={"file": ("new.txt", TEXT.encode())},
        data={"folder": folder},
    )
    contract_id = sent[-1]["redirect"].removeprefix("/documents/")
    assert client.ws.contract(contract_id).folder_id == deals.id
    assert f'href="/documents/{contract_id}"' in client.get(f"/files/folders/{deals.id}").text
    assert f'href="/documents/{contract_id}"' not in client.get("/files").text


def test_a_sample_from_a_folder_lands_in_it(make_client, database, settings, deals):
    client = make_client(extractor=Extractor(FakeQA(), [], settings, "fake"))
    page = client.get(f"/files/folders/{deals.id}").text
    assert 'action="/samples/medium_sponsorship_agreement.txt"' in page
    assert 'action="/samples/short_supply_agreement.txt"' not in page
    sent = frames(
        client, "POST", "/samples/medium_sponsorship_agreement.txt", data={"folder": deals.id}
    )
    contract_id = sent[-1]["redirect"].removeprefix("/documents/")
    assert contract_id != MEDIUM_DOC_ID
    assert client.ws.contract(contract_id).folder_id == deals.id
