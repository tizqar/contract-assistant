"""Who sees and changes what in the web app: owners, shares, admins and the review lock."""

import pytest

from contract_assistant import access, audit, files
from contract_assistant.access import ReviewStep
from contract_assistant.db.models import EDIT, VIEW
from contract_assistant.extract.extractor import Extractor
from tests.support.db import add_user
from tests.support.fakes import EchoTextClient, FakeQA
from tests.web.conftest import (
    ADMIN_EMAIL,
    MEDIUM_DOC_ID,
    SHORT_DOC_ID,
    frames,
    seed_demo,
    sign_in,
)

DOC = f"/documents/{SHORT_DOC_ID}"
TEXT = "Preamble. This agreement is governed by the laws of Vanuatu. Final words."


@pytest.fixture
def shared_folder(database, settings, reader):
    """The reader's folder "Deals" with the short contract in it."""
    seed_demo(database, settings, reader)
    with database.session() as session:
        folder = files.create_folder(session, reader, None, "Deals")
        files.move_contract(session, reader, SHORT_DOC_ID, folder.id)
    return folder


@pytest.fixture
def colleague(database):
    return add_user(database, "colleague@example.com", name="Colleague", user_id="colleague")


def share(database, reader, folder, colleague, level: str) -> None:
    with database.session() as session:
        files.share_folder(session, reader, folder.id, colleague.email, level)


def as_user(make_client, database, actor, **kwargs):
    client = make_client(**kwargs)
    return sign_in(client, database, actor)


def to_check(client) -> tuple[str, int, int]:
    """Return a clause type of the short contract with a passage to check, and its span."""
    cache = client.ws.load(SHORT_DOC_ID)
    names = client.ws.names
    name = next(n for n in names if cache.clauses.get(n) and cache.clauses[n][0].conf < 0.91)
    top = cache.clauses[name][0]
    return name, top.start, top.end


def decide(client, name: str, start: int, end: int):
    data = {"clause_type": name, "verdict": "confirmed", "start": start, "end": end}
    return client.post(f"{DOC}/decide", data=data, follow_redirects=False)


def test_someone_else_cannot_open_a_contract_or_know_it_exists(make_client, database, colleague):
    client = as_user(make_client, database, colleague)
    assert client.get(DOC).status_code == 404
    assert client.get(f"{DOC}/view").status_code == 404
    assert client.get(f"{DOC}/analysis.json").status_code == 404
    assert client.post(f"{DOC}/decide/undo", data={"clause_type": "Parties"}).status_code == 404
    assert client.ws.documents() == []
    assert "Read a contract with a local assistant" in client.get("/").text


def test_a_folder_shared_to_view_opens_but_cannot_be_changed(
    make_client, database, reader, shared_folder, colleague
):
    share(database, reader, shared_folder, colleague, VIEW)
    owner = make_client()
    name, start, end = to_check(owner)
    client = as_user(make_client, database, colleague)
    [entry] = client.ws.documents()
    assert entry.doc_id == SHORT_DOC_ID and not entry.mine and entry.owner_name == "Reader"
    pane = client.get(f"{DOC}/view?label={name}&start={start}&end={end}&status=check").text
    assert ">Check this</span>" in pane and ">Confirm</button>" not in pane
    refused = decide(client, name, start, end)
    assert refused.status_code == 403 and refused.text == access.NOT_ALLOWED
    assert client.get(f"/documents/{MEDIUM_DOC_ID}").status_code == 404


def test_a_folder_shared_to_edit_lets_the_colleague_decide_under_their_name(
    make_client, database, reader, shared_folder, colleague
):
    share(database, reader, shared_folder, colleague, EDIT)
    client = as_user(make_client, database, colleague)
    name, start, end = to_check(client)
    assert decide(client, name, start, end).status_code == 303
    owner = sign_in(make_client(), database, reader)
    pane = owner.get(f"{DOC}/view?label={name}&start={start}&end={end}&status=confirmed").text
    assert "by Colleague" in pane
    with database.session() as session:
        [event] = audit.events(session, audit.EventFilter(actions=[audit.CLAUSE_CONFIRMED]))
    assert event.actor_name == "Colleague" and event.contract_id == SHORT_DOC_ID
    assert event.details["clause"] == name and event.details["quote"]


def test_an_admin_sees_every_contract_and_can_decide(make_client, database, admin):
    client = as_user(make_client, database, admin)
    assert {entry.doc_id for entry in client.ws.documents()} >= {SHORT_DOC_ID, MEDIUM_DOC_ID}
    name, start, end = to_check(client)
    assert decide(client, name, start, end).status_code == 303
    html = client.get(DOC).text
    assert ">Admin</a>" in html and ADMIN_EMAIL in html and ">Administrator</span>" in html


def test_a_contract_sent_for_approval_is_locked(make_client, database, reader):
    client = make_client()
    name, start, end = to_check(client)
    with database.session() as session:
        files.review(session, reader, SHORT_DOC_ID, ReviewStep.SUBMIT)
    pane = client.get(f"{DOC}/view?label={name}&start={start}&end={end}&status=check").text
    assert ">Confirm</button>" not in pane
    refused = decide(client, name, start, end)
    assert refused.status_code == 409 and refused.text == access.LOCKED


def test_a_decided_passage_hides_undo_once_locked(make_client, database, reader):
    client = make_client()
    name, start, end = to_check(client)
    decide(client, name, start, end)
    url = f"{DOC}/view?label={name}&start={start}&end={end}&status=confirmed"
    assert ">Undo<" in client.get(url).text
    with database.session() as session:
        files.review(session, reader, SHORT_DOC_ID, ReviewStep.SUBMIT)
    locked = client.get(url).text
    assert ">Undo<" not in locked and "by Reader" in locked


def test_an_upload_goes_to_the_folder_asked_for(make_client, database, settings, reader):
    with database.session() as session:
        folder = files.create_folder(session, reader, None, "Inbox")
    client = make_client(extractor=Extractor(FakeQA(), [], settings, "fake"))
    sent = frames(
        client, "POST", "/upload", files={"file": ("new.txt", TEXT.encode())},
        data={"folder": folder.id},
    )
    contract_id = sent[-1]["redirect"].removeprefix("/documents/")
    contract = client.ws.contract(contract_id)
    assert contract.folder_id == folder.id and contract.name == "new.txt"


def test_an_upload_to_a_folder_shared_to_view_is_refused_before_reading(
    make_client, database, settings, reader, shared_folder, colleague
):
    share(database, reader, shared_folder, colleague, VIEW)
    qa = FakeQA()
    client = as_user(
        make_client, database, colleague, extractor=Extractor(qa, [], settings, "fake")
    )
    response = client.post(
        "/upload", files={"file": ("new.txt", TEXT.encode())}, data={"folder": shared_folder.id}
    )
    assert response.status_code == 403 and qa.calls == []


def test_each_user_has_their_own_chat(
    make_client, database, reader, shared_folder, colleague
):
    share(database, reader, shared_folder, colleague, VIEW)
    owner = make_client(gen=EchoTextClient())
    frames(owner, "POST", f"{DOC}/ask", data={"question": "Who are the parties?"})
    assert len(owner.ws.conversation(SHORT_DOC_ID).messages) == 2
    client = as_user(make_client, database, colleague)
    assert "Who are the parties?" not in client.get(f"{DOC}/chat").text
    reopened = sign_in(make_client(), database, reader)
    assert "Who are the parties?" in reopened.get(f"{DOC}/chat").text


def test_downloads_are_logged(client, database):
    assert client.get(f"{DOC}/analysis.json").status_code == 200
    assert client.get("/portfolio.csv").status_code == 200
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(actions=[audit.CONTRACT_EXPORTED]))
    assert [event.details["what"] for event in logged] == ["the portfolio", "the analysis"]
    assert logged[1].contract_id == SHORT_DOC_ID and logged[1].actor_name == "Reader"
