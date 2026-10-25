"""The trash: what deleting hides, what restoring brings back, and deleting for good."""

from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from contract_assistant import audit, files
from contract_assistant.access import Conflict, Forbidden, Level, NotFound
from contract_assistant.db.models import (
    CONFIRMED,
    EDIT,
    Analysis,
    Blob,
    ChatMessage,
    Contract,
    Event,
    Feedback,
    Folder,
    FolderShare,
    Generated,
)
from contract_assistant.db.models import Decision as DecisionRow
from tests.support.db import add_admin, add_user

DOC = "sha256:" + "c" * 64
OTHER_DOC = "sha256:" + "d" * 64
FILE = "e" * 64


def run(database, function, *args, **kwargs):
    """Call a files function in a session of its own, committed when it returns."""
    with database.session() as session:
        return function(session, *args, **kwargs)


@pytest.fixture
def people(database):
    with database.session() as session:
        for doc_id in (DOC, OTHER_DOC):
            session.add(Analysis(doc_id=doc_id, data={"doc_id": doc_id}, extractor="test"))
        session.add(Blob(sha256=FILE, media_type="application/pdf", size=3, data=b"pdf"))
    return SimpleNamespace(
        ana=add_user(database, "ana@example.com", name="Ana"),
        bob=add_user(database, "bob@example.com", name="Bob"),
        admin=add_admin(database, name="Admin"),
    )


def folder(database, actor, name, parent=None):
    return run(database, files.create_folder, actor, parent.id if parent else None, name)


def contract(database, actor, name, where=None, doc_id=DOC, file_sha256=None):
    folder_id = where.id if where else None
    return run(database, files.add_contract, actor, folder_id, name, doc_id, "pdf", file_sha256)


def count(database, model, *where):
    with database.session() as session:
        return session.scalar(select(func.count()).select_from(model).where(*where))


def test_a_contract_inside_a_deleted_folder_is_hidden(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    lease = contract(database, people.ana, "lease.pdf", acme)
    run(database, files.share_folder, people.ana, clients.id, "bob@example.com", EDIT)
    run(database, files.delete_folder, people.ana, clients.id)

    for actor in (people.ana, people.bob, people.admin):
        with pytest.raises(NotFound):
            run(database, files.contract, actor, lease.id)
        with pytest.raises(NotFound):
            run(database, files.listing, actor, acme.id)
        assert run(database, files.visible_contracts, actor) == []
    assert run(database, files.listing, people.ana).folders == []
    with pytest.raises(NotFound):
        run(database, files.restore_folder, people.bob, clients.id)


def test_restoring_a_folder_brings_its_subtree_back(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    lease = contract(database, people.ana, "lease.pdf", acme)
    run(database, files.delete_folder, people.ana, clients.id)
    restored = run(database, files.restore_folder, people.ana, clients.id)
    assert (restored.parent_id, restored.deleted_at) == (None, None)
    assert run(database, files.contract, people.ana, lease.id).folder_id == acme.id
    assert [
        view.name for view in run(database, files.listing, people.ana, clients.id).folders
    ] == ["Acme"]
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(folder_id=clients.id), limit=2)
    assert [(e.action, e.details) for e in logged] == [
        (audit.FOLDER_RESTORED, {"name": "Clients"}),
        (audit.FOLDER_DELETED, {"name": "Clients"}),
    ]


def test_a_folder_restored_out_of_a_deleted_parent_goes_to_the_top(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    lease = contract(database, people.ana, "lease.pdf", clients)
    run(database, files.delete_folder, people.ana, acme.id)
    run(database, files.delete_folder, people.ana, clients.id)
    folder(database, people.ana, "Acme")

    restored = run(database, files.restore_folder, people.ana, acme.id)
    assert (restored.parent_id, restored.name) == (None, "Acme (2)")
    back = run(database, files.restore_contract, people.ana, lease.id)
    assert back.folder_id is None
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(folder_id=acme.id), limit=1)
    assert logged[0].details == {
        "name": "Acme (2)",
        "old_name": "Acme",
        "target": files.TOP_LEVEL,
    }


def test_the_trash_lists_only_the_top_most_deleted_things(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    kept = folder(database, people.ana, "Kept")
    inner = contract(database, people.ana, "inner.pdf", acme)
    loose = contract(database, people.ana, "loose.pdf")
    single = contract(database, people.ana, "single.pdf", kept)
    run(database, files.delete_contract, people.ana, inner.id)
    run(database, files.delete_folder, people.ana, acme.id)
    run(database, files.delete_folder, people.ana, clients.id)
    run(database, files.delete_contract, people.ana, loose.id)
    run(database, files.delete_contract, people.ana, single.id)

    binned = run(database, files.trash, people.ana)
    assert [view.name for view in binned.folders] == ["Clients"]
    assert [view.name for view in binned.contracts] == ["single.pdf", "loose.pdf"]
    assert binned.folders[0].deleted_at is not None
    for_admin = run(database, files.trash, people.admin, people.ana.id)
    assert ([view.name for view in for_admin.folders], for_admin.level) == (
        ["Clients"],
        Level.ADMIN,
    )
    assert binned.level is Level.OWN
    assert run(database, files.trash, people.bob).folders == []

    run(database, files.restore_folder, people.ana, clients.id)
    after = run(database, files.trash, people.ana)
    assert [view.name for view in after.folders] == ["Acme"]
    assert [view.name for view in after.contracts] == ["single.pdf", "loose.pdf"]


def test_restoring_what_is_not_in_the_trash_conflicts(database, people):
    clients = folder(database, people.ana, "Clients")
    lease = contract(database, people.ana, "lease.pdf", clients)
    with pytest.raises(Conflict) as folder_refused:
        run(database, files.restore_folder, people.ana, clients.id)
    assert folder_refused.value.message == files.NOT_IN_TRASH
    with pytest.raises(Conflict):
        run(database, files.restore_contract, people.ana, lease.id)


def test_a_restored_contract_gets_a_free_name(database, people):
    lease = contract(database, people.ana, "lease.pdf")
    run(database, files.delete_contract, people.ana, lease.id)
    contract(database, people.ana, "lease.pdf")
    assert run(database, files.restore_contract, people.ana, lease.id).name == "lease (2).pdf"


def test_only_an_admin_purges_and_only_from_the_trash(database, people):
    clients = folder(database, people.ana, "Clients")
    lease = contract(database, people.ana, "lease.pdf", clients)
    with pytest.raises(Conflict):
        run(database, files.purge_contract, people.admin, lease.id)
    with pytest.raises(Conflict):
        run(database, files.purge_folder, people.admin, clients.id)
    run(database, files.delete_contract, people.ana, lease.id)
    run(database, files.delete_folder, people.ana, clients.id)
    with pytest.raises(Forbidden):
        run(database, files.purge_contract, people.ana, lease.id)
    with pytest.raises(Forbidden):
        run(database, files.purge_folder, people.ana, clients.id)
    with pytest.raises(NotFound):
        run(database, files.purge_folder, people.bob, clients.id)
    assert count(database, Contract) == 1


def add_dependents(database, contract_id, user_id):
    """Give a contract a chat message, a generated text and a feedback row."""
    with database.session() as session:
        session.add(
            ChatMessage(
                contract_id=contract_id, user_id=user_id, position=0, role="user", payload=b"x"
            )
        )
        session.add(
            Generated(kind="summary", key=f"{contract_id}-1", doc_id=contract_id, data={})
        )
        session.add(
            Feedback(
                contract_id=contract_id,
                user_id=user_id,
                question="Who pays?",
                route_kind="clause",
                clause_types=["Insurance"],
                liked=True,
            )
        )


def test_a_purge_removes_what_is_the_contracts_and_keeps_what_others_use(database, people):
    first = contract(database, people.ana, "first.pdf", file_sha256=FILE)
    second = contract(database, people.ana, "second.pdf", file_sha256=FILE)
    run(database, files.decide, people.ana, first.id, "Insurance", CONFIRMED, 0, 5, "Lease")
    add_dependents(database, first.id, people.ana.id)
    run(database, files.delete_contract, people.ana, first.id)
    run(database, files.purge_contract, people.admin, first.id)

    assert count(database, Contract) == 1
    assert count(database, DecisionRow) == 0
    assert count(database, ChatMessage) == 0
    assert count(database, Generated) == 0
    assert count(database, Feedback, Feedback.contract_id.is_(None)) == 1
    assert count(database, Event, Event.contract_id == first.id) == 0
    assert count(database, Analysis, Analysis.doc_id == DOC) == 1
    assert count(database, Blob) == 1

    run(database, files.delete_contract, people.ana, second.id)
    run(database, files.purge_contract, people.admin, second.id)
    assert count(database, Contract) == 0
    assert count(database, Analysis, Analysis.doc_id == DOC) == 0
    assert count(database, Analysis, Analysis.doc_id == OTHER_DOC) == 1
    assert count(database, Blob) == 0


def test_the_log_still_names_a_purged_contract(database, people):
    lease = contract(database, people.ana, "lease.pdf")
    run(database, files.rename_contract, people.ana, lease.id, "Lease 2024.pdf")
    run(database, files.decide, people.ana, lease.id, "Insurance", CONFIRMED, 0, 5, "Lease")
    run(database, files.delete_contract, people.ana, lease.id)
    run(database, files.purge_contract, people.admin, lease.id, ip="10.0.0.1")
    with database.session() as session:
        logged = audit.events(session)
    assert [audit.describe(event) for event in logged] == [
        "Admin deleted Lease 2024.pdf for good",
        "Ana moved Lease 2024.pdf to the trash",
        "Ana confirmed the Insurance passage of Lease 2024.pdf",
        "Ana renamed lease.pdf to Lease 2024.pdf",
        "Ana added lease.pdf",
    ]
    assert {event.contract_id for event in logged} == {None}
    assert (logged[0].ip, logged[0].details["id"]) == ("10.0.0.1", lease.id)


def test_purging_a_folder_takes_everything_inside_it(database, people):
    binned = folder(database, people.ana, "Binned")
    inner = folder(database, people.ana, "Inner", binned)
    kept = folder(database, people.ana, "Kept")
    contract(database, people.ana, "a.pdf", binned)
    gone_first = contract(database, people.ana, "b.pdf", inner, doc_id=OTHER_DOC)
    stays = contract(database, people.ana, "c.pdf", kept)
    run(database, files.share_folder, people.ana, inner.id, "bob@example.com", EDIT)
    run(database, files.delete_contract, people.ana, gone_first.id)
    run(database, files.delete_folder, people.ana, binned.id)
    run(database, files.purge_folder, people.admin, binned.id)

    assert count(database, Folder) == 1
    assert count(database, FolderShare) == 0
    with database.session() as session:
        assert session.scalars(select(Contract.id)).all() == [stays.id]
        assert session.scalars(select(Analysis.doc_id)).all() == [DOC]
        assert (
            session.scalar(
                select(func.count())
                .select_from(Event)
                .where(Event.folder_id.in_([binned.id, inner.id]))
            )
            == 0
        )
        logged = audit.events(session, limit=3)
    assert [audit.describe(event) for event in logged] == [
        "Admin deleted the folder Binned for good",
        "Admin deleted b.pdf for good",
        "Admin deleted a.pdf for good",
    ]
    assert logged[0].details == {
        "name": "Binned",
        "id": binned.id,
        "folders": 2,
        "contracts": 2,
    }
    assert logged[1].details["folder"] == "Inner"
    with database.session() as session:
        older = audit.events(session, audit.EventFilter(actions=[audit.FOLDER_CREATED]))
    assert [audit.describe(event) for event in older] == [
        "Ana created the folder Kept",
        "Ana created the folder Inner",
        "Ana created the folder Binned",
    ]


def test_a_folder_inside_the_trash_can_be_purged_on_its_own(database, people):
    binned = folder(database, people.ana, "Binned")
    inner = folder(database, people.ana, "Inner", binned)
    run(database, files.delete_folder, people.ana, binned.id)
    run(database, files.purge_folder, people.admin, inner.id)
    with database.session() as session:
        assert session.scalars(select(Folder.id)).all() == [binned.id]
