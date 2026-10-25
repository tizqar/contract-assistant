"""Folders and contracts on Postgres: the parts whose SQL could differ from SQLite's."""

import pytest
from sqlalchemy import func, select

from contract_assistant import audit, files
from contract_assistant.access import Conflict, Level, NotFound
from contract_assistant.db.models import (
    CONFIRMED,
    EDIT,
    VIEW,
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

pytestmark = pytest.mark.pg

DOC = "sha256:" + "1" * 64
FILE = "2" * 64


def run(database, function, *args, **kwargs):
    """Call a files function in a session of its own, committed when it returns."""
    with database.session() as session:
        return function(session, *args, **kwargs)


def count(database, model, *where):
    with database.session() as session:
        return session.scalar(select(func.count()).select_from(model).where(*where))


@pytest.fixture
def ana(pg_database):
    with pg_database.session() as session:
        session.add(Analysis(doc_id=DOC, data={"doc_id": DOC}, extractor="test"))
        session.add(Blob(sha256=FILE, media_type="application/pdf", size=3, data=b"pdf"))
    return add_user(pg_database, "ana@example.com", name="Ana")


def test_names_are_unique_ignoring_case_beyond_ascii(pg_database, ana):
    run(pg_database, files.create_folder, ana, None, "Ugovori Čačak")
    with pytest.raises(Conflict):
        run(pg_database, files.create_folder, ana, None, "UGOVORI ČAČAK")
    names = [
        run(pg_database, files.add_contract, ana, None, name, DOC, "pdf").name
        for name in ("Đurđevdan.pdf", "đURĐEVDAN.pdf")
    ]
    assert names == ["Đurđevdan.pdf", "đURĐEVDAN (2).pdf"]


def test_shares_reach_down_and_the_trash_hides_on_postgres(pg_database, ana):
    bob = add_user(pg_database, "bob@example.com", name="Bob")
    grand = run(pg_database, files.create_folder, ana, None, "Clients")
    parent = run(pg_database, files.create_folder, ana, grand.id, "Acme")
    child = run(pg_database, files.create_folder, ana, parent.id, "2024")
    lease = run(pg_database, files.add_contract, ana, child.id, "lease.pdf", DOC, "pdf")
    run(pg_database, files.share_folder, ana, grand.id, "bob@example.com", VIEW)
    run(pg_database, files.share_folder, ana, parent.id, "bob@example.com", EDIT)

    assert run(pg_database, files.contract, bob, lease.id).level is Level.EDIT
    assert [view.id for view in run(pg_database, files.visible_contracts, bob)] == [lease.id]
    assert [view.name for view in run(pg_database, files.shared_with_me, bob)] == ["Clients"]

    run(pg_database, files.delete_folder, ana, grand.id)
    with pytest.raises(NotFound):
        run(pg_database, files.contract, bob, lease.id)
    restored = run(pg_database, files.restore_folder, ana, child.id)
    assert restored.parent_id is None
    assert run(pg_database, files.contract, ana, lease.id).level is Level.OWN
    with pytest.raises(NotFound):
        run(pg_database, files.contract, bob, lease.id)


def test_a_purge_clears_the_foreign_keys_on_postgres(pg_database, ana):
    admin = add_admin(pg_database, name="Admin")
    binned = run(pg_database, files.create_folder, ana, None, "Binned")
    inner = run(pg_database, files.create_folder, ana, binned.id, "Inner")
    lease = run(pg_database, files.add_contract, ana, inner.id, "lease.pdf", DOC, "pdf", FILE)
    run(pg_database, files.decide, ana, lease.id, "Insurance", CONFIRMED, 0, 5, "Lease")
    run(pg_database, files.share_folder, ana, inner.id, "admin@example.com", EDIT)
    with pg_database.session() as session:
        session.add(
            ChatMessage(
                contract_id=lease.id, user_id=ana.id, position=0, role="user", payload=b"x"
            )
        )
        session.add(Generated(kind="summary", key=f"{lease.id}-1", doc_id=lease.id, data={}))
        session.add(
            Feedback(
                contract_id=lease.id,
                user_id=ana.id,
                question="Who pays?",
                route_kind="clause",
                clause_types=[],
                liked=False,
            )
        )
    run(pg_database, files.delete_folder, ana, binned.id)
    run(pg_database, files.purge_folder, admin, binned.id)

    for model in (Folder, FolderShare, Contract, DecisionRow, ChatMessage, Generated, Analysis):
        assert count(pg_database, model) == 0, model.__name__
    assert count(pg_database, Blob) == 0
    assert count(pg_database, Feedback, Feedback.contract_id.is_(None)) == 1
    assert count(pg_database, Event, Event.contract_id.is_not(None)) == 0
    assert count(pg_database, Event, Event.folder_id.is_not(None)) == 0
    with pg_database.session() as session:
        logged = audit.events(session, limit=100)
    assert [audit.describe(event) for event in logged][:3] == [
        "Admin deleted the folder Binned for good",
        "Admin deleted lease.pdf for good",
        "Ana moved the folder Binned to the trash",
    ]
    assert "Ana added lease.pdf" in [audit.describe(event) for event in logged]
