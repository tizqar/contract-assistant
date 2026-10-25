"""Sharing folders: with whom, at which level, and what the recipient then sees."""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from contract_assistant import audit, files
from contract_assistant.access import Forbidden, Level, NotFound
from contract_assistant.db.models import EDIT, VIEW, Analysis, FolderShare, User
from tests.support.db import add_admin, add_user

DOC = "sha256:" + "b" * 64


def run(database, function, *args, **kwargs):
    """Call a files function in a session of its own, committed when it returns."""
    with database.session() as session:
        return function(session, *args, **kwargs)


@pytest.fixture
def people(database):
    with database.session() as session:
        session.add(Analysis(doc_id=DOC, data={}, extractor="test"))
    return SimpleNamespace(
        ana=add_user(database, "ana@example.com", name="Ana"),
        bob=add_user(database, "bob@example.com", name="Bob"),
        cara=add_user(database, "cara@example.com", name="Cara"),
        admin=add_admin(database, name="Admin"),
    )


def folder(database, actor, name, parent=None):
    return run(database, files.create_folder, actor, parent.id if parent else None, name)


def shares(database):
    with database.session() as session:
        return [(s.user_id, s.level) for s in session.scalars(select(FolderShare))]


def test_a_folder_is_not_shared_with_its_owner(database, people):
    leases = folder(database, people.ana, "Leases")
    for actor in (people.ana, people.admin):
        with pytest.raises(files.FilesError) as refused:
            run(database, files.share_folder, actor, leases.id, "ana@example.com", VIEW)
        assert (refused.value.message, refused.value.status_code) == (files.OWNER_ALREADY, 400)
    assert shares(database) == []


def test_an_unknown_or_inactive_email_is_refused(database, people):
    leases = folder(database, people.ana, "Leases")
    with pytest.raises(NotFound) as unknown:
        run(database, files.share_folder, people.ana, leases.id, "nobody@example.com", VIEW)
    assert unknown.value.message == files.NO_SUCH_USER
    with database.session() as session:
        session.get(User, people.bob.id).active = False
    with pytest.raises(NotFound):
        run(database, files.share_folder, people.ana, leases.id, "bob@example.com", VIEW)
    with pytest.raises(files.FilesError) as level:
        run(database, files.share_folder, people.ana, leases.id, "cara@example.com", "owner")
    assert level.value.message == files.UNKNOWN_LEVEL
    assert shares(database) == []


def test_sharing_again_replaces_the_level(database, people):
    leases = folder(database, people.ana, "Leases")
    first = run(database, files.share_folder, people.ana, leases.id, " BOB@Example.com ", VIEW)
    assert (first.user_id, first.name, first.level, first.granted_by_name) == (
        people.bob.id,
        "Bob",
        VIEW,
        "Ana",
    )
    run(database, files.share_folder, people.ana, leases.id, "bob@example.com", EDIT)
    run(database, files.share_folder, people.ana, leases.id, "bob@example.com", EDIT)
    assert shares(database) == [(people.bob.id, EDIT)]
    assert run(database, files.listing, people.bob, leases.id).level is Level.EDIT
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(actions=[audit.FOLDER_SHARED]))
    assert [(e.subject_user_id, e.details) for e in logged] == [
        (people.bob.id, {"name": "Leases", "level": EDIT}),
        (people.bob.id, {"name": "Leases", "level": VIEW}),
    ]
    assert audit.describe(logged[0]) == "Ana shared the folder Leases with Bob (edit)"


def test_unsharing_takes_the_folder_away(database, people):
    leases = folder(database, people.ana, "Leases")
    run(database, files.share_folder, people.ana, leases.id, "bob@example.com", VIEW)
    run(database, files.unshare_folder, people.ana, leases.id, people.bob.id)
    run(database, files.unshare_folder, people.ana, leases.id, people.bob.id)
    with pytest.raises(NotFound):
        run(database, files.listing, people.bob, leases.id)
    assert run(database, files.shared_with_me, people.bob) == []
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(actions=[audit.FOLDER_UNSHARED]))
    assert [(e.actor_id, e.subject_user_id, e.details) for e in logged] == [
        (people.ana.id, people.bob.id, {"name": "Leases", "level": VIEW})
    ]


def test_only_the_owner_or_an_admin_shares(database, people):
    leases = folder(database, people.ana, "Leases")
    run(database, files.share_folder, people.ana, leases.id, "bob@example.com", VIEW)
    run(database, files.share_folder, people.ana, leases.id, "cara@example.com", EDIT)
    for actor in (people.bob, people.cara):
        with pytest.raises(Forbidden):
            run(database, files.share_folder, actor, leases.id, "admin@example.com", VIEW)
        with pytest.raises(Forbidden):
            run(database, files.unshare_folder, actor, leases.id, people.bob.id)
        with pytest.raises(Forbidden):
            run(database, files.shares_of, actor, leases.id)
    run(database, files.share_folder, people.admin, leases.id, "bob@example.com", EDIT)
    listed = run(database, files.shares_of, people.ana, leases.id)
    assert [(s.name, s.email, s.level, s.granted_by_name) for s in listed] == [
        ("Bob", "bob@example.com", EDIT, "Admin"),
        ("Cara", "cara@example.com", EDIT, "Ana"),
    ]


def test_a_shared_folder_lists_for_the_recipient_with_its_sub_folders(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    deep = folder(database, people.ana, "2024", acme)
    folder(database, people.ana, "Private")
    lease = run(database, files.add_contract, people.ana, deep.id, "lease.pdf", DOC, "pdf")
    run(database, files.share_folder, people.ana, clients.id, "bob@example.com", VIEW)
    run(database, files.share_folder, people.ana, acme.id, "bob@example.com", EDIT)
    folder(database, people.bob, "Bob's own")

    tops = run(database, files.shared_with_me, people.bob)
    assert [(view.name, view.owner_name, view.level, view.shared_with_me) for view in tops] == [
        ("Clients", "Ana", Level.VIEW, VIEW)
    ]
    inside = run(database, files.listing, people.bob, clients.id)
    assert [(view.name, view.level) for view in inside.folders] == [("Acme", Level.EDIT)]
    below = run(database, files.listing, people.bob, acme.id)
    assert [view.name for view in below.folders] == ["2024"]
    assert run(database, files.contract, people.bob, lease.id).level is Level.EDIT

    run(database, files.delete_folder, people.ana, clients.id)
    assert run(database, files.shared_with_me, people.bob) == []
    with pytest.raises(NotFound):
        run(database, files.listing, people.bob, acme.id)


def test_a_sub_folder_shared_alone_is_a_top_of_its_own(database, people):
    clients = folder(database, people.ana, "Clients")
    acme = folder(database, people.ana, "Acme", clients)
    other = folder(database, people.cara, "Other")
    run(database, files.share_folder, people.ana, acme.id, "bob@example.com", EDIT)
    run(database, files.share_folder, people.cara, other.id, "bob@example.com", VIEW)
    tops = run(database, files.shared_with_me, people.bob)
    assert [(view.owner_name, view.name, view.level) for view in tops] == [
        ("Ana", "Acme", Level.EDIT),
        ("Cara", "Other", Level.VIEW),
    ]
    assert run(database, files.shared_with_me, people.ana) == []
