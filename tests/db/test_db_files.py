"""Folders and contracts: who sees what at which level, the names, the moves, the listings."""

from types import SimpleNamespace

import pytest
from sqlalchemy import event, select

from contract_assistant import audit, files
from contract_assistant.access import Action, Conflict, Forbidden, Level, NotFound
from contract_assistant.db.models import EDIT, VIEW, Analysis, Event, Folder
from contract_assistant.web.i18n import SERBIAN, missing
from tests.support.db import add_admin, add_user

DOC = "sha256:" + "a" * 64


def add_analysis(database, doc_id=DOC):
    """Store an analysis row for contracts to point at; files.py never reads its content."""
    with database.session() as session:
        if session.get(Analysis, doc_id) is None:
            session.add(Analysis(doc_id=doc_id, data={"doc_id": doc_id}, extractor="test"))
    return doc_id


def run(database, function, *args, **kwargs):
    """Call a files function in a session of its own, committed when it returns."""
    with database.session() as session:
        return function(session, *args, **kwargs)


@pytest.fixture
def people(database):
    add_analysis(database)
    return SimpleNamespace(
        ana=add_user(database, "ana@example.com", name="Ana"),
        bob=add_user(database, "bob@example.com", name="Bob"),
        cara=add_user(database, "cara@example.com", name="Cara"),
        admin=add_admin(database, name="Admin"),
    )


def folder(database, actor, name, parent=None):
    return run(database, files.create_folder, actor, parent.id if parent else None, name)


def contract(database, actor, name, where=None):
    return run(
        database, files.add_contract, actor, where.id if where else None, name, DOC, "pdf"
    )


def share(database, owner, where, user, level):
    return run(database, files.share_folder, owner, where.id, user.email, level)


def level(database, actor, contract_id):
    return run(database, files.contract, actor, contract_id).level


# Levels


def test_the_owner_owns_and_an_admin_administers(database, people):
    leases = folder(database, people.ana, "Leases")
    lease = contract(database, people.ana, "lease.pdf", leases)
    assert level(database, people.ana, lease.id) is Level.OWN
    assert level(database, people.admin, lease.id) is Level.ADMIN
    assert run(database, files.listing, people.admin, leases.id).level is Level.ADMIN


def test_without_a_share_everything_is_not_found_never_forbidden(database, people):
    leases = folder(database, people.ana, "Leases")
    lease = contract(database, people.ana, "lease.pdf", leases)
    for attempt in (
        lambda s: files.contract(s, people.bob, lease.id),
        lambda s: files.contract(s, people.bob, lease.id, Action.RENAME),
        lambda s: files.listing(s, people.bob, leases.id),
        lambda s: files.listing(s, people.bob, owner_id=people.ana.id),
        lambda s: files.rename_folder(s, people.bob, leases.id, "Mine"),
        lambda s: files.delete_contract(s, people.bob, lease.id),
        lambda s: files.trash(s, people.bob, owner_id=people.ana.id),
        lambda s: files.contract(s, people.ana, "no-such-contract"),
    ):
        with pytest.raises(NotFound) as refused, database.session() as session:
            attempt(session)
        assert refused.value.status_code == 404


def test_a_view_share_reads_and_an_edit_share_also_changes(database, people):
    leases = folder(database, people.ana, "Leases")
    lease = contract(database, people.ana, "lease.pdf", leases)
    share(database, people.ana, leases, people.bob, VIEW)
    share(database, people.ana, leases, people.cara, EDIT)

    assert level(database, people.bob, lease.id) is Level.VIEW
    with pytest.raises(Forbidden):
        run(database, files.contract, people.bob, lease.id, Action.DECIDE)
    with pytest.raises(Forbidden):
        run(database, files.rename_contract, people.bob, lease.id, "mine.pdf")
    with pytest.raises(Forbidden):
        folder(database, people.bob, "Inside", leases)

    assert level(database, people.cara, lease.id) is Level.EDIT
    assert run(
        database, files.rename_contract, people.cara, lease.id, "Lease 2024.pdf"
    ).name == ("Lease 2024.pdf")
    inside = folder(database, people.cara, "Inside", leases)
    assert inside.owner_id == people.ana.id
    assert inside.level is Level.EDIT
    added = contract(database, people.cara, "annex.pdf", inside)
    assert (added.owner_id, added.created_by, added.created_by_name) == (
        people.ana.id,
        people.cara.id,
        "Cara",
    )
    with pytest.raises(Forbidden):
        run(database, files.delete_contract, people.cara, added.id)


def test_a_share_on_a_grandparent_reaches_everything_below_it(database, people):
    grand = folder(database, people.ana, "Clients")
    parent = folder(database, people.ana, "Acme", grand)
    child = folder(database, people.ana, "2024", parent)
    deep = contract(database, people.ana, "lease.pdf", child)
    share(database, people.ana, grand, people.bob, VIEW)

    assert level(database, people.bob, deep.id) is Level.VIEW
    seen = run(database, files.listing, people.bob, child.id)
    assert seen.level is Level.VIEW
    assert [step.name for step in seen.path] == ["Clients", "Acme", "2024"]
    assert seen.folder.shared_with_me == VIEW
    assert [view.name for view in seen.contracts] == ["lease.pdf"]


def test_the_path_of_a_recipient_starts_at_the_shared_folder(database, people):
    grand = folder(database, people.ana, "Clients")
    parent = folder(database, people.ana, "Acme", grand)
    child = folder(database, people.ana, "2024", parent)
    share(database, people.ana, parent, people.bob, VIEW)

    seen = run(database, files.listing, people.bob, child.id)
    assert [step.name for step in seen.path] == ["Acme", "2024"]
    with pytest.raises(NotFound):
        run(database, files.listing, people.bob, grand.id)
    owner_view = run(database, files.listing, people.ana, child.id)
    assert [step.name for step in owner_view.path] == ["Clients", "Acme", "2024"]
    assert owner_view.folder.id == child.id


def test_the_highest_of_two_shares_wins(database, people):
    grand = folder(database, people.ana, "Clients")
    parent = folder(database, people.ana, "Acme", grand)
    child = folder(database, people.ana, "2024", parent)
    deep = contract(database, people.ana, "lease.pdf", child)
    share(database, people.ana, grand, people.bob, VIEW)
    share(database, people.ana, parent, people.bob, EDIT)
    share(database, people.ana, grand, people.cara, EDIT)
    share(database, people.ana, child, people.cara, VIEW)

    assert level(database, people.bob, deep.id) is Level.EDIT
    assert level(database, people.cara, deep.id) is Level.EDIT
    above = run(database, files.listing, people.bob, grand.id)
    assert above.level is Level.VIEW
    assert [(view.name, view.level) for view in above.folders] == [("Acme", Level.EDIT)]


def test_a_contract_at_the_top_of_a_space_is_for_its_owner_and_admins(database, people):
    leases = folder(database, people.ana, "Leases")
    share(database, people.ana, leases, people.bob, EDIT)
    loose = contract(database, people.ana, "loose.pdf")
    assert loose.folder_id is None
    assert level(database, people.ana, loose.id) is Level.OWN
    assert level(database, people.admin, loose.id) is Level.ADMIN
    with pytest.raises(NotFound):
        run(database, files.contract, people.bob, loose.id)


def test_only_an_admin_opens_another_space(database, people):
    folder(database, people.ana, "Leases")
    contract(database, people.ana, "loose.pdf")
    with pytest.raises(NotFound):
        run(database, files.listing, people.bob, owner_id=people.ana.id)
    seen = run(database, files.listing, people.admin, owner_id=people.ana.id)
    assert (seen.owner_id, seen.owner_name, seen.level) == (people.ana.id, "Ana", Level.ADMIN)
    assert [view.name for view in seen.folders] == ["Leases"]
    assert [view.name for view in seen.contracts] == ["loose.pdf"]
    assert run(database, files.listing, people.admin).folders == []
    with pytest.raises(NotFound):
        run(database, files.listing, people.admin, owner_id="nobody")


def test_an_admin_adds_at_the_top_of_another_space(database, people):
    made = run(
        database, files.create_folder, people.admin, None, "From admin", owner_id=people.ana.id
    )
    added = run(
        database,
        files.add_contract,
        people.admin,
        None,
        "a.pdf",
        DOC,
        "pdf",
        owner_id=people.ana.id,
    )
    assert made.owner_id == added.owner_id == people.ana.id
    with pytest.raises(NotFound):
        run(database, files.create_folder, people.bob, None, "Sneaky", owner_id=people.ana.id)


def test_a_listing_is_sorted_and_says_what_is_shared(database, people):
    for name in ("beta", "Alpha", "gamma"):
        folder(database, people.ana, name)
    for name in ("b.pdf", "A.pdf"):
        contract(database, people.ana, name)
    alpha = run(database, files.listing, people.ana).folders[0]
    share(database, people.ana, alpha, people.bob, VIEW)
    top = run(database, files.listing, people.ana)
    assert top.folder is None and top.path == [] and top.level is Level.OWN
    assert [view.name for view in top.folders] == ["Alpha", "beta", "gamma"]
    assert [view.has_shares for view in top.folders] == [True, False, False]
    assert [view.shared_with_me for view in top.folders] == [None, None, None]
    assert [view.name for view in top.contracts] == ["A.pdf", "b.pdf"]
    assert {view.owner_name for view in top.contracts} == {"Ana"}


# Names


def test_sibling_folder_names_are_unique_ignoring_case(database, people):
    leases = folder(database, people.ana, "Leases")
    with pytest.raises(Conflict) as taken:
        folder(database, people.ana, "  leases ")
    assert taken.value.message == files.FOLDER_NAME_TAKEN
    folder(database, people.bob, "Leases")
    inside = folder(database, people.ana, "Leases", leases)
    other = folder(database, people.ana, "Other")
    with pytest.raises(Conflict):
        run(database, files.rename_folder, people.ana, other.id, "LEASES")
    assert run(database, files.rename_folder, people.ana, leases.id, "LEASES").name == "LEASES"
    run(database, files.delete_folder, people.ana, leases.id)
    assert folder(database, people.ana, "leases").name == "leases"
    assert inside.parent_id == leases.id


def test_names_are_tidied_and_must_be_short_and_not_empty(database, people):
    assert folder(database, people.ana, "  Two   words \n").name == "Two words"
    with pytest.raises(files.FilesError) as empty:
        folder(database, people.ana, "   ")
    assert (empty.value.message, empty.value.status_code) == (files.EMPTY_NAME, 400)
    with pytest.raises(files.FilesError) as long:
        folder(database, people.ana, "x" * 256)
    assert long.value.message == files.NAME_TOO_LONG


def test_a_taken_contract_name_is_numbered_before_its_extension(database, people):
    leases = folder(database, people.ana, "Leases")
    names = [
        contract(database, people.ana, name, leases).name
        for name in ("lease.pdf", "lease.pdf", "LEASE.pdf", "Contract no. 5", "Contract no. 5")
    ]
    assert names == [
        "lease.pdf",
        "lease (2).pdf",
        "LEASE (3).pdf",
        "Contract no. 5",
        "Contract no. 5 (2)",
    ]
    assert contract(database, people.ana, "lease.pdf").name == "lease.pdf"
    second = run(database, files.listing, people.ana, leases.id).contracts[1]
    with pytest.raises(Conflict) as taken:
        run(database, files.rename_contract, people.ana, second.id, "Lease.PDF")
    assert taken.value.message == files.CONTRACT_NAME_TAKEN
    long = contract(database, people.ana, "y" * 251 + ".pdf", leases)
    again = contract(database, people.ana, long.name, leases)
    assert len(again.name) == 255 and again.name.endswith(" (2).pdf")


def test_a_deleted_contract_frees_its_name(database, people):
    lease = contract(database, people.ana, "lease.pdf")
    run(database, files.delete_contract, people.ana, lease.id)
    assert contract(database, people.ana, "lease.pdf").name == "lease.pdf"


def test_a_contract_needs_a_stored_analysis_and_a_known_kind(database, people):
    with pytest.raises(files.FilesError) as no_analysis:
        run(database, files.add_contract, people.ana, None, "a.pdf", "sha256:none", "pdf")
    assert no_analysis.value.message == files.NO_ANALYSIS
    with pytest.raises(files.FilesError):
        run(database, files.add_contract, people.ana, None, "a.doc", DOC, "doc")
    with pytest.raises(files.FilesError):
        run(database, files.add_contract, people.ana, None, "a.pdf", DOC, "pdf", "f" * 64)
    chosen = run(
        database,
        files.add_contract,
        people.ana,
        None,
        "a.pdf",
        DOC,
        "pdf",
        contract_id="chosen-id",
    )
    assert chosen.id == "chosen-id"
    with pytest.raises(Conflict):
        run(
            database,
            files.add_contract,
            people.ana,
            None,
            "b.pdf",
            DOC,
            "pdf",
            contract_id="chosen-id",
        )


# Moves


def test_a_folder_cannot_move_into_itself_or_below_itself(database, people):
    top = folder(database, people.ana, "Top")
    below = folder(database, people.ana, "Below", top)
    for target in (top, below):
        with pytest.raises(files.FilesError) as refused:
            run(database, files.move_folder, people.ana, top.id, target.id)
        assert refused.value.message == files.INTO_ITSELF


def test_moves_stay_inside_one_space(database, people):
    mine = folder(database, people.ana, "Mine")
    loose = contract(database, people.ana, "loose.pdf")
    theirs = folder(database, people.bob, "Theirs")
    with pytest.raises(NotFound):
        run(database, files.move_folder, people.ana, mine.id, theirs.id)
    share(database, people.bob, theirs, people.ana, EDIT)
    for attempt in (
        lambda s: files.move_folder(s, people.ana, mine.id, theirs.id),
        lambda s: files.move_contract(s, people.ana, loose.id, theirs.id),
        lambda s: files.move_contract(s, people.admin, loose.id, theirs.id),
    ):
        with pytest.raises(files.FilesError) as refused, database.session() as session:
            attempt(session)
        assert refused.value.message == files.OTHER_SPACE


def test_an_editor_cannot_move_but_the_owner_can(database, people):
    shared = folder(database, people.ana, "Shared")
    first = folder(database, people.ana, "First", shared)
    second = folder(database, people.ana, "Second", shared)
    lease = contract(database, people.ana, "lease.pdf", first)
    share(database, people.ana, shared, people.cara, EDIT)
    with pytest.raises(Forbidden):
        run(database, files.move_contract, people.cara, lease.id, second.id)
    with pytest.raises(Forbidden):
        run(database, files.move_folder, people.cara, first.id, second.id)
    moved = run(database, files.move_folder, people.ana, first.id, second.id)
    assert moved.parent_id == second.id
    assert run(database, files.contract, people.cara, lease.id).level is Level.EDIT


def test_a_move_numbers_a_taken_name_and_is_logged(database, people):
    first = folder(database, people.ana, "First")
    second = folder(database, people.ana, "Second")
    contract(database, people.ana, "lease.pdf", second)
    lease = contract(database, people.ana, "lease.pdf", first)
    moved = run(database, files.move_contract, people.ana, lease.id, second.id)
    assert (moved.folder_id, moved.name) == (second.id, "lease (2).pdf")
    folder(database, people.ana, "Inner")
    inner = folder(database, people.ana, "Inner", first)
    to_top = run(database, files.move_folder, people.ana, inner.id, None)
    assert (to_top.parent_id, to_top.name) == (None, "Inner (2)")
    with database.session() as session:
        logged = audit.events(
            session, audit.EventFilter(actions=[audit.CONTRACT_MOVED, audit.FOLDER_MOVED])
        )
    assert [(e.action, e.details) for e in logged] == [
        (
            audit.FOLDER_MOVED,
            {"name": "Inner (2)", "target": files.TOP_LEVEL, "old_name": "Inner"},
        ),
        (
            audit.CONTRACT_MOVED,
            {"name": "lease (2).pdf", "target": "Second", "old_name": "lease.pdf"},
        ),
    ]
    assert audit.describe(logged[1]) == "Ana moved lease (2).pdf to Second"


def test_every_folder_change_is_logged_with_its_name(database, people):
    leases = folder(database, people.ana, "Leases")
    run(database, files.rename_folder, people.ana, leases.id, "Rentals")
    run(database, files.rename_folder, people.ana, leases.id, "Rentals")
    with database.session() as session:
        logged = audit.events(session, audit.EventFilter(folder_id=leases.id))
    assert [(e.action, e.actor_id, e.details) for e in logged] == [
        (audit.FOLDER_RENAMED, people.ana.id, {"name": "Rentals", "old_name": "Leases"}),
        (audit.FOLDER_CREATED, people.ana.id, {"name": "Leases"}),
    ]
    assert audit.describe(logged[0]) == "Ana renamed the folder Leases to Rentals"


# Everything one can see


def test_visible_contracts_are_own_and_shared_newest_first(database, people):
    shared = folder(database, people.ana, "Shared")
    private = folder(database, people.ana, "Private")
    first = contract(database, people.ana, "first.pdf", shared)
    contract(database, people.ana, "hidden.pdf", private)
    own = contract(database, people.bob, "own.pdf")
    share(database, people.ana, shared, people.bob, VIEW)
    later = contract(database, people.ana, "later.pdf", shared)
    gone = contract(database, people.bob, "gone.pdf")
    run(database, files.delete_contract, people.bob, gone.id)

    seen = run(database, files.visible_contracts, people.bob)
    assert [(view.name, view.level) for view in seen] == [
        ("later.pdf", Level.VIEW),
        ("own.pdf", Level.OWN),
        ("first.pdf", Level.VIEW),
    ]
    assert {view.id for view in seen} == {first.id, own.id, later.id}
    everything = run(database, files.visible_contracts, people.admin)
    assert len(everything) == 4
    assert {view.level for view in everything} == {Level.ADMIN}
    with pytest.raises(Forbidden):
        run(database, files.spaces, people.bob)


def test_spaces_count_what_is_out_of_the_trash(database, people):
    leases = folder(database, people.ana, "Leases")
    contract(database, people.ana, "a.pdf", leases)
    contract(database, people.ana, "b.pdf")
    binned = folder(database, people.ana, "Binned")
    contract(database, people.ana, "c.pdf", binned)
    run(database, files.delete_folder, people.ana, binned.id)
    assert run(database, files.spaces, people.admin) == [
        (people.admin.id, "Admin", 0),
        (people.ana.id, "Ana", 2),
        (people.bob.id, "Bob", 0),
        (people.cara.id, "Cara", 0),
    ]


def count_queries(database, function, *args):
    statements = []

    def counted(conn, cursor, statement, *_):
        statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", counted)
    try:
        with database.session() as session:
            result = function(session, *args)
    finally:
        event.remove(database.engine, "before_cursor_execute", counted)
    return len(statements), result


def deep_tree(database, owner, depth, prefix=""):
    """Make a chain of depth folders with a contract in each; return the deepest folder."""
    above = None
    for step in range(depth):
        above = folder(database, owner, f"{prefix}Level {step}", above)
        contract(database, owner, f"c{step}.pdf", above)
    return above


def test_a_deep_folder_lists_in_the_same_few_queries_as_a_shallow_one(database, people):
    shallow = deep_tree(database, people.ana, 2)
    deep = deep_tree(database, people.bob, 12)
    shallow_count, _ = count_queries(database, files.listing, people.ana, shallow.id)
    deep_count, seen = count_queries(database, files.listing, people.bob, deep.id)
    assert len(seen.path) == 12
    assert deep_count == shallow_count <= 8


def test_many_shares_cost_a_handful_of_queries(database, people):
    owners = [add_user(database, f"owner{n}@example.com") for n in range(4)]
    for owner in owners:
        for branch in range(3):
            leaf = deep_tree(database, owner, 4, prefix=f"b{branch} ")
            top = run(database, files.listing, owner, leaf.id).path[0]
            if branch == 0:
                share(database, owner, top, people.bob, VIEW)
            share(database, owner, leaf, people.bob, EDIT)
    count, seen = count_queries(database, files.visible_contracts, people.bob)
    assert count <= 6
    assert len(seen) == 4 * (4 + 1 + 1)
    assert run(database, files.shared_with_me, people.bob)[0].level is Level.VIEW


def test_every_message_has_its_serbian(database):
    assert missing(SERBIAN, list(files.TRANSLATED)) == []


def test_folder_rows_keep_the_creator(database, people):
    shared = folder(database, people.ana, "Shared")
    share(database, people.ana, shared, people.cara, EDIT)
    inside = folder(database, people.cara, "Inside", shared)
    with database.session() as session:
        row = session.get(Folder, inside.id)
        assert (row.owner_id, row.created_by) == (people.ana.id, people.cara.id)
        assert session.scalar(select(Event.actor_id).where(Event.folder_id == inside.id)) == (
            people.cara.id
        )
