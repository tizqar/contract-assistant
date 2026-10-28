"""The activity log as the pages show it: sentences and times in the reader's language."""

from datetime import UTC, datetime

from contract_assistant import audit
from contract_assistant.audit import LoggedEvent
from contract_assistant.web import activity


def event(action: str, **details) -> LoggedEvent:
    return LoggedEvent(
        id=1,
        at=datetime(2026, 10, 28, 13, 9, tzinfo=UTC),
        action=action,
        actor_id="ana",
        actor_name="Ana",
        actor_email="ana@example.com",
        contract_id="c1",
        contract_name="supply.pdf",
        folder_id=None,
        folder_name=None,
        subject_user_id="marko",
        subject_name="Marko",
        ip=None,
        details=details,
    )


def test_a_decision_names_the_clause_as_the_pages_do():
    confirmed = event(audit.CLAUSE_CONFIRMED, clause="Governing Law", name="supply.pdf")
    assert activity.sentence(confirmed, "en") == (
        "Ana confirmed the Governing Law passage of supply.pdf"
    )
    assert activity.sentence(confirmed, "sr") == (
        "Ana: potvrđen odlomak Merodavno pravo u ugovoru supply.pdf"
    )


def test_roles_levels_and_places_are_translated():
    shared = event(audit.FOLDER_SHARED, level="edit")
    assert activity.sentence(shared, "en").endswith("with Marko (can edit)")
    made = event(audit.ROLE_CHANGED, role="admin")
    assert activity.sentence(made, "sr") == "Ana: Marko dobija ulogu Administrator"
    moved = event(audit.CONTRACT_MOVED, target="the top level")
    assert activity.sentence(moved, "en") == "Ana moved supply.pdf to the top level"
    downloaded = event(audit.CONTRACT_EXPORTED, what="the report")
    assert activity.sentence(downloaded, "sr") == "Ana: preuzeto izveštaj za supply.pdf"


def test_times_are_local_and_written_the_language_way():
    at = datetime(2026, 10, 28, 13, 9, tzinfo=UTC)
    local = at.astimezone()
    assert activity.when(at, "en") == f"28 Oct 2026, {local:%H:%M}"
    assert activity.when(at, "sr").startswith("28. okt 2026.")


def test_rows_keep_the_order_and_the_group():
    [row] = activity.rows([event(audit.REVIEW_APPROVED)], "en")
    assert row.group == "review" and row.sentence == "Ana approved supply.pdf"


def test_a_filter_group_lists_its_actions():
    assert audit.CLAUSE_UNDONE in activity.actions_of("clause")
    assert all(action.startswith("review.") for action in activity.actions_of("review"))
    assert activity.actions_of("nothing") == []
