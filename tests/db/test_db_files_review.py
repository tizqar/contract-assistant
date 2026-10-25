"""Decisions on passages and the review of a contract: who may, when, and what is logged."""

from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from contract_assistant import audit, decisions, files
from contract_assistant.access import (
    LOCKED,
    WRONG_STEP,
    Conflict,
    Forbidden,
    NotFound,
    ReviewStep,
)
from contract_assistant.config import Settings
from contract_assistant.db.models import (
    APPROVED,
    CONFIRMED,
    EDIT,
    IN_REVIEW,
    REJECTED,
    SUBMITTED,
    VIEW,
    Analysis,
)
from contract_assistant.db.models import Decision as DecisionRow
from tests.support.caches import make_cache
from tests.support.db import add_admin, add_user

LAW = "This agreement is governed by the laws of Vanuatu."
OTHER = "Disputes go to arbitration in Geneva."
TEXT = f"Preamble. {LAW} {OTHER} Final words."
DOC = "sha256:" + "f" * 64


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
        viewer=add_user(database, "vera@example.com", name="Vera"),
        editor=add_user(database, "edo@example.com", name="Edo"),
        stranger=add_user(database, "sam@example.com", name="Sam"),
        admin=add_admin(database, name="Admin"),
    )


@pytest.fixture
def lease(database, people):
    shared = run(database, files.create_folder, people.ana, None, "Shared")
    run(database, files.share_folder, people.ana, shared.id, "vera@example.com", VIEW)
    run(database, files.share_folder, people.ana, shared.id, "edo@example.com", EDIT)
    return run(database, files.add_contract, people.ana, shared.id, "lease.pdf", DOC, "pdf")


def span(text):
    start = TEXT.find(text)
    return start, start + len(text)


def decide(database, actor, contract_id, verdict, text=LAW, clause="Governing Law"):
    start, end = span(text)
    return run(database, files.decide, actor, contract_id, clause, verdict, start, end, text)


def step(database, actor, contract_id, which, note=None):
    return run(database, files.review, actor, contract_id, which, note)


def logged(database, *actions):
    with database.session() as session:
        return audit.events(session, audit.EventFilter(actions=actions))


# Decisions


def test_deciding_needs_edit(database, people, lease):
    with pytest.raises(Forbidden):
        decide(database, people.viewer, lease.id, CONFIRMED)
    with pytest.raises(NotFound):
        decide(database, people.stranger, lease.id, CONFIRMED)
    made = decide(database, people.editor, lease.id, CONFIRMED)
    assert (made.by_id, made.by_name, made.decision.verdict) == (
        people.editor.id,
        "Edo",
        CONFIRMED,
    )
    with pytest.raises(Forbidden):
        run(database, files.undo_decision, people.viewer, lease.id, "Governing Law")


def test_a_contract_out_of_review_has_its_decisions_locked(database, people, lease):
    decide(database, people.ana, lease.id, CONFIRMED)
    step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    with pytest.raises(Conflict) as locked:
        decide(database, people.ana, lease.id, REJECTED)
    assert locked.value.message == LOCKED
    with pytest.raises(Conflict):
        run(database, files.undo_decision, people.admin, lease.id, "Governing Law")
    step(database, people.admin, lease.id, ReviewStep.APPROVE)
    with pytest.raises(Conflict):
        decide(database, people.admin, lease.id, REJECTED)
    step(database, people.admin, lease.id, ReviewStep.REOPEN)
    assert decide(database, people.ana, lease.id, REJECTED).decision.verdict == REJECTED


def test_one_decision_per_clause_type_and_the_log_keeps_the_history(database, people, lease):
    decide(database, people.editor, lease.id, CONFIRMED)
    decide(database, people.editor, lease.id, CONFIRMED)
    decide(database, people.ana, lease.id, REJECTED, OTHER)
    decide(database, people.ana, lease.id, CONFIRMED, OTHER, clause="Dispute Resolution")
    with database.session() as session:
        rows = session.scalar(select(func.count()).select_from(DecisionRow))
    assert rows == 2
    current = run(database, files.decisions_of, lease.id)
    assert {name: view.decision.verdict for name, view in current.items()} == {
        "Dispute Resolution": CONFIRMED,
        "Governing Law": REJECTED,
    }
    assert current["Governing Law"].by_name == "Ana"
    history = logged(database, audit.CLAUSE_CONFIRMED, audit.CLAUSE_REJECTED)
    assert [(e.actor_name, e.details["clause"], e.details["previous"]) for e in history] == [
        ("Ana", "Dispute Resolution", None),
        ("Ana", "Governing Law", CONFIRMED),
        ("Edo", "Governing Law", None),
    ]
    start, end = span(OTHER)
    assert history[1].details == {
        "clause": "Governing Law",
        "verdict": REJECTED,
        "start": start,
        "end": end,
        "quote": OTHER,
        "previous": CONFIRMED,
        "name": "lease.pdf",
    }
    assert audit.describe(history[1]) == "Ana rejected the Governing Law passage of lease.pdf"


def test_undo_logs_the_verdict_and_quote_it_takes_back(database, people, lease):
    decide(database, people.editor, lease.id, REJECTED)
    run(database, files.undo_decision, people.editor, lease.id, "Governing Law")
    run(database, files.undo_decision, people.editor, lease.id, "Governing Law")
    assert run(database, files.decisions_of, lease.id) == {}
    undone = logged(database, audit.CLAUSE_UNDONE)
    assert len(undone) == 1
    start, end = span(LAW)
    assert undone[0].details == {
        "clause": "Governing Law",
        "previous": REJECTED,
        "start": start,
        "end": end,
        "quote": LAW,
        "name": "lease.pdf",
    }
    assert audit.describe(undone[0]) == "Edo undid the decision on Governing Law in lease.pdf"


def test_a_decision_needs_a_verdict_a_clause_type_and_a_span(database, people, lease):
    for clause, verdict, start, end in (
        ("Governing Law", "maybe", 0, 5),
        ("", CONFIRMED, 0, 5),
        ("Governing Law", CONFIRMED, -1, 5),
        ("Governing Law", CONFIRMED, 5, 5),
    ):
        with pytest.raises(files.FilesError) as refused:
            run(database, files.decide, people.ana, lease.id, clause, verdict, start, end, "x")
        assert refused.value.status_code == 400
    assert logged(database, audit.CLAUSE_CONFIRMED, audit.CLAUSE_REJECTED) == []


def test_the_decisions_of_a_contract_apply_to_its_cache(database, people, lease):
    cache = make_cache(
        TEXT,
        Settings(),
        {
            "Governing Law": [("Preamble.", 0.6), (LAW, 0.3)],
            "Dispute Resolution": [(OTHER, 0.7)],
        },
    )
    decide(database, people.ana, lease.id, CONFIRMED)
    decide(database, people.ana, lease.id, REJECTED, OTHER, clause="Dispute Resolution")
    current = run(database, files.decisions_of, lease.id)
    applied = decisions.apply(cache, {name: view.decision for name, view in current.items()})
    top = applied.clauses["Governing Law"][0]
    assert (top.text, top.confirmed, top.conf) == (LAW, True, decisions.CONFIRMED_CONF)
    assert applied.clauses["Dispute Resolution"] == []
    assert current["Governing Law"].decision.at.endswith("+00:00")


# Review


def test_an_editor_sends_for_approval_and_an_admin_approves(database, people, lease):
    sent = step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    assert (sent.review_status, sent.review_changed_by_name) == (SUBMITTED, "Edo")
    assert sent.review_changed_at is not None
    approved = step(database, people.admin, lease.id, ReviewStep.APPROVE)
    assert (approved.review_status, approved.review_changed_by_name) == (APPROVED, "Admin")


def test_only_an_admin_approves_sends_back_or_reopens(database, people, lease):
    with pytest.raises(Forbidden):
        step(database, people.viewer, lease.id, ReviewStep.SUBMIT)
    step(database, people.ana, lease.id, ReviewStep.SUBMIT)
    for actor in (people.ana, people.editor):
        for which in (ReviewStep.APPROVE, ReviewStep.RETURN):
            with pytest.raises(Forbidden):
                step(database, actor, lease.id, which, note="Fix it")
    step(database, people.admin, lease.id, ReviewStep.APPROVE)
    with pytest.raises(Forbidden):
        step(database, people.ana, lease.id, ReviewStep.REOPEN)
    with pytest.raises(NotFound):
        step(database, people.stranger, lease.id, ReviewStep.REOPEN)
    assert step(database, people.admin, lease.id, ReviewStep.REOPEN).review_status == IN_REVIEW


def test_sending_back_needs_a_note_that_the_next_submission_clears(database, people, lease):
    step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    with pytest.raises(files.FilesError) as no_note:
        step(database, people.admin, lease.id, ReviewStep.RETURN, note="   ")
    assert no_note.value.message == files.NOTE_NEEDED
    with pytest.raises(files.FilesError):
        step(database, people.admin, lease.id, ReviewStep.RETURN, note="x" * 2001)
    back = step(
        database, people.admin, lease.id, ReviewStep.RETURN, note=" Check the insurance. "
    )
    assert (back.review_status, back.review_note) == (IN_REVIEW, "Check the insurance.")
    again = step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    assert (again.review_status, again.review_note) == (SUBMITTED, None)


def test_a_step_from_the_wrong_status_conflicts(database, people, lease):
    for actor, which in (
        (people.admin, ReviewStep.APPROVE),
        (people.admin, ReviewStep.RETURN),
        (people.admin, ReviewStep.REOPEN),
    ):
        with pytest.raises(Conflict) as wrong:
            step(database, actor, lease.id, which, note="note")
        assert wrong.value.message == WRONG_STEP
    step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    with pytest.raises(Conflict):
        step(database, people.editor, lease.id, ReviewStep.SUBMIT)


def test_every_step_is_logged_with_its_actor(database, people, lease):
    step(database, people.editor, lease.id, ReviewStep.SUBMIT)
    step(database, people.admin, lease.id, ReviewStep.RETURN, note="Check the insurance.")
    step(database, people.ana, lease.id, ReviewStep.SUBMIT)
    step(database, people.admin, lease.id, ReviewStep.APPROVE)
    step(database, people.admin, lease.id, ReviewStep.REOPEN)
    history = logged(
        database,
        audit.REVIEW_SUBMITTED,
        audit.REVIEW_RETURNED,
        audit.REVIEW_APPROVED,
        audit.REVIEW_REOPENED,
    )
    assert [(e.action, e.actor_id, e.contract_id) for e in history] == [
        (audit.REVIEW_REOPENED, people.admin.id, lease.id),
        (audit.REVIEW_APPROVED, people.admin.id, lease.id),
        (audit.REVIEW_SUBMITTED, people.ana.id, lease.id),
        (audit.REVIEW_RETURNED, people.admin.id, lease.id),
        (audit.REVIEW_SUBMITTED, people.editor.id, lease.id),
    ]
    assert history[3].details == {
        "name": "lease.pdf",
        "from": SUBMITTED,
        "to": IN_REVIEW,
        "note": "Check the insurance.",
    }
    assert audit.describe(history[3]) == "Admin sent lease.pdf back: Check the insurance."


def test_awaiting_approval_lists_the_longest_waiting_first(database, people, lease):
    other = run(database, files.add_contract, people.ana, None, "other.pdf", DOC, "pdf")
    binned = run(database, files.add_contract, people.ana, None, "binned.pdf", DOC, "pdf")
    for contract_id in (other.id, lease.id, binned.id):
        step(database, people.ana, contract_id, ReviewStep.SUBMIT)
    run(database, files.delete_contract, people.ana, binned.id)
    waiting = run(database, files.awaiting_approval, people.admin)
    assert [view.name for view in waiting] == ["other.pdf", "lease.pdf"]
    with pytest.raises(Forbidden):
        run(database, files.awaiting_approval, people.ana)
