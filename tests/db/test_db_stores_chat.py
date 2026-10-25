"""The chat log and the feedback log in the database, and the payloads the chat log refuses."""

import logging
import os
import pickle
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from contract_assistant.access import ReviewStep
from contract_assistant.answer import (
    Answer,
    AnswerItem,
    Citation,
    Definition,
    GeneratedAnswer,
    Passage,
)
from contract_assistant.db.models import TXT, Analysis, ChatMessage, Contract
from contract_assistant.db.stores import (
    ChatLog,
    FeedbackLog,
    RefusedClassError,
    dump_message,
    load_message,
)
from contract_assistant.explain import Explanation
from contract_assistant.feedback import Feedback
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.summary import Summary
from contract_assistant.web.chat import Asked, Bubble, Card, Reply, Said, Shown
from tests.support.db import add_user

QUOTE = "This Agreement shall be governed by the laws of the State of Ohio."


def add_contract(database, owner, name="supply.txt") -> str:
    """Add a contract of owner, with a placeholder analysis, and return its id."""
    with database.session() as session:
        if session.get(Analysis, "sha256:test") is None:
            session.add(Analysis(doc_id="sha256:test", data={}, extractor="test"))
            session.flush()
        contract = Contract(
            owner_id=owner.id, name=name, doc_id="sha256:test", kind=TXT, created_by=owner.id
        )
        session.add(contract)
        session.flush()
        return contract.id


@pytest.fixture
def ana(database):
    return add_user(database, "ana@example.com")


@pytest.fixture
def contract(database, ana):
    return add_contract(database, ana)


def answer_bubble(index=1) -> Bubble:
    """An answer with a citation, drawn as cards, the way the chat builds it."""
    citation = Citation(quote=QUOTE, page=3, conf=0.87, start=100, end=100 + len(QUOTE))
    item = AnswerItem(
        clause_type="Governing Law", status="found", best_conf=0.87, citation=citation
    )
    route = RouteResult(clause_types=["Governing Law"], kind="llm", latency_ms=12.5)
    result = Answer(
        question="Which law applies?",
        route=route,
        tau_low=0.1,
        tau_high=0.5,
        items=[item],
        dropped_quotes=0,
    )
    shown = Shown("Governing Law", "Governing Law", QUOTE, citation, "found")
    cards = [Card("found", "Governing Law", quote=QUOTE, page=3, shown=shown)]
    turn = Turn(question="Which law applies?", clause_types=["Governing Law"])
    reply = Reply("chat/answer.html", {"cards": cards}, answer=result, turn=turn, shown=shown)
    return Bubble("assistant", "<div>answer</div>", answer=result, reply=reply, index=index)


def passages_bubble() -> Bubble:
    """An answer from the closest passages, with the text the model wrote from them."""
    passage = Passage(quote=QUOTE, page=None, score=0.42, start=100, end=100 + len(QUOTE))
    generated = GeneratedAnswer(model="qwen", text="Ohio law applies.", latency_ms=800.0)
    route = RouteResult(clause_types=[], kind="llm", latency_ms=3.0, raw="none")
    result = Answer(
        question="Where are disputes heard?",
        route=route,
        tau_low=0.1,
        tau_high=0.5,
        items=[],
        dropped_quotes=0,
        passages=[passage],
        generated=generated,
    )
    shown = Shown("Related passage", "Related passage", QUOTE, None, "related")
    context = {
        "cards": [Card("related", quote=QUOTE, shown=shown)],
        "generated": "Ohio law applies.",
        "intro": "This is not one of the clauses the assistant checks.",
        "topics": [("Dates", ["Agreement Date", "Expiration Date"])],
    }
    reply = Reply("chat/answer.html", context, answer=result, shown=shown)
    return Bubble("assistant", "", answer=result, reply=reply, index=3)


def generated_bubbles() -> list[Bubble]:
    """An explanation, a summary and a definition, with the pydantic records behind them."""
    explanation = Explanation(
        clause_type="Governing Law",
        quote=QUOTE,
        language="sr",
        model="qwen",
        text="Ugovor prati zakone Ohaja.",
        latency_ms=900.0,
    )
    summary = Summary(
        doc_id="sha256:test",
        model="qwen",
        tau_low=0.1,
        tau_high=0.5,
        latency_ms=1500.0,
        items=[],
        text="A supply agreement.",
    )
    definition = Definition(term="Products", quote="the goods", page=1, start=5, end=14)
    defined = Answer(
        question="Products",
        route=RouteResult(clause_types=[], kind="definition", latency_ms=0.0),
        tau_low=0.1,
        tau_high=0.5,
        items=[],
        dropped_quotes=0,
        definition=definition,
    )
    shown = Shown("Governing Law", "Governing Law", QUOTE)
    explained = {"shown": shown, "explanation": explanation}
    summarized = {"summary": summary, "based_on": [("Governing Law", "found")], "hidden": 0}
    return [
        Bubble("assistant", "", reply=Reply("chat/explanation.html", explained, shown=shown)),
        Bubble("assistant", "", reply=Reply("chat/summary.html", summarized)),
        Bubble("assistant", "", answer=defined, reply=Reply("chat/answer.html", {}, defined)),
    ]


# The chat log


def test_real_chat_messages_load_back_equal(database, ana, contract):
    log = ChatLog(database)
    messages = [
        ("user", Said("user", "<p>Which law applies?</p>", asked=Asked("Which law applies?"))),
        ("assistant", answer_bubble()),
        ("user", Said("user", "", asked=Asked(label="Governing Law", language="sr"))),
        ("assistant", passages_bubble()),
        *(("assistant", bubble) for bubble in generated_bubbles()),
    ]
    for role, message in messages:
        log.append(contract, ana.id, role, message)
    assert log.load(contract, ana.id) == messages


def test_a_loaded_bubble_keeps_one_answer_for_the_bubble_and_its_reply(database, ana, contract):
    log = ChatLog(database)
    log.append(contract, ana.id, "assistant", answer_bubble())
    [(_, bubble)] = log.load(contract, ana.id)
    assert bubble.answer is bubble.reply.answer
    assert bubble.reply.context["cards"][0].shown is bubble.reply.shown


def test_each_user_and_contract_has_a_chat_of_its_own_in_order(database, ana, contract):
    bo = add_user(database, "bo@example.com")
    other = add_contract(database, ana, "other.txt")
    log = ChatLog(database)
    for number in range(3):
        log.append(contract, ana.id, "user", Said("user", f"ana {number}"))
        log.append(contract, bo.id, "user", Said("user", f"bo {number}"))
        log.append(other, ana.id, "user", Said("user", f"other {number}"))
    assert [m.html for _, m in log.load(contract, ana.id)] == ["ana 0", "ana 1", "ana 2"]
    assert [m.html for _, m in log.load(contract, bo.id)] == ["bo 0", "bo 1", "bo 2"]
    assert [m.html for _, m in log.load(other, ana.id)] == ["other 0", "other 1", "other 2"]
    assert log.load(other, bo.id) == []


def test_clear_removes_only_that_chat(database, ana, contract):
    bo = add_user(database, "bo@example.com")
    log = ChatLog(database)
    log.append(contract, ana.id, "user", Said("user", "ana"))
    log.append(contract, ana.id, "assistant", answer_bubble())
    log.append(contract, bo.id, "user", Said("user", "bo"))
    assert log.clear(contract, ana.id) == 2
    assert log.load(contract, ana.id) == []
    assert [m.html for _, m in log.load(contract, bo.id)] == ["bo"]
    log.append(contract, ana.id, "user", Said("user", "again"))
    assert [m.html for _, m in log.load(contract, ana.id)] == ["again"]


def test_replace_changes_one_message_in_its_place(database, ana, contract):
    log = ChatLog(database)
    log.append(contract, ana.id, "user", Said("user", "question"))
    log.append(contract, ana.id, "assistant", answer_bubble())
    log.append(contract, ana.id, "user", Said("user", "next"))
    [_, (_, bubble), _] = log.load(contract, ana.id)
    bubble.thanked = True
    assert log.replace(contract, ana.id, 1, bubble)
    loaded = log.load(contract, ana.id)
    assert [m.html for _, m in loaded] == ["question", "<div>answer</div>", "next"]
    assert loaded[1][1].thanked
    assert not log.replace(contract, ana.id, 3, bubble)
    assert not log.replace(contract, ana.id, -1, bubble)


def test_replace_counts_the_messages_as_load_returns_them(database, ana, contract):
    log = ChatLog(database)
    log.append(contract, ana.id, "user", Said("user", "first"))
    insert_payload(database, contract, ana.id, b"not a pickle")
    log.append(contract, ana.id, "user", Said("user", "third"))
    assert log.replace(contract, ana.id, 1, Said("user", "changed"))
    assert [m.html for _, m in log.load(contract, ana.id)] == ["first", "changed"]


def test_a_chat_is_deleted_with_its_contract(database, ana, contract):
    log = ChatLog(database)
    log.append(contract, ana.id, "user", Said("user", "question"))
    with database.session() as session:
        session.delete(session.get(Contract, contract))
    with database.session() as session:
        assert session.scalar(select(func.count()).select_from(ChatMessage)) == 0


def test_plain_values_load_back(database, ana, contract):
    belgrade = timezone(timedelta(hours=2))
    context = {
        "at": datetime(2026, 10, 25, 12, 30, tzinfo=belgrade),
        "day": datetime(2026, 10, 25, tzinfo=UTC).date(),
        "step": ReviewStep.SUBMIT,
        "tags": {"a", "b"},
        "frozen": frozenset({1}),
        "pair": (1.5, None, True),
        "bytes": b"\x00\x01",
    }
    log = ChatLog(database)
    bubble = Bubble("assistant", "", reply=Reply("chat/text.html", context))
    log.append(contract, ana.id, "assistant", bubble)
    [(_, bubble)] = log.load(contract, ana.id)
    assert bubble.reply.context == context
    assert bubble.reply.context["step"] is ReviewStep.SUBMIT


def test_append_refuses_a_message_the_log_could_not_load(database, ana, contract):
    log = ChatLog(database)
    with pytest.raises(RefusedClassError, match="pathlib"):
        log.append(contract, ana.id, "user", Said("user", "", asked=Asked(Path("x"))))
    assert log.load(contract, ana.id) == []


# Payloads the chat log refuses


def insert_payload(database, contract_id, user_id, payload: bytes) -> None:
    """Put a payload in the chat as if it had been saved: after the messages there."""
    with database.session() as session:
        last = session.scalar(select(func.max(ChatMessage.position))) or 0
        session.add(
            ChatMessage(
                contract_id=contract_id,
                user_id=user_id,
                position=last + 1,
                role="assistant",
                payload=payload,
            )
        )


def stack_global(module: str, name: str, argument: str) -> bytes:
    """A protocol 4 pickle that calls module.name(argument)."""

    def text(value: str) -> bytes:
        data = value.encode()
        return b"\x8c" + bytes([len(data)]) + data

    return b"\x80\x04" + text(module) + text(name) + b"\x93" + text(argument) + b"\x85R."


class RunsACommand:
    def __init__(self, command: str):
        self.command = command

    def __reduce__(self):
        return (os.system, (self.command,))


def test_a_payload_that_calls_os_system_is_skipped_with_a_warning(
    database, ana, contract, tmp_path, caplog
):
    marker = tmp_path / "ran"
    log = ChatLog(database)
    log.append(contract, ana.id, "user", Said("user", "before"))
    for protocol in (0, 2, 5):
        payload = pickle.dumps(RunsACommand(f"touch {marker}"), protocol=protocol)
        insert_payload(database, contract, ana.id, payload)
    log.append(contract, ana.id, "user", Said("user", "after"))
    with caplog.at_level(logging.WARNING, logger="contract_assistant.db.stores"):
        loaded = log.load(contract, ana.id)
    assert [m.html for _, m in loaded] == ["before", "after"]
    assert not marker.exists()
    assert caplog.text.count("is not allowed in a chat message") == 3


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("os", "system"),
        ("posix", "system"),
        ("builtins", "eval"),
        ("subprocess", "getoutput"),
        # Reached through a module of the package that imports os.
        ("contract_assistant.generation", "os.system"),
        ("contract_assistant.generation", "os"),
        # A function of the package, not a class.
        ("contract_assistant.feedback", "record_feedback"),
        # A class the package imports from elsewhere.
        ("contract_assistant.generation", "Path"),
        # A class of the package that is not a data class.
        ("contract_assistant.db.engine", "Database"),
        ("contract_assistant_other", "Thing"),
    ],
)
def test_a_payload_naming_anything_else_is_refused_without_running_it(module, name, tmp_path):
    marker = tmp_path / "ran"
    with pytest.raises(RefusedClassError):
        load_message(stack_global(module, name, f"touch {marker}"))
    assert not marker.exists()


def test_a_payload_whose_class_no_longer_exists_is_skipped(database, ana, contract, caplog):
    payload = dump_message(Said("user", "gone"))
    assert payload.count(b"\x8c\x04Said") == 1
    renamed = payload.replace(b"\x8c\x04Said", b"\x8c\x04Gone")
    insert_payload(database, contract, ana.id, renamed)
    ChatLog(database).append(contract, ana.id, "user", Said("user", "kept"))
    with caplog.at_level(logging.WARNING, logger="contract_assistant.db.stores"):
        loaded = ChatLog(database).load(contract, ana.id)
    assert [m.html for _, m in loaded] == ["kept"]
    assert "contract_assistant.web.chat.Gone no longer exists" in caplog.text


def test_a_payload_of_a_module_that_is_gone_or_not_loaded_is_refused():
    with pytest.raises(RefusedClassError, match="no longer exists"):
        load_message(stack_global("contract_assistant.removed", "Thing", "x"))


def test_a_payload_that_is_not_a_pickle_is_skipped(database, ana, contract, caplog):
    insert_payload(database, contract, ana.id, b"\x80\x05garbage")
    with caplog.at_level(logging.WARNING, logger="contract_assistant.db.stores"):
        assert ChatLog(database).load(contract, ana.id) == []
    assert "skipping chat message" in caplog.text


# The feedback log


def verdict(doc_id: str, at: str, liked=True, **changes) -> Feedback:
    fields = {
        "doc_id": doc_id,
        "question": "Which law applies?",
        "route_kind": "llm",
        "clause_types": ["Governing Law"],
        "liked": liked,
        "at": at,
    }
    return Feedback(**{**fields, **changes})


def test_feedback_loads_back_oldest_first_per_contract_or_all(database, ana, contract):
    other = add_contract(database, ana, "other.txt")
    log = FeedbackLog(database)
    late = verdict(contract, "2026-10-25T12:00:00+00:00", liked=False, comment="Wrong page.")
    early = verdict(contract, "2026-10-25T09:00:00+00:00")
    elsewhere = verdict(other, "2026-10-25T10:00:00+00:00", clause_types=[])
    for item in (late, early, elsewhere):
        log.record(item.doc_id, ana.id, item)
    assert log.read(contract) == [early, late]
    assert log.read(other) == [elsewhere]
    assert log.read() == [early, elsewhere, late]


def test_feedback_without_a_time_zone_is_taken_as_utc(database, ana, contract):
    log = FeedbackLog(database)
    log.record(contract, ana.id, verdict(contract, "2026-10-25T09:00:00"))
    assert log.read(contract)[0].at == "2026-10-25T09:00:00+00:00"


def test_feedback_of_no_contract_and_no_user_reads_with_an_empty_doc_id(database):
    log = FeedbackLog(database)
    log.record(None, None, verdict("sha256:old", "2026-10-25T09:00:00+00:00"))
    assert log.read()[0].doc_id == ""


@pytest.mark.pg
def test_postgres_keeps_chat_payloads_and_feedback(pg_database):
    ana = add_user(pg_database, "ana@example.com")
    contract = add_contract(pg_database, ana)
    log = ChatLog(pg_database)
    messages = [("user", Said("user", "q", asked=Asked("q"))), ("assistant", answer_bubble())]
    for role, message in messages:
        log.append(contract, ana.id, role, message)
    assert log.load(contract, ana.id) == messages
    feedback = FeedbackLog(pg_database)
    item = verdict(contract, "2026-10-25T09:00:00+00:00")
    feedback.record(contract, ana.id, item)
    assert feedback.read(contract) == [item]
