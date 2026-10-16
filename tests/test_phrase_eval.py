import json
import logging

import pytest

from contract_assistant.config import Settings
from contract_assistant.evaluation import phrase_eval
from contract_assistant.evaluation.phrase_eval import PhrasingReport, evaluate_phrasing
from contract_assistant.evaluation.questions import EvalQuestion, QuestionKind
from contract_assistant.router.clients import LLMError
from contract_assistant.store import DocumentCache
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

LAW_ALPHA = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
LAW_BETA = "This Agreement is governed by the laws of Texas."
TAU_LOW = 0.2
TAU_HIGH = 0.8


def reply(paragraph: str) -> str:
    return json.dumps({"paragraph": paragraph})


FAITHFUL = f'Delaware law governs: "{LAW_ALPHA}"'
UNQUOTED = "Texas law governs."
ALTERED = 'Delaware law governs: "laws of Texas"'
INVENTED = '"laws of Ohio" and "five months"'
# One reply per call: the questions in order, each on alpha and then on beta.
REPLIES = [
    reply(FAITHFUL),
    reply(UNQUOTED),
    reply(ALTERED),
    LLMError("timed out"),
    "not json",
    reply(INVENTED),
]


def make_question(number: int, kind: QuestionKind, gold: list[str]) -> EvalQuestion:
    return EvalQuestion(
        id=f"q{number}", question=f"question {number}", gold=gold, kind=kind, split="test"
    )


QUESTIONS = [
    make_question(1, "single", ["Governing Law"]),
    make_question(2, "compound", ["Governing Law", "Term"]),
    make_question(3, "out_of_scope", []),
    make_question(4, "single", ["Term"]),
]


@pytest.fixture
def caches() -> dict[str, DocumentCache]:
    settings = Settings()
    alpha = {"Governing Law": [(LAW_ALPHA, 0.9)], "Term": [(TERM, 0.9)]}
    beta = {"Governing Law": [(LAW_BETA, 0.9)], "Term": []}
    return {
        "alpha": make_cache(f"{LAW_ALPHA} {TERM}", settings, alpha),
        "beta": make_cache(LAW_BETA, settings, beta),
    }


def mixed_client() -> FakeLLMClient:
    return FakeLLMClient(REPLIES, model="qwen-test")


def run(
    client: FakeLLMClient,
    caches: dict[str, DocumentCache],
    questions: list[EvalQuestion],
    tau_low: float = TAU_LOW,
    tau_high: float = TAU_HIGH,
) -> PhrasingReport:
    return evaluate_phrasing(client, questions, caches, tau_low, tau_high)


@pytest.fixture
def report(caches) -> PhrasingReport:
    return run(mixed_client(), caches, QUESTIONS)


def test_every_in_scope_question_is_asked_on_every_document(report):
    assert report.calls == 6


def test_report_names_the_model_of_the_client(report):
    assert report.model == "qwen-test"


def test_faithful_and_unquoted_replies_are_accepted(report):
    assert report.accepted == 2


def test_replies_with_unfaithful_spans_are_violations(report):
    assert report.violations == 2


def test_replies_without_a_quoted_span_are_counted(report):
    assert report.unquoted_replies == 1


def test_llm_errors_are_counted_as_failures(report):
    assert report.llm_failures == 1


def test_replies_off_the_schema_are_counted_as_invalid(report):
    assert report.invalid_replies == 1


def test_every_call_ends_in_exactly_one_outcome(report):
    outcomes = (
        report.accepted + report.violations + report.llm_failures + report.invalid_replies
    )
    assert outcomes == report.calls


def test_violation_rate_leaves_out_the_failed_calls(report):
    assert report.violation_rate == pytest.approx(2 / 5)


def test_violation_rate_is_over_all_calls_when_none_failed(caches):
    client = FakeLLMClient([reply(ALTERED), reply(FAITHFUL)])
    report = run(client, {"alpha": caches["alpha"]}, QUESTIONS[:2])
    assert report.violation_rate == 0.5


def test_violation_rate_is_none_when_every_call_failed(caches):
    client = FakeLLMClient([LLMError("down")] * 6)
    report = run(client, caches, QUESTIONS)
    assert (report.llm_failures, report.violation_rate) == (6, None)


def test_out_of_scope_questions_are_not_asked(caches):
    client = mixed_client()
    run(client, caches, QUESTIONS)
    assert len(client.calls) == 6
    assert not any("question 3" in user for _, user, _ in client.calls)


def test_questions_are_asked_in_order_each_on_every_document(caches):
    client = mixed_client()
    run(client, caches, QUESTIONS)
    asked = [user.splitlines()[0] for _, user, _ in client.calls]
    assert asked == [f"Question: question {number}" for number in (1, 1, 2, 2, 4, 4)]


def test_route_given_to_answer_is_the_gold_set_of_the_question(monkeypatch, caches):
    routes = []
    real_answer = phrase_eval.answer

    def spy(question, route, cache, tau_low, tau_high):
        routes.append(route)
        return real_answer(question, route, cache, tau_low, tau_high)

    monkeypatch.setattr(phrase_eval, "answer", spy)
    run(mixed_client(), caches, QUESTIONS)
    assert [(route.kind, route.clause_types) for route in routes] == [
        ("oracle", ["Governing Law"]),
        ("oracle", ["Governing Law"]),
        ("oracle", ["Governing Law", "Term"]),
        ("oracle", ["Governing Law", "Term"]),
        ("oracle", ["Term"]),
        ("oracle", ["Term"]),
    ]


def test_thresholds_decide_the_status_the_model_is_given(caches):
    client = FakeLLMClient([reply(UNQUOTED)])
    run(client, {"alpha": caches["alpha"]}, QUESTIONS[:1], tau_low=0.95, tau_high=0.99)
    _, user, _ = client.calls[0]
    assert "- Governing Law (not found)" in user


def test_violation_record_carries_the_offending_span(report):
    record = report.violation_records[0]
    assert (record.question_id, record.document) == ("q2", "alpha")
    assert record.spans == ["laws of Texas"]
    assert record.paragraph == ALTERED


def test_every_span_is_offending_when_the_answer_has_no_verified_quote(report):
    record = report.violation_records[1]
    assert (record.question_id, record.document) == ("q4", "beta")
    assert record.spans == ["laws of Ohio", "five months"]


def test_there_is_one_record_per_violation(report):
    assert len(report.violation_records) == report.violations


def test_failed_call_is_logged_with_question_document_and_error(caches, caplog):
    client = FakeLLMClient([LLMError("timed out")])
    with caplog.at_level(logging.WARNING):
        run(client, {"alpha": caches["alpha"]}, QUESTIONS[:1])
    [record] = caplog.records
    for part in ("q1", "alpha", "timed out"):
        assert part in record.getMessage()


def test_no_in_scope_questions_raises_before_asking_the_model(caches):
    client = mixed_client()
    with pytest.raises(ValueError, match="no in-scope questions"):
        run(client, caches, [QUESTIONS[2]])
    assert client.calls == []


def test_no_documents_raises_before_asking_the_model():
    client = mixed_client()
    with pytest.raises(ValueError, match="no in-scope questions or documents"):
        run(client, {}, QUESTIONS)
    assert client.calls == []


def test_report_survives_a_json_round_trip(report):
    assert PhrasingReport.model_validate_json(report.model_dump_json(indent=2)) == report
