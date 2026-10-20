import pytest

from contract_assistant.answer import Passage
from contract_assistant.config import Settings
from contract_assistant.evaluation.passage_eval import evaluate_passage_answers
from contract_assistant.evaluation.questions import EvalQuestion
from contract_assistant.router.clients import LLMError
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

TEXT = "The Term is five years. Fees are due within 30 days."
FEES = "Fees are due within 30 days."
FAITHFUL = 'It says "five years".'
MISQUOTE = 'It says "ten years".'


class FakeSearch:
    """Search returning prepared passages per question; records the questions it saw."""

    def __init__(self, passages: dict[str, list[Passage]]):
        self.passages = passages
        self.questions: list[str] = []

    def search(self, cache, question: str) -> list[Passage]:
        self.questions.append(question)
        return self.passages.get(question, [])


def make_question(id_: str, kind: str = "out_of_scope") -> EvalQuestion:
    return EvalQuestion(id=id_, question=f"question {id_}", gold=[], kind=kind, split="test")


def passage(quote: str = "The Term is five years.") -> Passage:
    return Passage(quote=quote, page=None, score=0.7, start=0, end=len(quote))


@pytest.fixture
def caches():
    return {"doc": make_cache(TEXT, Settings())}


def evaluate(client, questions, caches, search, attempts=2):
    return evaluate_passage_answers(client, questions, caches, search, attempts)


def test_only_out_of_scope_questions_are_asked(caches):
    search = FakeSearch({})
    questions = [
        make_question("a", "single"),
        make_question("b"),
        make_question("c", "compound"),
    ]
    evaluate(FakeLLMClient([]), questions, caches, search)
    assert search.questions == ["question b"]


def test_evaluating_without_out_of_scope_questions_raises(caches):
    with pytest.raises(ValueError):
        evaluate(FakeLLMClient([]), [make_question("a", "single")], caches, FakeSearch({}))


def test_evaluating_without_caches_raises():
    with pytest.raises(ValueError):
        evaluate(FakeLLMClient([]), [make_question("a")], {}, FakeSearch({}))


def test_a_question_without_passages_gives_an_empty_record_and_no_call(caches):
    client = FakeLLMClient([])
    report = evaluate(client, [make_question("a")], caches, FakeSearch({}))
    (record,) = report.records
    assert record.passages == 0
    assert client.text_calls == []
    assert report.no_passages == 1
    assert report.calls == 0


def test_an_answered_question_records_passages_quoted_spans_and_attempts(caches):
    search = FakeSearch({"question a": [passage(), passage(FEES)]})
    report = evaluate(FakeLLMClient([FAITHFUL]), [make_question("a")], caches, search)
    (record,) = report.records
    assert (record.question_id, record.document) == ("a", "doc")
    assert record.passages == 2
    assert record.withheld is False
    assert record.quoted_spans == 1
    assert record.attempts == 1
    assert record.reason is None
    assert report.answered == 1


def test_a_withheld_answer_records_the_reason_and_counts_as_withheld(caches):
    search = FakeSearch({"question a": [passage()]})
    client = FakeLLMClient([MISQUOTE, MISQUOTE])
    report = evaluate(client, [make_question("a")], caches, search)
    (record,) = report.records
    assert record.withheld is True
    assert "passages" in record.reason
    assert record.attempts == 2
    assert (report.withheld, report.answered) == (1, 0)


def test_an_llm_error_counts_as_a_failure_and_gives_no_record(caches):
    search = FakeSearch({"question a": [passage()]})
    report = evaluate(FakeLLMClient([LLMError("down")]), [make_question("a")], caches, search)
    assert report.llm_failures == 1
    assert report.records == []


def test_calls_count_the_questions_with_passages(caches):
    search = FakeSearch({"question a": [passage()], "question c": [passage()]})
    questions = [make_question("a"), make_question("b"), make_question("c")]
    report = evaluate(FakeLLMClient([FAITHFUL, FAITHFUL]), questions, caches, search)
    assert (report.calls, report.no_passages, report.answered) == (2, 1, 2)
    assert len(report.records) == 3


def test_mean_latency_covers_only_records_with_passages(caches):
    search = FakeSearch({"question a": [passage()]})
    questions = [make_question("a"), make_question("b")]
    report = evaluate(FakeLLMClient([FAITHFUL]), questions, caches, search)
    answered = next(r for r in report.records if r.passages)
    assert report.mean_latency_ms == answered.latency_ms


def test_mean_latency_is_none_without_records_with_passages(caches):
    report = evaluate(FakeLLMClient([]), [make_question("a")], caches, FakeSearch({}))
    assert report.mean_latency_ms is None


def test_the_report_names_the_model(caches):
    client = FakeLLMClient([], model="model-x")
    report = evaluate(client, [make_question("a")], caches, FakeSearch({}))
    assert report.model == "model-x"
