import pytest

from contract_assistant.config import Settings
from contract_assistant.evaluation.summary_eval import evaluate_summaries
from contract_assistant.router.clients import LLMError
from contract_assistant.summary import summary_items
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

LAW = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
EXCLUSIVITY = "Supplier shall be the exclusive supplier."
INSURANCE = "Each party shall carry liability insurance."
NOTICE = "Notices must be in writing."
TEXT = f"{LAW} {TERM} {EXCLUSIVITY} {INSURANCE} {NOTICE}"
TYPES = ["Governing Law", "Term", "Insurance", "Exclusivity", "Notice"]
TAU_LOW = 0.2
TAU_HIGH = 0.8
ECHO_QUOTES = 3
MISQUOTE = 'It says "The Term is ten years."'


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def full_cache(settings):
    return make_cache(
        TEXT,
        settings,
        {
            "Governing Law": [(LAW, 0.9)],
            "Term": [(TERM, 0.5)],
            "Insurance": [(INSURANCE, 0.9)],
            "Exclusivity": [(EXCLUSIVITY, 0.01)],
            "Notice": [(NOTICE, 0.7)],
        },
    )


@pytest.fixture
def empty_cache(settings):
    return make_cache("Nothing here.", settings, {"Term": [("Nothing here.", 0.01)]})


def evaluate(client, caches):
    return evaluate_summaries(client, caches, TYPES, TAU_LOW, TAU_HIGH)


def test_evaluating_without_caches_raises():
    with pytest.raises(ValueError):
        evaluate(EchoTextClient(), {})


def test_one_record_per_document_in_cache_order(full_cache, empty_cache):
    report = evaluate(EchoTextClient(), {"b": full_cache, "a": empty_cache})
    assert [record.document for record in report.records] == ["b", "a"]
    assert report.documents == 2


def test_document_without_items_is_withheld_without_a_model_call(empty_cache):
    client = FakeLLMClient([])
    report = evaluate(client, {"empty": empty_cache})
    [record] = report.records
    assert record.items == 0
    assert record.withheld is True
    assert client.text_calls == []


def test_echo_client_gives_a_record_with_the_summary_numbers(full_cache):
    report = evaluate(EchoTextClient(), {"doc": full_cache})
    [record] = report.records
    assert record.items == len(summary_items(full_cache, TYPES, TAU_LOW, TAU_HIGH))
    assert record.quoted_spans == ECHO_QUOTES
    assert record.sentences == ECHO_QUOTES
    assert record.chars > 0
    assert record.withheld is False
    assert record.unfaithful_spans == []


def test_record_chars_is_the_length_of_the_summary_text(full_cache):
    client = FakeLLMClient(["A short summary."])
    [record] = evaluate(client, {"doc": full_cache}).records
    assert record.chars == len("A short summary.")


def test_twice_misquoting_reply_is_a_withheld_record_with_its_spans(full_cache):
    report = evaluate(FakeLLMClient([MISQUOTE, MISQUOTE]), {"doc": full_cache})
    [record] = report.records
    assert record.withheld is True
    assert record.unfaithful_spans == ["The Term is ten years."]
    assert report.withheld == 1


def test_withheld_counts_documents_without_items_too(full_cache, empty_cache):
    report = evaluate(EchoTextClient(), {"full": full_cache, "empty": empty_cache})
    assert report.withheld == 1


def test_llm_failure_is_counted_and_gives_no_record(full_cache):
    report = evaluate(FakeLLMClient([LLMError("down")]), {"doc": full_cache})
    assert report.llm_failures == 1
    assert report.records == []
    assert report.documents == 1


def test_mean_latency_is_none_when_no_document_produced_a_summary(full_cache, empty_cache):
    report = evaluate(FakeLLMClient([LLMError("down")]), {"a": full_cache, "b": empty_cache})
    assert report.mean_latency_ms is None


def test_mean_latency_is_the_mean_over_documents_with_items(full_cache, empty_cache):
    report = evaluate(EchoTextClient(), {"a": full_cache, "b": empty_cache, "c": full_cache})
    with_items = [record.latency_ms for record in report.records if record.items]
    assert report.mean_latency_ms == pytest.approx(sum(with_items) / 2)


def test_report_names_the_model_of_the_client(full_cache):
    assert evaluate(EchoTextClient(), {"doc": full_cache}).model == "fake-gen"


def test_second_reply_is_recorded_as_a_retry(full_cache):
    report = evaluate(FakeLLMClient([MISQUOTE, "A short summary."]), {"doc": full_cache})
    [record] = report.records
    assert (record.withheld, record.attempts) == (False, 2)
    assert (report.retried, report.withheld) == (1, 0)


def test_first_good_reply_is_one_attempt_and_no_retry(full_cache):
    report = evaluate(FakeLLMClient(["A short summary."]), {"doc": full_cache})
    assert (report.records[0].attempts, report.retried) == (1, 0)


def test_attempts_can_be_limited_to_one(full_cache):
    client = FakeLLMClient([MISQUOTE])
    report = evaluate_summaries(client, {"doc": full_cache}, TYPES, TAU_LOW, TAU_HIGH, attempts=1)
    assert (report.withheld, report.retried, len(client.text_calls)) == (1, 0, 1)
