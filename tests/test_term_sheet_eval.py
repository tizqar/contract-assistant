import pytest

from contract_assistant.config import Settings
from contract_assistant.evaluation.term_sheet_eval import (
    evaluate_term_sheets,
    load_demo_gold,
)
from contract_assistant.router.clients import LLMError
from contract_assistant.store import read_cache
from contract_assistant.term_sheet import FIELDS
from tests.conftest import ROOT
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

TAU_LOW = 0.1
TAU_HIGH = 0.8
DEMO = ROOT / "demo"
PARTIES = "Acme Corp. and Beta LLC"
LAW = "This Agreement is governed by the Laws of England."
RENEWAL = "It renews for ten (10) years."
TEXT = f"Contract between {PARTIES}. {LAW} {RENEWAL}"
LAW_REPLY = '{"governing_law": "Laws of England"}'
NOTHING = "{}"


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def cache(settings):
    return make_cache(
        TEXT,
        settings,
        {
            "Parties": [(PARTIES, 0.9)],
            "Governing Law": [(LAW, 0.9)],
            "Renewal Term": [(RENEWAL, 0.9)],
        },
    )


@pytest.fixture
def itemless_cache(settings):
    return make_cache("Nothing here.", settings, {"Insurance": [("Nothing here.", 0.9)]})


def evaluate(replies, caches, gold=None, attempts=1):
    client = replies if hasattr(replies, "text_calls") else FakeLLMClient(replies)
    return evaluate_term_sheets(
        client, caches, gold or {}, TAU_LOW, TAU_HIGH, attempts=attempts
    )


def record_of(report, field, document="doc"):
    return next(r for r in report.records if r.document == document and r.field == field)


def test_load_demo_gold_reads_the_three_demo_documents():
    gold = load_demo_gold(DEMO / "demo_documents.json")
    assert set(gold) == {
        "short_supply_agreement",
        "medium_sponsorship_agreement",
        "long_outsourcing_agreement",
    }
    (span,) = gold["short_supply_agreement"]["Governing Law"]
    assert "Laws of England" in span


def test_evaluating_without_caches_raises():
    with pytest.raises(ValueError):
        evaluate([], {})


def test_one_record_per_field_in_fields_order(cache):
    report = evaluate([NOTHING, NOTHING], {"a": cache, "b": cache})
    assert [r.document for r in report.records] == ["a"] * len(FIELDS) + ["b"] * len(FIELDS)
    assert [r.field for r in report.records[: len(FIELDS)]] == list(FIELDS)
    assert len(FIELDS) == 8


def test_a_value_agrees_when_a_gold_span_contains_it(cache):
    gold = {"doc": {"Governing Law": [LAW]}}
    report = evaluate([LAW_REPLY], {"doc": cache}, gold)
    assert record_of(report, "governing_law").agreeing == 1
    assert report.agreeing == 1


def test_a_value_agrees_when_it_contains_a_gold_span(cache):
    gold = {"doc": {"Governing Law": ["laws of england"]}}
    report = evaluate(['{"governing_law": "the Laws of England"}'], {"doc": cache}, gold)
    assert record_of(report, "governing_law").agreeing == 1


def test_agreement_ignores_case_and_whitespace(cache):
    gold = {"doc": {"Governing Law": ["LAWS   OF\nEngland"]}}
    report = evaluate([LAW_REPLY], {"doc": cache}, gold)
    assert record_of(report, "governing_law").agreeing == 1


def test_a_value_does_not_agree_with_an_unrelated_gold_span(cache):
    gold = {"doc": {"Governing Law": ["the Laws of France"]}}
    report = evaluate([LAW_REPLY], {"doc": cache}, gold)
    assert record_of(report, "governing_law").agreeing == 0
    assert report.agreeing == 0


def test_a_value_does_not_agree_with_gold_of_another_clause_type(cache):
    gold = {"doc": {"Parties": [LAW]}}
    report = evaluate([LAW_REPLY], {"doc": cache}, gold)
    assert record_of(report, "governing_law").agreeing == 0


def test_gold_present_reflects_the_gold_of_the_source_types(cache):
    gold = {"doc": {"Renewal Term": ["renews"], "Insurance": ["x"]}}
    report = evaluate([NOTHING], {"doc": cache}, gold)
    present = {r.field for r in report.records if r.gold_present}
    assert present == {"term", "renewal_term"}


def test_missing_counts_fields_with_gold_but_no_values_or_rejected_ones(cache):
    gold = {"doc": {"Governing Law": [LAW], "Parties": [PARTIES]}}
    report = evaluate(['{"parties": ["Zeta"]}'], {"doc": cache}, gold)
    assert report.missing == 1
    assert record_of(report, "parties").rejected == ["Zeta"]


def test_rejected_counts_the_rejected_values(cache):
    report = evaluate(['{"parties": ["Zeta", "Omega"]}'], {"doc": cache})
    assert report.rejected == 2


def test_a_document_without_items_is_withheld_and_gives_no_records(itemless_cache):
    client = FakeLLMClient([])
    report = evaluate(client, {"doc": itemless_cache})
    assert report.withheld == 1
    assert report.records == []
    assert client.text_calls == []


def test_an_invalid_reply_is_withheld_and_gives_records_without_values(cache):
    report = evaluate(["bad", "bad"], {"doc": cache}, attempts=2)
    assert report.withheld == 1
    assert len(report.records) == len(FIELDS)
    assert report.values == 0


def test_an_llm_error_counts_as_a_failure_and_gives_no_records(cache):
    report = evaluate([LLMError("down")], {"doc": cache})
    assert report.llm_failures == 1
    assert report.records == []


def test_mean_latency_is_none_without_sheets(itemless_cache):
    assert evaluate([], {"doc": itemless_cache}).mean_latency_ms is None


def test_mean_latency_is_a_number_with_sheets(cache):
    assert evaluate([NOTHING], {"doc": cache}).mean_latency_ms >= 0


def test_the_report_counts_documents_fields_values_and_the_model(cache, itemless_cache):
    report = evaluate([LAW_REPLY], {"a": cache, "b": itemless_cache})
    assert report.documents == 2
    assert report.fields == len(FIELDS)
    assert report.values == 1
    assert report.model == "fake-llm"


def test_the_echo_client_on_the_demo_documents_finishes():
    settings = Settings()
    documents = load_demo_gold(DEMO / "demo_documents.json")
    caches = {slug: read_cache(DEMO / "cache" / f"{slug}.json", settings) for slug in documents}
    report = evaluate_term_sheets(EchoTextClient(), caches, documents, TAU_LOW, TAU_HIGH)
    assert report.documents == 3
    assert report.fields == 3 * len(FIELDS)
