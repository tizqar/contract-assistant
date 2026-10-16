import json
import logging

import pytest

from contract_assistant.answer import Answer, answer
from contract_assistant.config import Settings
from contract_assistant.phrase import (
    PHRASE_SCHEMA,
    cited_quotes,
    quoted_spans,
    request_paragraph,
    unfaithful_spans,
)
from contract_assistant.router.base import RouteResult
from contract_assistant.router.clients import LLMError, OllamaClient
from contract_assistant.store import read_cache
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

QUESTION = "Which law governs this agreement?"
LAW = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
EXCLUSIVITY = "Supplier shall be the exclusive supplier."
INSURANCE = "Each party shall carry liability insurance."
TEXT = f"{LAW} {TERM} {EXCLUSIVITY} {INSURANCE}"
WRAPPED_LAW = "This Agreement is governed by\n    the laws of Delaware."
DEFINED_TERM = 'This Agreement is made by Acme (the "Company") and Beta.'
CURLY_TERM = "The “Company” shall pay the fee."
TAU_LOW = 0.2
TAU_HIGH = 0.8
DEMO_CACHE = "short_supply_agreement.json"


@pytest.fixture
def settings() -> Settings:
    return Settings()


def answer_to(settings: Settings, clauses: dict[str, list[tuple[str, float]]]) -> Answer:
    cache = make_cache(TEXT, settings, clauses)
    route = RouteResult(clause_types=list(clauses), kind="oracle", latency_ms=0.0)
    return answer(QUESTION, route, cache, TAU_LOW, TAU_HIGH)


@pytest.fixture
def law_answer(settings) -> Answer:
    return answer_to(settings, {"Governing Law": [(LAW, 0.9)]})


@pytest.fixture
def mixed_answer(settings) -> Answer:
    cache = make_cache(
        TEXT,
        settings,
        {
            "Governing Law": [(LAW, 0.9)],
            "Term": [(TERM, 0.5)],
            "Exclusivity": [(EXCLUSIVITY, 0.01)],
            "Insurance": [(INSURANCE, 0.9)],
        },
    )
    cache.clauses["Insurance"][0].start += 1
    route = RouteResult(clause_types=list(cache.clauses), kind="oracle", latency_ms=0.0)
    return answer(QUESTION, route, cache, TAU_LOW, TAU_HIGH)


def reply(paragraph: str) -> str:
    return json.dumps({"paragraph": paragraph})


def sent_request(result: Answer) -> tuple[str, str, dict]:
    client = FakeLLMClient([reply("Delaware law governs.")])
    request_paragraph(result, client)
    return client.calls[0]


def test_valid_reply_gives_the_paragraph_as_written(law_answer):
    paragraph = f'The contract is governed by Delaware law: "{LAW}"'
    assert request_paragraph(law_answer, FakeLLMClient([reply(paragraph)])) == paragraph


def test_paragraph_is_returned_without_surrounding_whitespace(law_answer):
    client = FakeLLMClient(['{"paragraph": "  Delaware law governs.\\n"}'])
    assert request_paragraph(law_answer, client) == "Delaware law governs."


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"paragraph": "Delaware law governs."',
        '{"text": "Delaware law governs."}',
        '{"paragraph": "Delaware law governs.", "extra": 1}',
        '{"paragraph": 5}',
        '{"paragraph": ""}',
        '{"paragraph": "   "}',
        '["Delaware law governs."]',
    ],
    ids=[
        "malformed",
        "truncated",
        "wrong_key",
        "extra_key",
        "number",
        "empty",
        "blank",
        "list",
    ],
)
def test_reply_that_does_not_follow_the_schema_gives_none(law_answer, raw):
    assert request_paragraph(law_answer, FakeLLMClient([raw])) is None


def test_reply_that_does_not_follow_the_schema_is_logged_with_the_model(law_answer, caplog):
    client = FakeLLMClient(["not json"], model="qwen-test")
    with caplog.at_level(logging.WARNING):
        request_paragraph(law_answer, client)
    [record] = caplog.records
    assert "qwen-test" in record.getMessage()


def test_llm_error_propagates(law_answer):
    with pytest.raises(LLMError):
        request_paragraph(law_answer, FakeLLMClient([LLMError("timed out")]))


def test_answer_without_items_raises_before_asking_the_model(settings):
    empty = answer_to(settings, {})
    client = FakeLLMClient([reply("nothing")])
    with pytest.raises(ValueError, match="no items"):
        request_paragraph(empty, client)
    assert client.calls == []


def test_the_model_is_asked_once(law_answer):
    client = FakeLLMClient([reply("Delaware law governs.")])
    request_paragraph(law_answer, client)
    assert len(client.calls) == 1


def test_user_message_has_the_question_and_each_verified_quote(mixed_answer):
    _, user, _ = sent_request(mixed_answer)
    assert user.startswith(f"Question: {QUESTION}\n")
    assert f'"{LAW}"' in user
    assert f'"{TERM}"' in user


def test_user_message_has_no_quote_text_of_not_found_or_withheld_items(mixed_answer):
    _, user, _ = sent_request(mixed_answer)
    assert EXCLUSIVITY not in user
    assert INSURANCE not in user


def test_user_message_lists_each_clause_type_with_its_status_in_order(mixed_answer):
    _, user, _ = sent_request(mixed_answer)
    lines = [line for line in user.splitlines() if line.startswith("- ")]
    assert [line.split(":")[0] for line in lines] == [
        "- Governing Law (found)",
        "- Term (possible, verify)",
        "- Exclusivity (not found)",
        "- Insurance (found, quote withheld)",
    ]


def test_system_prompt_asks_for_word_for_word_quotes_in_double_quotes(law_answer):
    system, _, _ = sent_request(law_answer)
    assert "word for word" in system
    assert "double quotes" in system


def test_schema_passed_to_the_client_is_the_phrasing_schema(law_answer):
    _, _, schema = sent_request(law_answer)
    assert schema == PHRASE_SCHEMA


def test_phrasing_schema_requires_a_paragraph_and_forbids_other_keys():
    assert PHRASE_SCHEMA == {
        "type": "object",
        "additionalProperties": False,
        "required": ["paragraph"],
        "properties": {"paragraph": {"type": "string"}},
    }


def test_cited_quotes_are_the_verified_quotes_in_item_order(mixed_answer):
    assert cited_quotes(mixed_answer) == [LAW, TERM]


def test_quoted_spans_reads_straight_and_curly_pairs_in_order():
    assert quoted_spans('He said "one" and “two” and "three"') == ["one", "two", "three"]


def test_quoted_spans_ignores_blank_spans_and_unpaired_marks():
    assert quoted_spans('An empty "" and a blank " " and "dangling') == []


def test_quoted_spans_keeps_line_breaks_inside_a_span():
    assert quoted_spans('"first\nsecond"') == ["first\nsecond"]


def test_exact_quote_has_no_unfaithful_span():
    paragraph = f'The contract is governed by Delaware law: "{LAW}"'
    assert unfaithful_spans(paragraph, [LAW]) == []


def test_paragraph_without_a_quoted_span_has_no_unfaithful_span():
    assert unfaithful_spans("Delaware law governs the agreement.", [LAW]) == []


def test_part_of_a_quote_word_for_word_is_faithful():
    paragraph = 'The agreement is "governed by the laws of Delaware".'
    assert unfaithful_spans(paragraph, [LAW]) == []


def test_span_from_any_of_the_quotes_is_faithful():
    assert unfaithful_spans(f'The duration is stated: "{TERM}"', [LAW, TERM]) == []


def test_line_wrap_inside_a_quote_is_tolerated():
    paragraph = 'It says "governed by the laws of Delaware".'
    assert unfaithful_spans(paragraph, [WRAPPED_LAW]) == []


def test_extra_whitespace_inside_a_span_is_tolerated():
    paragraph = 'It says "governed  by\nthe laws of   Delaware".'
    assert unfaithful_spans(paragraph, [LAW]) == []


def test_altered_word_makes_the_span_unfaithful():
    span = "This Agreement is governed by the laws of Texas."
    assert unfaithful_spans(f'It says "{span}"', [LAW]) == [span]


def test_word_added_to_a_quote_makes_the_span_unfaithful():
    span = "This Agreement is governed by the laws of the State of Delaware."
    assert unfaithful_spans(f'It says "{span}"', [LAW]) == [span]


def test_only_the_altered_span_among_faithful_ones_is_returned():
    paragraph = f'"{LAW}" and "The Term is ten years."'
    assert unfaithful_spans(paragraph, [LAW, TERM]) == ["The Term is ten years."]


@pytest.mark.parametrize("text", [EXCLUSIVITY, INSURANCE], ids=["not_found", "withheld"])
def test_document_text_that_is_not_a_cited_quote_is_unfaithful(mixed_answer, text):
    assert unfaithful_spans(f'It also says "{text}"', cited_quotes(mixed_answer)) == [text]


def test_faithful_curly_quotes_are_checked_like_straight_ones():
    assert unfaithful_spans(f"The contract says “{LAW}”", [LAW]) == []


def test_altered_curly_quotes_make_the_span_unfaithful():
    span = "This Agreement is governed by the laws of Texas."
    assert unfaithful_spans(f"The contract says “{span}”", [LAW]) == [span]


def test_curly_quotes_inside_a_straight_quoted_span_belong_to_the_span():
    assert unfaithful_spans(f'It says "{CURLY_TERM}"', [CURLY_TERM]) == []


def test_verbatim_copy_of_a_quote_that_contains_double_quotes_is_faithful():
    assert unfaithful_spans(f'It says "{DEFINED_TERM}"', [DEFINED_TERM]) == []


def test_change_after_a_double_quote_inside_a_quote_is_unfaithful():
    paragraph = 'It says "This Agreement is made by Acme (the "Company") and Gamma."'
    assert unfaithful_spans(paragraph, [DEFINED_TERM]) == [") and Gamma."]


def test_unfaithful_spans_gives_the_offending_span_as_written():
    assert unfaithful_spans('"a  b" and "c  x"', ["a b c d"]) == ["c  x"]


def test_every_span_is_unfaithful_without_any_quote():
    assert unfaithful_spans('"a" and “b”', []) == ["a", "b"]


def test_unfaithful_spans_compares_words_exactly():
    spans = unfaithful_spans('"laws of Delaware"', ["the Laws of Delaware"])
    assert spans == ["laws of Delaware"]


@pytest.mark.slow
def test_local_model_returns_a_paragraph_for_the_governing_law_clause(settings):
    client = OllamaClient.from_settings(settings)
    if not client.available():
        pytest.skip(f"{settings.llm_model} is not available from the local Ollama server")
    cache = read_cache(settings.demo_cache_dir / DEMO_CACHE, settings)
    route = RouteResult(clause_types=["Governing Law"], kind="oracle", latency_ms=0.0)
    result = answer(QUESTION, route, cache, settings.tau_low, settings.tau_high)
    paragraph = request_paragraph(result, client)
    assert paragraph
    spans = unfaithful_spans(paragraph, cited_quotes(result))
    assert all(span in paragraph for span in spans)
