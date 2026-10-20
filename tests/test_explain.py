import logging

import pytest

from contract_assistant.explain import (
    DEFAULT_LANGUAGE,
    EMPTY_REASON,
    LANGUAGES,
    MAX_SENTENCES,
    MAX_TOKENS,
    MISQUOTE_REASON,
    UnknownLanguageError,
    request_explanation,
    unsupported_numbers,
)
from contract_assistant.phrase import unsupported_numbers as phrase_unsupported_numbers
from contract_assistant.router.clients import LLMError
from tests.support.fakes import EchoTextClient, FakeLLMClient

CLAUSE_TYPE = "Term"
QUOTE = "The Term is five years from the Effective Date."
WRAPPED_QUOTE = "The Term is five\n    years from the Effective Date."
NUMERIC_QUOTE = "The fee is ten (10) dollars, or 1.5 percent of 1,000 units."
RESTATEMENT = "The agreement lasts five years from the day it starts."
MODEL = "gen-test"


def explain(reply: str, quote: str = QUOTE, language: str = "en"):
    client = FakeLLMClient([reply], model=MODEL)
    return request_explanation(CLAUSE_TYPE, quote, language, client)


def test_numbers_missing_from_the_quote_are_listed_in_order():
    assert unsupported_numbers("It is 5 years, then 7 or 9.", "It is 7 years.") == ["5", "9"]


def test_a_number_written_in_digits_is_supported_by_the_quote():
    assert unsupported_numbers("It lasts 10 years.", "for ten (10) years") == []


@pytest.mark.parametrize("number", ["1.5", "1,000"])
def test_a_decimal_or_thousands_number_is_one_number(number):
    assert unsupported_numbers(f"Pay {number} now.", "nothing here") == [number]


def test_text_without_digits_has_no_unsupported_numbers():
    assert unsupported_numbers("Five years.", "ten years") == []


def test_languages_offer_english_and_serbian():
    assert {"en", "sr"} <= set(LANGUAGES)


def test_default_language_is_english():
    assert DEFAULT_LANGUAGE == "en"


def test_unknown_language_raises_before_any_call():
    client = FakeLLMClient([RESTATEMENT])
    with pytest.raises(UnknownLanguageError, match="de"):
        request_explanation(CLAUSE_TYPE, QUOTE, "de", client)
    assert client.text_calls == []


def test_system_prompt_names_english_and_the_sentence_limit():
    client = FakeLLMClient([RESTATEMENT])
    request_explanation(CLAUSE_TYPE, QUOTE, "en", client)
    system = client.text_calls[0][0]
    assert "English" in system
    assert f"at most {MAX_SENTENCES} sentences" in system


def test_system_prompt_names_serbian_for_the_serbian_language():
    client = FakeLLMClient([RESTATEMENT])
    request_explanation(CLAUSE_TYPE, QUOTE, "sr", client)
    assert "Serbian" in client.text_calls[0][0]


def test_user_message_has_the_clause_type_and_the_quote_in_double_quotes():
    client = FakeLLMClient([RESTATEMENT])
    request_explanation(CLAUSE_TYPE, QUOTE, "en", client)
    assert client.text_calls[0][1] == f'Clause type: {CLAUSE_TYPE}\nClause: "{QUOTE}"'


def test_user_message_collapses_whitespace_in_the_quote():
    client = FakeLLMClient([RESTATEMENT])
    request_explanation(CLAUSE_TYPE, WRAPPED_QUOTE, "en", client)
    assert f'"{QUOTE}"' in client.text_calls[0][1]


def test_explanation_asks_for_the_token_limit():
    client = FakeLLMClient([RESTATEMENT])
    request_explanation(CLAUSE_TYPE, QUOTE, "en", client)
    assert client.text_calls[0][2] == MAX_TOKENS


def test_plain_restatement_is_returned_stripped():
    explanation = explain(f"  {RESTATEMENT}\n")
    assert explanation.text == RESTATEMENT
    assert explanation.reason is None


def test_reply_quoting_text_not_in_the_clause_is_withheld():
    explanation = explain('It says "The Term is ten years."')
    assert explanation.text is None
    assert explanation.reason == MISQUOTE_REASON


def test_reply_quoting_part_of_the_clause_word_for_word_is_kept():
    reply = 'The agreement runs for "five years from the Effective Date".'
    assert explain(reply).text == reply


def test_reply_naming_a_number_absent_from_the_clause_is_withheld_with_it():
    explanation = explain("The agreement lasts 7 years.")
    assert explanation.text is None
    assert "7" in explanation.reason


def test_whitespace_only_reply_is_withheld_as_empty():
    explanation = explain(" \n ")
    assert explanation.text is None
    assert explanation.reason == EMPTY_REASON


def test_explanation_records_what_it_was_made_from():
    explanation = explain(RESTATEMENT, language="sr")
    assert explanation.clause_type == CLAUSE_TYPE
    assert explanation.quote == QUOTE
    assert explanation.language == "sr"
    assert explanation.model == MODEL
    assert explanation.latency_ms >= 0


def test_llm_error_propagates():
    client = FakeLLMClient([LLMError("down")])
    with pytest.raises(LLMError, match="down"):
        request_explanation(CLAUSE_TYPE, QUOTE, "en", client)


def test_withheld_explanation_warns_naming_the_model(caplog):
    with caplog.at_level(logging.WARNING):
        explain('It says "invented".')
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert MODEL in record.getMessage()


def test_echo_client_gives_a_faithful_explanation_for_a_quote_with_digits():
    explanation = request_explanation(CLAUSE_TYPE, NUMERIC_QUOTE, "en", EchoTextClient())
    assert explanation.text is not None
    assert explanation.reason is None


def test_phrase_numbers_supported_by_any_quote_are_not_listed():
    assert phrase_unsupported_numbers("5 and 7 and 9", ["of 5", "of 7"]) == ["9"]


def test_phrase_numbers_are_listed_in_order():
    assert phrase_unsupported_numbers("3 then 1.5 then 2", ["2"]) == ["3", "1.5"]


def test_phrase_text_without_quotes_has_every_number_unsupported():
    assert phrase_unsupported_numbers("4 and 4", []) == ["4", "4"]
