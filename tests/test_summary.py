import logging

import pytest

from contract_assistant.answer import STATUS_WORDS
from contract_assistant.config import Settings
from contract_assistant.phrase import unsupported_numbers
from contract_assistant.router.clients import LLMError
from contract_assistant.summary import (
    EMPTY_REASON,
    MAX_QUOTE_CHARS,
    MAX_TOKENS,
    MISQUOTE_REASON,
    NUMBERS_REASON,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Summary,
    SummaryStore,
    complete_sentences,
    quotes_of,
    render_markdown,
    request_summary,
    summary_items,
)
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

LAW = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
EXCLUSIVITY = "Supplier shall be the exclusive supplier."
INSURANCE = "Each party shall carry liability insurance."
TEXT = f"{LAW} {TERM} {EXCLUSIVITY} {INSURANCE}"
TYPES = ["Governing Law", "Term", "Exclusivity", "Insurance"]
TAU_LOW = 0.2
TAU_HIGH = 0.8
DOC_ID = "sha256:abc123"
MODEL = "gen-test"
WRAPPED_LAW = "This Agreement is governed by\n    the laws of Delaware."
FAITHFUL = f'The agreement is governed by Delaware law ("{LAW}") (Governing Law).'


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def mixed_cache(settings):
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
    return cache


@pytest.fixture
def items(mixed_cache):
    return summary_items(mixed_cache, TYPES, TAU_LOW, TAU_HIGH)


def summarize(items, reply: str = FAITHFUL) -> Summary:
    return request_summary(DOC_ID, items, FakeLLMClient([reply], model=MODEL), TAU_LOW, TAU_HIGH)


def make_summary(items, **changes) -> Summary:
    fields = {
        "doc_id": DOC_ID,
        "model": MODEL,
        "tau_low": TAU_LOW,
        "tau_high": TAU_HIGH,
        "items": items,
        "text": FAITHFUL,
        "latency_ms": 1500.0,
    }
    return Summary(**{**fields, **changes})


def test_summary_items_keep_found_and_check_items_with_a_verified_quote(items):
    assert [item.clause_type for item in items] == ["Governing Law", "Term"]


def test_summary_items_drop_not_found_items(items):
    assert "Exclusivity" not in [item.clause_type for item in items]


def test_summary_items_drop_withheld_items(items):
    assert "Insurance" not in [item.clause_type for item in items]


def test_summary_items_follow_the_order_of_the_clause_types(mixed_cache):
    items = summary_items(mixed_cache, ["Term", "Governing Law"], TAU_LOW, TAU_HIGH)
    assert [item.clause_type for item in items] == ["Term", "Governing Law"]


def test_summary_items_ignore_clause_types_absent_from_the_cache(mixed_cache):
    items = summary_items(mixed_cache, ["Cap On Liability", "Term"], TAU_LOW, TAU_HIGH)
    assert [item.clause_type for item in items] == ["Term"]


def test_unvalidated_clause_type_is_capped_at_check(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.99)]})
    [item] = summary_items(cache, ["Governing Law"], TAU_LOW, TAU_HIGH, {"Governing Law"})
    assert item.status == "check"


def test_confident_clause_type_is_found(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.99)]})
    [item] = summary_items(cache, ["Governing Law"], TAU_LOW, TAU_HIGH)
    assert item.status == "found"


def test_summary_items_reject_a_low_threshold_above_the_high_one(mixed_cache):
    with pytest.raises(ValueError):
        summary_items(mixed_cache, TYPES, TAU_HIGH, TAU_LOW)


def test_quotes_of_returns_the_quotes_in_item_order(items):
    assert quotes_of(items) == [LAW, TERM]


def test_summary_of_no_items_raises_and_makes_no_call():
    client = FakeLLMClient([FAITHFUL])
    with pytest.raises(ValueError):
        request_summary(DOC_ID, [], client, TAU_LOW, TAU_HIGH)
    assert client.text_calls == []


def test_summary_makes_one_call_with_the_system_prompt_and_token_limit(items):
    client = FakeLLMClient([FAITHFUL])
    request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    [(system, _, max_tokens)] = client.text_calls
    assert system == SYSTEM_PROMPT
    assert max_tokens == MAX_TOKENS


def test_summary_request_lists_every_item_in_order_with_its_status_word(items):
    client = FakeLLMClient([FAITHFUL])
    request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    user = client.text_calls[0][1]
    found, check = STATUS_WORDS["found"], STATUS_WORDS["check"]
    assert user == (
        f'Items:\n- Governing Law ({found}): "{LAW}"\n- Term ({check}): "{TERM}"'
    )


def test_summary_request_collapses_whitespace_inside_a_quote(settings):
    text = f"{WRAPPED_LAW} {TERM}"
    cache = make_cache(text, settings, {"Governing Law": [(WRAPPED_LAW, 0.9)]})
    items = summary_items(cache, ["Governing Law"], TAU_LOW, TAU_HIGH)
    client = FakeLLMClient([FAITHFUL])
    request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    assert f'"{LAW}"' in client.text_calls[0][1]


def test_long_quote_is_cut_at_a_word_boundary_outside_the_quotes(settings):
    long_quote = " ".join(["word"] * MAX_QUOTE_CHARS)
    cache = make_cache(long_quote, settings, {"Term": [(long_quote, 0.9)]})
    items = summary_items(cache, ["Term"], TAU_LOW, TAU_HIGH)
    client = FakeLLMClient(["Nothing quoted."])
    summary = request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    user = client.text_calls[0][1]
    sent = user.split('"')[1]
    assert user.endswith('" (cut)')
    assert len(sent) <= MAX_QUOTE_CHARS
    assert long_quote.startswith(sent)
    assert sent.endswith("word")
    assert summary.items[0].citation.quote == long_quote


def test_faithful_reply_is_kept_stripped(items):
    summary = summarize(items, f"  {FAITHFUL}\n")
    assert summary.text == FAITHFUL
    assert summary.reason is None
    assert summary.unfaithful_spans == []


def test_reply_without_quotes_is_faithful(items):
    assert summarize(items, "A Delaware contract.").text == "A Delaware contract."


def test_reply_quoting_part_of_a_quote_is_faithful(items):
    reply = 'It is governed by "the laws of Delaware".'
    assert summarize(items, reply).text == reply


def test_reply_quoting_altered_text_is_withheld_with_the_spans(items):
    altered = "This Agreement is governed by the laws of Texas."
    summary = summarize(items, f'It says "{altered}" and "{TERM}".')
    assert summary.text is None
    assert summary.reason == MISQUOTE_REASON
    assert summary.unfaithful_spans == [altered]


def test_reply_quoting_a_not_found_clause_is_unfaithful(items):
    summary = summarize(items, f'It also says "{EXCLUSIVITY}".')
    assert summary.text is None
    assert summary.unfaithful_spans == [EXCLUSIVITY]


def test_whitespace_only_reply_is_withheld_as_empty(items):
    summary = summarize(items, "  \n ")
    assert summary.text is None
    assert summary.reason == EMPTY_REASON


def test_reply_naming_a_number_no_quote_contains_is_withheld_with_it(settings):
    cache = make_cache("The term is ten (10) years.", settings, {"Term": [("ten (10) years", 0.9)]})
    items = summary_items(cache, ["Term"], TAU_LOW, TAU_HIGH)
    summary = summarize(items, "The term is 12 years.")
    assert summary.text is None
    assert summary.reason == NUMBERS_REASON.format(numbers="12")


def test_reply_repeating_a_number_of_a_quote_is_kept(settings):
    cache = make_cache("The term is ten (10) years.", settings, {"Term": [("ten (10) years", 0.9)]})
    items = summary_items(cache, ["Term"], TAU_LOW, TAU_HIGH)
    assert summarize(items, "It lasts ten (10) years.").text == "It lasts ten (10) years."


def test_misquote_reason_wins_over_the_numbers_reason(items):
    summary = summarize(items, 'It says "The Term is 99 years." for 99 years.')
    assert summary.reason == MISQUOTE_REASON


def test_summary_text_ends_at_the_last_complete_sentence(items):
    assert summarize(items, "The law is English. The term").text == "The law is English."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two", "One."),
        ("One. Two!", "One. Two!"),
        ('One. "Two." Three', 'One. "Two."'),
        ("One (see Two.) Three", "One (see Two.)"),
        ("No sentence end", "No sentence end"),
        ("", ""),
        ("It ends with a quote\u201d", "It ends with a quote\u201d"),
    ],
    ids=["cut", "complete", "closing_quote", "closing_paren", "no_end", "empty", "closer"],
)
def test_complete_sentences_cuts_an_unfinished_last_sentence(text, expected):
    assert complete_sentences(text) == expected


def test_summary_records_its_keys(items):
    summary = summarize(items)
    assert summary.model == MODEL
    assert summary.doc_id == DOC_ID
    assert (summary.tau_low, summary.tau_high) == (TAU_LOW, TAU_HIGH)
    assert summary.prompt_version == PROMPT_VERSION
    assert summary.items == items


def test_summary_latency_is_not_negative(items):
    assert summarize(items).latency_ms >= 0


def test_llm_error_propagates(items):
    client = FakeLLMClient([LLMError("down")])
    with pytest.raises(LLMError, match="down"):
        request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH)


def test_withheld_summary_warns_naming_the_model(items, caplog):
    with caplog.at_level(logging.WARNING):
        summarize(items, 'It says "invented".')
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert MODEL in record.getMessage()


def test_markdown_shows_the_text_the_sources_and_the_model(items):
    markdown = render_markdown(make_summary(items))
    found, check = STATUS_WORDS["found"], STATUS_WORDS["check"]
    assert markdown.startswith(FAITHFUL)
    assert f"Based on: Governing Law ({found}), Term ({check})" in markdown
    assert f"model: {MODEL}" in markdown
    assert "1.5 s" in markdown


def test_markdown_of_a_withheld_summary_lists_the_unfaithful_spans(items):
    summary = make_summary(
        items, text=None, reason=MISQUOTE_REASON, unfaithful_spans=["one", "two"]
    )
    assert render_markdown(summary) == (
        f'Summary withheld: {MISQUOTE_REASON}.\n- "one"\n- "two"'
    )


@pytest.fixture
def store(tmp_path) -> SummaryStore:
    return SummaryStore(tmp_path / "summaries")


def test_load_of_an_unknown_key_is_none(store):
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None


def test_saved_summary_loads_equal(store, items):
    summary = make_summary(items)
    store.save(summary)
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) == summary


def test_saved_file_is_named_after_the_doc_id_without_the_prefix(store, items):
    path = store.save(make_summary(items))
    assert path.name.startswith("abc123-")
    assert path.suffix == ".json"


def test_save_creates_the_directory(store, items):
    path = store.save(make_summary(items))
    assert path.parent == store.directory
    assert path.is_file()


@pytest.mark.parametrize(
    ("model", "tau_low", "tau_high"),
    [("other", TAU_LOW, TAU_HIGH), (MODEL, 0.3, TAU_HIGH), (MODEL, TAU_LOW, 0.9)],
    ids=["model", "tau_low", "tau_high"],
)
def test_other_keys_are_a_miss_in_another_file(store, items, model, tau_low, tau_high):
    path = store.save(make_summary(items))
    assert store.load(DOC_ID, model, tau_low, tau_high) is None
    other = store.save(make_summary(items, model=model, tau_low=tau_low, tau_high=tau_high))
    assert other != path


def test_invalid_json_file_gives_none_and_a_warning(store, items, caplog):
    path = store.save(make_summary(items))
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None
    assert str(path) in caplog.text


def test_summary_of_another_prompt_version_gives_none(store, items):
    store.save(make_summary(items, prompt_version=PROMPT_VERSION + 1))
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None


def test_summary_saved_under_the_path_of_other_keys_gives_none(store, items):
    path = store.save(make_summary(items))
    other = make_summary(items, model="other")
    path.write_text(other.model_dump_json(), encoding="utf-8")
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None


def test_request_summary_asks_again_when_the_first_reply_is_withheld(items):
    client = FakeLLMClient(['The term is "ten (10) weeks".', FAITHFUL])
    summary = request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts=2)
    assert summary.text == FAITHFUL
    assert (summary.reason, summary.attempts, len(client.text_calls)) == (None, 2, 2)


def test_request_summary_stops_after_the_given_attempts(items):
    misquote = 'The term is "ten (10) weeks".'
    client = FakeLLMClient([misquote, misquote, FAITHFUL])
    summary = request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts=2)
    assert summary.text is None
    assert (summary.attempts, len(client.text_calls)) == (2, 2)


def test_request_summary_asks_once_by_default_and_records_one_attempt(items):
    client = FakeLLMClient([FAITHFUL])
    assert request_summary(DOC_ID, items, client, TAU_LOW, TAU_HIGH).attempts == 1
    assert len(client.text_calls) == 1


def test_request_summary_rejects_zero_attempts(items):
    with pytest.raises(ValueError, match="attempts"):
        request_summary(DOC_ID, items, FakeLLMClient([]), TAU_LOW, TAU_HIGH, attempts=0)


def test_number_in_words_with_other_digits_is_unsupported():
    quotes = ["A six (6) months notice", "a term of ten (10) years"]
    assert unsupported_numbers("Notice is six (10) years; the term is ten (10) years.", quotes) == [
        "six (10)"
    ]


def test_number_in_words_is_matched_without_regard_to_case():
    assert unsupported_numbers("Notice is six (6) months.", ["Six (6) months notice"]) == []


def test_summary_mixing_a_word_and_digit_pair_is_withheld(settings):
    text = "The term is ten (10) years with six (6) months notice."
    cache = make_cache(text, settings, {"Term": [(text, 0.9)]})
    items = summary_items(cache, ["Term"], TAU_LOW, TAU_HIGH)
    summary = summarize(items, "Termination needs a six (10) year notice (Term).")
    assert summary.text is None
    assert summary.reason == NUMBERS_REASON.format(numbers="six (10)")
