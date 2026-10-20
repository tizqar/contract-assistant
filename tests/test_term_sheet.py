import json

import pytest

from contract_assistant.config import Settings
from contract_assistant.router.clients import LLMError
from contract_assistant.store import read_cache
from contract_assistant.term_sheet import (
    FIELDS,
    INVALID_REASON,
    LABELS,
    MAX_TOKENS,
    NOT_STATED,
    NOT_VERIFIED,
    REJECTED_REASON,
    SYSTEM_PROMPT,
    TermSheet,
    TermSheetReply,
    TermSheetStore,
    TermValue,
    cell,
    parse_reply,
    render_markdown,
    request_term_sheet,
    term_sheet_items,
    verify,
)
from tests.conftest import ROOT
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

TAU_LOW = 0.2
TAU_HIGH = 0.8
# The parties of the short demo contract score 0.18, below the default low threshold.
DEMO_TAU_LOW = 0.1
DOC_ID = "sha256:abc123"
PARTIES = "Acme Corp. and Beta LLC"
AGREEMENT = "made on the 15th  day\nof January 2018"
INITIAL = "The initial term is ten (10) years."
RENEWAL = "It renews for ten (10) years."
LAW = "This Agreement is governed by the Laws of England."
TEXT = f"Contract between {PARTIES}, {AGREEMENT}. {INITIAL} {RENEWAL} {LAW}"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"


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
            "Agreement Date": [(AGREEMENT, 0.9)],
            "Expiration Date": [(INITIAL, 0.9)],
            "Renewal Term": [(RENEWAL, 0.9)],
            "Governing Law": [(LAW, 0.9)],
        },
    )


@pytest.fixture
def items(cache):
    return term_sheet_items(cache, TAU_LOW, TAU_HIGH)


def reply_json(**fields) -> str:
    return json.dumps(fields)


def request(items, replies, attempts=1):
    client = FakeLLMClient(replies)
    sheet = request_term_sheet(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts)
    return sheet, client


def make_sheet(terms=None, rejected=None, reason=None) -> TermSheet:
    return TermSheet(
        doc_id=DOC_ID,
        model="fake-llm",
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        latency_ms=1.0,
        items=[],
        terms={field: [] for field in FIELDS} | (terms or {}),
        rejected=rejected or {},
        reason=reason,
    )


def term_value(value, clause_type="Governing Law") -> TermValue:
    return TermValue(value=value, clause_type=clause_type, page=None, start=0, end=len(value))


def test_term_sheet_items_keep_only_the_field_clause_types_in_fields_order(settings):
    text = "Alpha. Beta. Gamma. Delta."
    cache = make_cache(
        text,
        settings,
        {
            "Insurance": [("Alpha.", 0.9)],
            "Governing Law": [("Beta.", 0.9)],
            "Parties": [("Gamma.", 0.9)],
            "Renewal Term": [("Delta.", 0.9)],
        },
    )
    items = term_sheet_items(cache, TAU_LOW, TAU_HIGH)
    assert [item.clause_type for item in items] == ["Parties", "Renewal Term", "Governing Law"]


def test_term_sheet_items_have_no_duplicate_clause_types(cache):
    names = [item.clause_type for item in term_sheet_items(cache, TAU_LOW, TAU_HIGH)]
    assert len(names) == len(set(names))


def test_term_sheet_items_leave_out_not_found_items(settings):
    cache = make_cache(
        "Alpha. Beta.",
        settings,
        {"Parties": [("Alpha.", 0.9)], "Governing Law": [("Beta.", 0.01)]},
    )
    items = term_sheet_items(cache, TAU_LOW, TAU_HIGH)
    assert [item.clause_type for item in items] == ["Parties"]


def test_term_sheet_items_cap_unvalidated_types_at_verify(cache):
    items = term_sheet_items(cache, TAU_LOW, TAU_HIGH, unvalidated=["Parties"])
    statuses = {item.clause_type: item.status for item in items}
    assert statuses["Parties"] == "check"
    assert statuses["Governing Law"] == "found"


def test_reply_turns_one_party_string_into_a_list():
    assert TermSheetReply(parties="Acme").parties == ["Acme"]


def test_reply_ignores_extra_keys():
    reply = TermSheetReply.model_validate({"governing_law": "England", "other": 1})
    assert reply.governing_law == "England"


def test_reply_with_a_number_for_a_date_fails_validation():
    with pytest.raises(ValueError):
        TermSheetReply.model_validate({"agreement_date": 2018})


def test_parse_reply_reads_a_plain_object():
    assert parse_reply('{"governing_law": "England"}').governing_law == "England"


def test_parse_reply_reads_an_object_in_json_fences():
    reply = parse_reply('```json\n{"governing_law": "England"}\n```')
    assert reply.governing_law == "England"


def test_parse_reply_reads_an_object_with_prose_around_it():
    reply = parse_reply('Here it is: {"governing_law": "England"} Hope it helps.')
    assert reply.governing_law == "England"


@pytest.mark.parametrize(
    "text", ["no object here", '{"governing_law": ', '{"agreement_date": 2018}']
)
def test_parse_reply_gives_none_for_no_object_broken_json_or_a_wrong_type(text):
    assert parse_reply(text) is None


def test_verify_locates_a_value_ignoring_case_and_whitespace(items, cache):
    terms, rejected = verify(
        TermSheetReply(agreement_date="MADE ON the 15th day of january 2018"), items
    )
    (value,) = terms["agreement_date"]
    assert value.value == AGREEMENT
    assert rejected == {}
    assert cache.text[value.start : value.end] == AGREEMENT
    assert value.start == TEXT.index(AGREEMENT)


def test_verify_offsets_are_the_citation_start_plus_the_offset_in_the_quote(items, cache):
    terms, _ = verify(TermSheetReply(governing_law="laws of england"), items)
    (value,) = terms["governing_law"]
    assert value.value == "Laws of England"
    assert cache.text[value.start : value.end] == "Laws of England"
    assert value.start == TEXT.index("Laws of England")


def test_verify_attributes_a_value_to_the_own_source_clause_type_first(items):
    terms, _ = verify(
        TermSheetReply(expiration_date="ten (10) years", renewal_term="ten (10) years"), items
    )
    assert terms["expiration_date"][0].clause_type == "Expiration Date"
    assert terms["renewal_term"][0].clause_type == "Renewal Term"
    assert terms["renewal_term"][0].start == TEXT.index(RENEWAL) + RENEWAL.index("ten")


def test_verify_accepts_a_value_found_only_in_another_items_quote(items):
    terms, rejected = verify(TermSheetReply(effective_date="January 2018"), items)
    assert terms["effective_date"][0].clause_type == "Agreement Date"
    assert rejected == {}


def test_verify_rejects_a_value_in_no_quote_under_its_field(items):
    terms, rejected = verify(TermSheetReply(governing_law="France"), items)
    assert terms["governing_law"] == []
    assert rejected == {"governing_law": ["France"]}


def test_verify_keeps_two_values_with_the_same_offsets_once(items):
    terms, _ = verify(TermSheetReply(parties=["Acme Corp.", "acme corp."]), items)
    assert len(terms["parties"]) == 1


def test_verify_skips_null_and_blank_values(items):
    terms, rejected = verify(
        TermSheetReply(parties=["", "  "], governing_law=None, term=" "), items
    )
    assert terms["parties"] == [] and terms["term"] == []
    assert rejected == {}


def test_verify_gives_every_field_a_list(items):
    terms, _ = verify(TermSheetReply(), items)
    assert list(terms) == list(FIELDS)
    assert all(values == [] for values in terms.values())


def test_request_without_items_raises_and_does_not_call():
    client = FakeLLMClient([])
    with pytest.raises(ValueError):
        request_term_sheet(DOC_ID, [], client, TAU_LOW, TAU_HIGH)
    assert client.text_calls == []


def test_request_lists_each_item_with_collapsed_whitespace(items):
    _, client = request(items, [reply_json()])
    user = client.text_calls[0][1]
    assert f'- Agreement Date: "{" ".join(AGREEMENT.split())}"' in user
    assert f'- Parties: "{PARTIES}"' in user
    assert f'- Governing Law: "{LAW}"' in user
    assert "\n" not in user.split("- Agreement Date:")[1].split("\n")[0]


def test_request_passes_the_system_prompt_and_max_tokens(items):
    _, client = request(items, [reply_json()])
    system, _, max_tokens = client.text_calls[0]
    assert (system, max_tokens) == (SYSTEM_PROMPT, MAX_TOKENS)


def test_request_asks_again_with_the_invalid_reason_for_an_invalid_reply(items):
    sheet, client = request(items, ["not json", reply_json(governing_law="Laws of England")], 2)
    assert INVALID_REASON in client.text_calls[1][1]
    assert sheet.attempts == 2
    assert sheet.terms["governing_law"][0].value == "Laws of England"


def test_request_gives_empty_terms_when_every_reply_is_invalid(items):
    sheet, _ = request(items, ["bad", "worse"], 2)
    assert sheet.terms == {field: [] for field in FIELDS}
    assert sheet.rejected == {}
    assert sheet.reason == INVALID_REASON
    assert sheet.attempts == 2


def test_request_asks_again_naming_a_rejected_value(items):
    good = reply_json(governing_law="Laws of England")
    _, client = request(items, [reply_json(governing_law="France"), good], 2)
    assert REJECTED_REASON.format(values="'France'") in client.text_calls[1][1]


def test_request_keeps_verified_values_and_lists_rejected_ones_without_a_reason(items):
    sheet, _ = request(items, [reply_json(governing_law="Laws of England", parties=["Zeta"])])
    assert sheet.terms["governing_law"][0].value == "Laws of England"
    assert sheet.rejected == {"parties": ["Zeta"]}
    assert sheet.reason is None


def test_request_parses_a_reply_that_does_not_end_in_a_sentence_end(items):
    sheet, _ = request(items, [reply_json(parties=["Acme Corp."])])
    assert sheet.terms["parties"][0].value == "Acme Corp."


def test_request_records_the_run(items):
    sheet, _ = request(items, [reply_json()])
    assert (sheet.attempts, sheet.model, sheet.doc_id) == (1, "fake-llm", DOC_ID)
    assert (sheet.tau_low, sheet.tau_high) == (TAU_LOW, TAU_HIGH)
    assert sheet.latency_ms >= 0
    assert sheet.items == items


def test_request_propagates_llm_errors(items):
    with pytest.raises(LLMError):
        request(items, [LLMError("down")])


def test_request_with_the_echo_client_on_the_short_demo_cache_verifies_every_value():
    cache = read_cache(DEMO_CACHE, Settings())
    items = term_sheet_items(cache, DEMO_TAU_LOW, TAU_HIGH)
    sheet = request_term_sheet(cache.doc_id, items, EchoTextClient(), DEMO_TAU_LOW, TAU_HIGH)
    assert sheet.terms["parties"]
    assert sheet.terms["agreement_date"]
    assert sheet.terms["governing_law"]
    assert sheet.rejected == {}
    assert sheet.reason is None


def test_cell_joins_values_with_collapsed_whitespace():
    sheet = make_sheet(
        {"parties": [term_value("Acme  Corp.", "Parties"), term_value("Beta\nLLC", "Parties")]}
    )
    assert cell(sheet, "parties") == "Acme Corp.; Beta LLC"


def test_cell_names_the_clause_type_when_it_is_not_a_source_of_the_field():
    sheet = make_sheet({"effective_date": [term_value("January 2018", "Agreement Date")]})
    assert cell(sheet, "effective_date") == "January 2018 (from Agreement Date)"


def test_cell_shows_rejected_values_in_quotes_as_not_verified():
    sheet = make_sheet(rejected={"governing_law": ["France"]})
    assert cell(sheet, "governing_law") == f"{NOT_VERIFIED} ('France')"


def test_cell_of_a_field_without_values_is_not_stated():
    assert cell(make_sheet(), "governing_law") == NOT_STATED


def test_render_markdown_has_a_header_a_rule_and_a_row_per_field():
    sheets = [("one", make_sheet()), ("two", make_sheet())]
    lines = render_markdown(sheets).splitlines()
    assert lines[0] == "| Term | one | two |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2:] == [
        f"| {LABELS[field]} | {NOT_STATED} | {NOT_STATED} |" for field in FIELDS
    ]


def test_render_markdown_adds_a_note_for_each_withheld_sheet():
    sheets = [("one", make_sheet()), ("two", make_sheet(reason=INVALID_REASON))]
    lines = render_markdown(sheets).splitlines()
    assert lines[-1] == f"two: term sheet withheld: {INVALID_REASON}"
    assert sum("withheld" in line for line in lines) == 1


def test_term_sheet_store_round_trips_a_sheet(tmp_path):
    store = TermSheetStore(tmp_path / "sheets")
    sheet = make_sheet({"governing_law": [term_value("Laws of England")]})
    store.save(sheet)
    assert store.load(DOC_ID, "fake-llm", TAU_LOW, TAU_HIGH) == sheet
