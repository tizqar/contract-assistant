import json

import pytest

from contract_assistant.config import Settings
from contract_assistant.obligations import (
    INVALID_REASON,
    MAX_ITEMS,
    MAX_TOKENS,
    REJECTED_REASON,
    SYSTEM_PROMPT,
    Obligation,
    ObligationReply,
    ObligationSheet,
    ObligationStore,
    by_party,
    obligation_items,
    obligations_request,
    parse_reply,
    render_markdown,
    request_obligations,
    verify,
)
from contract_assistant.summary import summary_items
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

TAU_LOW = 0.2
TAU_HIGH = 0.8
PAY = "The Customer shall pay the fees within thirty days."
LICENSE = "The Supplier grants the Customer\n   a non-exclusive licence."
NONCOMPETE = "The Supplier may not hire the staff of the Customer."
AUDIT = "Either party may audit the books."
PREFIX = "Preamble text. "
TEXT = f"{PREFIX}{PAY} {LICENSE} {NONCOMPETE} {AUDIT}"
CLAUSES = {
    "Payment Terms": [(PAY, 0.9)],
    "License Grant": [(LICENSE, 0.9)],
    "Non-Compete": [(NONCOMPETE, 0.9)],
    "Audit Rights": [(AUDIT, 0.9)],
}
TYPES = list(CLAUSES)
DOC_ID = "sha256:abc"
MODEL = "fake-llm"


def entry(clause_type, party, modality, action) -> dict[str, str]:
    return {"clause_type": clause_type, "party": party, "modality": modality, "action": action}


GOOD = entry("Payment Terms", "the Customer", "shall", "pay the fees")
GOOD_REPLY = json.dumps([GOOD])
BAD = entry("Payment Terms", "the Customer", "must", "deliver the goods")


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def cache(settings):
    return make_cache(TEXT, settings, CLAUSES)


@pytest.fixture
def items(cache):
    return summary_items(cache, TYPES, TAU_LOW, TAU_HIGH)


def replies(*entries) -> list[ObligationReply]:
    return [ObligationReply.model_validate(e) for e in entries]


def verified(*entries, items):
    return verify(replies(*entries), items)


def test_obligation_items_put_found_items_before_verify_items(settings):
    cache = make_cache(
        TEXT, settings, {"Payment Terms": [(PAY, 0.5)], "License Grant": [(LICENSE, 0.9)]}
    )
    result = obligation_items(cache, ["Payment Terms", "License Grant"], TAU_LOW, TAU_HIGH)
    assert [(i.clause_type, i.status) for i in result] == [
        ("License Grant", "found"),
        ("Payment Terms", "check"),
    ]


def test_obligation_items_are_capped_at_the_maximum(settings):
    sentences = {f"Type {n}": f"Party {n} shall act number {n}." for n in range(MAX_ITEMS + 4)}
    text = " ".join(sentences.values())
    clauses = {name: [(sentence, 0.9)] for name, sentence in sentences.items()}
    cache = make_cache(text, settings, clauses)
    result = obligation_items(cache, list(clauses), TAU_LOW, TAU_HIGH)
    assert len(result) == MAX_ITEMS


def test_obligation_items_keep_found_items_when_the_cap_cuts_verify_items(settings):
    sentences = {f"Type {n}": f"Party {n} shall act number {n}." for n in range(MAX_ITEMS + 2)}
    text = " ".join(sentences.values())
    clauses = {name: [(s, 0.5)] for name, s in sentences.items()}
    last = f"Type {MAX_ITEMS + 1}"
    clauses[last] = [(sentences[last], 0.9)]
    cache = make_cache(text, settings, clauses)
    result = obligation_items(cache, list(clauses), TAU_LOW, TAU_HIGH)
    assert result[0].clause_type == last
    assert all(i.status == "check" for i in result[1:])


def test_obligation_items_drop_withheld_and_not_found_items(settings):
    cache = make_cache(
        TEXT, settings, {"Payment Terms": [(PAY, 0.9)], "Audit Rights": [(AUDIT, 0.01)]}
    )
    result = obligation_items(
        cache, ["Payment Terms", "Audit Rights", "Missing"], TAU_LOW, TAU_HIGH
    )
    assert [i.clause_type for i in result] == ["Payment Terms"]
    assert all(i.citation is not None and not i.withheld for i in result)


def test_parse_reply_reads_a_json_array():
    assert parse_reply(GOOD_REPLY) == replies(GOOD)


def test_parse_reply_reads_an_array_in_json_fences():
    assert parse_reply(f"```json\n{GOOD_REPLY}\n```") == replies(GOOD)


def test_parse_reply_reads_an_array_with_prose_around_it():
    assert parse_reply(f"Here you go: {GOOD_REPLY} Hope this helps.") == replies(GOOD)


def test_parse_reply_gives_an_empty_list_for_an_empty_array():
    assert parse_reply("[]") == []


def test_parse_reply_gives_none_when_the_reply_is_not_an_array():
    assert parse_reply(json.dumps(GOOD)) is None


def test_parse_reply_gives_none_for_broken_json():
    assert parse_reply('[{"clause_type": "Payment Terms", "party": ') is None


def test_parse_reply_gives_none_when_an_entry_misses_a_key():
    partial = {k: v for k, v in GOOD.items() if k != "action"}
    assert parse_reply(json.dumps([GOOD, partial])) is None


def test_parse_reply_ignores_extra_keys():
    assert parse_reply(json.dumps([{**GOOD, "confidence": 0.9}])) == replies(GOOD)


def test_verify_keeps_the_exact_slices_of_the_quote_with_document_offsets(items, cache):
    kept, rejected = verified(GOOD, items=items)
    assert rejected == []
    obligation = kept[0]
    assert (obligation.party, obligation.modality, obligation.action) == (
        "The Customer",
        "must",
        "pay the fees",
    )
    assert cache.text[obligation.start : obligation.end] == "pay the fees"
    assert obligation.start == TEXT.index("pay the fees")
    assert obligation.clause_type == "Payment Terms"
    assert obligation.page is None


def test_verify_offsets_are_the_citation_start_plus_the_offset_in_the_quote(items):
    citation = next(i for i in items if i.clause_type == "Payment Terms").citation
    kept, _ = verified(GOOD, items=items)
    assert kept[0].start == citation.start + citation.quote.index("pay the fees")


def test_verify_matches_case_and_whitespace_insensitively_and_keeps_the_quote_text(items):
    reply = entry("License Grant", "THE  customer", "grants", "a   NON-EXCLUSIVE\nlicence")
    kept, rejected = verified(reply, items=items)
    assert rejected == []
    assert kept[0].party == "the Customer"
    assert kept[0].action == "a non-exclusive licence"


def test_verify_keeps_the_original_whitespace_of_the_slice(items, cache):
    reply = entry(
        "License Grant", "the Supplier", "grants", "the Customer a non-exclusive licence"
    )
    kept, _ = verified(reply, items=items)
    assert kept[0].action == "the Customer\n   a non-exclusive licence"
    assert cache.text[kept[0].start : kept[0].end] == kept[0].action


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("shall", "must"),
        ("agrees to", "must"),
        ("grants", "must"),
        ("must", "must"),
        ("may not", "must not"),
        ("shall not", "must not"),
        ("cannot", "must not"),
        ("may", "may"),
        ("is entitled to", "may"),
        ("MAY", "may"),
    ],
)
def test_verify_normalises_the_modality_word(items, word, expected):
    kept, rejected = verified(
        entry("Payment Terms", "the Customer", word, "pay the fees"), items=items
    )
    assert rejected == []
    assert kept[0].modality == expected


def test_verify_rejects_an_unknown_modality_word(items):
    kept, rejected = verified(
        entry("Payment Terms", "the Customer", "perhaps", "pay the fees"), items=items
    )
    assert kept == []
    assert rejected[0]["modality"] == "perhaps"


def test_verify_rejects_a_party_that_is_not_in_the_quote(items):
    kept, rejected = verified(
        entry("Payment Terms", "the Supplier", "shall", "pay the fees"), items=items
    )
    assert (kept, len(rejected)) == ([], 1)


def test_verify_rejects_an_action_that_is_not_in_the_quote(items):
    kept, rejected = verified(BAD, items=items)
    assert (kept, len(rejected)) == ([], 1)


def test_verify_rejects_an_unknown_clause_type(items):
    kept, rejected = verified(
        entry("Warranty", "the Customer", "shall", "pay the fees"), items=items
    )
    assert (kept, len(rejected)) == ([], 1)


def test_verify_rejects_an_action_taken_from_another_items_quote(items):
    mixed = entry("Payment Terms", "the Customer", "may", "audit the books")
    kept, rejected = verified(mixed, items=items)
    assert (kept, len(rejected)) == ([], 1)


def test_verify_keeps_two_entries_with_the_same_party_modality_and_slice_once(items):
    twin = {**GOOD, "party": "THE CUSTOMER", "modality": "must"}
    kept, rejected = verified(GOOD, twin, items=items)
    assert len(kept) == 1
    assert rejected == []


def test_verify_keeps_entries_with_another_modality_for_the_same_slice(items):
    other = {**GOOD, "modality": "may"}
    kept, _ = verified(GOOD, other, items=items)
    assert [o.modality for o in kept] == ["must", "may"]


def test_verify_returns_rejected_entries_as_dicts_and_keeps_the_good_ones(items):
    kept, rejected = verified(GOOD, BAD, items=items)
    assert len(kept) == 1
    assert rejected == [BAD]


def test_verify_rejects_an_entry_with_a_blank_action(items):
    kept, rejected = verified({**GOOD, "action": "  "}, items=items)
    assert (kept, len(rejected)) == ([], 1)


def test_obligations_request_needs_items():
    with pytest.raises(ValueError, match="no items"):
        obligations_request(DOC_ID, [])


def test_obligations_request_lists_each_item_with_its_collapsed_quote(items):
    request = obligations_request(DOC_ID, items)
    assert request.user.startswith("Items:\n")
    assert (
        '- License Grant: "The Supplier grants the Customer a non-exclusive licence."'
        in request.user
    )
    assert f'- Payment Terms: "{PAY}"' in request.user


def test_obligations_request_is_not_trimmed_and_has_the_prompt_and_token_limit(items):
    request = obligations_request(DOC_ID, items)
    assert request.trim is False
    assert request.system == SYSTEM_PROMPT
    assert request.max_tokens == MAX_TOKENS
    assert DOC_ID in request.label


def test_obligations_check_gives_the_invalid_reason_for_a_reply_that_is_not_an_array(items):
    check = obligations_request(DOC_ID, items).check
    assert check(json.dumps(GOOD)) == INVALID_REASON


def test_obligations_check_accepts_a_reply_whose_entries_all_verify(items):
    assert obligations_request(DOC_ID, items).check(GOOD_REPLY) is None


def test_obligations_check_accepts_a_reply_where_only_some_entries_verify(items):
    assert obligations_request(DOC_ID, items).check(json.dumps([GOOD, BAD])) is None


def test_obligations_check_names_the_entries_when_none_verify(items):
    reason = obligations_request(DOC_ID, items).check(json.dumps([BAD]))
    assert reason == REJECTED_REASON.format(entries="'the Customer: deliver the goods'")


def test_request_obligations_returns_the_verified_obligations_of_a_good_reply(items):
    client = FakeLLMClient([GOOD_REPLY])
    sheet = request_obligations(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    assert sheet.reason is None
    assert sheet.attempts == 1
    assert [o.action for o in sheet.obligations] == ["pay the fees"]
    assert sheet.rejected == []
    assert sheet.items == items


def test_request_obligations_records_the_model_thresholds_and_latency(items):
    client = FakeLLMClient([GOOD_REPLY], model="other-model")
    sheet = request_obligations(DOC_ID, items, client, TAU_LOW, TAU_HIGH)
    assert (sheet.doc_id, sheet.model) == (DOC_ID, "other-model")
    assert (sheet.tau_low, sheet.tau_high) == (TAU_LOW, TAU_HIGH)
    assert sheet.latency_ms >= 0


def test_request_obligations_gives_the_invalid_reason_after_two_invalid_replies(items):
    client = FakeLLMClient(["not json", "still not json"])
    sheet = request_obligations(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts=2)
    assert sheet.obligations == []
    assert sheet.reason == INVALID_REASON
    assert sheet.attempts == 2
    assert len(client.text_calls) == 2


def test_request_obligations_keeps_a_reply_with_one_good_and_one_bad_entry_at_once(items):
    client = FakeLLMClient([json.dumps([GOOD, BAD])])
    sheet = request_obligations(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts=2)
    assert sheet.attempts == 1
    assert len(client.text_calls) == 1
    assert [o.action for o in sheet.obligations] == ["pay the fees"]
    assert sheet.rejected == [BAD]
    assert sheet.reason is None


def test_request_obligations_asks_again_when_no_entry_verifies(items):
    client = FakeLLMClient([json.dumps([BAD]), GOOD_REPLY])
    sheet = request_obligations(DOC_ID, items, client, TAU_LOW, TAU_HIGH, attempts=2)
    assert sheet.attempts == 2
    assert len(sheet.obligations) == 1


def obligation(party, action="pay", modality="must", clause_type="Payment Terms") -> Obligation:
    return Obligation(
        party=party,
        modality=modality,
        action=action,
        clause_type=clause_type,
        page=None,
        start=0,
        end=3,
    )


def sheet_of(*obligations, rejected=(), reason=None) -> ObligationSheet:
    return ObligationSheet(
        doc_id=DOC_ID,
        model=MODEL,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        latency_ms=1.0,
        items=[],
        obligations=list(obligations),
        rejected=list(rejected),
        reason=reason,
    )


def test_by_party_groups_in_order_of_first_appearance():
    sheet = sheet_of(obligation("B", "one"), obligation("A", "two"), obligation("B", "three"))
    groups = by_party(sheet)
    assert list(groups) == ["B", "A"]
    assert [o.action for o in groups["B"]] == ["one", "three"]


def test_render_markdown_says_when_the_obligations_are_withheld():
    text = render_markdown(sheet_of(reason=INVALID_REASON))
    assert text == f"Obligations withheld: {INVALID_REASON}."


def test_render_markdown_says_when_no_obligation_was_read():
    assert render_markdown(sheet_of()) == "No obligation was read from the clauses found."


def test_render_markdown_groups_by_party_and_counts_the_dropped_entries():
    sheet = sheet_of(
        obligation("the Customer", "pay  the\nfees"),
        obligation("the Supplier", "hire", "must not", "Non-Compete"),
        obligation("the Customer", "audit", "may", "Audit Rights"),
        rejected=[BAD, BAD],
    )
    lines = render_markdown(sheet).splitlines()
    assert lines[0] == "**the Customer**"
    assert '- must: "pay the fees" (Payment Terms)' in lines
    assert '- may: "audit" (Audit Rights)' in lines
    assert lines.index("**the Supplier**") > lines.index('- may: "audit" (Audit Rights)')
    assert '- must not: "hire" (Non-Compete)' in lines
    assert lines[-1] == "2 entries of the model were not in the clauses and were dropped."


def test_obligation_store_saves_and_loads_a_sheet(tmp_path):
    store = ObligationStore(tmp_path / "o")
    sheet = sheet_of(obligation("the Customer"), rejected=[BAD])
    path = store.save(sheet)
    assert path.is_file()
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) == sheet


def test_obligation_store_misses_for_another_model(tmp_path):
    store = ObligationStore(tmp_path / "o")
    store.save(sheet_of(obligation("the Customer")))
    assert store.load(DOC_ID, "another", TAU_LOW, TAU_HIGH) is None
