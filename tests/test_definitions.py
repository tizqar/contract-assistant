from pathlib import Path

import pytest

from contract_assistant.answer import Definition
from contract_assistant.definitions import (
    MAX_DEFINITION_CHARS,
    MAX_EXTENSIONS,
    MAX_TERM_CHARS,
    MIN_CONTEXT_CHARS,
    MIN_TERM_CHARS,
    STOP_FIRST_WORDS,
    STOP_TERMS,
    asked_term,
    find_definitions,
    lookup,
)

DEMO = Path(__file__).resolve().parent.parent / "demo"
# A sentence opening long enough that the defining sentence needs no extension.
PAD = "In consideration of the mutual promises set out in this document and other value, "
ADDRESS_TEXT = "ACME INC. 2nd Floor, Main Street. Vanuatu ('the Supplier') of one part."


def terms_of(text):
    return [definition.term for definition in find_definitions(text)]


def define(term, quote="A quote.", page=1):
    return Definition(term=term, quote=quote, page=page, start=0, end=len(quote))


@pytest.mark.parametrize(
    "form",
    [
        '("Supplier")',
        '(the "Supplier")',
        "('the Supplier')",
        '(hereinafter the "Supplier")',
        "(hereinafter referred to as the “Supplier”)",
    ],
)
def test_parenthetical_definition_gives_the_term_without_the_article(form):
    assert terms_of(f"{PAD}Acme Corp {form} will deliver the goods.") == ["Supplier"]


@pytest.mark.parametrize(
    "phrase",
    [
        '"Products" means the goods listed in schedule A',
        '"Products" shall mean the goods listed in schedule A',
        "Products means the goods listed in schedule A",
    ],
)
def test_means_definition_gives_the_term(phrase):
    assert terms_of(f"{PAD}{phrase}.") == ["Products"]


def test_quote_is_the_sentence_containing_the_match_and_equals_its_slice():
    text = f'First sentence here. {PAD}Acme Corp (the "Supplier") will deliver. Last one.'
    [definition] = find_definitions(text)
    assert definition.quote == f'{PAD}Acme Corp (the "Supplier") will deliver.'
    assert definition.quote == text[definition.start : definition.end]


def test_short_sentence_is_extended_backwards_over_earlier_sentence_ends():
    [definition] = find_definitions(ADDRESS_TEXT)
    assert definition.quote == ADDRESS_TEXT
    assert (definition.start, definition.end) == (0, len(ADDRESS_TEXT))


def test_extension_stops_after_the_maximum_number_of_sentence_ends():
    sentences = ["One.", "Two.", "Three.", "Four.", "Five."]
    text = " ".join(sentences) + ' Acme Corp (the "Supplier") will deliver.'
    [definition] = find_definitions(text)
    assert definition.start == text.index("Three.")
    assert MAX_EXTENSIONS == 3 and MIN_CONTEXT_CHARS > len(text[: definition.start])


def test_long_sentence_is_not_extended():
    text = f'Earlier sentence. {PAD}Acme Corp (the "Supplier") will deliver.'
    [definition] = find_definitions(text)
    assert definition.quote.startswith("In consideration")


def test_quote_is_capped_and_cut_at_a_word_boundary_after_the_match():
    text = f'{PAD}Acme Corp (the "Supplier") ' + "word " * 400 + "end."
    [definition] = find_definitions(text)
    assert len(definition.quote) <= MAX_DEFINITION_CHARS
    assert definition.end > text.index("Supplier") + len("Supplier")
    assert text[definition.end] == " "
    assert text[definition.start : definition.end] == definition.quote


def test_each_term_appears_once_and_the_first_definition_wins_ignoring_case():
    text = (
        f'{PAD}Acme Corp (the "Supplier") will deliver. '
        f'{PAD}Beta Ltd (the "SUPPLIER") will not.'
    )
    [definition] = find_definitions(text)
    assert definition.term == "Supplier"
    assert definition.start == 0


@pytest.mark.parametrize("term", sorted(STOP_TERMS))
def test_stop_terms_are_skipped(term):
    assert terms_of(f'{PAD}this document (the "{term}") is signed.') == []


def test_term_starting_with_a_stop_word_is_skipped():
    assert "The" in STOP_FIRST_WORDS
    assert terms_of(f'{PAD}"The following terms" means these words.') == []


def test_term_shorter_than_the_minimum_is_skipped():
    assert terms_of(f'{PAD}Acme Corp ("{"A" * (MIN_TERM_CHARS - 1)}") will deliver.') == []


def test_term_longer_than_the_maximum_is_skipped():
    long_term = "Alpha " * 5 + "B" * MAX_TERM_CHARS
    assert terms_of(f'{PAD}Acme Corp ("{long_term}") will deliver.') == []


def test_results_are_ordered_by_position_not_by_pattern():
    text = (
        f'{PAD}Products means the goods. {PAD}Acme Corp (the "Supplier") will deliver. '
        f'{PAD}"Services" means the work.'
    )
    assert terms_of(text) == ["Products", "Supplier", "Services"]


def test_page_is_none_without_page_starts():
    assert [d.page for d in find_definitions(f'{PAD}Acme Corp ("Supplier") delivers.')] == [
        None
    ]


def test_page_is_the_one_based_page_of_the_quote_start():
    first = f'{PAD}Acme Corp ("Supplier") delivers.'
    second = f'{PAD}Beta Ltd ("Buyer") pays.'
    text = f"{first}\n\n{second}"
    starts = [0, len(first) + 2]
    pages = {d.term: d.page for d in find_definitions(text, starts)}
    assert pages == {"Supplier": 1, "Buyer": 2}


def test_text_without_definitions_gives_none():
    assert find_definitions("Nothing is defined in this sentence.") == []


def demo_definitions(name):
    text = (DEMO / name).read_text(encoding="utf-8")
    return text, find_definitions(text)


def test_short_demo_contract_defines_manufacturer_customer_and_rma():
    _, definitions = demo_definitions("short_supply_agreement.txt")
    assert [d.term for d in definitions] == ["Manufacturer", "Customer", "RMA"]


def test_long_demo_contract_has_55_terms_starting_with_m_and_i_affiliate_bank():
    _, definitions = demo_definitions("long_outsourcing_agreement.txt")
    assert len(definitions) == 55
    assert [d.term for d in definitions][:3] == ["M&I", "Affiliate", "Bank"]


def test_medium_demo_contract_has_ten_terms_starting_with_effective_date():
    _, definitions = demo_definitions("medium_sponsorship_agreement.txt")
    assert len(definitions) == 10
    assert [d.term for d in definitions][:3] == ["Effective Date", "Excite", "Sponsor"]


@pytest.mark.parametrize(
    "name",
    [
        "short_supply_agreement.txt",
        "medium_sponsorship_agreement.txt",
        "long_outsourcing_agreement.txt",
    ],
)
def test_every_demo_quote_is_its_slice_of_the_text(name):
    text, definitions = demo_definitions(name)
    assert all(text[d.start : d.end] == d.quote for d in definitions)
    assert all(len(d.quote) <= MAX_DEFINITION_CHARS for d in definitions)
    assert all(d.start < d.end for d in definitions)


@pytest.mark.parametrize(
    ("question", "term"),
    [
        ("What is an RMA?", "RMA"),
        ("Who is the Customer?", "Customer"),
        ("what does 'the Manufacturer' mean", "the Manufacturer"),
        ("define RMA", "RMA"),
        ("Šta je RMA?", "RMA"),
        ("What is the notice period for termination?", "notice period for termination"),
    ],
)
def test_asked_term_is_the_term_the_question_asks_about(question, term):
    assert asked_term(question) == term


def test_asked_term_drops_the_word_term_in_what_does_the_term_mean():
    assert asked_term("What does the term Products mean?") == "Products"


def test_asked_term_is_none_for_a_question_that_is_routed():
    assert asked_term("Which law governs the contract?") is None


DEFINITIONS = [define("the Customer"), define("RMA"), define("Manufacturer")]


def test_lookup_matches_the_exact_term_ignoring_case_and_a_leading_article():
    assert lookup(DEFINITIONS, "What is an rma?") is DEFINITIONS[1]
    assert lookup(DEFINITIONS, "Who is the Manufacturer?") is DEFINITIONS[2]


def test_lookup_matches_when_the_asked_term_contains_a_known_term_as_whole_words():
    assert lookup(DEFINITIONS, "What is the RMA number?") is DEFINITIONS[1]


def test_lookup_matches_when_a_known_term_contains_the_asked_term():
    assert lookup(DEFINITIONS, "What is Customer?") is DEFINITIONS[0]


def test_lookup_does_not_match_inside_a_word():
    assert lookup(DEFINITIONS, "What is an RMAs?") is None


def test_lookup_is_none_for_an_unknown_term():
    assert lookup(DEFINITIONS, "What is the notice period for termination?") is None


def test_lookup_is_none_for_a_question_that_is_not_a_definition_question():
    assert lookup(DEFINITIONS, "Which law governs the contract?") is None


def test_lookup_prefers_an_exact_match_over_an_earlier_contained_one():
    definitions = [define("Customer Data"), define("Data")]
    assert lookup(definitions, "What is Data?") is definitions[1]
