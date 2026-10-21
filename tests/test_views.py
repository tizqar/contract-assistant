import dataclasses
import datetime
import json
import re
import time

import pytest

from contract_assistant.answer import (
    Answer,
    AnswerItem,
    Citation,
    Definition,
    GeneratedAnswer,
    Passage,
    answer,
)
from contract_assistant.compare import ChecklistRow, TermRow
from contract_assistant.config import ROOT, Settings
from contract_assistant.dates import KeyDate
from contract_assistant.explain import Explanation
from contract_assistant.fulltext import FullTextSummary, SectionSummary
from contract_assistant.obligations import Obligation, ObligationSheet
from contract_assistant.review import Flag
from contract_assistant.router.base import RouteResult
from contract_assistant.store import read_cache
from contract_assistant.summary import Summary
from contract_assistant.term_sheet import (
    FIELDS,
    LABELS,
    NOT_STATED,
    TermSheet,
    TermValue,
)
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    CLAUSE_HEADERS,
    CONTEXT_CHARS,
    EXCERPT_LABEL,
    NO_PAGE,
    OTHER_TOPIC,
    PASSAGE_PREFIX,
    RELATED_LABEL,
    STATUS_LABELS,
    SUGGESTED_QUESTIONS,
    TABLE_PASSAGE_CHARS,
    TEXT_START_LABEL,
    chat_reply,
    checklist_intro,
    clause_rows,
    compare_clause_rows,
    compare_term_rows,
    date_rows,
    definition_block,
    definition_citation,
    definition_rows,
    display_name,
    document_name,
    document_title,
    document_view,
    elapsed_text,
    explanation_markdown,
    fulltext_markdown,
    highlighted_quote,
    obligation_rows,
    page_of,
    passage_choices,
    passage_citation,
    review_rows,
    section_rows,
    sources_json,
    streaming_markdown,
    summary_markdown,
    technical_markdown,
    term_rows,
    top_citation,
)
from tests.support.caches import make_cache

TAU_LOW = 0.2
TAU_HIGH = 0.8
DEMO_CACHE = ROOT / "demo/cache/short_supply_agreement.json"
LINE = "The quick brown fox jumps over the lazy dog.\n"


@pytest.fixture
def settings() -> Settings:
    return Settings()


def long_text(lines: int = 400) -> str:
    return "".join(f"{LINE[:-1]} {i}\n" for i in range(lines))


def view_of(text, settings, needle, clause="Clause"):
    cache = make_cache(text, settings, {clause: [(needle, 0.9)]})
    return cache, top_citation(cache, clause)


def clause_types() -> list[str]:
    path = ROOT / "data/clause_types.json"
    return [entry["name"] for entry in json.loads(path.read_text())["clause_types"]]


def test_groups_cover_every_clause_type_once_in_file_order():
    grouped = [name for names in CLAUSE_GROUPS.values() for name in names]
    assert grouped == clause_types()


def test_display_name_lowercases_small_words():
    assert display_name("Cap On Liability") == "Cap on Liability"
    assert display_name("Change Of Control") == "Change of Control"
    assert display_name("Covenant Not To Sue") == "Covenant Not to Sue"
    assert display_name("Notice Period To Terminate Renewal") == (
        "Notice Period to Terminate Renewal"
    )
    assert display_name("Termination For Convenience") == "Termination for Convenience"
    assert display_name("No-Solicit Of Customers") == "No-Solicit of Customers"


def test_display_name_keeps_plain_names():
    assert display_name("Governing Law") == "Governing Law"


@pytest.mark.parametrize(
    ("clause_type", "shown"),
    [
        ("Rofr/Rofo/Rofn", "Right of First Refusal, Offer or Negotiation"),
        ("Unlimited/All-You-Can-Eat-License", "Unlimited Usage License"),
        ("Affiliate License-Licensor", "License from the Licensor's Affiliates"),
        ("Affiliate License-Licensee", "License to the Licensee's Affiliates"),
    ],
)
def test_display_name_overrides_unreadable_names(clause_type, shown):
    assert display_name(clause_type) == shown


def test_display_names_are_unique():
    shown = [display_name(name) for name in clause_types()]
    assert len(set(shown)) == len(shown)


def test_document_name_from_snake_case_text_file():
    assert document_name("short_supply_agreement.txt") == "Short supply agreement"


def test_document_name_keeps_capitals_and_splits_hyphens():
    assert document_name("ACME_Supply-Agreement.pdf") == "ACME Supply Agreement"


def test_document_name_collapses_spaces():
    assert document_name("a__b - c.txt") == "A b c"


def test_title_for_text_file_gives_rounded_words(settings):
    cache = make_cache("word " * 3_449, settings)
    assert document_title(cache) == "**Test** · about 3,400 words"


def test_title_for_short_text_gives_exact_words(settings):
    cache = make_cache("one two three", settings)
    assert document_title(cache) == "**Test** · 3 words"


def test_title_for_pdf_gives_page_count(settings):
    cache, _ = pdf_cache(settings, "two")
    source = cache.source.model_copy(update={"filename": "supply_contract.pdf", "n_pages": 23})
    cache = cache.model_copy(update={"source": source})
    assert document_title(cache) == "**Supply contract** · 23 pages"


def test_rows_follow_topic_order_with_a_row_per_clause_type(settings):
    cache = make_cache("a b", settings, {"Parties": [("a", 0.9)]})
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH)
    assert [row[1] for row in rows] == [display_name(n) for n in clause_types()]
    assert all(len(row) == len(CLAUSE_HEADERS) for row in rows)


def test_topic_is_shown_on_first_row_of_each_topic_only(settings):
    cache = make_cache("a b", settings)
    topics = [row[0] for row in clause_rows(cache, TAU_LOW, TAU_HIGH)]
    assert [t for t in topics if t] == list(CLAUSE_GROUPS)
    assert topics[0] == "General terms"
    assert topics[1] == ""


def test_rows_retier_when_thresholds_move(settings):
    cache = make_cache("a b c d", settings, {"Parties": [("b c", 0.5)]})
    before = clause_rows(cache, TAU_LOW, TAU_HIGH)[1][2]
    after = clause_rows(cache, 0.6, TAU_HIGH)[1][2]
    assert STATUS_LABELS["check"] in before
    assert STATUS_LABELS["not_found"] in after


def test_status_cell_carries_tier_class(settings):
    cache = make_cache("a b", settings, {"Parties": [("a", 0.9)]})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[1][2] == (
        '<span class="tier tier-found">Found</span>'
    )


def test_confidence_is_a_whole_percentage(settings):
    cache = make_cache("a b", settings, {"Parties": [("a", 0.944)]})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[1][3] == "94%"


def test_status_filter_keeps_only_matching_rows_and_shows_topic(settings):
    clauses = {"Parties": [("a", 0.9)], "Audit Rights": [("b", 0.9)], "Insurance": [("c", 0.5)]}
    cache = make_cache("a b c", settings, clauses)
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH, status="found")
    assert [row[1] for row in rows] == ["Parties", "Audit Rights"]
    assert [row[0] for row in rows] == ["General terms", "Liability and other terms"]


def test_every_row_of_the_demo_cache_has_a_status_label(settings):
    cache = read_cache(DEMO_CACHE, settings)
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH)
    assert len(rows) == len(clause_types())
    assert all(any(word in row[2] for word in STATUS_LABELS.values()) for row in rows)


def test_long_passage_is_truncated_with_ellipsis(settings):
    quote = "word " * 100
    cache = make_cache(quote, settings, {"Parties": [(quote.strip(), 0.9)]})
    shown = clause_rows(cache, TAU_LOW, TAU_HIGH)[1][5]
    assert len(shown) == TABLE_PASSAGE_CHARS
    assert shown.endswith("…")


def test_passage_whitespace_is_collapsed(settings):
    text = "alpha \n\n  beta"
    cache = make_cache(text, settings, {"Parties": [(text, 0.9)]})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[1][5] == "alpha beta"


def test_clause_without_candidates_is_not_found_without_page(settings):
    cache = make_cache("text", settings, {"Parties": []})
    row = clause_rows(cache, TAU_LOW, TAU_HIGH)[1]
    assert row[1] == "Parties"
    assert STATUS_LABELS["not_found"] in row[2]
    assert row[3:] == ["0%", NO_PAGE, ""]


def test_page_column_shows_the_page_number(settings):
    cache, _ = pdf_cache(settings, "two")
    candidate = cache.clauses["Term"][0].model_copy(update={"page": 2})
    cache = cache.model_copy(update={"clauses": {"Parties": [candidate]}})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[1][4] == "2"


def test_top_citation_is_none_for_empty_n_best(settings):
    cache = make_cache("text", settings, {"Term": []})
    assert top_citation(cache, "Term") is None


def test_top_citation_ignores_tier(settings):
    cache = make_cache("a b", settings, {"Term": [("a", 0.01)]})
    assert top_citation(cache, "Term").quote == "a"


def test_highlighted_segment_equals_cited_text(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "fox jumps over the lazy dog. 200")
    _, segments = document_view(cache, "Clause", citation)
    highlighted = [text for text, label in segments if label == "Clause"]
    assert highlighted == [cache.text[citation.start : citation.end]]


def test_text_window_starts_and_ends_on_line_breaks(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "fox jumps over the lazy dog. 200")
    _, segments = document_view(cache, "Clause", citation)
    shown = "".join(part for part, _ in segments)
    start = text.index(shown)
    end = start + len(shown)
    assert start == 0 or text[start - 1] == "\n"
    assert end == len(text) or text[end] == "\n"


def test_text_label_with_citation_names_the_excerpt(settings):
    cache, citation = view_of(long_text(), settings, "fox jumps over the lazy dog. 200")
    label, _ = document_view(cache, "Clause", citation)
    assert label == EXCERPT_LABEL


def test_window_near_start_does_not_run_before_text(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "quick brown fox jumps over the lazy dog. 0")
    _, segments = document_view(cache, "Clause", citation)
    assert "".join(part for part, _ in segments).startswith(text[:20])


def test_window_near_end_does_not_run_past_text(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "fox jumps over the lazy dog. 399")
    _, segments = document_view(cache, "Clause", citation)
    assert "".join(part for part, _ in segments).endswith(text[-20:])


def test_window_covers_citation_longer_than_context(settings):
    text = long_text(400)
    needle = text[100 : 100 + 3 * CONTEXT_CHARS]
    cache, citation = view_of(text, settings, needle)
    _, segments = document_view(cache, "Clause", citation)
    assert next(part for part, label in segments if label) == needle


def test_window_of_text_without_line_breaks_stays_bounded(settings):
    text = "word " * 20_000
    start = len(text) // 2
    needle = text[start : start + 50]
    cache = make_cache(text, settings)
    citation = Citation(quote=needle, page=None, conf=0.9, start=start, end=start + 50)
    _, segments = document_view(cache, "Clause", citation)
    shown = "".join(part for part, _ in segments)
    assert len(shown) <= 2 * CONTEXT_CHARS + len(needle)


def pdf_cache(settings, citation_text):
    pages = ["page one text", "page two text", "page three text"]
    text = "\n".join(pages)
    starts = [0, len(pages[0]) + 1, len(pages[0]) + len(pages[1]) + 2]
    cache = make_cache(text, settings, {"Term": [(citation_text, 0.9)]})
    source = cache.source.model_copy(update={"kind": "pdf", "n_pages": 3})
    cache = cache.model_copy(update={"page_starts": starts, "source": source})
    return cache, starts


def test_pdf_view_covers_exactly_the_citation_page(settings):
    cache, starts = pdf_cache(settings, "two")
    label, segments = document_view(cache, "Term", top_citation(cache, "Term"))
    assert label == "Page 2"
    assert "".join(part for part, _ in segments) == cache.text[starts[1] : starts[2]]


def test_pdf_citation_across_pages_gives_page_range(settings):
    cache, _ = pdf_cache(settings, "text\npage two")
    label, _ = document_view(cache, "Term", top_citation(cache, "Term"))
    assert label == "Pages 1–2"


def test_no_citation_shows_text_start_without_highlight(settings):
    text = long_text()
    cache = make_cache(text, settings)
    label, segments = document_view(cache, None, None)
    assert label == TEXT_START_LABEL
    assert all(name is None for _, name in segments)
    assert segments[0][0].startswith(text[:20])


def test_no_citation_in_pdf_shows_first_page(settings):
    cache, starts = pdf_cache(settings, "two")
    label, segments = document_view(cache, None, None)
    assert label == "Page 1"
    assert segments == [(cache.text[: starts[1]], None)]


def test_technical_details_for_llm_matching(settings):
    text = technical_markdown(settings, "llm")
    assert f"`{settings.extractor_name}`" in text
    assert f"`{settings.llm_model}`" in text
    assert "Ollama" in text
    assert settings.tau_source in text


def test_technical_details_name_the_openai_compatible_api(settings):
    settings = dataclasses.replace(settings, llm_backend="openai")
    assert "OpenAI-compatible API" in technical_markdown(settings, "llm")


def test_technical_details_for_embedding_matching(settings):
    text = technical_markdown(settings, "embedding")
    assert f"`{settings.embedding_model_id}`" in text
    assert settings.llm_model not in text


def make_answer(cache, names):
    route = RouteResult(clause_types=names, kind="embedding", latency_ms=41.6)
    return answer("question", route, cache, TAU_LOW, TAU_HIGH)


def reply_for(settings, clause, conf, text="a b c", needle="b"):
    cache = make_cache(text, settings, {clause: [(needle, conf)]})
    return chat_reply(make_answer(cache, [clause]))


def test_found_reply_has_badge_name_quote_and_meta(settings):
    reply = reply_for(settings, "Governing Law", 0.944)
    assert reply.startswith('<span class="tier tier-found">Found</span> **Governing Law**\n> b')
    assert "<small>94% confidence</small>" in reply


def test_verify_reply_has_verify_badge(settings):
    reply = reply_for(settings, "Insurance", 0.5)
    assert '<span class="tier tier-check">Verify</span> **Insurance**' in reply


def test_reply_uses_display_names(settings):
    assert "**Cap on Liability**" in reply_for(settings, "Cap On Liability", 0.9)


def test_meta_line_has_page_for_a_pdf(settings):
    cache, _ = pdf_cache(settings, "two")
    candidate = cache.clauses["Term"][0].model_copy(update={"page": 12})
    cache = cache.model_copy(update={"clauses": {"Insurance": [candidate]}})
    reply = chat_reply(make_answer(cache, ["Insurance"]))
    assert "<small>Page 12 · 90% confidence</small>" in reply


def test_quote_is_escaped_and_cut_like_the_cli(settings):
    text = "*" + "x" * 500
    reply = reply_for(settings, "Insurance", 0.9, text, text)
    quote = next(line for line in reply.splitlines() if line.startswith("> "))
    assert quote.startswith("> \\*")
    assert quote.endswith("…")


def test_not_found_reply_says_the_clause_was_checked(settings):
    reply = reply_for(settings, "Insurance", 0.001)
    assert '<span class="tier tier-not_found">Not found</span> **Insurance**' in reply
    assert "The contract was checked for this clause; no passage matches it." in reply
    assert "highlighted" not in reply


def test_withheld_reply_keeps_the_badge_and_hides_the_quote(settings):
    cache = make_cache("a b c", settings, {"Insurance": [("b", 0.9)]})
    broken = cache.model_copy(update={"text": "x y z"})
    reply = chat_reply(make_answer(broken, ["Insurance"]))
    assert 'tier-found">Found</span> **Insurance**' in reply
    assert "could not be verified word for word" in reply
    assert "\n> " not in reply
    assert "highlighted" not in reply


def test_reply_points_to_the_highlighted_passage(settings):
    reply = reply_for(settings, "Insurance", 0.9)
    assert reply.endswith(
        "<small>The passage is highlighted in the document on the right.</small>"
    )


def test_reply_for_several_passages_uses_the_plural(settings):
    clauses = {"Insurance": [("a", 0.9)], "Audit Rights": [("b", 0.9)]}
    cache = make_cache("a b c", settings, clauses)
    reply = chat_reply(make_answer(cache, list(clauses)))
    assert "The first passage is highlighted in the document on the right;" in reply
    assert "choose another under \u201cPassage shown\u201d." in reply


def test_empty_route_lists_every_clause_under_its_topic(settings):
    cache = make_cache("a b c", settings)
    reply = chat_reply(make_answer(cache, []))
    assert reply.startswith("This question does not match any of the clauses I check.")
    for topic, names in CLAUSE_GROUPS.items():
        line = next(line for line in reply.splitlines() if line.startswith(f"- **{topic}:**"))
        assert all(display_name(name) in line for name in names)
    assert SUGGESTED_QUESTIONS[0] in reply


@pytest.mark.parametrize("conf", [0.001, 0.5, 1.0])
def test_reply_has_no_developer_wording(settings, conf):
    reply = reply_for(settings, "Insurance", conf)
    assert not re.search(r"router|p\. n/a|confidence \d\.\d\d| ms\b", reply, re.IGNORECASE)


def test_sources_json_lists_other_candidates(settings):
    cache = make_cache("a b c", settings, {"Term": [("a", 0.9), ("b", 0.4), ("c", 0.1)]})
    sources = sources_json(make_answer(cache, ["Term"]), cache)
    others = sources["other_candidates"]["Term"]
    assert [item["text"] for item in others] == ["b", "c"]
    assert set(others[0]) == {"text", "conf", "start", "end", "page"}
    json.dumps(sources)


def test_highlighted_segment_is_labelled_with_the_display_name(settings):
    cache, citation = view_of("a Cap On Liability b", settings, "Cap On", "Cap On Liability")
    _, segments = document_view(cache, "Cap On Liability", citation)
    assert [label for _, label in segments if label] == ["Cap on Liability"]


def test_demo_governing_law_view_highlights_top_citation(settings):
    cache = read_cache(DEMO_CACHE, settings)
    citation = top_citation(cache, "Governing Law")
    _, segments = document_view(cache, "Governing Law", citation)
    highlighted = [part for part, label in segments if label == "Governing Law"]
    assert highlighted == [citation.quote]


def passages_answer(passages: list[Passage]) -> Answer:
    route = RouteResult(clause_types=[], kind="embedding", latency_ms=1.0)
    return answer(
        "question", route, make_cache("a b", Settings()), TAU_LOW, TAU_HIGH
    ).model_copy(update={"passages": passages})


def a_passage(quote="Invoices are payable <b>in sixty days", page=None) -> Passage:
    return Passage(quote=quote, page=page, score=0.7, start=0, end=len(quote))


def test_reply_with_passages_explains_and_shows_related_badges():
    reply = chat_reply(passages_answer([a_passage(), a_passage("Second")]))
    assert "does not match a clause I check" in reply
    assert reply.count('<span class="tier tier-related">Related</span>') == 2
    assert "The first passage is highlighted" in reply


def test_reply_with_passages_escapes_the_quote_and_shows_the_page():
    reply = chat_reply(passages_answer([a_passage(page=4)]))
    assert "> Invoices are payable \\<b\\>in sixty days" in reply
    assert "<small>Page 4</small>" in reply


def test_reply_without_passages_keeps_the_empty_route_text():
    assert chat_reply(passages_answer([])).startswith("This question does not match any")


def test_passage_choices_number_the_related_passages():
    choices = passage_choices([a_passage(), a_passage()])
    assert choices == [
        ("Related passage 1", f"{PASSAGE_PREFIX}0"),
        ("Related passage 2", f"{PASSAGE_PREFIX}1"),
    ]


def test_passage_citation_uses_the_score_as_confidence():
    assert passage_citation(a_passage()).conf == 0.7


def test_unvalidated_type_meta_says_it_was_not_validated(settings):
    cache = make_cache("a b c", settings, {"Payment Terms": [("b", 0.86)]})
    reply = chat_reply(make_answer(cache, ["Payment Terms"]), {"Payment Terms"})
    assert "<small>Not validated on test data · 86% confidence</small>" in reply


def test_checklist_adds_a_topic_for_extra_types_present_in_the_cache(settings):
    cache = make_cache("a b", settings, {"Payment Terms": [("a", 0.97)], "Delivery": []})
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH, unvalidated={"Payment Terms", "Delivery"})
    extra = [row for row in rows if row[1] in ("Payment Terms", "Delivery")]
    assert [row[0] for row in extra] == [OTHER_TOPIC, ""]
    assert "tier-check" in extra[0][2]


def test_checklist_leaves_out_extra_types_missing_from_the_cache(settings):
    cache = make_cache("a b", settings)
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH, unvalidated={"Payment Terms"})
    assert len(rows) == len(clause_types())


def test_technical_details_list_the_unvalidated_types(settings):
    text = technical_markdown(settings, "embedding", {"Payment Terms"})
    assert "Not validated on test data:** Payment Terms" in text


def test_checklist_intro_counts_the_types_listed_and_names_the_extras(settings):
    cache = make_cache("a b", settings, {"Payment Terms": [], "Delivery": []})
    intro = checklist_intro(cache, {"Payment Terms", "Delivery"})
    assert intro.startswith(f"All {len(clause_types()) + 2} clauses")
    assert "without validation data" in intro


def test_checklist_intro_without_extra_types_does_not_mention_them(settings):
    intro = checklist_intro(make_cache("a b", settings), {"Payment Terms"})
    assert intro.startswith(f"All {len(clause_types())} clauses")
    assert "validation" not in intro


def test_empty_route_lists_the_extra_types_after_the_other_topics(settings):
    cache = make_cache("a b c", settings)
    reply = chat_reply(make_answer(cache, []), {"Payment Terms", "Delivery"})
    lines = [line for line in reply.splitlines() if line.startswith("- **")]
    assert lines[-1].startswith(f"- **{OTHER_TOPIC}:**")
    assert "Payment Terms" in lines[-1] and "Delivery" in lines[-1]


def summary_of(**changes) -> Summary:
    fields = {
        "doc_id": "sha256:x",
        "model": "gen-1",
        "tau_low": 0.2,
        "tau_high": 0.8,
        "items": [
            AnswerItem(
                clause_type="Governing Law",
                status="found",
                best_conf=0.9,
                citation=Citation(quote="Vanuatu law.", page=None, conf=0.9, start=0, end=12),
            ),
            AnswerItem(
                clause_type="Payment Terms",
                status="check",
                best_conf=0.5,
                citation=Citation(
                    quote="Pay in 30 days.", page=None, conf=0.5, start=13, end=28
                ),
            ),
        ],
        "text": "Vanuatu law governs.",
        "latency_ms": 4600.0,
    }
    return Summary(**{**fields, **changes})


def explanation_of(**changes) -> Explanation:
    fields = {
        "clause_type": "Governing Law",
        "quote": "Vanuatu law.",
        "language": "en",
        "model": "gen-1",
        "text": "Vanuatu law applies.",
        "latency_ms": 100.0,
    }
    return Explanation(**{**fields, **changes})


def test_highlighted_quote_reads_the_dict_form_gradio_passes():
    segments = [
        {"token": "Before ", "class_or_confidence": None},
        {"token": "Clause.", "class_or_confidence": "Parties"},
        {"token": "Other.", "class_or_confidence": "Governing Law"},
    ]
    assert highlighted_quote(segments) == ("Clause.", "Parties")


def test_highlighted_quote_reads_tuples():
    assert highlighted_quote([("a", None), ("b", "Parties")]) == ("b", "Parties")


def test_highlighted_quote_is_none_without_a_label():
    assert highlighted_quote([("a", None), {"token": "b", "class_or_confidence": None}]) is None


def test_highlighted_quote_is_none_for_none_and_empty():
    assert highlighted_quote(None) is None
    assert highlighted_quote([]) is None


def test_summary_shows_the_text_the_model_and_the_counts():
    markdown = summary_markdown(summary_of())
    assert markdown.startswith("Vanuatu law governs.")
    assert "gen-1" in markdown and "2 clause(s)" in markdown and "5 s" in markdown
    assert "verified word for word" in markdown


def test_summary_lists_the_sources_with_badges_in_order():
    based_on = summary_markdown(summary_of()).splitlines()[-1]
    assert based_on.index("Governing Law") < based_on.index("Payment Terms")
    assert "tier-found" in based_on and "tier-check" in based_on


def test_summary_marks_unvalidated_sources():
    based_on = summary_markdown(summary_of(), {"Payment Terms"}).splitlines()[-1]
    assert based_on.count("Not validated on test data") == 1
    assert based_on.index("Not validated") > based_on.index("Payment Terms")


def test_withheld_summary_gives_the_reason_and_counts_the_spans_without_printing_them():
    summary = summary_of(text=None, reason="it quotes text that is not in the contract")
    summary = summary.model_copy(update={"unfaithful_spans": ["Made up one", "Made up two"]})
    markdown = summary_markdown(summary)
    assert "not shown because it quotes text that is not in the contract." in markdown
    assert "2 quoted passage(s)" in markdown
    assert "Made up" not in markdown


def test_withheld_summary_without_spans_has_no_count_line():
    markdown = summary_markdown(summary_of(text=None, reason="the model gave an empty reply"))
    assert "<small>" not in markdown


def test_explanation_has_heading_text_and_restatement_note():
    markdown = explanation_markdown(explanation_of())
    assert markdown.startswith("**Governing Law** · in plain English")
    assert "Vanuatu law applies." in markdown
    assert "gen-1" in markdown and "not a quote" in markdown


def test_explanation_names_serbian_without_the_script_note():
    heading = explanation_markdown(explanation_of(language="sr")).splitlines()[0]
    assert heading.endswith("in plain Serbian")


def test_withheld_explanation_gives_the_reason_only():
    markdown = explanation_markdown(explanation_of(text=None, reason="it names numbers: 9"))
    assert markdown == "The explanation is not shown because it names numbers: 9."


def test_explanation_of_a_related_passage_keeps_the_label():
    markdown = explanation_markdown(explanation_of(clause_type=RELATED_LABEL))
    assert markdown.startswith(f"**{RELATED_LABEL}** · in plain English")


def test_technical_details_name_the_generation_model(settings):
    text = technical_markdown(settings, "embedding", gen_model="gen-1")
    assert "**Summaries and explanations:** `gen-1` through Ollama" in text


def test_technical_details_say_generation_is_off(settings):
    assert "**Summaries and explanations:** off" in technical_markdown(settings, "embedding")


def generated_of(text: str | None = "Payable in sixty days.", reason=None) -> GeneratedAnswer:
    return GeneratedAnswer(model="gen-1", text=text, reason=reason, latency_ms=5.0)


def with_generated(generated: GeneratedAnswer | None) -> Answer:
    answer_ = passages_answer([a_passage()])
    return answer_.model_copy(update={"generated": generated})


def test_reply_shows_the_generated_answer_before_the_passages():
    reply = chat_reply(with_generated(generated_of()))
    assert (
        '<span class="tier tier-generated">Generated</span> **Answer from the passages**'
        in reply
    )
    assert reply.index("Payable in sixty days.") < reply.index("does not match a clause")
    assert "Written by the language model gen-1 from the passages below." in reply
    assert "This is not a clause match." in reply
    assert "tier-related" in reply and "The passage is highlighted" in reply


def test_reply_with_a_withheld_generated_answer_gives_only_the_reason():
    reply = chat_reply(with_generated(generated_of(None, "it adds numbers: 9")))
    assert reply.startswith(
        "<small>The language model's answer is not shown because it adds numbers: 9.</small>"
    )
    assert "tier-generated" not in reply and "tier-related" in reply


def test_reply_without_a_generated_answer_has_no_generated_block():
    assert "tier-generated" not in chat_reply(with_generated(None))


def sheet_of(terms=None, rejected=None) -> TermSheet:
    return TermSheet(
        doc_id="d",
        model="gen-1",
        tau_low=0.2,
        tau_high=0.8,
        latency_ms=1.0,
        items=[],
        terms={field: [] for field in FIELDS} | (terms or {}),
        rejected=rejected or {},
    )


def term_value(value="Vanuatu", clause_type="Governing Law", page=3) -> TermValue:
    return TermValue(value=value, clause_type=clause_type, page=page, start=0, end=len(value))


def test_term_rows_have_one_labelled_row_per_field_in_order():
    rows = term_rows(sheet_of())
    assert [row[0] for row in rows] == [LABELS[field] for field in FIELDS]
    assert all(len(row) == 4 for row in rows)


def test_term_row_shows_value_clause_name_and_page():
    rows = term_rows(sheet_of({"governing_law": [term_value()]}))
    assert rows[-1] == ["Governing law", "Vanuatu", "Governing Law", "3"]


def test_term_row_joins_clause_names_and_takes_the_first_page():
    values = [term_value("A", "Parties", 2), term_value("B", "Agreement Date", None)]
    row = term_rows(sheet_of({"parties": values}))[0]
    assert row[2] == "Parties, Agreement Date" and row[3] == "2"


def test_term_row_without_a_page_shows_the_dash():
    row = term_rows(sheet_of({"governing_law": [term_value(page=None)]}))[-1]
    assert row[3] == NO_PAGE


def test_term_row_not_stated_has_no_clause_and_no_page():
    assert term_rows(sheet_of())[0] == ["Parties", NOT_STATED, "", NO_PAGE]


def test_term_row_not_verified_names_the_rejected_value():
    row = term_rows(sheet_of(rejected={"governing_law": ["Mars"]}))[-1]
    assert row[1].startswith("not verified") and "Mars" in row[1]


def test_elapsed_text_counts_whole_seconds_since_the_start():
    assert elapsed_text(time.perf_counter() - 4.2) == "4 s"


def test_streaming_markdown_shows_the_status_then_the_text():
    started = time.perf_counter() - 4
    assert streaming_markdown("Writing the summary", "The law", started) == (
        "Writing the summary · 4 s ▌\n\nThe law"
    )


def test_streaming_markdown_without_text_is_only_the_status():
    started = time.perf_counter()
    assert (
        streaming_markdown("Writing the summary", "", started) == "Writing the summary · 0 s ▌"
    )


def test_streaming_markdown_names_why_it_asks_again():
    started = time.perf_counter() - 12
    text = streaming_markdown("Writing the summary", "New text", started, "it is empty")
    assert text.startswith("The first reply was not shown because it is empty; asking again")
    assert "· 12 s ▌" in text and text.endswith("New text")


def test_technical_details_say_generated_text_is_streamed_and_checked(settings):
    text = technical_markdown(settings, "embedding", gen_model="gen-1")
    assert "shown as it is written, then checked" in text


def test_technical_details_without_a_model_do_not_mention_streaming(settings):
    assert "shown as it is written" not in technical_markdown(settings, "embedding")


def make_definition(
    page: int | None = 3, quote: str = "RMA means a return number."
) -> Definition:
    return Definition(term="RMA", quote=quote, page=page, start=10, end=10 + len(quote))


def test_definition_block_shows_badge_term_quote_and_page():
    block = definition_block(make_definition())
    assert 'class="tier tier-definition">Definition</span> **RMA**' in block
    assert "> RMA means a return number." in block
    assert "The sentence of the contract that defines the term" in block
    assert "Page 3" in block


def test_definition_block_without_a_page_names_no_page():
    assert "Page" not in definition_block(make_definition(page=None))


def test_definition_citation_is_the_sentence_with_full_confidence():
    citation = definition_citation(make_definition())
    assert (citation.quote, citation.page, citation.conf) == (
        "RMA means a return number.",
        3,
        1.0,
    )
    assert (citation.start, citation.end) == (10, 36)


def test_definition_rows_shorten_the_quote_and_mark_a_missing_page():
    long = "RMA means " + "x" * 200
    rows = definition_rows([make_definition(), make_definition(page=None, quote=long)])
    assert rows[0] == ["RMA", "RMA means a return number.", "3"]
    assert len(rows[1][1]) == TABLE_PASSAGE_CHARS and rows[1][1].endswith("…")
    assert rows[1][2] == NO_PAGE


def test_chat_reply_for_a_definition_answer_is_the_definition_block():
    definition = make_definition()
    route = RouteResult(clause_types=[], kind="definition", latency_ms=0.0)
    result = Answer(
        question="What is an RMA?",
        route=route,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        items=[],
        dropped_quotes=0,
        definition=definition,
    )
    assert chat_reply(result) == definition_block(definition)


def make_flag(**changes) -> Flag:
    item = AnswerItem(
        clause_type="Cap On Liability",
        status="check",
        best_conf=0.5,
        citation=Citation(quote="Liability is capped.", page=1, conf=0.5, start=0, end=20),
    )
    fields = {
        "rule": "r",
        "kind": "present",
        "severity": "high",
        "title": "Title",
        "note": "A note.",
        "status": "check",
        "items": [item],
        "unvalidated": False,
    }
    return Flag(**{**fields, **changes})


def test_review_rows_show_severity_title_clauses_and_note():
    [row] = review_rows([make_flag()])
    assert row == [
        '<span class="tier tier-high">High</span>',
        "Title",
        "Cap on Liability (verify)",
        "A note.",
    ]


def test_review_rows_mark_a_flag_on_an_unvalidated_clause_type():
    [row] = review_rows([make_flag(unvalidated=True, severity="low")])
    assert row[1] == "Title (not validated on test data)" and "tier-low" in row[0]


def test_review_rows_of_no_flags_are_empty():
    assert review_rows([]) == []


def test_compare_term_rows_mark_a_difference():
    rows = [
        TermRow(field="parties", label="Parties", left="A", right="B", differs=True),
        TermRow(
            field="governing_law", label="Governing law", left="X", right="X", differs=False
        ),
    ]
    assert compare_term_rows(rows) == [
        ["Parties", "A", "B", "≠"],
        ["Governing law", "X", "X", ""],
    ]


def test_compare_clause_rows_show_status_words_and_a_mark():
    rows = [
        ChecklistRow(clause_type="Insurance", left="found", right="not_found", differs=True),
        ChecklistRow(clause_type="Audit Rights", left="check", right="check", differs=False),
    ]
    assert compare_clause_rows(rows) == [
        ["Insurance", "Found", "Not found", "≠"],
        ["Audit Rights", "Verify", "Verify", ""],
    ]


def test_date_rows_show_the_iso_date_the_source_and_the_formula():
    dates = [
        KeyDate(
            label="Agreement date",
            date=datetime.date(2020, 1, 2),
            source="January 2, 2020",
            clause_type="Agreement Date",
        ),
        KeyDate(
            label="Initial term ends",
            date=datetime.date(2021, 1, 2),
            source="one year",
            clause_type="Expiration Date",
            formula="agreement date + 1 year",
        ),
    ]
    assert date_rows(dates) == [
        ["Agreement date", "2020-01-02", "January 2, 2020 (Agreement Date)", "stated"],
        [
            "Initial term ends",
            "2021-01-02",
            "one year (Expiration Date)",
            "agreement date + 1 year",
        ],
    ]


def test_date_rows_say_when_a_date_is_not_readable():
    unreadable = KeyDate(
        label="Effective date", date=None, source="upon signing", clause_type=None
    )
    assert date_rows([unreadable]) == [
        ["Effective date", "not readable", "upon signing", "stated"]
    ]


def test_date_rows_of_nothing_are_empty():
    assert date_rows([]) == []


def test_summary_markdown_says_when_it_is_written_in_serbian():
    assert "written in Serbian" in summary_markdown(summary_of(language="sr"))


def test_summary_markdown_in_english_does_not_name_a_language():
    assert "written in" not in summary_markdown(summary_of())


def obligation(party="The Customer", action="pay  the\ninvoices", page=2) -> Obligation:
    return Obligation(
        party=party,
        modality="must",
        action=action,
        clause_type="Governing Law",
        page=page,
        start=0,
        end=5,
    )


def obligation_sheet(obligations) -> ObligationSheet:
    return ObligationSheet(
        doc_id="d",
        model="gen-1",
        tau_low=0.2,
        tau_high=0.8,
        latency_ms=1.0,
        items=[],
        obligations=obligations,
    )


def test_obligation_rows_show_party_modality_collapsed_action_clause_and_page():
    rows = obligation_rows(obligation_sheet([obligation()]))
    assert rows == [["The Customer", "must", "pay the invoices", "Governing Law", "2"]]


def test_obligation_rows_group_the_entries_by_party():
    sheet = obligation_sheet(
        [obligation("A"), obligation("B", page=None), obligation("A", action="act")]
    )
    rows = obligation_rows(sheet)
    assert [row[0] for row in rows] == ["A", "A", "B"]
    assert rows[2][4] == NO_PAGE


def test_obligation_rows_of_an_empty_sheet_are_empty():
    assert obligation_rows(obligation_sheet([])) == []


def fulltext_summary(sections, text="Overview.", reason=None) -> FullTextSummary:
    return FullTextSummary(
        doc_id="d",
        model="gen-1",
        tau_low=0.2,
        tau_high=0.8,
        latency_ms=3000.0,
        sections=sections,
        text=text,
        reason=reason,
    )


def test_section_rows_number_the_sections_and_show_page_and_text():
    sections = [
        SectionSummary(start=0, end=5, page=1, text="First.", attempts=1),
        SectionSummary(start=5, end=9, page=None, text="Second.", attempts=1),
    ]
    assert section_rows(fulltext_summary(sections)) == [
        ["1", "1", "First."],
        ["2", NO_PAGE, "Second."],
    ]


def test_section_rows_say_why_a_section_is_withheld():
    section = SectionSummary(start=0, end=5, page=1, text=None, reason="it was off", attempts=2)
    assert section_rows(fulltext_summary([section]))[0][2] == "withheld: it was off"


def test_fulltext_markdown_names_the_model_sections_and_seconds():
    section = SectionSummary(start=0, end=5, page=1, text="First.", attempts=1)
    markdown = fulltext_markdown(fulltext_summary([section]))
    assert markdown.startswith("Overview.")
    assert "gen-1" in markdown and "1 section(s) in 3 s" in markdown


def test_fulltext_markdown_gives_the_reason_when_withheld():
    summary = fulltext_summary([], text=None, reason="no section could be summarized")
    assert fulltext_markdown(summary) == (
        "The overview is not shown because no section could be summarized."
    )


def test_page_of_maps_offsets_to_one_based_pages():
    starts = [0, 10, 25]
    assert [page_of(starts, offset) for offset in (0, 9, 10, 24, 25, 99)] == [1, 1, 2, 2, 3, 3]


def test_page_of_is_page_one_without_page_starts():
    assert page_of([], 5) == 1


def test_technical_details_name_the_documents_folder(settings):
    assert "cache/documents/" in technical_markdown(settings, "llm")
