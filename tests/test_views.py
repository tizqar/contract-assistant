import dataclasses
import json
import re

import pytest

from contract_assistant.answer import Citation, answer
from contract_assistant.config import ROOT, Settings
from contract_assistant.router.base import RouteResult
from contract_assistant.store import read_cache
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    CLAUSE_HEADERS,
    CONTEXT_CHARS,
    EXCERPT_LABEL,
    NO_PAGE,
    STATUS_LABELS,
    SUGGESTED_QUESTIONS,
    TABLE_PASSAGE_CHARS,
    TEXT_START_LABEL,
    chat_reply,
    clause_rows,
    display_name,
    document_name,
    document_title,
    document_view,
    sources_json,
    technical_markdown,
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
    assert len(rows) == len(cache.clauses)
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
