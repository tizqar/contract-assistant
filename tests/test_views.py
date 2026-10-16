import json

import pytest

from contract_assistant.answer import Citation, answer
from contract_assistant.config import ROOT, Settings
from contract_assistant.router.base import RouteResult
from contract_assistant.store import read_cache
from contract_assistant.ui.views import (
    CONTEXT_CHARS,
    TABLE_QUOTE_CHARS,
    TABLE_STATUS_WORDS,
    chat_reply,
    clause_rows,
    document_title,
    document_view,
    header_markdown,
    sources_json,
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


def test_rows_retier_when_thresholds_move(settings):
    cache = make_cache("a b c d", settings, {"Term": [("b c", 0.5)]})
    before = clause_rows(cache, TAU_LOW, TAU_HIGH)[0][1]
    after = clause_rows(cache, 0.6, TAU_HIGH)[0][1]
    assert TABLE_STATUS_WORDS["check"] in before
    assert TABLE_STATUS_WORDS["not_found"] in after


def test_rows_sorted_by_tier_then_confidence(settings):
    clauses = {
        "A": [("a", 0.1)],
        "B": [("b", 0.9)],
        "C": [("c", 0.5)],
        "D": [("d", 0.95)],
        "E": [("e", 0.4)],
    }
    cache = make_cache("a b c d e", settings, clauses)
    names = [row[0] for row in clause_rows(cache, TAU_LOW, TAU_HIGH)]
    assert names == ["D", "B", "C", "E", "A"]


def test_every_row_has_a_status_word(settings):
    cache = read_cache(DEMO_CACHE, settings)
    words = list(TABLE_STATUS_WORDS.values())
    rows = clause_rows(cache, TAU_LOW, TAU_HIGH)
    assert len(rows) == len(cache.clauses)
    assert all(any(word in row[1] for word in words) for row in rows)


def test_status_cell_carries_tier_class(settings):
    cache = make_cache("a b", settings, {"Term": [("a", 0.9)]})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[0][1] == (
        '<span class="tier tier-found">found</span>'
    )


def test_long_quote_is_truncated_with_ellipsis(settings):
    quote = "word " * 100
    cache = make_cache(quote, settings, {"Term": [(quote.strip(), 0.9)]})
    shown = clause_rows(cache, TAU_LOW, TAU_HIGH)[0][4]
    assert len(shown) == TABLE_QUOTE_CHARS
    assert shown.endswith("…")


def test_quote_whitespace_is_collapsed(settings):
    text = "alpha \n\n  beta"
    cache = make_cache(text, settings, {"Term": [(text, 0.9)]})
    assert clause_rows(cache, TAU_LOW, TAU_HIGH)[0][4] == "alpha beta"


def test_empty_n_best_row_is_not_found(settings):
    cache = make_cache("text", settings, {"Term": []})
    row = clause_rows(cache, TAU_LOW, TAU_HIGH)[0]
    assert row[0] == "Term"
    assert TABLE_STATUS_WORDS["not_found"] in row[1]
    assert row[2:] == [0.0, "n/a", ""]


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


def test_text_window_label_gives_character_range(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "fox jumps over the lazy dog. 200")
    label, segments = document_view(cache, "Clause", citation)
    shown = "".join(part for part, _ in segments)
    start = text.index(shown)
    assert label == f"characters {start:,}–{start + len(shown):,} of {len(text):,}"


def test_window_near_start_does_not_run_before_text(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "quick brown fox jumps over the lazy dog. 0")
    label, segments = document_view(cache, "Clause", citation)
    assert label.startswith("characters 0–")
    assert "".join(part for part, _ in segments).startswith(text[:20])


def test_window_near_end_does_not_run_past_text(settings):
    text = long_text()
    cache, citation = view_of(text, settings, "fox jumps over the lazy dog. 399")
    label, segments = document_view(cache, "Clause", citation)
    assert label.endswith(f"–{len(text):,} of {len(text):,}")
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
    assert label == "page 2"
    assert "".join(part for part, _ in segments) == cache.text[starts[1] : starts[2]]


def test_pdf_citation_across_pages_gives_page_range(settings):
    cache, _ = pdf_cache(settings, "text\npage two")
    label, _ = document_view(cache, "Term", top_citation(cache, "Term"))
    assert label == "pages 1–2"


def test_no_citation_shows_text_start_without_highlight(settings):
    text = long_text()
    cache = make_cache(text, settings)
    label, segments = document_view(cache, None, None)
    assert label.startswith("characters 0–")
    assert all(name is None for _, name in segments)
    assert segments[0][0].startswith(text[:20])


def test_no_citation_in_pdf_shows_first_page(settings):
    cache, starts = pdf_cache(settings, "two")
    label, segments = document_view(cache, None, None)
    assert label == "page 1"
    assert segments == [(cache.text[: starts[1]], None)]


def test_header_names_extractor_and_router(settings):
    header = header_markdown(settings, "embedding")
    assert f"`{settings.extractor_name}`" in header
    assert header.endswith("router: embedding")


def test_header_shows_llm_model_for_llm_router(settings):
    assert header_markdown(settings, "llm").endswith(f"router: {settings.llm_model}")


def test_title_for_text_file(settings):
    cache = make_cache("hello", settings)
    assert document_title(cache) == "test.txt · 5 chars"


def test_title_for_pdf_has_page_count(settings):
    cache, _ = pdf_cache(settings, "two")
    source = cache.source.model_copy(update={"filename": "contract.pdf"})
    cache = cache.model_copy(update={"source": source})
    assert document_title(cache) == f"contract.pdf · 3 pages · {len(cache.text):,} chars"


def make_answer(cache, names, latency=41.6):
    route = RouteResult(clause_types=names, kind="embedding", latency_ms=latency)
    return answer("question", route, cache, TAU_LOW, TAU_HIGH)


def test_chat_reply_has_answer_and_router_line(settings):
    cache = make_cache("a b c", settings, {"Term": [("b", 0.9)]})
    reply = chat_reply(make_answer(cache, ["Term"]), ["Term"])
    assert "**Term**" in reply
    assert reply.endswith("\n\n*router: embedding · 42 ms*")


def test_chat_reply_reports_withheld_quotes(settings):
    cache = make_cache("a b c", settings, {"Term": [("b", 0.9)]})
    broken = cache.model_copy(update={"text": "x y z"})
    reply = chat_reply(make_answer(broken, ["Term"]), ["Term"])
    assert reply.endswith("· 1 quote withheld*")


def test_sources_json_lists_other_candidates(settings):
    cache = make_cache("a b c", settings, {"Term": [("a", 0.9), ("b", 0.4), ("c", 0.1)]})
    sources = sources_json(make_answer(cache, ["Term"]), cache)
    others = sources["other_candidates"]["Term"]
    assert [item["text"] for item in others] == ["b", "c"]
    assert set(others[0]) == {"text", "conf", "start", "end", "page"}
    json.dumps(sources)


def test_demo_governing_law_view_highlights_top_citation(settings):
    cache = read_cache(DEMO_CACHE, settings)
    citation = top_citation(cache, "Governing Law")
    _, segments = document_view(cache, "Governing Law", citation)
    highlighted = [part for part, label in segments if label == "Governing Law"]
    assert highlighted == [citation.quote]
