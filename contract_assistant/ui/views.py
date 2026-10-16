"""Pure functions from caches and answers to Gradio component values."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from contract_assistant.answer import ELLIPSIS, Answer, Citation, Status, render_markdown, tier
from contract_assistant.config import Settings
from contract_assistant.store import CachedCandidate, DocumentCache

CLAUSE_HEADERS = ["clause type", "status", "conf", "page", "quote"]
CONF_DECIMALS = 2
TABLE_QUOTE_CHARS = 120
CONTEXT_CHARS = 1500
# Longer than any line of the demo texts; a text without line breaks is cut mid-line.
MAX_WIDEN_CHARS = 3000
NO_PAGE = "n/a"
TABLE_STATUS_WORDS: dict[Status, str] = {
    "found": "found",
    "check": "verify",
    "not_found": "not found",
}
STATUS_ORDER: dict[Status, int] = {"found": 0, "check": 1, "not_found": 2}

Segments = list[tuple[str, str | None]]


def clause_rows(cache: DocumentCache, tau_low: float, tau_high: float) -> list[list]:
    """Return one table row per clause type, sorted by tier, then confidence."""
    keyed = [
        _row(name, candidates, tau_low, tau_high) for name, candidates in cache.clauses.items()
    ]
    keyed.sort(key=lambda item: item[0])
    return [row for _, row in keyed]


def _row(
    name: str, candidates: list[CachedCandidate], tau_low: float, tau_high: float
) -> tuple[tuple[int, float, str], list]:
    if not candidates:
        key = (STATUS_ORDER["not_found"], 0.0, name)
        return key, [name, _status_cell("not_found"), 0.0, NO_PAGE, ""]
    top = candidates[0]
    status = tier(top.conf, tau_low, tau_high)
    page = NO_PAGE if top.page is None else top.page
    cells = [name, _status_cell(status), round(top.conf, CONF_DECIMALS), page]
    return (STATUS_ORDER[status], -top.conf, name), [*cells, _short_quote(top.text)]


def _status_cell(status: Status) -> str:
    return f'<span class="tier tier-{status}">{TABLE_STATUS_WORDS[status]}</span>'


def _short_quote(text: str) -> str:
    quote = " ".join(text.split())
    if len(quote) <= TABLE_QUOTE_CHARS:
        return quote
    return quote[: TABLE_QUOTE_CHARS - len(ELLIPSIS)] + ELLIPSIS


def top_citation(cache: DocumentCache, clause_type: str) -> Citation | None:
    """Return the top candidate of a clause type as a citation, whatever its tier."""
    candidates = cache.clauses.get(clause_type, [])
    if not candidates:
        return None
    top = candidates[0]
    return Citation(quote=top.text, page=top.page, conf=top.conf, start=top.start, end=top.end)


def document_view(
    cache: DocumentCache, clause_type: str | None, citation: Citation | None
) -> tuple[str, Segments]:
    """Return the label and highlighted segments for the part of the text around a citation."""
    if cache.page_starts:
        first, last, lo, hi = _page_window(cache, citation)
        label = f"page {first}" if first == last else f"pages {first}–{last}"
    else:
        lo, hi = _text_window(cache.text, citation)
        label = f"characters {lo:,}–{hi:,} of {len(cache.text):,}"
    return label, _segments(cache.text, lo, hi, clause_type, citation)


def _text_window(text: str, citation: Citation | None) -> tuple[int, int]:
    first = 0 if citation is None else citation.start - CONTEXT_CHARS
    last = 2 * CONTEXT_CHARS if citation is None else citation.end + CONTEXT_CHARS
    return _widen_left(text, max(first, 0)), _widen_right(text, min(last, len(text)))


def _widen_left(text: str, position: int) -> int:
    newline = text.rfind("\n", max(position - MAX_WIDEN_CHARS, 0), position)
    if newline >= 0:
        return newline + 1
    return 0 if position <= MAX_WIDEN_CHARS else position


def _widen_right(text: str, position: int) -> int:
    newline = text.find("\n", position, position + MAX_WIDEN_CHARS)
    if newline >= 0:
        return newline
    return len(text) if position + MAX_WIDEN_CHARS >= len(text) else position


def _page_window(cache: DocumentCache, citation: Citation | None) -> tuple[int, int, int, int]:
    starts = cache.page_starts
    first = 1 if citation is None else bisect_right(starts, citation.start)
    last = (
        first
        if citation is None
        else bisect_right(starts, max(citation.end - 1, citation.start))
    )
    first = max(first, 1)
    last = max(last, first)
    hi = starts[last] if last < len(starts) else len(cache.text)
    return first, last, starts[first - 1], hi


def _segments(
    text: str, lo: int, hi: int, clause_type: str | None, citation: Citation | None
) -> Segments:
    if citation is None:
        return [(text[lo:hi], None)]
    parts = [
        (text[lo : citation.start], None),
        (text[citation.start : citation.end], clause_type),
        (text[citation.end : hi], None),
    ]
    return [part for part in parts if part[0]]


def header_markdown(settings: Settings, route_kind: str) -> str:
    """Return the header line naming the extractor run and the router."""
    router = settings.llm_model if route_kind == "llm" else route_kind
    return f"**Contract assistant** · extractor: `{settings.extractor_name}` · router: {router}"


def document_title(cache: DocumentCache) -> str:
    """Return the file name with its page and character counts."""
    source = cache.source
    parts = [source.filename]
    if source.n_pages is not None:
        parts.append(f"{source.n_pages} pages")
    parts.append(f"{source.n_chars:,} chars")
    return " · ".join(parts)


def chat_reply(answer: Answer, clause_type_names: Sequence[str]) -> str:
    """Return the rendered answer followed by a line with the router and its latency."""
    line = f"router: {answer.route.kind} · {round(answer.route.latency_ms)} ms"
    if answer.dropped_quotes:
        noun = "quote" if answer.dropped_quotes == 1 else "quotes"
        line += f" · {answer.dropped_quotes} {noun} withheld"
    return f"{render_markdown(answer, clause_type_names)}\n\n*{line}*"


def sources_json(answer: Answer, cache: DocumentCache) -> dict:
    """Return the answer and the other n-best candidates of each of its clause types."""
    others = {
        item.clause_type: [
            {"text": c.text, "conf": c.conf, "start": c.start, "end": c.end, "page": c.page}
            for c in cache.clauses[item.clause_type][1:]
        ]
        for item in answer.items
    }
    return {"answer": answer.model_dump(mode="json"), "other_candidates": others}
