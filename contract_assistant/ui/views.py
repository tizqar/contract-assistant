"""Pure functions from caches and answers to Gradio component values."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

from contract_assistant.answer import (
    ELLIPSIS,
    Answer,
    AnswerItem,
    Citation,
    Status,
    blockquote,
    tier,
)
from contract_assistant.config import Settings
from contract_assistant.store import DocumentCache

CLAUSE_GROUPS: dict[str, tuple[str, ...]] = {
    "General terms": (
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Governing Law",
    ),
    "Competition, termination and assignment": (
        "Most Favored Nation",
        "Non-Compete",
        "Exclusivity",
        "No-Solicit Of Customers",
        "Competitive Restriction Exception",
        "No-Solicit Of Employees",
        "Non-Disparagement",
        "Termination For Convenience",
        "Rofr/Rofo/Rofn",
        "Change Of Control",
        "Anti-Assignment",
    ),
    "Revenue and pricing": (
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
    ),
    "Intellectual property and licences": (
        "IP Ownership Assignment",
        "Joint IP Ownership",
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Source Code Escrow",
    ),
    "Liability and other terms": (
        "Post-Termination Services",
        "Audit Rights",
        "Uncapped Liability",
        "Cap On Liability",
        "Liquidated Damages",
        "Warranty Duration",
        "Insurance",
        "Covenant Not To Sue",
        "Third Party Beneficiary",
    ),
}
STATUS_LABELS: dict[Status, str] = {
    "found": "Found",
    "check": "Verify",
    "not_found": "Not found",
}
CLAUSE_HEADERS = ["Topic", "Clause", "Status", "Confidence", "Page", "Passage"]
SUGGESTED_QUESTIONS = [
    "Who are the parties?",
    "Which law governs the contract?",
    "When does the agreement expire, and does it renew?",
    "Can either party terminate without cause?",
    "Is there a non-compete or exclusivity obligation?",
    "Is liability capped?",
]
SMALL_WORDS = frozenset({"Of", "To", "On", "Or", "For"})
DISPLAY_OVERRIDES = {
    "Rofr/Rofo/Rofn": "Right of First Refusal, Offer or Negotiation",
    "Unlimited/All-You-Can-Eat-License": "Unlimited Usage License",
    "Affiliate License-Licensor": "License from the Licensor's Affiliates",
    "Affiliate License-Licensee": "License to the Licensee's Affiliates",
}
NOT_FOUND_TEXT = "The contract was checked for this clause; no passage matches it."
WITHHELD_TEXT = (
    "A matching passage was found but it could not be verified word for word "
    "against the contract, so it is not shown."
)
EMPTY_ROUTE_TEXT = "This question does not match any of the clauses I check."
HIGHLIGHT_TEXT = {
    1: "The passage is highlighted in the document on the right.",
    2: (
        "The first passage is highlighted in the document on the right; "
        "choose another under \u201cPassage shown\u201d."
    ),
}
NO_PAGE = "—"
EXCERPT_LABEL = "Excerpt around the highlighted passage"
TEXT_START_LABEL = "Beginning of the contract"
TABLE_PASSAGE_CHARS = 120
WORD_ROUNDING = 100
CONTEXT_CHARS = 1500
# Longer than any line of the demo texts; a text without line breaks is cut mid-line.
MAX_WIDEN_CHARS = 3000
EXAMPLE_QUESTIONS = 2

Segments = list[tuple[str, str | None]]


def display_name(clause_type: str) -> str:
    """Return the name of a clause type as it is shown to the user."""
    if clause_type in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[clause_type]
    first, *rest = clause_type.split(" ")
    return " ".join([first, *(word.lower() if word in SMALL_WORDS else word for word in rest)])


def document_name(filename: str) -> str:
    """Return a file name as a readable title."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    name = " ".join(stem.split())
    return name[:1].upper() + name[1:]


def document_title(cache: DocumentCache) -> str:
    """Return the document name with its page count or approximate word count."""
    source = cache.source
    name = f"**{document_name(source.filename)}**"
    if source.n_pages is not None:
        return f"{name} · {source.n_pages} pages"
    words = len(cache.text.split())
    if words >= WORD_ROUNDING:
        words = round(words / WORD_ROUNDING) * WORD_ROUNDING
        return f"{name} · about {words:,} words"
    return f"{name} · {words} words"


def _badge(status: Status) -> str:
    return f'<span class="tier tier-{status}">{STATUS_LABELS[status]}</span>'


def _percent(conf: float) -> str:
    return f"{conf:.0%}"


def chat_reply(answer: Answer) -> str:
    """Return the Markdown shown in the chat for an answer."""
    if not answer.items:
        return _empty_route_reply()
    blocks = [_item_block(item) for item in answer.items]
    shown = sum(item.citation is not None for item in answer.items)
    if shown:
        blocks.append(f"<small>{HIGHLIGHT_TEXT[min(shown, 2)]}</small>")
    return "\n\n".join(blocks)


def _item_block(item: AnswerItem) -> str:
    heading = f"{_badge(item.status)} **{display_name(item.clause_type)}**"
    if item.withheld:
        return f"{heading}\n\n{WITHHELD_TEXT}"
    if item.citation is None:
        return f"{heading}\n\n{NOT_FOUND_TEXT}"
    citation = item.citation
    meta = f"{_percent(citation.conf)} confidence"
    if citation.page is not None:
        meta = f"Page {citation.page} · {meta}"
    return f"{heading}\n{blockquote(citation.quote)}\n\n<small>{meta}</small>"


def _empty_route_reply() -> str:
    topics = "\n".join(
        f"- **{topic}:** {', '.join(display_name(name) for name in names)}"
        for topic, names in CLAUSE_GROUPS.items()
    )
    examples = " or ".join(f"\u201c{q}\u201d" for q in SUGGESTED_QUESTIONS[:EXAMPLE_QUESTIONS])
    return f"{EMPTY_ROUTE_TEXT}\n\nI can answer questions about:\n\n{topics}\n\nFor example: {examples}"


def clause_rows(
    cache: DocumentCache, tau_low: float, tau_high: float, status: Status | None = None
) -> list[list[str]]:
    """Return one row per clause type in topic order, optionally only those with a status."""
    rows = []
    for topic, names in CLAUSE_GROUPS.items():
        for position, name in enumerate(names):
            candidates = cache.clauses.get(name, [])
            row_status = (
                tier(candidates[0].conf, tau_low, tau_high) if candidates else "not_found"
            )
            if status is not None and row_status != status:
                continue
            label = topic if position == 0 or status is not None else ""
            rows.append(_row(label, name, row_status, candidates))
    return rows


def _row(topic: str, name: str, status: Status, candidates: list) -> list[str]:
    if not candidates:
        return [topic, display_name(name), _badge(status), _percent(0.0), NO_PAGE, ""]
    top = candidates[0]
    page = NO_PAGE if top.page is None else str(top.page)
    return [
        topic,
        display_name(name),
        _badge(status),
        _percent(top.conf),
        page,
        _short_passage(top.text),
    ]


def _short_passage(text: str) -> str:
    passage = " ".join(text.split())
    if len(passage) <= TABLE_PASSAGE_CHARS:
        return passage
    return passage[: TABLE_PASSAGE_CHARS - len(ELLIPSIS)] + ELLIPSIS


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
        label = f"Page {first}" if first == last else f"Pages {first}–{last}"
    else:
        lo, hi = _text_window(cache.text, citation)
        label = EXCERPT_LABEL if citation else TEXT_START_LABEL
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
        (text[citation.start : citation.end], clause_type and display_name(clause_type)),
        (text[citation.end : hi], None),
    ]
    return [part for part in parts if part[0]]


def technical_markdown(settings: Settings, route_kind: str) -> str:
    """Return the technical facts behind the answers as a short Markdown list."""
    return "\n".join(
        [
            f"- **Extraction model:** `{settings.extractor_name}`",
            f"- **Question matching:** {_matching(settings, route_kind)}",
            f"- **Found from:** {_percent(settings.tau_high)} confidence ({settings.tau_source})",
            f"- **Verify from:** {_percent(settings.tau_low)} confidence ({settings.tau_source})",
        ]
    )


def _matching(settings: Settings, route_kind: str) -> str:
    if route_kind == "llm":
        backend = "Ollama" if settings.llm_backend == "ollama" else "an OpenAI-compatible API"
        return f"`{settings.llm_model}` through {backend}"
    if route_kind == "embedding":
        return f"embedding model `{settings.embedding_model_id}`"
    return route_kind


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
