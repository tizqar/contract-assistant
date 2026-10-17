"""Tiered answers with verbatim citations, and their Markdown rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from contract_assistant.router.base import RouteResult
from contract_assistant.store import DocumentCache

Status = Literal["found", "check", "not_found"]

MAX_CHAT_QUOTE_CHARS = 400
ELLIPSIS = "…"
MARKDOWN_SPECIAL = "\\`*_#><[]|"
STATUS_WORDS: dict[Status, str] = {
    "found": "found",
    "check": "possible, verify",
    "not_found": "not found",
}
EMPTY_ROUTE_MESSAGE = "I can only answer about these clause types:"


class Citation(BaseModel):
    """A quoted span of the document."""

    quote: str
    page: int | None
    conf: float
    start: int
    end: int


class AnswerItem(BaseModel):
    """The outcome for one clause type."""

    clause_type: str
    status: Status
    best_conf: float
    citation: Citation | None
    withheld: bool = False


class Answer(BaseModel):
    """The items for one question over one document."""

    question: str
    route: RouteResult
    tau_low: float
    tau_high: float
    items: list[AnswerItem]
    dropped_quotes: int


def _check_thresholds(tau_low: float, tau_high: float) -> None:
    if tau_low > tau_high:
        raise ValueError(f"tau_low {tau_low} is above tau_high {tau_high}")


def tier(conf: float, tau_low: float, tau_high: float) -> Status:
    """Return the status of a top-candidate confidence."""
    _check_thresholds(tau_low, tau_high)
    if conf >= tau_high:
        return "found"
    if conf >= tau_low:
        return "check"
    return "not_found"


def compose(
    cache: DocumentCache, clause_types: list[str], tau_low: float, tau_high: float
) -> list[AnswerItem]:
    """Build one item per clause type from the top candidate in the cache."""
    _check_thresholds(tau_low, tau_high)
    return [_item(name, cache, tau_low, tau_high) for name in clause_types]


def _item(name: str, cache: DocumentCache, tau_low: float, tau_high: float) -> AnswerItem:
    candidates = cache.clauses[name]
    if not candidates:
        return AnswerItem(clause_type=name, status="not_found", best_conf=0.0, citation=None)
    top = candidates[0]
    status = tier(top.conf, tau_low, tau_high)
    citation = None
    if status != "not_found":
        citation = Citation(
            quote=top.text, page=top.page, conf=top.conf, start=top.start, end=top.end
        )
    return AnswerItem(clause_type=name, status=status, best_conf=top.conf, citation=citation)


def verify_quotes(items: list[AnswerItem], text: str) -> tuple[list[AnswerItem], int]:
    """Withhold citations that are not verbatim in text; return the items and the count."""
    verified = []
    dropped = 0
    for item in items:
        if item.citation is None or _is_verbatim(item.citation, text):
            verified.append(item)
            continue
        verified.append(item.model_copy(update={"citation": None, "withheld": True}))
        dropped += 1
    return verified, dropped


def _is_verbatim(citation: Citation, text: str) -> bool:
    return text[citation.start : citation.end] == citation.quote


def answer(
    question: str, route: RouteResult, cache: DocumentCache, tau_low: float, tau_high: float
) -> Answer:
    """Compose the items for a route and verify every quote against the document."""
    items = compose(cache, route.clause_types, tau_low, tau_high)
    items, dropped = verify_quotes(items, cache.text)
    return Answer(
        question=question,
        route=route,
        tau_low=tau_low,
        tau_high=tau_high,
        items=items,
        dropped_quotes=dropped,
    )


def render_markdown(answer: Answer, clause_type_names: Sequence[str]) -> str:
    """Render an answer as Markdown; an empty route lists the supported clause types."""
    if not answer.items:
        names = "\n".join(f"- {name}" for name in clause_type_names)
        return f"{EMPTY_ROUTE_MESSAGE}\n\n{names}"
    return "\n\n".join(_render_item(item) for item in answer.items)


def _render_item(item: AnswerItem) -> str:
    word = STATUS_WORDS[item.status]
    heading = f"**{item.clause_type}** — "
    if item.withheld:
        return f"{heading}{word} · quote withheld (failed verification)"
    if item.citation is None:
        return f"{heading}{word} (checked; best candidate {item.best_conf:.2f})"
    page = "n/a" if item.citation.page is None else item.citation.page
    meta = f"{word} · p. {page} · confidence {item.citation.conf:.2f}"
    return f"{heading}{meta}\n{blockquote(item.citation.quote)}"


def blockquote(quote: str) -> str:
    """Return a quote cut to the chat limit, escaped, as a Markdown blockquote."""
    if len(quote) > MAX_CHAT_QUOTE_CHARS:
        quote = quote[: MAX_CHAT_QUOTE_CHARS - len(ELLIPSIS)] + ELLIPSIS
    # Indentation would turn a line into a Markdown code block.
    lines = [line.strip() for line in _escape(quote).split("\n")]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _escape(text: str) -> str:
    return "".join(f"\\{char}" if char in MARKDOWN_SPECIAL else char for char in text)
