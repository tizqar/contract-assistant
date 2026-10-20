"""A summary of a contract written from its verified clause quotes, and the check on it."""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from pydantic import Field

from contract_assistant import generation
from contract_assistant.answer import STATUS_WORDS, AnswerItem, compose, verify_quotes
from contract_assistant.generation import (
    MS_PER_S,
    GeneratedRecord,
    GeneratedStore,
    complete_sentences,
    faithfulness_reason,
    generate,
)
from contract_assistant.phrase import unfaithful_spans
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DocumentCache

__all__ = ["complete_sentences"]

logger = logging.getLogger(__name__)

PROMPT_VERSION = 3
MAX_TOKENS = 600
# The app and the CLI ask again once when a reply is withheld, naming the reason; the
# evaluation asks once.
APP_ATTEMPTS = 2
MAX_SENTENCES = 8
# A longer quote is cut in the request; the check still compares against the whole quote.
MAX_QUOTE_CHARS = 600
CUT_MARK = "(cut)"
SUMMARY_DIR = "summaries"
KIND = "summary"
SOURCE = "contract"
MISQUOTE_REASON = generation.MISQUOTE_REASON.format(source=SOURCE)
NUMBERS_REASON = generation.NUMBERS_REASON.format(source=SOURCE, numbers="{numbers}")
EMPTY_REASON = generation.EMPTY_REASON
SYSTEM_PROMPT = (
    "Write a summary of a contract for a lawyer from the clause items given: at most "
    f"{MAX_SENTENCES} sentences in one or two short paragraphs, and nothing else, "
    "no headings, lists or notes. "
    "Cover the parties and dates first, then the main rights, obligations and restrictions, "
    "then termination, liability and governing law, as far as the items allow. "
    "Quote the contract only with the quotes given, word for word, in double quotes, "
    "and use double quotes for nothing else. "
    "Copy every number, date and amount exactly as the items state it, and do not add "
    "facts, parties, dates, amounts or section numbers that the items do not state. "
    "Where a statement rests on an item marked 'possible, verify', say that it is to be "
    "verified. Reply with the summary only."
)


class Summary(GeneratedRecord):
    """The summary of one document by one model under one pair of thresholds."""

    prompt_version: int = PROMPT_VERSION
    items: list[AnswerItem]
    text: str | None
    reason: str | None = None
    unfaithful_spans: list[str] = Field(default_factory=list)


def summary_items(
    cache: DocumentCache,
    clause_types: Sequence[str],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> list[AnswerItem]:
    """Return the found and verify items of the cache with a verified quote, in the given order."""
    names = [name for name in clause_types if name in cache.clauses]
    items = compose(cache, names, tau_low, tau_high, unvalidated)
    items, _ = verify_quotes(items, cache.text)
    return [item for item in items if item.citation is not None]


def quotes_of(items: Sequence[AnswerItem]) -> list[str]:
    """Return the verified quotes of the items, in order."""
    return [item.citation.quote for item in items if item.citation is not None]


def request_summary(
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    attempts: int = 1,
) -> Summary:
    """Ask the model to summarize the items, up to attempts times while the reply is withheld."""
    items = [item for item in items if item.citation is not None]
    if not items:
        raise ValueError("no items to summarize")
    quotes = quotes_of(items)
    result = generate(
        client,
        SYSTEM_PROMPT,
        _user_message(items),
        MAX_TOKENS,
        lambda text: faithfulness_reason(text, quotes, SOURCE),
        source=SOURCE,
        attempts=attempts,
        label=f"summary of {doc_id}",
    )
    return Summary(
        doc_id=doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        items=items,
        text=result.text,
        reason=result.reason,
        unfaithful_spans=unfaithful_spans(result.reply, quotes),
        attempts=result.attempts,
        latency_ms=result.latency_ms,
    )


def _user_message(items: Sequence[AnswerItem]) -> str:
    return "Items:\n" + "\n".join(_describe(item) for item in items)


def _describe(item: AnswerItem) -> str:
    quote = " ".join(item.citation.quote.split()) if item.citation else ""
    cut = ""
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0]
        cut = f" {CUT_MARK}"
    return f'- {item.clause_type} ({STATUS_WORDS[item.status]}): "{quote}"{cut}'


def render_markdown(summary: Summary) -> str:
    """Render a summary for the command line: the text, its sources and the model."""
    if summary.text is None:
        lines = [f"Summary withheld: {summary.reason}."]
        lines.extend(f'- "{span}"' for span in summary.unfaithful_spans)
        return "\n".join(lines)
    based = ", ".join(
        f"{item.clause_type} ({STATUS_WORDS[item.status]})" for item in summary.items
    )
    seconds = summary.latency_ms / MS_PER_S
    return f"{summary.text}\n\nBased on: {based}\n\nmodel: {summary.model} · {seconds:.1f} s"


class SummaryStore(GeneratedStore[Summary]):
    """Summaries saved as JSON files, one per document, model, thresholds and prompt version."""

    def __init__(self, directory: Path):
        super().__init__(directory, KIND, PROMPT_VERSION, Summary)
