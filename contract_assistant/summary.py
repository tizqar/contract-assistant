"""A summary of a contract written from its verified clause quotes, and the check on it."""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from pydantic import Field

from contract_assistant import generation
from contract_assistant.answer import STATUS_WORDS, AnswerItem, compose, verify_quotes
from contract_assistant.explain import DEFAULT_LANGUAGE, LANGUAGES, UnknownLanguageError
from contract_assistant.generation import (
    MS_PER_S,
    GeneratedRecord,
    GeneratedStore,
    Generation,
    Request,
    complete_sentences,
    faithful_check,
    run,
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
# A summary in another language has a store kind of its own and this sentence in its prompt.
LANGUAGE_INSTRUCTION = (
    " Write the summary in {language}; keep every quote in the language of the contract."
)
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
    language: str = DEFAULT_LANGUAGE
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


def system_prompt(language: str = DEFAULT_LANGUAGE) -> str:
    """Return the system prompt, asking for another language than English when given one."""
    if language not in LANGUAGES:
        raise UnknownLanguageError(
            f"unknown language {language!r}; use one of {', '.join(LANGUAGES)}"
        )
    if language == DEFAULT_LANGUAGE:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + LANGUAGE_INSTRUCTION.format(language=LANGUAGES[language])


def summary_request(
    doc_id: str, items: Sequence[AnswerItem], language: str = DEFAULT_LANGUAGE
) -> Request:
    """Return the request that writes the summary of the items, with its faithfulness check."""
    items = cited(items)
    return Request(
        system_prompt(language),
        _user_message(items),
        MAX_TOKENS,
        faithful_check(quotes_of(items), SOURCE),
        SOURCE,
        f"summary of {doc_id}",
    )


def summary_from(
    result: Generation,
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    language: str = DEFAULT_LANGUAGE,
) -> Summary:
    """Return the Summary that a generation outcome gives for the items."""
    items = cited(items)
    return Summary(
        doc_id=doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        language=language,
        items=items,
        text=result.text,
        reason=result.reason,
        unfaithful_spans=unfaithful_spans(result.reply, quotes_of(items)),
        attempts=result.attempts,
        latency_ms=result.latency_ms,
    )


def cited(items: Sequence[AnswerItem]) -> list[AnswerItem]:
    """Return the items that kept a citation; raise when none did."""
    kept = [item for item in items if item.citation is not None]
    if not kept:
        raise ValueError("no items to summarize")
    return kept


def request_summary(
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    attempts: int = 1,
    language: str = DEFAULT_LANGUAGE,
) -> Summary:
    """Ask the model to summarize the items, up to attempts times while the reply is withheld."""
    result = run(client, summary_request(doc_id, items, language), attempts)
    return summary_from(result, doc_id, items, client, tau_low, tau_high, language)


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
    """Summaries saved as JSON files, one per document, model, thresholds, language and version."""

    def __init__(
        self, location: Path | generation.RecordBackend, language: str = DEFAULT_LANGUAGE
    ):
        kind = KIND if language == DEFAULT_LANGUAGE else f"{KIND}-{language}"
        super().__init__(location, kind, PROMPT_VERSION, Summary)
        self.language = language
