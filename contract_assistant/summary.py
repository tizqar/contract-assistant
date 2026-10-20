"""A summary of a contract written from its verified clause quotes, and the check on it."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Collection, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from contract_assistant.answer import STATUS_WORDS, AnswerItem, compose, verify_quotes
from contract_assistant.phrase import unfaithful_spans, unsupported_numbers
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DOC_ID_PREFIX, DocumentCache

logger = logging.getLogger(__name__)

PROMPT_VERSION = 3
MAX_TOKENS = 600
# The app and the CLI ask again once when a reply is withheld, naming the reason; the
# evaluation asks once.
APP_ATTEMPTS = 2
REJECTION = (
    "\n\nYour previous summary was rejected because {reason}. It read:\n{text}\n\n"
    "Write the summary again and follow the rules above. Copy each number with the words "
    "and digits the items use, such as 'six (6) months' or 'ten (10) years', and quote only "
    "the quotes given."
)
MAX_SENTENCES = 8
# A longer quote is cut in the request; the check still compares against the whole quote.
MAX_QUOTE_CHARS = 600
CUT_MARK = "(cut)"
SUMMARY_DIR = "summaries"
KEY_CHARS = 16
JSON_INDENT = 1
MS_PER_S = 1000
MISQUOTE_REASON = "it quotes text that is not in the contract"
NUMBERS_REASON = "it names numbers the clauses do not: {numbers}"
EMPTY_REASON = "the model gave an empty reply"
SENTENCE_END = ".!?"
CLOSERS = "\"\u201d)"
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


class Summary(BaseModel):
    """The summary of one document by one model under one pair of thresholds."""

    doc_id: str
    model: str
    tau_low: float
    tau_high: float
    prompt_version: int = PROMPT_VERSION
    items: list[AnswerItem]
    text: str | None
    reason: str | None = None
    unfaithful_spans: list[str] = []
    latency_ms: float
    attempts: int = 1


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
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    user = _user_message(items)
    quotes = quotes_of(items)
    start = time.perf_counter()
    for attempt in range(1, attempts + 1):
        text = complete_sentences(client.complete_text(SYSTEM_PROMPT, user, MAX_TOKENS).strip())
        spans = unfaithful_spans(text, quotes)
        reason = _reason(text, spans, unsupported_numbers(text, quotes))
        if reason is None:
            break
        logger.warning(
            "summary of %s by %s withheld (attempt %d): %s", doc_id, client.model, attempt, reason
        )
        user = _user_message(items) + REJECTION.format(reason=reason, text=text)
    return Summary(
        doc_id=doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        items=items,
        text=None if reason else text,
        reason=reason,
        unfaithful_spans=spans,
        latency_ms=(time.perf_counter() - start) * MS_PER_S,
        attempts=attempt,
    )


def _reason(text: str, spans: list[str], numbers: list[str]) -> str | None:
    if not text:
        return EMPTY_REASON
    if spans:
        return MISQUOTE_REASON
    if numbers:
        return NUMBERS_REASON.format(numbers=", ".join(numbers))
    return None


def complete_sentences(text: str) -> str:
    """Return text without an unfinished last sentence, as when the model ran out of tokens."""
    if not text or text[-1] in SENTENCE_END + CLOSERS:
        return text
    end = max(text.rfind(char) for char in SENTENCE_END)
    if end < 0:
        return text
    while end + 1 < len(text) and text[end + 1] in CLOSERS:
        end += 1
    return text[: end + 1]


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


class SummaryStore:
    """Summaries saved as JSON files, one per document, model, thresholds and prompt version."""

    def __init__(self, directory: Path):
        self.directory = directory

    def load(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> Summary | None:
        """Return the saved summary for these keys, or None."""
        path = self._path(doc_id, model, tau_low, tau_high)
        if not path.is_file():
            return None
        try:
            summary = Summary.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning("skipping invalid summary file: %s", path)
            return None
        keys = (summary.doc_id, summary.model, summary.tau_low, summary.tau_high)
        if keys != (doc_id, model, tau_low, tau_high):
            return None
        return summary if summary.prompt_version == PROMPT_VERSION else None

    def save(self, summary: Summary) -> Path:
        """Write the summary atomically and return its path."""
        path = self._path(summary.doc_id, summary.model, summary.tau_low, summary.tau_high)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(summary.model_dump_json(indent=JSON_INDENT))
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def _path(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> Path:
        key = json.dumps([model, tau_low, tau_high, PROMPT_VERSION]).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:KEY_CHARS]
        return self.directory / f"{doc_id.removeprefix(DOC_ID_PREFIX)}-{digest}.json"
