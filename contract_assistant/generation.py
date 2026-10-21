"""The checked generation loop and the store that summaries, term sheets and answers share."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from contract_assistant.phrase import unfaithful_spans, unsupported_numbers
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DOC_ID_PREFIX

logger = logging.getLogger(__name__)

MS_PER_S = 1000
SENTENCE_END = ".!?"
CLOSERS = '"”)'
KEY_CHARS = 16
JSON_INDENT = 1
EMPTY_REASON = "the model gave an empty reply"
MISQUOTE_REASON = "it quotes text that is not in the {source}"
NUMBERS_REASON = "it names numbers not in the {source}: {numbers}"
REJECTION = (
    "\n\nYour previous reply was rejected because {reason}. It read:\n{text}\n\n"
    "Write it again and follow the rules above. Copy each number with the words and digits "
    "the {source} uses, such as 'six (6) months' or 'ten (10) years', and quote only the "
    "text given."
)

Check = Callable[[str], str | None]


@dataclass(frozen=True)
class Request:
    """What one checked generation needs: the prompts, the token limit, the check and labels."""

    system: str
    user: str
    max_tokens: int
    check: Check
    source: str
    label: str = "reply"
    trim: bool = True


class Partial(BaseModel):
    """The text of an attempt as it grows, and why the attempt before it was withheld."""

    text: str
    attempt: int
    previous_reason: str | None = None


class Generation(BaseModel):
    """The outcome of the loop: the accepted text, or the last reply and why it is withheld."""

    text: str | None
    reply: str
    reason: str | None = None
    attempts: int
    latency_ms: float


class GeneratedRecord(BaseModel):
    """What every saved generation carries: the keys of its store entry and its cost."""

    doc_id: str
    model: str
    tau_low: float
    tau_high: float
    prompt_version: int
    attempts: int = 1
    latency_ms: float


def generate_stream(
    client: LLMClient,
    system: str,
    user: str,
    max_tokens: int,
    check: Check,
    *,
    source: str,
    attempts: int = 1,
    label: str = "reply",
    trim: bool = True,
) -> Iterator[Partial | Generation]:
    """Yield each attempt's text as it grows, then the Generation; the reason names a retry."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    message = user
    previous: str | None = None
    start = time.perf_counter()
    for attempt in range(1, attempts + 1):
        pieces: list[str] = []
        for piece in client.stream_text(system, message, max_tokens):
            pieces.append(piece)
            yield Partial(text="".join(pieces), attempt=attempt, previous_reason=previous)
        reply = "".join(pieces).strip()
        if trim:
            reply = complete_sentences(reply)
        reason = check(reply)
        if reason is None:
            break
        logger.warning("%s by %s withheld (attempt %d): %s", label, client.model, attempt, reason)
        previous = reason
        message = user + REJECTION.format(reason=reason, text=reply, source=source)
    yield Generation(
        text=None if reason else reply,
        reply=reply,
        reason=reason,
        attempts=attempt,
        latency_ms=(time.perf_counter() - start) * MS_PER_S,
    )


def generate(
    client: LLMClient,
    system: str,
    user: str,
    max_tokens: int,
    check: Check,
    *,
    source: str,
    attempts: int = 1,
    label: str = "reply",
    trim: bool = True,
) -> Generation:
    """Ask the model up to attempts times, naming the reason while check rejects the reply."""
    stream = generate_stream(
        client,
        system,
        user,
        max_tokens,
        check,
        source=source,
        attempts=attempts,
        label=label,
        trim=trim,
    )
    outcome: Partial | Generation | None = None
    for item in stream:
        outcome = item
    if not isinstance(outcome, Generation):
        raise TypeError("the generation stream ended without an outcome")
    return outcome


def stream(client: LLMClient, request: Request, attempts: int = 1) -> Iterator[Partial | Generation]:
    """Stream a request: the growing text of each attempt, then the Generation."""
    return generate_stream(
        client,
        request.system,
        request.user,
        request.max_tokens,
        request.check,
        source=request.source,
        attempts=attempts,
        label=request.label,
        trim=request.trim,
    )


def run(client: LLMClient, request: Request, attempts: int = 1) -> Generation:
    """Run a request to its outcome."""
    return generate(
        client,
        request.system,
        request.user,
        request.max_tokens,
        request.check,
        source=request.source,
        attempts=attempts,
        label=request.label,
        trim=request.trim,
    )


def faithful_check(quotes: Sequence[str], source: str) -> Check:
    """Return the check that withholds text which misquotes or adds numbers to the quotes."""
    return lambda text: faithfulness_reason(text, quotes, source)


def faithfulness_reason(text: str, quotes: Sequence[str], source: str) -> str | None:
    """Why text is withheld against the quotes: empty, a misquote or a number not in them."""
    if not text.strip():
        return EMPTY_REASON
    if unfaithful_spans(text, quotes):
        return MISQUOTE_REASON.format(source=source)
    numbers = unsupported_numbers(text, quotes)
    if numbers:
        return NUMBERS_REASON.format(source=source, numbers=", ".join(numbers))
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


class GeneratedStore[T: GeneratedRecord]:
    """Records saved as JSON files, one per document, model, thresholds, kind and version."""

    def __init__(self, directory: Path, kind: str, version: int, record_type: type[T]):
        self.directory = directory
        self.kind = kind
        self.version = version
        self.record_type = record_type

    def load(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> T | None:
        """Return the saved record for these keys, or None."""
        path = self._path(doc_id, model, tau_low, tau_high)
        if not path.is_file():
            return None
        try:
            record = self.record_type.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning("skipping invalid %s file: %s", self.kind, path)
            return None
        keys = (record.doc_id, record.model, record.tau_low, record.tau_high)
        if keys != (doc_id, model, tau_low, tau_high):
            return None
        return record if record.prompt_version == self.version else None

    def path_of(self, record: T) -> Path:
        """Return the path the record is saved under, whether or not it exists yet."""
        return self._path(record.doc_id, record.model, record.tau_low, record.tau_high)

    def save(self, record: T) -> Path:
        """Write the record atomically and return its path."""
        path = self.path_of(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(record.model_dump_json(indent=JSON_INDENT))
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return path

    def _path(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> Path:
        key = json.dumps([self.kind, model, tau_low, tau_high, self.version]).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:KEY_CHARS]
        return self.directory / f"{doc_id.removeprefix(DOC_ID_PREFIX)}-{digest}.json"
