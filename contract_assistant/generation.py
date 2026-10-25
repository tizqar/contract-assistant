"""The checked generation loop and the store that summaries, term sheets and answers share."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


class RecordBackend(Protocol):
    """Where a GeneratedStore keeps its records: JSON text by kind and key."""

    def read(self, kind: str, key: str) -> str | None:
        """Return the record saved under kind and key, or None."""
        ...

    def write(self, kind: str, key: str, doc_id: str, payload: str) -> None:
        """Save a record of doc_id under kind and key, replacing an earlier one."""
        ...

    def delete_doc(self, doc_id: str) -> int:
        """Delete every record of doc_id; return how many there were."""
        ...


class FileRecords:
    """Records as JSON files in one directory, named by key; the kind is part of the key."""

    def __init__(self, directory: Path):
        self.directory = directory

    def path(self, key: str) -> Path:
        """Return the file of a key, whether or not it exists yet."""
        return self.directory / f"{key}.json"

    def read(self, kind: str, key: str) -> str | None:
        path = self.path(key)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def write(self, kind: str, key: str, doc_id: str, payload: str) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def delete_doc(self, doc_id: str) -> int:
        if not self.directory.is_dir():
            return 0
        stem = re.escape(doc_id.removeprefix(DOC_ID_PREFIX))
        name = re.compile(rf"{stem}-[0-9a-f]{{{KEY_CHARS}}}\.json")
        paths = [path for path in self.directory.glob("*.json") if name.fullmatch(path.name)]
        for path in paths:
            path.unlink(missing_ok=True)
        return len(paths)


class GeneratedStore[T: GeneratedRecord]:
    """Records of one kind and version, one per document, model and thresholds.

    They are JSON files in a directory when the location is a path, else rows of the backend
    given, such as the database's. The key of a record is the same on every backend.
    """

    def __init__(
        self, location: Path | RecordBackend, kind: str, version: int, record_type: type[T]
    ):
        if isinstance(location, str | os.PathLike):
            location = FileRecords(Path(location))
        self.records: RecordBackend = location
        self.kind = kind
        self.version = version
        self.record_type = record_type

    @property
    def directory(self) -> Path:
        """The directory of the files; only a store on files has one."""
        return self._files().directory

    def key_of(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> str:
        """Return the key of a record: the doc id without its prefix, a digest of the rest."""
        key = json.dumps([self.kind, model, tau_low, tau_high, self.version]).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:KEY_CHARS]
        return f"{doc_id.removeprefix(DOC_ID_PREFIX)}-{digest}"

    def load(self, doc_id: str, model: str, tau_low: float, tau_high: float) -> T | None:
        """Return the saved record for these keys, or None."""
        key = self.key_of(doc_id, model, tau_low, tau_high)
        payload = self.records.read(self.kind, key)
        if payload is None:
            return None
        try:
            record = self.record_type.model_validate_json(payload)
        except ValidationError:
            logger.warning("skipping invalid %s %s", self.kind, self._where(key))
            return None
        keys = (record.doc_id, record.model, record.tau_low, record.tau_high)
        if keys != (doc_id, model, tau_low, tau_high):
            return None
        return record if record.prompt_version == self.version else None

    def path_of(self, record: T) -> Path:
        """Return the path the record is saved under, whether or not it exists yet."""
        return self._files().path(self._key(record))

    def save(self, record: T) -> Path | None:
        """Save the record, replacing an earlier one; return its path when it is a file."""
        key = self._key(record)
        payload = record.model_dump_json(indent=JSON_INDENT)
        self.records.write(self.kind, key, record.doc_id, payload)
        return self.records.path(key) if isinstance(self.records, FileRecords) else None

    def _key(self, record: T) -> str:
        return self.key_of(record.doc_id, record.model, record.tau_low, record.tau_high)

    def _files(self) -> FileRecords:
        if not isinstance(self.records, FileRecords):
            raise TypeError(f"the {self.kind} records are not kept in files")
        return self.records

    def _where(self, key: str) -> str:
        if isinstance(self.records, FileRecords):
            return f"file: {self.records.path(key)}"
        return f"record: {key}"
