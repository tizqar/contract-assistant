"""Cache schema and the store that reads and writes cached extraction results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from contract_assistant.config import Settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DOC_ID_PREFIX = "sha256:"
JSON_INDENT = 1
HARNESS_IMPL = 3
# The harness stamp does not record n_best_size; the harness always uses its default.
HARNESS_N_BEST_SIZE = 20
HARNESS_MODEL_PATTERN = re.compile(r"runs/([^/]+)/final")


class CachedCandidate(BaseModel):
    """One answer candidate with its offsets in the document text."""

    text: str
    start: int
    end: int
    page: int | None
    conf: float
    # Set by the reader's confirmation, never by the model: such a candidate is never capped.
    confirmed: bool = False


class ModelInfo(BaseModel):
    """The extractor run and decode settings a cache was written with."""

    name: str
    path: str
    max_seq_len: int
    doc_stride: int
    max_answer_len: int
    top_k: int
    n_best_size: int
    device: str
    origin: Literal["extractor", "harness"]


class SourceInfo(BaseModel):
    """The file a cache was extracted from."""

    filename: str
    kind: Literal["pdf", "txt"]
    n_chars: int
    n_pages: int | None


class DocumentCache(BaseModel):
    """The n-best candidates for every clause type of one document."""

    schema_version: Literal[1] = SCHEMA_VERSION
    doc_id: str
    source: SourceInfo
    model: ModelInfo
    page_starts: list[int]
    text: str
    clauses: dict[str, list[CachedCandidate]]
    timing: dict[str, float]


@dataclass(frozen=True)
class CacheEntry:
    """A cache file and what it was extracted from."""

    doc_id: str
    source: SourceInfo
    path: Path


class CacheMismatchError(ValueError):
    """Raised when a cache was written with other model or decode settings."""


class CacheIntegrityError(ValueError):
    """Raised when a cache is internally inconsistent."""


def doc_id_of(text: str) -> str:
    """Return the document id: sha256 of the UTF-8 text."""
    return DOC_ID_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_cache(cache: DocumentCache, path: Path) -> None:
    """Write a cache to path atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cache.model_dump_json(indent=JSON_INDENT))
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_cache(path: Path, settings: Settings) -> DocumentCache:
    """Read a cache and check it against the settings and its own contents."""
    try:
        cache = DocumentCache.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise CacheIntegrityError(f"not a valid cache: {path.name}") from error
    _check_settings(cache, settings)
    _check_integrity(cache)
    return cache


def _differing_keys(settings: Settings, actual: dict) -> list[str]:
    expected = settings.decode_settings()
    return sorted(key for key in expected if expected[key] != actual[key])


def _check_settings(cache: DocumentCache, settings: Settings) -> None:
    differing = _differing_keys(settings, cache.model.model_dump())
    if differing:
        raise CacheMismatchError(
            f"cache was written with other settings: {', '.join(differing)}"
        )


def _check_integrity(cache: DocumentCache) -> None:
    if doc_id_of(cache.text) != cache.doc_id:
        raise CacheIntegrityError("text does not match doc_id")
    for clause, candidates in cache.clauses.items():
        for candidate in candidates:
            if cache.text[candidate.start : candidate.end] != candidate.text:
                raise CacheIntegrityError(f"offsets do not match text in clause {clause}")


class Store:
    """Caches in a writable directory plus a read-only demo directory."""

    def __init__(self, cache_dir: Path, demo_dir: Path):
        self.cache_dir = cache_dir
        self.demo_dir = demo_dir

    def save(self, cache: DocumentCache) -> Path:
        """Write the cache to cache_dir, named by its document hash."""
        path = self.cache_dir / f"{cache.doc_id.removeprefix(DOC_ID_PREFIX)}.json"
        write_cache(cache, path)
        return path

    def find(self, doc_id: str) -> Path | None:
        """Return the path of the cache for doc_id in cache_dir or demo_dir, or None."""
        path = self.cache_dir / f"{doc_id.removeprefix(DOC_ID_PREFIX)}.json"
        if path.is_file():
            return path
        return next((e.path for e in _scan(self.demo_dir) if e.doc_id == doc_id), None)

    def load(self, doc_id: str, settings: Settings) -> DocumentCache | None:
        """Return the cache for doc_id from cache_dir or demo_dir, or None."""
        path = self.find(doc_id)
        if path is None:
            return None
        cache = read_cache(path, settings)
        if cache.doc_id != doc_id:
            raise CacheIntegrityError(f"cache {path.name} belongs to {cache.doc_id}")
        return cache

    def load_path(self, path: Path, settings: Settings) -> DocumentCache:
        """Read a cache from an explicit file."""
        return read_cache(path, settings)

    def list(self) -> list[CacheEntry]:
        """List the caches in cache_dir, then demo_dir."""
        return [*_scan(self.cache_dir), *_scan(self.demo_dir)]


def _scan(directory: Path) -> list[CacheEntry]:
    entries = []
    for path in sorted(directory.glob("*.json")):
        try:
            cache = DocumentCache.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning("skipping invalid cache file: %s", path)
            continue
        entries.append(CacheEntry(cache.doc_id, cache.source, path))
    return entries


class HarnessImportError(ValueError):
    """Raised when thesis predictions cannot be imported as caches."""


def import_harness_preds(
    preds_path: Path,
    cuad_json: Path,
    settings: Settings,
    titles: Collection[str] | None = None,
) -> dict[str, DocumentCache]:
    """Import thesis predictions as caches keyed by CUAD title."""
    stamp = json.loads(preds_path.with_suffix(".config.json").read_text(encoding="utf-8"))
    decode = _stamp_decode_settings(stamp, settings)
    texts = _cuad_texts(cuad_json)
    caches = {}
    for title, preds in _read_preds(preds_path, titles):
        if title not in texts:
            raise HarnessImportError(f"contract not in CUAD json: {title}")
        caches[title] = _harness_cache(title, texts[title], preds, decode, stamp["model"])
    missing = sorted(set(titles or ()) - set(caches))
    if missing:
        raise HarnessImportError(f"no predictions for: {', '.join(missing)}")
    return caches


def _stamp_decode_settings(stamp: dict, settings: Settings) -> dict[str, str | int]:
    if stamp.get("impl") != HARNESS_IMPL:
        raise HarnessImportError(f"harness impl {stamp.get('impl')} is not {HARNESS_IMPL}")
    match = HARNESS_MODEL_PATTERN.fullmatch(str(stamp.get("model")))
    if match is None:
        raise HarnessImportError(f"cannot read run name from model {stamp.get('model')}")
    decode = {
        "name": match.group(1),
        "max_seq_len": stamp["max_seq_len"],
        "doc_stride": stamp["doc_stride"],
        "max_answer_len": stamp["max_answer_len"],
        "top_k": stamp["top_k"],
        "n_best_size": HARNESS_N_BEST_SIZE,
    }
    differing = _differing_keys(settings, decode)
    if differing:
        raise HarnessImportError(
            f"predictions were written with other settings: {', '.join(differing)}"
        )
    return decode


def _cuad_texts(cuad_json: Path) -> dict[str, str]:
    data = json.loads(cuad_json.read_text(encoding="utf-8"))["data"]
    return {entry["title"]: entry["paragraphs"][0]["context"] for entry in data}


def _read_preds(
    preds_path: Path, titles: Collection[str] | None
) -> Iterator[tuple[str, dict[str, list[list]]]]:
    with preds_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if titles is None or record["contract_id"] in titles:
                yield record["contract_id"], record["preds"]


def _harness_cache(
    title: str,
    text: str,
    preds: dict[str, list[list]],
    decode: dict[str, str | int],
    model_path: str,
) -> DocumentCache:
    return DocumentCache(
        doc_id=doc_id_of(text),
        source=SourceInfo(filename=f"{title}.txt", kind="txt", n_chars=len(text), n_pages=None),
        model=ModelInfo(**decode, path=model_path, device="unknown", origin="harness"),
        page_starts=[],
        text=text,
        clauses={
            clause: [_harness_candidate(title, clause, text, *pred) for pred in candidates]
            for clause, candidates in preds.items()
        },
        timing={},
    )


def _harness_candidate(
    title: str, clause: str, text: str, answer: str, conf: float
) -> CachedCandidate:
    # The harness does not record offsets; the first occurrence is used.
    start = text.find(answer)
    if start < 0:
        raise HarnessImportError(f"predicted text not in contract {title} ({clause})")
    return CachedCandidate(
        text=answer, start=start, end=start + len(answer), page=None, conf=conf
    )
