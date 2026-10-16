"""Cache schema and the store that reads and writes cached extraction results."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_assistant.config import Settings

SCHEMA_VERSION = 1
DOC_ID_PREFIX = "sha256:"
JSON_INDENT = 1


class CachedCandidate(BaseModel):
    """One answer candidate with its offsets in the document text."""

    text: str
    start: int
    end: int
    page: int | None
    conf: float


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
    cache = DocumentCache.model_validate_json(path.read_text(encoding="utf-8"))
    _check_settings(cache, settings)
    _check_integrity(cache)
    return cache


def _check_settings(cache: DocumentCache, settings: Settings) -> None:
    expected = settings.decode_settings()
    actual = cache.model.model_dump()
    differing = sorted(key for key in expected if expected[key] != actual[key])
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

    def load(self, doc_id: str, settings: Settings) -> DocumentCache | None:
        """Return the cache for doc_id from cache_dir or demo_dir, or None."""
        path = self.cache_dir / f"{doc_id.removeprefix(DOC_ID_PREFIX)}.json"
        if path.is_file():
            return read_cache(path, settings)
        for entry in _scan(self.demo_dir):
            if entry.doc_id == doc_id:
                return read_cache(entry.path, settings)
        return None

    def load_path(self, path: Path, settings: Settings) -> DocumentCache:
        """Read a cache from an explicit file."""
        return read_cache(path, settings)

    def list(self) -> list[CacheEntry]:
        """List the caches in cache_dir, then demo_dir."""
        return [*_scan(self.cache_dir), *_scan(self.demo_dir)]


def _scan(directory: Path) -> list[CacheEntry]:
    entries = []
    for path in sorted(directory.glob("*.json")):
        cache = DocumentCache.model_validate_json(path.read_text(encoding="utf-8"))
        entries.append(CacheEntry(cache.doc_id, cache.source, path))
    return entries
