"""Run the extractor over the clause types of one document."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from contract_assistant.clause_types import ClauseType, app_clause_types
from contract_assistant.config import ROOT, Settings
from contract_assistant.extract.qa_core import Candidate, QAModel
from contract_assistant.pdf import Document, page_of
from contract_assistant.store import (
    CachedCandidate,
    DocumentCache,
    ModelInfo,
    SourceInfo,
    doc_id_of,
)

ProgressCallback = Callable[[int, int, str], None]

TIMING_DECIMALS = 1

logger = logging.getLogger(__name__)


class QAFunction(Protocol):
    """Extractive QA over a whole document, returning ranked candidates."""

    def __call__(
        self,
        question: str,
        context: str,
        *,
        max_seq_len: int,
        doc_stride: int,
        max_answer_len: int,
        top_k: int,
        n_best_size: int,
    ) -> list[Candidate]: ...


class ModelNotFoundError(FileNotFoundError):
    """Raised when the extractor model folder does not exist."""


class Extractor:
    """Asks every clause type's question and collects the n-best answers."""

    def __init__(
        self,
        qa: QAFunction,
        clause_types: list[ClauseType],
        settings: Settings,
        device: str,
    ):
        self.qa = qa
        self.clause_types = clause_types
        self.settings = settings
        self.device = device

    @classmethod
    def from_settings(cls, settings: Settings) -> Extractor:
        """Load the model named by the settings; raises ModelNotFoundError if absent."""
        path = settings.extractor_path
        if not path.is_dir():
            raise ModelNotFoundError(f"model folder not found: {path}")
        qa = QAModel(
            str(path),
            device=settings.device,
            batch_size=settings.batch_size,
            max_seq_len=settings.max_seq_len,
            doc_stride=settings.doc_stride,
        )
        clause_types = app_clause_types(settings)
        return cls(qa, clause_types, settings, device=str(qa.device))

    def extract(self, doc: Document, progress: ProgressCallback | None = None) -> DocumentCache:
        """Extract the candidates of every clause type from doc."""
        started = time.perf_counter()
        clauses = self._extract_types(doc, self.clause_types, progress)
        elapsed = time.perf_counter() - started
        logger.info("extracted %s: %d chars in %.1f s", doc.filename, len(doc.text), elapsed)
        return DocumentCache(
            doc_id=doc_id_of(doc.text),
            source=SourceInfo(
                filename=doc.filename,
                kind=doc.kind,
                n_chars=len(doc.text),
                n_pages=len(doc.page_starts) or None,
            ),
            model=ModelInfo(
                **self.settings.decode_settings(),
                path=_model_path(self.settings.extractor_path),
                device=self.device,
                origin="extractor",
            ),
            page_starts=doc.page_starts,
            text=doc.text,
            clauses=clauses,
            timing={"extract_s": round(elapsed, TIMING_DECIMALS)},
        )

    def extend(
        self, cache: DocumentCache, progress: ProgressCallback | None = None
    ) -> DocumentCache:
        """Add the clause types missing from cache; return cache itself if none is missing."""
        missing = [c for c in self.clause_types if c.name not in cache.clauses]
        if not missing:
            return cache
        started = time.perf_counter()
        doc = Document(
            cache.text,
            cache.page_starts,
            cache.source.filename,
            cache.source.kind,
            cache.doc_id,
        )
        added = self._extract_types(doc, missing, progress)
        elapsed = time.perf_counter() - started
        logger.info(
            "extended %s: %d clause types in %.1f s",
            cache.source.filename,
            len(missing),
            elapsed,
        )
        timing = {
            **cache.timing,
            "extract_s": round(cache.timing.get("extract_s", 0.0) + elapsed, TIMING_DECIMALS),
        }
        return cache.model_copy(
            update={"clauses": {**cache.clauses, **added}, "timing": timing}
        )

    def _extract_types(
        self,
        doc: Document,
        clause_types: list[ClauseType],
        progress: ProgressCallback | None,
    ) -> dict[str, list[CachedCandidate]]:
        clauses = {}
        for done, clause_type in enumerate(clause_types, start=1):
            clauses[clause_type.name] = self._candidates(doc, clause_type)
            if progress:
                progress(done, len(clause_types), clause_type.name)
        return clauses

    def _candidates(self, doc: Document, clause_type: ClauseType) -> list[CachedCandidate]:
        found = self.qa(
            clause_type.cuad_question,
            doc.text,
            max_seq_len=self.settings.max_seq_len,
            doc_stride=self.settings.doc_stride,
            max_answer_len=self.settings.max_answer_len,
            top_k=self.settings.top_k,
            n_best_size=self.settings.n_best_size,
        )
        return [
            CachedCandidate(
                text=c.text,
                start=c.start,
                end=c.end,
                page=page_of(doc, c.start),
                conf=c.conf,
            )
            for c in found
        ]


def _model_path(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return str(path)
