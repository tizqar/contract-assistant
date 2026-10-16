"""Embedding router: clause types ranked by similarity to the question."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from contract_assistant.clause_types import ClauseType
from contract_assistant.router.base import RouteResult

EMBEDDING_MAX_TYPES = 5
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MS_PER_S = 1000


class RouterDataError(ValueError):
    """Raised when the router descriptions do not match the clause types."""


class Encoder(Protocol):
    """Turns texts into L2-normalised vectors."""

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """A sentence-transformers model that prefixes queries with the retrieval instruction."""

    def __init__(self, model_id: str):
        self.model = SentenceTransformer(model_id)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([QUERY_INSTRUCTION + text for text in texts])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class EmbeddingRouter:
    """Routes a question to the clause types whose descriptions are close to it."""

    kind = "embedding"

    def __init__(
        self,
        encoder: Encoder,
        descriptions: dict[str, list[str]],
        threshold: float,
        max_types: int,
    ):
        self.encoder = encoder
        self.threshold = threshold
        self.max_types = max_types
        self.names = list(descriptions)
        texts = [text for name in self.names for text in descriptions[name]]
        self.owners = np.array(
            [index for index, name in enumerate(self.names) for _ in descriptions[name]]
        )
        self.vectors = encoder.encode_documents(texts)

    def scores(self, question: str) -> dict[str, float]:
        """Per clause type, the highest cosine similarity over its texts."""
        query = self.encoder.encode_queries([question])[0]
        similarities = self.vectors @ query
        return {
            name: float(similarities[self.owners == index].max())
            for index, name in enumerate(self.names)
        }

    def route(self, question: str) -> RouteResult:
        """Return the types scoring at least the threshold, best first."""
        start = time.perf_counter()
        scored = sorted(self.scores(question).items(), key=lambda item: -item[1])
        chosen = [name for name, score in scored if score >= self.threshold]
        return RouteResult(
            clause_types=chosen[: self.max_types],
            kind="embedding",
            latency_ms=(time.perf_counter() - start) * MS_PER_S,
        )


def load_descriptions(clause_types: list[ClauseType], path: Path) -> dict[str, list[str]]:
    """Each type's CUAD description followed by its hand-written paraphrases."""
    paraphrases = json.loads(path.read_text(encoding="utf-8"))
    expected = {clause_type.name for clause_type in clause_types}
    unknown = sorted(set(paraphrases) - expected)
    if unknown:
        raise RouterDataError(f"unknown clause types in {path.name}: {unknown}")
    missing = sorted(expected - set(paraphrases))
    if missing:
        raise RouterDataError(f"clause types without paraphrases in {path.name}: {missing}")
    return {
        clause_type.name: [clause_type.description, *paraphrases[clause_type.name]]
        for clause_type in clause_types
    }
