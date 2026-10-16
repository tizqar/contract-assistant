from typing import Literal

from contract_assistant.config import Settings
from contract_assistant.store import (
    CachedCandidate,
    DocumentCache,
    ModelInfo,
    SourceInfo,
    doc_id_of,
)


def make_cache(
    text: str,
    settings: Settings,
    clauses: dict[str, list[tuple[str, float]]] | None = None,
    origin: Literal["extractor", "harness"] = "extractor",
) -> DocumentCache:
    """Build a valid txt cache; each candidate is found in text by its string."""
    candidates = {
        clause: [_candidate(text, answer, conf) for answer, conf in answers]
        for clause, answers in (clauses or {}).items()
    }
    return DocumentCache(
        doc_id=doc_id_of(text),
        source=SourceInfo(filename="test.txt", kind="txt", n_chars=len(text), n_pages=None),
        model=ModelInfo(
            path="models/test",
            device="cpu",
            origin=origin,
            **settings.decode_settings(),
        ),
        page_starts=[],
        text=text,
        clauses=candidates,
        timing={},
    )


def _candidate(text: str, answer: str, conf: float) -> CachedCandidate:
    start = text.find(answer)
    return CachedCandidate(
        text=answer, start=start, end=start + len(answer), page=None, conf=conf
    )
