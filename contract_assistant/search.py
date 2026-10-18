"""Passage search: the parts of a document closest to a question."""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from collections import OrderedDict

import numpy as np

from contract_assistant.answer import Passage
from contract_assistant.router.embedding import Encoder
from contract_assistant.store import DocumentCache

logger = logging.getLogger(__name__)

MAX_PASSAGES = 3
MIN_PASSAGE_SCORE = 0.55  # chosen by hand on the demo contracts
MIN_PASSAGE_CHARS = 200
MAX_PASSAGE_CHARS = 600
MIN_INDEXED_CHARS = 40
MAX_CACHED_DOCUMENTS = 4

BLOCK = re.compile(r"\S(?:.*?\S)?(?=[ \t]*\n[ \t]*\n|\s*\Z)", re.DOTALL)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])[ \t]+|\n")

Span = tuple[int, int]


def split_passages(text: str) -> list[Span]:
    """Offsets of passages of 200-600 characters that never cross a blank line."""
    spans: list[Span] = []
    for block in BLOCK.finditer(text):
        spans.extend(_group(text, _sentences(text, block.start(), block.end())))
    return spans


def _sentences(text: str, start: int, end: int) -> list[Span]:
    spans: list[Span] = []
    position = start
    for match in SENTENCE_BREAK.finditer(text, start, end):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, end))
    return [_trim(text, s, e) for s, e in spans if text[s:e].strip()]


def _trim(text: str, start: int, end: int) -> Span:
    while text[start].isspace():
        start += 1
    while text[end - 1].isspace():
        end -= 1
    return start, end


def _group(text: str, sentences: list[Span]) -> list[Span]:
    passages: list[Span] = []
    current: Span | None = None
    for start, end in sentences:
        if end - start > MAX_PASSAGE_CHARS:
            passages.extend(_flush(current))
            passages.extend(_cut_long(text, start, end))
            current = None
            continue
        if current is not None and end - current[0] > MAX_PASSAGE_CHARS:
            passages.append(current)
            current = None
        current = (start if current is None else current[0], end)
        if current[1] - current[0] >= MIN_PASSAGE_CHARS:
            passages.append(current)
            current = None
    passages.extend(_flush(current))
    return _merge_short_tail(passages)


def _flush(span: Span | None) -> list[Span]:
    return [] if span is None else [span]


def _cut_long(text: str, start: int, end: int) -> list[Span]:
    pieces: list[Span] = []
    while end - start > MAX_PASSAGE_CHARS:
        limit = start + MAX_PASSAGE_CHARS
        cut = text.rfind(" ", start, limit)
        if cut <= start:
            cut = limit
        pieces.append(_trim(text, start, cut))
        start = cut
        while text[start].isspace():
            start += 1
    pieces.append((start, end))
    return pieces


def _merge_short_tail(passages: list[Span]) -> list[Span]:
    if len(passages) < 2:
        return passages
    (start, _), (_, end) = passages[-2], passages[-1]
    short = passages[-1][1] - passages[-1][0] < MIN_PASSAGE_CHARS
    if short and end - start <= MAX_PASSAGE_CHARS:
        return [*passages[:-2], (start, end)]
    return passages


class PassageSearch:
    """Finds the passages of a document closest to a question."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.documents: OrderedDict[str, tuple[list[Span], np.ndarray]] = OrderedDict()

    def search(self, cache: DocumentCache, question: str) -> list[Passage]:
        """Up to MAX_PASSAGES non-overlapping passages scoring at least MIN_PASSAGE_SCORE."""
        spans, vectors = self._index(cache)
        if not spans:
            return []
        query = self.encoder.encode_queries([question])[0]
        scores = vectors @ query
        chosen: list[Span] = []
        passages: list[Passage] = []
        for index in np.argsort(-scores):
            score = float(scores[index])
            if score < MIN_PASSAGE_SCORE or len(passages) == MAX_PASSAGES:
                break
            start, end = spans[index]
            if any(
                start < other_end and other_start < end for other_start, other_end in chosen
            ):
                continue
            chosen.append((start, end))
            passages.append(
                Passage(
                    quote=cache.text[start:end],
                    page=_page_of(cache.page_starts, start),
                    score=score,
                    start=start,
                    end=end,
                )
            )
        return passages

    def _index(self, cache: DocumentCache) -> tuple[list[Span], np.ndarray]:
        if cache.doc_id in self.documents:
            self.documents.move_to_end(cache.doc_id)
            return self.documents[cache.doc_id]
        spans = [(s, e) for s, e in split_passages(cache.text) if e - s >= MIN_INDEXED_CHARS]
        vectors = (
            self.encoder.encode_documents([cache.text[s:e] for s, e in spans])
            if spans
            else np.empty((0, 0))
        )
        logger.info("indexed %d passages of %s", len(spans), cache.doc_id)
        self.documents[cache.doc_id] = (spans, vectors)
        if len(self.documents) > MAX_CACHED_DOCUMENTS:
            self.documents.popitem(last=False)
        return spans, vectors


def _page_of(page_starts: list[int], offset: int) -> int | None:
    if not page_starts:
        return None
    return max(bisect_right(page_starts, offset), 1)
