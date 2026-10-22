"""Every occurrence of a term in a contract's text, with its page and a snippet around it."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence

from pydantic import BaseModel

MIN_QUERY_CHARS = 2
MAX_MATCHES = 200
SNIPPET_CHARS = 60


class Match(BaseModel):
    """One occurrence: its span, its page and the text around it on one line."""

    start: int
    end: int
    page: int | None
    before: str
    text: str
    after: str


def pattern_of(query: str) -> re.Pattern[str] | None:
    """Return the case-insensitive pattern of a query, None when it is too short."""
    words = query.split()
    if len("".join(words)) < MIN_QUERY_CHARS:
        return None
    return re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)


def occurrences(
    text: str, query: str, page_starts: Sequence[int] = (), limit: int = MAX_MATCHES
) -> list[Match]:
    """Return the first `limit` occurrences of the query in the text, in document order."""
    pattern = pattern_of(query)
    if pattern is None:
        return []
    matches = []
    for found in pattern.finditer(text):
        matches.append(_match(text, found.start(), found.end(), page_starts))
        if len(matches) >= limit:
            break
    return matches


def count(text: str, query: str) -> int:
    """Return how many times the query occurs in the text."""
    pattern = pattern_of(query)
    return 0 if pattern is None else sum(1 for _ in pattern.finditer(text))


def _match(text: str, start: int, end: int, page_starts: Sequence[int]) -> Match:
    page = bisect_right(page_starts, start) if page_starts else None
    return Match(
        start=start,
        end=end,
        page=page or None,
        before=_squash(text[max(start - SNIPPET_CHARS, 0) : start]),
        text=text[start:end],
        after=_squash(text[end : end + SNIPPET_CHARS]),
    )


def _squash(fragment: str) -> str:
    return " ".join(fragment.split())
