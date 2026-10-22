"""The reader's decisions on passages: a passage to check confirmed as found, or rejected.

A decision names a clause type and the span it is about. Applied to a cache, a confirmed span
moves to the top of its clause type with full confidence, so every status, count and flag
treats it as found; a rejected span is removed, so the next candidate, if any, takes its place.
Decisions are saved per contract under the cache folder and survive restarts.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_assistant.store import DOC_ID_PREFIX, CachedCandidate, DocumentCache

DECISIONS_DIR = "decisions"
CONFIRMED_CONF = 1.0
JSON_INDENT = 1

Verdict = Literal["confirmed", "rejected"]


class Decision(BaseModel):
    """What the reader decided about one passage of one clause type."""

    clause_type: str
    verdict: Verdict
    start: int
    end: int
    at: str


def now() -> str:
    """Return the current time as an ISO string in UTC."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


class DecisionStore:
    """The decisions of every contract, one JSON file per contract."""

    def __init__(self, directory: Path):
        self.directory = directory

    def path_of(self, doc_id: str) -> Path:
        """Return the file that holds the decisions of a contract."""
        return self.directory / f"{doc_id.removeprefix(DOC_ID_PREFIX)}.json"

    def load(self, doc_id: str) -> dict[str, Decision]:
        """Return the decisions of a contract by clause type; none when nothing was decided."""
        path = self.path_of(doc_id)
        if not path.is_file():
            return {}
        entries = json.loads(path.read_text(encoding="utf-8"))
        decisions = [Decision.model_validate(entry) for entry in entries]
        return {decision.clause_type: decision for decision in decisions}

    def decide(self, doc_id: str, decision: Decision) -> dict[str, Decision]:
        """Keep a decision, replacing an earlier one on the same clause type."""
        decisions = self.load(doc_id)
        decisions[decision.clause_type] = decision
        self._write(doc_id, decisions)
        return decisions

    def forget(self, doc_id: str, clause_type: str) -> dict[str, Decision]:
        """Drop the decision on a clause type, so the passage shows as the model found it."""
        decisions = self.load(doc_id)
        decisions.pop(clause_type, None)
        self._write(doc_id, decisions)
        return decisions

    def _write(self, doc_id: str, decisions: dict[str, Decision]) -> None:
        path = self.path_of(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [decision.model_dump() for decision in decisions.values()]
        path.write_text(json.dumps(entries, indent=JSON_INDENT), encoding="utf-8")


def apply(cache: DocumentCache, decisions: dict[str, Decision]) -> DocumentCache:
    """Return the cache as the decisions make it; the cache itself when there are none."""
    if not decisions:
        return cache
    clauses = dict(cache.clauses)
    for name, decision in decisions.items():
        candidates = list(clauses.get(name, []))
        same = [c for c in candidates if (c.start, c.end) == (decision.start, decision.end)]
        others = [c for c in candidates if (c.start, c.end) != (decision.start, decision.end)]
        if decision.verdict == "rejected":
            clauses[name] = others
            continue
        confirmed = same[0] if same else _candidate(cache, decision)
        if confirmed is not None:
            marked = confirmed.model_copy(update={"conf": CONFIRMED_CONF, "confirmed": True})
            clauses[name] = [marked, *others]
    return cache.model_copy(update={"clauses": clauses})


def _candidate(cache: DocumentCache, decision: Decision) -> CachedCandidate | None:
    """Build the candidate of a confirmed span that is no longer among the model's."""
    if not 0 <= decision.start < decision.end <= len(cache.text):
        return None
    page = bisect_right(cache.page_starts, decision.start) if cache.page_starts else None
    return CachedCandidate(
        text=cache.text[decision.start : decision.end],
        start=decision.start,
        end=decision.end,
        page=page or None,
        conf=CONFIRMED_CONF,
        confirmed=True,
    )
