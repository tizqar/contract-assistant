"""Feedback on answers, kept as one JSON line per verdict for the evaluation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

FEEDBACK_FILE = "feedback.jsonl"


class Feedback(BaseModel):
    """One verdict on one answer."""

    doc_id: str
    question: str
    route_kind: str
    clause_types: list[str]
    liked: bool
    comment: str | None = None
    at: str


def record_feedback(path: Path, verdict: Feedback) -> Path:
    """Append a verdict to the feedback file, creating it when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(verdict.model_dump_json() + "\n")
    return path


def read_feedback(path: Path) -> list[Feedback]:
    """Read every verdict of the feedback file, oldest first; an absent file is empty."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Feedback.model_validate_json(line) for line in lines if line.strip()]


def now() -> str:
    """Return the current time in ISO form, in UTC."""
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def summary(verdicts: list[Feedback]) -> dict[str, int]:
    """Return the counts of liked and disliked verdicts."""
    liked = sum(verdict.liked for verdict in verdicts)
    return {"liked": liked, "disliked": len(verdicts) - liked}


def dump_json(verdicts: list[Feedback]) -> str:
    """Return the verdicts as a JSON list."""
    return json.dumps([verdict.model_dump() for verdict in verdicts], indent=2)
