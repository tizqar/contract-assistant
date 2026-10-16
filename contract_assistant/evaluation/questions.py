"""The router evaluation questions and their gold clause types."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

QuestionKind = Literal["single", "compound", "out_of_scope"]
Split = Literal["dev", "test"]


class EvalQuestion(BaseModel):
    """A question with the clause types a correct router returns for it."""

    id: str
    question: str
    gold: list[str]
    kind: QuestionKind
    split: Split


def load_questions(path: Path, split: Split | None = None) -> list[EvalQuestion]:
    """Read the questions in file order, optionally only one split."""
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]
    return [q for q in questions if split is None or q.split == split]
