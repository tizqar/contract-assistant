"""The 41 CUAD clause types and their questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

N_CLAUSE_TYPES = 41


@dataclass(frozen=True)
class ClauseType:
    """One CUAD clause type and the exact question the extractor was trained on."""

    id: str
    name: str
    cuad_question: str
    description: str


class ClauseTypeError(ValueError):
    """Raised when the clause type file is not a valid CUAD list."""


def load_clause_types(path: Path) -> list[ClauseType]:
    """Read the clause types in file order and check count and names."""
    entries = json.loads(path.read_text())["clause_types"]
    if len(entries) != N_CLAUSE_TYPES:
        raise ClauseTypeError(f"expected {N_CLAUSE_TYPES} clause types, got {len(entries)}")
    types = [ClauseType(**entry) for entry in entries]
    seen: set[str] = set()
    for clause_type in types:
        if clause_type.name in seen:
            raise ClauseTypeError(f"duplicate clause type: {clause_type.name}")
        seen.add(clause_type.name)
    return types
