"""The 41 CUAD clause types, the extra ones, and their questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contract_assistant.config import Settings

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
    return _parse(entries)


def load_extra_clause_types(path: Path) -> list[ClauseType]:
    """Read the application's extra clause types in file order and check names."""
    return _parse(json.loads(path.read_text())["clause_types"])


def app_clause_types(settings: Settings) -> list[ClauseType]:
    """Return the 41 CUAD clause types followed by the extra ones."""
    cuad = load_clause_types(settings.clause_types_path)
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    cuad_names = {clause_type.name for clause_type in cuad}
    for clause_type in extra:
        if clause_type.name in cuad_names:
            raise ClauseTypeError(f"extra clause type clashes with CUAD: {clause_type.name}")
    return cuad + extra


def _parse(entries: list[dict]) -> list[ClauseType]:
    types = [ClauseType(**entry) for entry in entries]
    seen: set[str] = set()
    for clause_type in types:
        if clause_type.name in seen:
            raise ClauseTypeError(f"duplicate clause type: {clause_type.name}")
        seen.add(clause_type.name)
    return types
