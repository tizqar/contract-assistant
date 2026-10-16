import json

import pytest

from contract_assistant.clause_types import (
    N_CLAUSE_TYPES,
    ClauseTypeError,
    load_clause_types,
)
from contract_assistant.config import load_settings


def _entry(name: str) -> dict:
    return {
        "id": name.lower(),
        "name": name,
        "cuad_question": f'about "{name}"',
        "description": "d",
    }


def _write(tmp_path, names):
    path = tmp_path / "types.json"
    path.write_text(json.dumps({"clause_types": [_entry(n) for n in names]}))
    return path


@pytest.fixture(scope="module")
def types():
    return load_clause_types(load_settings().clause_types_path)


def test_loads_all_types(types):
    assert len(types) == N_CLAUSE_TYPES


def test_keeps_file_order(types):
    assert types[0].name == "Document Name"
    assert types[-1].name == "Third Party Beneficiary"


def test_names_are_unique(types):
    assert len({t.name for t in types}) == N_CLAUSE_TYPES


def test_question_quotes_the_name(types):
    assert all(f'"{t.name}"'.lower() in t.cuad_question.lower() for t in types)


def test_wrong_count_raises(tmp_path):
    path = _write(tmp_path, [f"T{i}" for i in range(40)])
    with pytest.raises(ClauseTypeError, match="expected 41 clause types, got 40"):
        load_clause_types(path)


def test_duplicate_name_raises(tmp_path):
    path = _write(tmp_path, [f"T{i}" for i in range(40)] + ["T0"])
    with pytest.raises(ClauseTypeError, match="duplicate clause type: T0"):
        load_clause_types(path)
