import json
import re
from dataclasses import replace

import pytest

from contract_assistant.clause_types import (
    N_CLAUSE_TYPES,
    ClauseTypeError,
    app_clause_types,
    load_clause_types,
    load_extra_clause_types,
)
from contract_assistant.config import Settings, load_settings

CUAD_QUESTION = re.compile(
    r'Highlight the parts \(if any\) of this contract related to "(?P<name>.+)" '
    r"that should be reviewed by a lawyer\. Details: (?P<details>.+)"
)
N_EXTRA_CLAUSE_TYPES = 8


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


@pytest.fixture(scope="module")
def extras():
    return load_extra_clause_types(load_settings().extra_clause_types_path)


def test_extra_file_loads_eight_types(extras):
    assert len(extras) == N_EXTRA_CLAUSE_TYPES


def test_extra_names_are_unique(extras):
    assert len({t.name for t in extras}) == N_EXTRA_CLAUSE_TYPES


def test_extra_names_do_not_clash_with_cuad(types, extras):
    assert not {t.name for t in types} & {t.name for t in extras}


def test_extra_questions_follow_the_cuad_template(extras):
    for clause_type in extras:
        match = CUAD_QUESTION.fullmatch(clause_type.cuad_question)
        assert match
        assert match["name"] == clause_type.name
        assert match["details"] == clause_type.description


def test_extra_loader_accepts_any_count(tmp_path):
    assert len(load_extra_clause_types(_write(tmp_path, ["A", "B"]))) == 2


def test_extra_duplicate_name_raises(tmp_path):
    with pytest.raises(ClauseTypeError, match="duplicate clause type: A"):
        load_extra_clause_types(_write(tmp_path, ["A", "A"]))


def test_app_types_are_the_cuad_types_then_the_extras(types, extras):
    assert app_clause_types(Settings()) == types + extras


def test_app_types_reject_an_extra_that_clashes_with_cuad(tmp_path):
    path = _write(tmp_path, ["Governing Law"])
    settings = replace(Settings(), extra_clause_types_path=path)
    with pytest.raises(ClauseTypeError, match="clashes with CUAD: Governing Law"):
        app_clause_types(settings)
