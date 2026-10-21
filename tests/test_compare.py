import pytest

from contract_assistant.compare import (
    DIFFERS,
    ChecklistRow,
    TermRow,
    compare_checklists,
    compare_terms,
    render_markdown,
    statuses,
)
from contract_assistant.config import Settings
from contract_assistant.term_sheet import FIELDS, LABELS, NOT_STATED, TermSheet, TermValue
from tests.support.caches import make_cache

TAU_LOW = 0.2
TAU_HIGH = 0.8
LAW = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
INSURANCE = "Each party shall carry insurance."
TEXT = f"{LAW} {TERM} {INSURANCE}"
LEFT_NAME = "Alpha"
RIGHT_NAME = "Beta"


@pytest.fixture
def settings() -> Settings:
    return Settings()


def status_of(settings, conf, quote=LAW, clause="Governing Law"):
    cache = make_cache(TEXT, settings, {clause: [(quote, conf)]})
    return statuses(cache, [clause], TAU_LOW, TAU_HIGH)[clause]


def sheet(**values: str) -> TermSheet:
    terms = {
        field: [TermValue(value=value, clause_type=FIELDS[field][0], page=None, start=0, end=1)]
        for field, value in values.items()
    }
    return TermSheet(
        doc_id="sha256:abc",
        model="gen-test",
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        prompt_version=1,
        latency_ms=0.0,
        items=[],
        terms=terms,
        rejected={},
    )


def test_statuses_are_found_at_or_above_tau_high(settings):
    assert status_of(settings, 0.9) == "found"


def test_statuses_are_check_between_the_thresholds(settings):
    assert status_of(settings, 0.5) == "check"


def test_statuses_are_not_found_below_tau_low(settings):
    assert status_of(settings, 0.1) == "not_found"


def test_statuses_call_a_type_absent_from_the_cache_not_found(settings):
    cache = make_cache(TEXT, settings, {})
    assert statuses(cache, ["Governing Law"], TAU_LOW, TAU_HIGH) == {
        "Governing Law": "not_found"
    }


def test_statuses_call_a_withheld_citation_not_found(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)]})
    cache.clauses["Governing Law"][0].start += 1
    assert statuses(cache, ["Governing Law"], TAU_LOW, TAU_HIGH)["Governing Law"] == "not_found"


def test_statuses_cap_unvalidated_types_at_check(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.95)]})
    result = statuses(cache, ["Governing Law"], TAU_LOW, TAU_HIGH, ["Governing Law"])
    assert result["Governing Law"] == "check"


def test_statuses_keep_the_order_of_the_clause_types(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)]})
    names = ["Insurance", "Governing Law", "Term"]
    assert list(statuses(cache, names, TAU_LOW, TAU_HIGH)) == names


def test_checklist_has_one_row_per_clause_type_in_order(settings):
    left = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)]})
    right = make_cache(TEXT, settings, {})
    rows = compare_checklists(left, right, ["Term", "Governing Law"], TAU_LOW, TAU_HIGH)
    assert [row.clause_type for row in rows] == ["Term", "Governing Law"]


def test_checklist_row_differs_when_the_statuses_differ(settings):
    left = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)], "Term": [(TERM, 0.9)]})
    right = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.5)], "Term": [(TERM, 0.95)]})
    rows = compare_checklists(left, right, ["Governing Law", "Term"], TAU_LOW, TAU_HIGH)
    assert rows == [
        ChecklistRow(clause_type="Governing Law", left="found", right="check", differs=True),
        ChecklistRow(clause_type="Term", left="found", right="found", differs=False),
    ]


def test_checklist_row_with_both_not_found_does_not_differ(settings):
    left = make_cache(TEXT, settings, {})
    right = make_cache(TEXT, settings, {"Term": [(TERM, 0.1)]})
    (row,) = compare_checklists(left, right, ["Term"], TAU_LOW, TAU_HIGH)
    assert (row.left, row.right, row.differs) == ("not_found", "not_found", False)


def test_terms_have_one_row_per_field_in_order_with_labels():
    rows = compare_terms(sheet(), sheet())
    assert [row.field for row in rows] == list(FIELDS)
    assert [row.label for row in rows] == [LABELS[field] for field in FIELDS]


def test_terms_carry_the_value_of_each_side():
    rows = {r.field: r for r in compare_terms(sheet(governing_law="Delaware"), sheet())}
    assert rows["governing_law"].left == "Delaware"
    assert rows["governing_law"].right == NOT_STATED


def test_terms_differ_when_the_values_differ():
    rows = {r.field: r for r in compare_terms(sheet(term="5 years"), sheet(term="3 years"))}
    assert rows["term"].differs


def test_terms_ignore_case_and_whitespace_when_comparing():
    left = sheet(parties="Acme  Corp.\nand Beta")
    right = sheet(parties="ACME Corp. and  BETA")
    rows = {r.field: r for r in compare_terms(left, right)}
    assert not rows["parties"].differs


def test_terms_not_stated_on_both_sides_do_not_differ():
    assert not any(row.differs for row in compare_terms(sheet(), sheet()))


def test_terms_stated_on_one_side_only_differ():
    rows = {r.field: r for r in compare_terms(sheet(term="5 years"), sheet())}
    assert rows["term"].differs


def rendered() -> list[str]:
    terms = [
        TermRow(
            field="term", label="Initial term", left="5 years", right="3 years", differs=True
        ),
        TermRow(field="parties", label="Parties", left="Acme", right="Acme", differs=False),
    ]
    checklist = [
        ChecklistRow(clause_type="Insurance", left="found", right="not_found", differs=True),
        ChecklistRow(clause_type="Term", left="check", right="check", differs=False),
    ]
    return render_markdown((LEFT_NAME, RIGHT_NAME), checklist, terms).splitlines()


def test_render_has_a_terms_table_and_a_clause_table_with_the_document_names():
    lines = rendered()
    assert lines[0] == f"| Term | {LEFT_NAME} | {RIGHT_NAME} | |"
    assert f"| Clause | {LEFT_NAME} | {RIGHT_NAME} | |" in lines
    assert sum(line.startswith("| --- |") for line in lines) == 2


def test_render_marks_only_the_differing_rows():
    lines = rendered()
    marked = [line for line in lines if line.endswith(f"| {DIFFERS} |")]
    assert marked == [
        f"| Initial term | 5 years | 3 years | {DIFFERS} |",
        f"| Insurance | found | not found | {DIFFERS} |",
    ]
    assert "| Parties | Acme | Acme |  |" in lines
    assert "| Term | possible, verify | possible, verify |  |" in lines


def test_render_uses_status_words_in_the_clause_table():
    assert "| Term | possible, verify | possible, verify |  |" in rendered()
