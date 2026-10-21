import csv
import io

import pytest

from contract_assistant.config import Settings
from contract_assistant.portfolio import (
    NOT_EXTRACTED,
    headers,
    portfolio_rows,
    render_markdown,
    table,
    to_csv,
)
from contract_assistant.review import Rule
from contract_assistant.store import Store
from contract_assistant.term_sheet import FIELDS, TermSheet, TermSheetStore, TermValue
from tests.support.caches import make_cache

LAW = "This Agreement is governed by the laws of Delaware."
TERM = "The Term is five years."
EXCLUSIVITY = "Supplier shall be the exclusive supplier."
TEXT_A = f"{LAW} {TERM}"
TEXT_B = f"{EXCLUSIVITY} Nothing else is stated here."
TEXT_C = "A third contract with a few words."
TYPES = ["Governing Law", "Term", "Exclusivity"]
FOUND = 0.95
CHECK = 0.5
MODEL = "fake-gen"
RULES = [
    Rule(id="law", kind="missing", clause_types=["Governing Law"], severity="high", title="t", note="n"),
    Rule(id="excl", kind="present", clause_types=["Exclusivity"], severity="medium", title="t", note="n"),
    Rule(id="ins", kind="missing", clause_types=["Insurance"], severity="low", title="t", note="n"),
]  # fmt: skip


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def cache_a(settings):
    return make_cache(
        TEXT_A, settings, {"Governing Law": [(LAW, FOUND)], "Term": [(TERM, CHECK)]}
    )


@pytest.fixture
def cache_b(settings):
    return make_cache(TEXT_B, settings, {"Exclusivity": [(EXCLUSIVITY, FOUND)]})


@pytest.fixture
def store(tmp_path, cache_a, cache_b) -> Store:
    store = Store(tmp_path / "caches", tmp_path / "demo")
    (tmp_path / "demo").mkdir()
    store.save(cache_a)
    store.save(cache_b)
    return store


@pytest.fixture
def sheets(tmp_path, cache_a) -> TermSheetStore:
    sheets = TermSheetStore(tmp_path / "term_sheets")
    start = TEXT_A.index("Delaware")
    value = TermValue(
        value="Delaware", clause_type="Governing Law", page=None, start=start, end=start + 8
    )
    terms = {field: [] for field in FIELDS}
    terms["governing_law"] = [value]
    sheets.save(
        TermSheet(
            doc_id=cache_a.doc_id,
            model=MODEL,
            tau_low=Settings().tau_low,
            tau_high=Settings().tau_high,
            latency_ms=1.0,
            items=[],
            terms=terms,
            rejected={},
        )
    )
    return sheets


@pytest.fixture
def rows(store, settings, sheets):
    return portfolio_rows(store, settings, TYPES, RULES, sheets, MODEL)


def row_of(rows, cache):
    return next(row for row in rows if row.doc_id == cache.doc_id)


def test_portfolio_has_one_row_per_document(rows, cache_a, cache_b):
    assert sorted(row.doc_id for row in rows) == sorted([cache_a.doc_id, cache_b.doc_id])


def test_portfolio_rows_follow_store_order(rows, store):
    assert [row.doc_id for row in rows] == [entry.doc_id for entry in store.list()]


def test_a_document_in_both_directories_is_listed_once(
    tmp_path, store, settings, sheets, cache_a
):
    Store(tmp_path / "demo", tmp_path / "demo").save(cache_a)
    rows = portfolio_rows(store, settings, TYPES, RULES, sheets, MODEL)
    assert [row.doc_id for row in rows].count(cache_a.doc_id) == 1


def test_size_of_a_text_cache_is_in_words(rows, cache_a):
    assert row_of(rows, cache_a).size == f"{len(TEXT_A.split())} words"


def test_clause_counts_split_into_found_check_and_not_found(rows, cache_a):
    row = row_of(rows, cache_a)
    assert (row.found, row.check, row.not_found) == (1, 1, 1)


def test_clause_counts_cover_only_the_given_clause_types(store, settings, sheets, cache_a):
    rows = portfolio_rows(store, settings, ["Governing Law"], RULES, sheets, MODEL)
    row = row_of(rows, cache_a)
    assert (row.found, row.check, row.not_found) == (1, 0, 0)


def test_key_terms_come_from_the_saved_sheet(rows, cache_a):
    assert row_of(rows, cache_a).terms["governing_law"] == "Delaware"


def test_key_terms_of_a_document_without_a_sheet_are_not_extracted(rows, cache_b):
    assert set(row_of(rows, cache_b).terms.values()) == {NOT_EXTRACTED}


def test_key_terms_without_a_model_are_not_extracted(store, settings, sheets, cache_a):
    rows = portfolio_rows(store, settings, TYPES, RULES, sheets, None)
    assert set(row_of(rows, cache_a).terms.values()) == {NOT_EXTRACTED}


def test_flags_are_counted_per_severity(rows, cache_a, cache_b):
    a, b = row_of(rows, cache_a), row_of(rows, cache_b)
    assert (a.flags_high, a.flags_medium, a.flags_low) == (0, 0, 1)
    assert (b.flags_high, b.flags_medium, b.flags_low) == (1, 1, 1)


def test_an_invalid_cache_file_is_skipped(store, settings, sheets):
    (store.cache_dir / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(portfolio_rows(store, settings, TYPES, RULES, sheets, MODEL)) == 2


def test_a_cache_with_other_settings_is_skipped(store, settings, sheets):
    other = make_cache(TEXT_C, Settings(extractor_name="another-extractor"))
    store.save(other)
    rows = portfolio_rows(store, settings, TYPES, RULES, sheets, MODEL)
    assert other.doc_id not in [row.doc_id for row in rows]


def test_headers_name_every_column_of_a_table_row(rows):
    assert all(len(row) == len(headers()) for row in table(rows))


def test_headers_start_with_the_contract_and_end_with_the_flags():
    assert headers()[0] == "Contract"
    assert headers()[-3:] == ["High flags", "Medium flags", "Low flags"]


def test_table_cells_are_strings_in_header_order(rows):
    first = table(rows)[0]
    assert all(isinstance(cell, str) for cell in first)
    assert first[0] == rows[0].name
    assert first[2:5] == [str(rows[0].found), str(rows[0].check), str(rows[0].not_found)]
    assert first[-3:] == [
        str(rows[0].flags_high),
        str(rows[0].flags_medium),
        str(rows[0].flags_low),
    ]


def test_csv_has_a_header_line_and_one_line_per_row(rows):
    parsed = list(csv.reader(io.StringIO(to_csv(rows))))
    assert parsed[0] == headers()
    assert parsed[1:] == table(rows)


def test_csv_of_no_rows_is_only_the_header():
    assert list(csv.reader(io.StringIO(to_csv([])))) == [headers()]


def test_markdown_has_one_row_per_document(rows):
    lines = render_markdown(rows).splitlines()
    assert len(lines) == 2 + len(rows)
    assert lines[0].startswith("| Contract |")


def test_markdown_replaces_pipes_in_values(rows):
    rows[0].terms["parties"] = "Acme | Beta"
    text = render_markdown(rows)
    assert "Acme / Beta" in text
    assert "Acme | Beta" not in text
