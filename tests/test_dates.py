from datetime import date

import pytest

from contract_assistant.dates import (
    MAX_FRAGMENT_CHARS,
    Duration,
    KeyDate,
    key_dates,
    parse_date,
    parse_duration,
    render_markdown,
)
from contract_assistant.term_sheet import FIELDS, TermSheet, TermValue

AGREEMENT = "15th day of January 2018"
EFFECTIVE = "1st day of March 2019"
EXPIRATION = "28th day of February 2022"
TEN_YEARS = "ten (10) years"


def sheet_of(**values: str) -> TermSheet:
    terms = {field: [] for field in FIELDS}
    for field, value in values.items():
        clause_type = FIELDS[field][0]
        terms[field] = [
            TermValue(value=value, clause_type=clause_type, page=None, start=0, end=1)
        ]
    return TermSheet(
        doc_id="sha256:abc",
        model="fake-gen",
        tau_low=0.2,
        tau_high=0.8,
        latency_ms=1.0,
        items=[],
        terms=terms,
        rejected={},
    )


def by_label(dates: list[KeyDate]) -> dict[str, KeyDate]:
    return {item.label: item for item in dates}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("15th day of January 2018", date(2018, 1, 15)),
        ("15t h day of January 2018", date(2018, 1, 15)),
        ("16th day of February, 1998", date(1998, 2, 16)),
        ("February 16, 1998", date(1998, 2, 16)),
        ("23rd day of September, 1997", date(1997, 9, 23)),
    ],
)
def test_parse_date_reads_a_stated_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", ["ten (10) years", "the first Monday"])
def test_parse_date_without_a_year_is_none(text):
    assert parse_date(text) is None


def test_parse_date_of_a_long_fragment_is_none():
    text = "January 2018 " + "x" * MAX_FRAGMENT_CHARS
    assert parse_date(text) is None


def test_parse_date_of_nonsense_with_a_year_does_not_raise():
    result = parse_date("not a date 2019 at all")
    assert result is None or isinstance(result, date)


@pytest.mark.parametrize(
    ("text", "count", "unit"),
    [
        ("ten (10) years", 10, "year"),
        ("one (1) twelve month period", 12, "month"),
        ("six (6) months", 6, "month"),
        ("thirty days", 30, "day"),
        ("period of two (2) years", 2, "year"),
    ],
)
def test_parse_duration_reads_the_number_nearest_the_unit(text, count, unit):
    assert parse_duration(text) == Duration(count=count, unit=unit)


def test_parse_duration_without_a_unit_is_none():
    assert parse_duration("not stated") is None


def test_duration_text_is_singular_for_one():
    assert Duration(count=1, unit="year").text() == "1 year"


def test_duration_text_is_plural_for_many():
    assert Duration(count=10, unit="year").text() == "10 years"


def test_duration_delta_adds_months_to_a_date():
    assert date(2018, 1, 31) + Duration(count=1, unit="month").delta() == date(2018, 2, 28)


def test_duration_delta_adds_years_to_a_date():
    assert date(2018, 1, 15) + Duration(count=10, unit="year").delta() == date(2028, 1, 15)


def test_stated_dates_have_no_formula_and_keep_their_clause_type():
    dates = by_label(
        key_dates(
            sheet_of(
                agreement_date=AGREEMENT, effective_date=EFFECTIVE, expiration_date=EXPIRATION
            )
        )
    )
    assert dates["Agreement date"].date == date(2018, 1, 15)
    assert dates["Effective date"].date == date(2019, 3, 1)
    assert dates["Expiration date"].date == date(2022, 2, 28)
    assert dates["Expiration date"].clause_type == "Expiration Date"
    assert all(item.formula is None for item in dates.values())


def test_unreadable_stated_date_keeps_its_source():
    dates = by_label(key_dates(sheet_of(effective_date="upon signature")))
    assert dates["Effective date"].date is None
    assert dates["Effective date"].source == "upon signature"


def test_initial_term_end_is_the_effective_date_plus_the_term():
    dates = by_label(key_dates(sheet_of(effective_date=EFFECTIVE, term=TEN_YEARS)))
    end = dates["Initial term ends"]
    assert end.date == date(2029, 3, 1)
    assert end.formula == "effective date + 10 years"
    assert end.source == TEN_YEARS


def test_initial_term_end_prefers_the_effective_date_over_the_agreement_date():
    dates = by_label(
        key_dates(sheet_of(agreement_date=AGREEMENT, effective_date=EFFECTIVE, term=TEN_YEARS))
    )
    assert dates["Initial term ends"].date == date(2029, 3, 1)


def test_initial_term_end_falls_back_to_the_agreement_date():
    dates = by_label(key_dates(sheet_of(agreement_date=AGREEMENT, term=TEN_YEARS)))
    assert dates["Initial term ends"].date == date(2028, 1, 15)
    assert dates["Initial term ends"].formula == "agreement date + 10 years"


def test_stated_expiration_date_is_the_end_instead_of_a_derived_one():
    dates = by_label(
        key_dates(
            sheet_of(effective_date=EFFECTIVE, expiration_date=EXPIRATION, term=TEN_YEARS)
        )
    )
    assert "Initial term ends" not in dates


def test_first_renewal_ends_one_renewal_term_after_the_end():
    dates = by_label(
        key_dates(sheet_of(expiration_date=EXPIRATION, renewal_term="one (1) year"))
    )
    renewal = dates["First renewal ends"]
    assert renewal.date == date(2023, 2, 28)
    assert renewal.formula == "end of the initial term + 1 year"


def test_last_day_to_give_notice_is_the_notice_period_before_the_end():
    dates = by_label(
        key_dates(sheet_of(expiration_date=EXPIRATION, notice_period="sixty (60) days"))
    )
    notice = dates["Last day to give notice"]
    assert notice.date == date(2021, 12, 30)
    assert notice.formula == "end of the initial term - 60 days"


def test_nothing_is_derived_without_a_start_date():
    assert key_dates(sheet_of(term=TEN_YEARS, renewal_term="one (1) year")) == []


def test_nothing_is_derived_without_a_term():
    dates = key_dates(sheet_of(effective_date=EFFECTIVE, notice_period="sixty (60) days"))
    assert [item.label for item in dates] == ["Effective date"]


def test_render_markdown_has_one_row_per_date():
    dates = key_dates(sheet_of(effective_date=EFFECTIVE, term=TEN_YEARS))
    lines = render_markdown(dates).splitlines()
    assert len(lines) == 2 + len(dates)
    assert "2029-03-01" in lines[-1]
    assert "effective date + 10 years" in lines[-1]


def test_render_markdown_says_not_readable_for_an_unreadable_date():
    text = render_markdown(key_dates(sheet_of(effective_date="upon signature")))
    assert "not readable" in text


def test_render_markdown_of_no_dates_says_so():
    assert render_markdown([]) == "No dates could be read from the key terms."
