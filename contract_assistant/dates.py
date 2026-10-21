"""Key dates: the dates a term sheet states and the ones that follow from its durations."""

from __future__ import annotations

import re
from datetime import date

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel

from contract_assistant.term_sheet import TermSheet

YEAR = re.compile(r"\b(1[89]|20)\d{2}\b")
UNIT = re.compile(r"\b(year|month|week|day)s?\b", re.IGNORECASE)
DIGITS = re.compile(r"\d+")
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18,
    "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36, "sixty": 60, "ninety": 90,
}  # fmt: skip
# The OCR of the demo contracts writes "15t h day"; the parser needs "15th day".
ORDINAL_GAP = re.compile(r"\b(\d{1,2})(st|nd|rd|th)?\s+h\b")
MAX_FRAGMENT_CHARS = 80


class Duration(BaseModel):
    """A duration read from a verbatim fragment."""

    count: int
    unit: str

    def delta(self) -> relativedelta:
        """Return the duration as a relativedelta."""
        return relativedelta(**{f"{self.unit}s": self.count})

    def text(self) -> str:
        """Return the duration in words, like '10 years'."""
        unit = self.unit if self.count == 1 else f"{self.unit}s"
        return f"{self.count} {unit}"


class KeyDate(BaseModel):
    """A date of the contract: stated in a clause, or computed from stated terms."""

    label: str
    date: date | None
    source: str
    clause_type: str | None
    formula: str | None = None


def parse_date(text: str) -> date | None:
    """Return the date a fragment states, or None when it states no full date."""
    cleaned = ORDINAL_GAP.sub(lambda m: f"{m.group(1)}th", text)
    if not YEAR.search(cleaned) or len(cleaned) > MAX_FRAGMENT_CHARS:
        return None
    try:
        return date_parser.parse(cleaned, fuzzy=True, dayfirst=False).date()
    except (ValueError, OverflowError):
        return None


def parse_duration(text: str) -> Duration | None:
    """Return the duration a fragment states, reading the number nearest to the unit word."""
    unit_match = UNIT.search(text)
    if unit_match is None:
        return None
    unit = unit_match.group(1).lower()
    tokens = re.findall(r"[a-z\-]+|\d+", text[: unit_match.start()].lower())
    # "one (1) twelve month period": the word next to the unit wins over an earlier number.
    for token in reversed(tokens):
        if token.isdigit():
            return Duration(count=int(token), unit=unit)
        if token in NUMBER_WORDS:
            return Duration(count=NUMBER_WORDS[token], unit=unit)
    return None


def key_dates(sheet: TermSheet) -> list[KeyDate]:
    """Return the stated dates of a term sheet and the dates its durations imply."""
    dates: list[KeyDate] = []
    stated = {
        field: _first(sheet, field) for field in ("agreement_date", "effective_date", "expiration_date")
    }
    for field, label in (
        ("agreement_date", "Agreement date"),
        ("effective_date", "Effective date"),
        ("expiration_date", "Expiration date"),
    ):
        value = stated[field]
        if value is not None:
            dates.append(
                KeyDate(label=label, date=parse_date(value[0]), source=value[0], clause_type=value[1])
            )
    start = _start(dates)
    term = _duration(sheet, "term")
    end = next((d.date for d in dates if d.label == "Expiration date" and d.date), None)
    if end is None and start is not None and term is not None:
        end = start.date + term[0].delta()
        dates.append(
            KeyDate(
                label="Initial term ends",
                date=end,
                source=term[1],
                clause_type=term[2],
                formula=f"{start.label.lower()} + {term[0].text()}",
            )
        )
    renewal = _duration(sheet, "renewal_term")
    if end is not None and renewal is not None:
        dates.append(
            KeyDate(
                label="First renewal ends",
                date=end + renewal[0].delta(),
                source=renewal[1],
                clause_type=renewal[2],
                formula=f"end of the initial term + {renewal[0].text()}",
            )
        )
    notice = _duration(sheet, "notice_period")
    if end is not None and notice is not None:
        dates.append(
            KeyDate(
                label="Last day to give notice",
                date=end - notice[0].delta(),
                source=notice[1],
                clause_type=notice[2],
                formula=f"end of the initial term - {notice[0].text()}",
            )
        )
    return dates


def _first(sheet: TermSheet, field: str) -> tuple[str, str] | None:
    values = sheet.terms.get(field, [])
    if not values:
        return None
    return " ".join(values[0].value.split()), values[0].clause_type


def _start(dates: list[KeyDate]) -> KeyDate | None:
    for label in ("Effective date", "Agreement date"):
        for item in dates:
            if item.label == label and item.date is not None:
                return item
    return None


def _duration(sheet: TermSheet, field: str) -> tuple[Duration, str, str] | None:
    value = _first(sheet, field)
    if value is None:
        return None
    duration = parse_duration(value[0])
    return None if duration is None else (duration, value[0], value[1])


def render_markdown(dates: list[KeyDate]) -> str:
    """Render the key dates as a Markdown table."""
    if not dates:
        return "No dates could be read from the key terms."
    lines = ["| Date | When | From | How |", "| --- | --- | --- | --- |"]
    for item in dates:
        when = item.date.isoformat() if item.date else "not readable"
        how = item.formula or "stated"
        lines.append(f"| {item.label} | {when} | {item.source} ({item.clause_type}) | {how} |")
    return "\n".join(lines)
