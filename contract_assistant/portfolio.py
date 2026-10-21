"""All cached contracts in one table: size, clause counts, key terms and review flags."""

from __future__ import annotations

import csv
import io
from collections.abc import Collection, Sequence

from pydantic import BaseModel

from contract_assistant.compare import statuses
from contract_assistant.config import Settings
from contract_assistant.review import Rule, review
from contract_assistant.store import CacheIntegrityError, CacheMismatchError, Store
from contract_assistant.term_sheet import FIELDS, LABELS, TermSheetStore, cell

CACHE_ERRORS = (CacheIntegrityError, CacheMismatchError)
NOT_EXTRACTED = "not extracted"


class PortfolioRow(BaseModel):
    """One contract of the portfolio."""

    doc_id: str
    name: str
    size: str
    found: int
    check: int
    not_found: int
    terms: dict[str, str]
    flags_high: int
    flags_medium: int
    flags_low: int


def portfolio_rows(
    store: Store,
    settings: Settings,
    clause_types: Sequence[str],
    rules: Sequence[Rule],
    term_sheets: TermSheetStore,
    model: str | None,
    unvalidated: Collection[str] = (),
) -> list[PortfolioRow]:
    """Return one row per cached document, in store order; damaged caches are skipped."""
    rows: list[PortfolioRow] = []
    seen: set[str] = set()
    for entry in store.list():
        if entry.doc_id in seen:
            continue
        seen.add(entry.doc_id)
        try:
            cache = store.load(entry.doc_id, settings)
        except CACHE_ERRORS:
            continue
        if cache is None:
            continue
        status = statuses(cache, clause_types, settings.tau_low, settings.tau_high, unvalidated)
        flags = review(cache, rules, settings.tau_low, settings.tau_high, unvalidated)
        sheet = (
            term_sheets.load(cache.doc_id, model, settings.tau_low, settings.tau_high)
            if model
            else None
        )
        terms = {
            field: (cell(sheet, field) if sheet else NOT_EXTRACTED) for field in FIELDS
        }
        rows.append(
            PortfolioRow(
                doc_id=cache.doc_id,
                name=entry.source.filename,
                size=_size(cache.source.n_pages, cache.text),
                found=sum(value == "found" for value in status.values()),
                check=sum(value == "check" for value in status.values()),
                not_found=sum(value == "not_found" for value in status.values()),
                terms=terms,
                flags_high=sum(flag.severity == "high" for flag in flags),
                flags_medium=sum(flag.severity == "medium" for flag in flags),
                flags_low=sum(flag.severity == "low" for flag in flags),
            )
        )
    return rows


def _size(n_pages: int | None, text: str) -> str:
    return f"{n_pages} pages" if n_pages is not None else f"{len(text.split()):,} words"


def headers() -> list[str]:
    """Return the column names of the portfolio table."""
    return [
        "Contract",
        "Size",
        "Found",
        "Verify",
        "Not found",
        *(LABELS[field] for field in FIELDS),
        "High flags",
        "Medium flags",
        "Low flags",
    ]


def table(rows: Sequence[PortfolioRow]) -> list[list[str]]:
    """Return the portfolio as rows of strings in the order of headers()."""
    return [
        [
            row.name,
            row.size,
            str(row.found),
            str(row.check),
            str(row.not_found),
            *(row.terms[field] for field in FIELDS),
            str(row.flags_high),
            str(row.flags_medium),
            str(row.flags_low),
        ]
        for row in rows
    ]


def to_csv(rows: Sequence[PortfolioRow]) -> str:
    """Return the portfolio as CSV text with a header line."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers())
    writer.writerows(table(rows))
    return buffer.getvalue()


def render_markdown(rows: Sequence[PortfolioRow]) -> str:
    """Render the portfolio as a Markdown table."""
    names = headers()
    lines = ["| " + " | ".join(names) + " |", "|" + " --- |" * len(names)]
    lines += ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in table(rows)]
    return "\n".join(lines)
