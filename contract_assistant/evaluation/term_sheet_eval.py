"""Measures the term sheets of the demo contracts against CUAD's gold spans."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.store import DocumentCache
from contract_assistant.term_sheet import (
    APP_ATTEMPTS,
    FIELDS,
    TermSheet,
    request_term_sheet,
    term_sheet_items,
)

logger = logging.getLogger(__name__)

Gold = dict[str, dict[str, list[str]]]


class FieldRecord(BaseModel):
    """One field of one document: its verified values and how they relate to the gold spans."""

    document: str
    field: str
    values: list[str]
    agreeing: int
    rejected: list[str]
    gold_present: bool


class TermSheetReport(BaseModel):
    """Counts over the fields of the term sheets of all documents for one model."""

    model: str
    documents: int
    fields: int
    values: int
    agreeing: int
    rejected: int
    missing: int
    withheld: int
    llm_failures: int
    mean_latency_ms: float | None
    records: list[FieldRecord]


def load_demo_gold(path: Path) -> Gold:
    """Read the gold spans of the demo documents, keyed by document slug and clause type."""
    documents = json.loads(path.read_text(encoding="utf-8"))
    return {document["slug"]: document["gold_spans"] for document in documents}


def evaluate_term_sheets(
    client: LLMClient,
    caches: dict[str, DocumentCache],
    gold: Gold,
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
    attempts: int = APP_ATTEMPTS,
) -> TermSheetReport:
    """Extract the term sheet of each document and compare every value with the gold spans."""
    if not caches:
        raise ValueError("no documents to evaluate")
    records: list[FieldRecord] = []
    failures = withheld = 0
    latencies: list[float] = []
    for document, cache in caches.items():
        items = term_sheet_items(cache, tau_low, tau_high, unvalidated)
        if not items:
            withheld += 1
            continue
        try:
            sheet = request_term_sheet(cache.doc_id, items, client, tau_low, tau_high, attempts)
        except LLMError as error:
            logger.warning("term sheet of %s failed: %s", document, error)
            failures += 1
            continue
        latencies.append(sheet.latency_ms)
        if sheet.reason:
            withheld += 1
        records.extend(_records(document, sheet, gold.get(document, {})))
    return TermSheetReport(
        model=client.model,
        documents=len(caches),
        fields=len(records),
        values=sum(len(record.values) for record in records),
        agreeing=sum(record.agreeing for record in records),
        rejected=sum(len(record.rejected) for record in records),
        missing=sum(
            record.gold_present and not record.values and not record.rejected
            for record in records
        ),
        withheld=withheld,
        llm_failures=failures,
        mean_latency_ms=mean(latencies) if latencies else None,
        records=records,
    )


def _records(document: str, sheet: TermSheet, gold: dict[str, list[str]]) -> list[FieldRecord]:
    records = []
    for field, sources in FIELDS.items():
        spans = [_collapse(span) for name in sources for span in gold.get(name, [])]
        values = [value.value for value in sheet.terms.get(field, [])]
        records.append(
            FieldRecord(
                document=document,
                field=field,
                values=values,
                agreeing=sum(_agrees(value, spans) for value in values),
                rejected=sheet.rejected.get(field, []),
                gold_present=bool(spans),
            )
        )
    return records


def _agrees(value: str, spans: list[str]) -> bool:
    needle = _collapse(value)
    return any(needle in span or span in needle for span in spans)


def _collapse(text: str) -> str:
    return " ".join(text.split()).lower()
