"""Measures the summaries of a set of contracts: misquotes, length and latency per document."""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Sequence
from statistics import mean

from pydantic import BaseModel

from contract_assistant.phrase import quoted_spans
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.store import DocumentCache
from contract_assistant.summary import APP_ATTEMPTS, request_summary, summary_items

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


class SummaryRecord(BaseModel):
    """The summary of one document in numbers; a document without items is withheld."""

    document: str
    items: int
    chars: int
    sentences: int
    quoted_spans: int
    unfaithful_spans: list[str]
    withheld: bool
    attempts: int
    latency_ms: float


class SummaryReport(BaseModel):
    """Outcome counts of the summary check over the documents for one model."""

    model: str
    documents: int
    withheld: int
    retried: int
    llm_failures: int
    mean_latency_ms: float | None
    records: list[SummaryRecord]


def evaluate_summaries(
    client: LLMClient,
    caches: dict[str, DocumentCache],
    clause_types: Sequence[str],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
    attempts: int = APP_ATTEMPTS,
) -> SummaryReport:
    """Summarize each document as the app does and record the checks on the summary."""
    if not caches:
        raise ValueError("no documents to evaluate")
    records: list[SummaryRecord] = []
    failures = 0
    for document, cache in caches.items():
        items = summary_items(cache, clause_types, tau_low, tau_high, unvalidated)
        if not items:
            records.append(_empty_record(document))
            continue
        try:
            summary = request_summary(
                cache.doc_id, items, client, tau_low, tau_high, attempts=attempts
            )
        except LLMError as error:
            logger.warning("summary of %s failed: %s", document, error)
            failures += 1
            continue
        text = summary.text or ""
        records.append(
            SummaryRecord(
                document=document,
                items=len(summary.items),
                chars=len(text),
                sentences=len(SENTENCE_END.findall(text)),
                quoted_spans=len(quoted_spans(text)),
                unfaithful_spans=summary.unfaithful_spans,
                withheld=summary.text is None,
                attempts=summary.attempts,
                latency_ms=summary.latency_ms,
            )
        )
    latencies = [record.latency_ms for record in records if record.items]
    return SummaryReport(
        model=client.model,
        documents=len(caches),
        withheld=sum(record.withheld for record in records),
        retried=sum(record.attempts > 1 for record in records),
        llm_failures=failures,
        mean_latency_ms=mean(latencies) if latencies else None,
        records=records,
    )


def _empty_record(document: str) -> SummaryRecord:
    return SummaryRecord(
        document=document,
        items=0,
        chars=0,
        sentences=0,
        quoted_spans=0,
        unfaithful_spans=[],
        withheld=True,
        attempts=0,
        latency_ms=0.0,
    )
