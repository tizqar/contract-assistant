"""Measures how often an LLM paragraph quotes text that is not a verified quote."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from pydantic import BaseModel

from contract_assistant.answer import Answer, answer
from contract_assistant.evaluation.questions import EvalQuestion
from contract_assistant.phrase import (
    cited_quotes,
    quoted_spans,
    request_paragraph,
    unfaithful_spans,
)
from contract_assistant.router.base import RouteResult
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.store import DocumentCache

logger = logging.getLogger(__name__)


class ViolationRecord(BaseModel):
    """A paragraph with quoted spans that are not part of a verified quote."""

    question_id: str
    document: str
    spans: list[str]
    paragraph: str


class PhrasingReport(BaseModel):
    """Outcome counts of the quote check over the phrased answers of one model."""

    model: str
    calls: int
    accepted: int
    violations: int
    unquoted_replies: int
    llm_failures: int
    invalid_replies: int
    violation_rate: float | None
    violation_records: list[ViolationRecord]


def evaluate_phrasing(
    client: LLMClient,
    questions: list[EvalQuestion],
    caches: dict[str, DocumentCache],
    tau_low: float,
    tau_high: float,
) -> PhrasingReport:
    """Phrase the gold-routed answer to each in-scope question on each document and check it."""
    in_scope = [question for question in questions if question.kind != "out_of_scope"]
    if not in_scope or not caches:
        raise ValueError("no in-scope questions or documents to evaluate")
    calls = accepted = unquoted_replies = llm_failures = invalid_replies = 0
    records: list[ViolationRecord] = []
    for question_id, document, result in _answers(in_scope, caches, tau_low, tau_high):
        calls += 1
        try:
            paragraph = request_paragraph(result, client)
        except LLMError as error:
            logger.warning("phrasing %s on %s failed: %s", question_id, document, error)
            llm_failures += 1
            continue
        if paragraph is None:
            invalid_replies += 1
            continue
        if not quoted_spans(paragraph):
            unquoted_replies += 1
        spans = unfaithful_spans(paragraph, cited_quotes(result))
        if spans:
            records.append(
                ViolationRecord(
                    question_id=question_id, document=document, spans=spans, paragraph=paragraph
                )
            )
        else:
            accepted += 1
    return PhrasingReport(
        model=client.model,
        calls=calls,
        accepted=accepted,
        violations=len(records),
        unquoted_replies=unquoted_replies,
        llm_failures=llm_failures,
        invalid_replies=invalid_replies,
        violation_rate=_rate(len(records), calls - llm_failures),
        violation_records=records,
    )


def _answers(
    questions: list[EvalQuestion],
    caches: dict[str, DocumentCache],
    tau_low: float,
    tau_high: float,
) -> Iterator[tuple[str, str, Answer]]:
    for question in questions:
        route = RouteResult(clause_types=question.gold, kind="oracle", latency_ms=0.0)
        for document, cache in caches.items():
            result = answer(question.question, route, cache, tau_low, tau_high)
            yield question.id, document, result


def _rate(violations: int, replies: int) -> float | None:
    return violations / replies if replies else None
