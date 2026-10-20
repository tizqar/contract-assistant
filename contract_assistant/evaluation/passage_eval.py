"""Measures the answers written from passages for the out-of-scope questions on the demo contracts."""

from __future__ import annotations

import logging
from statistics import mean

from pydantic import BaseModel

from contract_assistant.evaluation.questions import EvalQuestion
from contract_assistant.passage_answer import APP_ATTEMPTS, request_passage_answer
from contract_assistant.phrase import quoted_spans
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.search import PassageSearch
from contract_assistant.store import DocumentCache

logger = logging.getLogger(__name__)

OUT_OF_SCOPE = "out_of_scope"


class PassageAnswerRecord(BaseModel):
    """One question on one document: the passages found and what the model made of them."""

    question_id: str
    document: str
    passages: int
    withheld: bool
    reason: str | None
    quoted_spans: int
    attempts: int
    latency_ms: float


class PassageAnswerReport(BaseModel):
    """Outcome counts over the out-of-scope questions on all documents for one model."""

    model: str
    calls: int
    answered: int
    withheld: int
    no_passages: int
    llm_failures: int
    mean_latency_ms: float | None
    records: list[PassageAnswerRecord]


def evaluate_passage_answers(
    client: LLMClient,
    questions: list[EvalQuestion],
    caches: dict[str, DocumentCache],
    search: PassageSearch,
    attempts: int = APP_ATTEMPTS,
) -> PassageAnswerReport:
    """Answer each out-of-scope question on each document from its passages and check it."""
    out_of_scope = [question for question in questions if question.kind == OUT_OF_SCOPE]
    if not out_of_scope or not caches:
        raise ValueError("no out-of-scope questions or documents to evaluate")
    records: list[PassageAnswerRecord] = []
    calls = failures = no_passages = 0
    for question in out_of_scope:
        for document, cache in caches.items():
            passages = search.search(cache, question.question)
            if not passages:
                no_passages += 1
                records.append(_empty_record(question.id, document))
                continue
            calls += 1
            try:
                answer = request_passage_answer(question.question, passages, client, attempts)
            except LLMError as error:
                logger.warning("answer to %s on %s failed: %s", question.id, document, error)
                failures += 1
                continue
            records.append(
                PassageAnswerRecord(
                    question_id=question.id,
                    document=document,
                    passages=len(passages),
                    withheld=answer.text is None,
                    reason=answer.reason,
                    quoted_spans=len(quoted_spans(answer.text or "")),
                    attempts=answer.attempts,
                    latency_ms=answer.latency_ms,
                )
            )
    answered = [record for record in records if record.passages and not record.withheld]
    latencies = [record.latency_ms for record in records if record.passages]
    return PassageAnswerReport(
        model=client.model,
        calls=calls,
        answered=len(answered),
        withheld=sum(record.withheld for record in records if record.passages),
        no_passages=no_passages,
        llm_failures=failures,
        mean_latency_ms=mean(latencies) if latencies else None,
        records=records,
    )


def _empty_record(question_id: str, document: str) -> PassageAnswerRecord:
    return PassageAnswerRecord(
        question_id=question_id,
        document=document,
        passages=0,
        withheld=False,
        reason=None,
        quoted_spans=0,
        attempts=0,
        latency_ms=0.0,
    )
