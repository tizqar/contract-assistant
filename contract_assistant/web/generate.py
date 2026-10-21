"""Generation steps the web routes stream: a status while the model writes, then the result.

Each `*_steps` generator yields `Step`s while the language model writes and ends with the
saved or newly written record. A saved record is yielded at once, without a step. A withheld
record (its text is None, its reason says why) is never saved, so the next request asks again.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from contract_assistant.explain import Explanation, explanation_from, explanation_request
from contract_assistant.generation import Generation, Request
from contract_assistant.generation import stream as generation_stream
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DocumentCache
from contract_assistant.summary import APP_ATTEMPTS as SUMMARY_ATTEMPTS
from contract_assistant.summary import Summary, summary_from, summary_items, summary_request
from contract_assistant.term_sheet import APP_ATTEMPTS as TERM_SHEET_ATTEMPTS
from contract_assistant.term_sheet import TermSheet, request_term_sheet, term_sheet_items
from contract_assistant.web.core import Thresholds, Workspace, WorkspaceError

WRITING_SUMMARY = "Writing the summary"
WRITING_EXPLANATION = "Writing the explanation"
EXTRACTING_TERMS = "Extracting the key terms. This can take up to half a minute."
NOTHING_TO_SUMMARIZE = "No clause was found in this contract, so there is nothing to summarize."
NO_TERM_CLAUSES = "None of the clauses the key terms are read from was found in this contract."
RETRY_TEXT = "The first reply was not shown because {reason}; asking again"


@dataclass(frozen=True)
class Step:
    """What the model has written so far, with the status to show above it."""

    what: str
    text: str
    elapsed_s: int
    previous_reason: str | None = None

    @property
    def status(self) -> str:
        """The status line: what is being written, or why it is being written again."""
        if self.previous_reason is None:
            return f"{self.what} · {self.elapsed_s} s"
        return f"{RETRY_TEXT.format(reason=self.previous_reason)} · {self.elapsed_s} s"


def checked(
    client: LLMClient, what: str, request: Request, attempts: int
) -> Iterator[Step | Generation]:
    """Yield a step per piece the model writes, then the checked Generation."""
    started = time.perf_counter()
    yield Step(what, "", 0)
    for item in generation_stream(client, request, attempts):
        if isinstance(item, Generation):
            yield item
        else:
            elapsed = round(time.perf_counter() - started)
            yield Step(what, item.text, elapsed, item.previous_reason)


def saved_summary(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, language: str
) -> Summary | None:
    """Return the saved summary of a contract in a language, or None without generating."""
    if ws.services.gen is None:
        return None
    return ws.summary_store(language).load(cache.doc_id, ws.services.gen.model, tau.low, tau.high)


def summary_steps(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, language: str
) -> Iterator[Step | Summary]:
    """Summarize the clauses found, reusing a saved summary; steps while the model writes."""
    gen = ws.gen()
    items = summary_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated)
    if not items:
        raise WorkspaceError(NOTHING_TO_SUMMARIZE)
    store = ws.summary_store(language)
    summary = store.load(cache.doc_id, gen.model, tau.low, tau.high)
    if summary is None:
        request = summary_request(cache.doc_id, items, language)
        for step in checked(gen, WRITING_SUMMARY, request, SUMMARY_ATTEMPTS):
            if isinstance(step, Generation):
                summary = summary_from(
                    step, cache.doc_id, items, gen, tau.low, tau.high, language=language
                )
            else:
                yield step
        if summary.text is not None:
            store.save(summary)
    yield summary


def saved_term_sheet(ws: Workspace, cache: DocumentCache, tau: Thresholds) -> TermSheet | None:
    """Return the saved term sheet of a contract, or None without generating."""
    if ws.services.gen is None:
        return None
    return ws.term_sheets.load(cache.doc_id, ws.services.gen.model, tau.low, tau.high)


def term_sheet_steps(
    ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[Step | TermSheet]:
    """Extract the key terms, reusing a saved term sheet; one step while the model works."""
    gen = ws.gen()
    items = term_sheet_items(cache, tau.low, tau.high, ws.services.unvalidated)
    if not items:
        raise WorkspaceError(NO_TERM_CLAUSES)
    sheet = ws.term_sheets.load(cache.doc_id, gen.model, tau.low, tau.high)
    if sheet is None:
        yield Step(EXTRACTING_TERMS, "", 0)
        sheet = request_term_sheet(
            cache.doc_id, items, gen, tau.low, tau.high, attempts=TERM_SHEET_ATTEMPTS
        )
        if sheet.reason is None:
            ws.term_sheets.save(sheet)
    yield sheet


def explanation_steps(
    ws: Workspace, clause_type: str, quote: str, language: str
) -> Iterator[Step | Explanation]:
    """Restate a quote in plain language; steps while the model writes, then the Explanation."""
    gen = ws.gen()
    request = explanation_request(clause_type, quote, language)
    explanation = None
    for step in checked(gen, WRITING_EXPLANATION, request, 1):
        if isinstance(step, Generation):
            explanation = explanation_from(step, clause_type, quote, language, gen)
        else:
            yield step
    if explanation is not None:
        yield explanation
