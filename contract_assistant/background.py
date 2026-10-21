"""Background work: summaries and term sheets made ahead of time, and models kept loaded."""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection, Iterable, Sequence

from contract_assistant.config import Settings
from contract_assistant.obligations import APP_ATTEMPTS as OBLIGATION_ATTEMPTS
from contract_assistant.obligations import (
    ObligationStore,
    obligation_items,
    request_obligations,
)
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    Store,
)
from contract_assistant.summary import APP_ATTEMPTS as SUMMARY_ATTEMPTS
from contract_assistant.summary import SummaryStore, request_summary, summary_items
from contract_assistant.term_sheet import APP_ATTEMPTS as TERM_SHEET_ATTEMPTS
from contract_assistant.term_sheet import TermSheetStore, request_term_sheet, term_sheet_items

logger = logging.getLogger(__name__)

CACHE_ERRORS = (CacheIntegrityError, CacheMismatchError)


def precompute(
    cache: DocumentCache,
    clause_types: Sequence[str],
    gen: LLMClient,
    summaries: SummaryStore,
    term_sheets: TermSheetStore,
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> None:
    """Make and save the summary and the term sheet of a document when they are missing."""
    keys = (cache.doc_id, gen.model, tau_low, tau_high)
    if summaries.load(*keys) is None:
        items = summary_items(cache, clause_types, tau_low, tau_high, unvalidated)
        if items:
            summary = request_summary(
                cache.doc_id, items, gen, tau_low, tau_high, attempts=SUMMARY_ATTEMPTS
            )
            if summary.text is not None:
                summaries.save(summary)
    if term_sheets.load(*keys) is None:
        items = term_sheet_items(cache, tau_low, tau_high, unvalidated)
        if items:
            sheet = request_term_sheet(
                cache.doc_id, items, gen, tau_low, tau_high, attempts=TERM_SHEET_ATTEMPTS
            )
            if sheet.reason is None:
                term_sheets.save(sheet)


def precompute_obligations(
    cache: DocumentCache,
    clause_types: Sequence[str],
    gen: LLMClient,
    obligations: ObligationStore,
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> None:
    """Make and save the obligations of a document when they are missing."""
    if obligations.load(cache.doc_id, gen.model, tau_low, tau_high) is not None:
        return
    items = obligation_items(cache, clause_types, tau_low, tau_high, unvalidated)
    if not items:
        return
    sheet = request_obligations(
        cache.doc_id, items, gen, tau_low, tau_high, attempts=OBLIGATION_ATTEMPTS
    )
    if sheet.reason is None:
        obligations.save(sheet)


def precompute_all(
    store: Store,
    settings: Settings,
    clause_types: Sequence[str],
    gen: LLMClient,
    summaries: SummaryStore,
    term_sheets: TermSheetStore,
    unvalidated: Collection[str] = (),
    obligations: ObligationStore | None = None,
) -> int:
    """Precompute every cached document in turn; return how many were done without error."""
    done = 0
    seen: set[str] = set()
    for entry in store.list():
        if entry.doc_id in seen:
            continue
        seen.add(entry.doc_id)
        try:
            cache = store.load(entry.doc_id, settings)
            if cache is None:
                continue
            precompute(
                cache,
                clause_types,
                gen,
                summaries,
                term_sheets,
                settings.tau_low,
                settings.tau_high,
                unvalidated,
            )
            if obligations is not None:
                precompute_obligations(
                    cache,
                    clause_types,
                    gen,
                    obligations,
                    settings.tau_low,
                    settings.tau_high,
                    unvalidated,
                )
        except (LLMError, *CACHE_ERRORS) as error:
            logger.warning("precompute of %s skipped: %s", entry.doc_id, error)
            continue
        done += 1
    logger.info("precomputed %d documents", done)
    return done


def warm(clients: Iterable[LLMClient | None]) -> None:
    """Load again every client that can be loaded, so idle models stay in memory."""
    for client in clients:
        load = getattr(client, "load", None)
        if load is None:
            continue
        try:
            load()
        except LLMError as error:
            logger.warning("keeping %s loaded failed: %s", client.model, error)


def run_in_background(target, *args) -> threading.Thread:
    """Run target(*args) in a daemon thread that logs a failure, and return the thread."""

    def guarded() -> None:
        try:
            target(*args)
        except Exception:
            logger.exception("background %s failed", getattr(target, "__name__", target))

    thread = threading.Thread(target=guarded, daemon=True)
    thread.start()
    return thread
