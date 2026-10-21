"""Everything an app needs from outside: caches, models, saved generations and settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from contract_assistant.clause_types import ClauseType
from contract_assistant.config import Settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.fulltext import FullTextStore
from contract_assistant.obligations import ObligationStore
from contract_assistant.review import Rule
from contract_assistant.router.base import Router
from contract_assistant.router.clients import LLMClient
from contract_assistant.search import PassageSearch
from contract_assistant.store import Store
from contract_assistant.summary import SummaryStore
from contract_assistant.term_sheet import TermSheetStore


@dataclass(frozen=True)
class Services:
    """Everything the app needs from outside: caches, models and settings."""

    store: Store
    router: Router
    extractor: Extractor | None
    clause_types: list[ClauseType]
    settings: Settings
    search: PassageSearch
    unvalidated: frozenset[str]
    gen: LLMClient | None = None
    summaries: SummaryStore | None = None
    term_sheets: TermSheetStore | None = None
    rules: Sequence[Rule] = ()
    obligations: ObligationStore | None = None
    fulltexts: FullTextStore | None = None
