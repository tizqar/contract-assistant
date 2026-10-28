"""Shared state and helpers of the web app: the workspace, thresholds, templates and streams.

The app is server-rendered. Every route returns HTML, a partial for htmx or a whole page,
except the streaming routes, which send newline-delimited JSON frames (see `frame`) that
`static/app.js` applies as they arrive. All URLs are free of the thresholds: the browser
appends `low` and `high` to every request from the hidden inputs of the page.

Everything the app keeps is in the database. `Workspace` holds the stores and the models for
the whole app; a request acts through a `UserWorkspace`, the workspace as the signed-in user
may see it. In the routes a contract's id is its `doc_id`: `UserWorkspace.load` returns the
analysis of a contract with the contract id in place of the text's hash and the contract's
name as its file name, so saved summaries, chats and every URL belong to the contract.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlencode

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
from sqlalchemy import select
from starlette.concurrency import iterate_in_threadpool

from contract_assistant import audit, files
from contract_assistant.access import (
    AccessError,
    Action,
    Actor,
    Permission,
    allowed,
    decisions_open,
    require,
    require_permission,
)
from contract_assistant.accounts import LoginThrottle
from contract_assistant.answer import Answer, Citation, Definition, tier
from contract_assistant.background import precompute, precompute_obligations
from contract_assistant.db.engine import Database
from contract_assistant.db.models import ADMIN, IN_REVIEW, PDF, Contract
from contract_assistant.db.models import Decision as DecisionRow
from contract_assistant.db.stores import (
    AnalysisStore,
    BlobStore,
    ChatLog,
    DbRecords,
    FeedbackLog,
)
from contract_assistant.decisions import Decision
from contract_assistant.decisions import apply as apply_decisions
from contract_assistant.definitions import find_definitions
from contract_assistant.feedback import Feedback
from contract_assistant.files import ContractView, DecisionView
from contract_assistant.fulltext import FullTextStore
from contract_assistant.obligations import ObligationStore
from contract_assistant.review import Flag, load_rules, review
from contract_assistant.router.base import Turn
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.services import Services
from contract_assistant.store import CacheIntegrityError, CacheMismatchError, DocumentCache
from contract_assistant.summary import SummaryStore
from contract_assistant.term_sheet import TermSheetStore
from contract_assistant.ui.views import SEVERITY_LABELS, display_name, document_name
from contract_assistant.web import copy
from contract_assistant.web.i18n import copy_for, language_of, translate

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
NDJSON = "application/x-ndjson"
ENGLISH = "en"
SERBIAN = "sr"
DEFAULT_DOCUMENT = "short_supply_agreement.txt"
MAX_REMEMBERED_TURNS = 8
MAX_CACHED_DEFINITIONS = 4
# Chats kept in memory, each of one user about one contract; the others are read back from
# the database when opened again.
MAX_CONVERSATIONS = 64
# Counts of the contract lists, per analysis and set of decisions.
MAX_COUNTS = 512
KEEP_WARM_S = 1500  # Ollama unloads an idle model after 30 minutes; load it again before then.
SHORT_CHARS = 120
ELLIPSIS = "…"
# Where the bytes of uploaded files are written for the page renderer, under the cache folder.
FILES_DIR = "files"
MEDIA_TYPES = {"pdf": "application/pdf", "txt": "text/plain; charset=utf-8"}

# Labels of a highlight that are not clause types.
DEFINITION_LABEL = "Definition"
RELATED_LABEL = "Related passage"
SECTION_LABEL = "Section {number}"

SELECT_CONTRACT = "Select a contract first."
NOT_ANALYSED = "This contract has not been analysed yet."
DAMAGED_ANALYSIS = "The saved analysis of this contract is damaged or inconsistent."
SUMMARIES_OFF = (
    "Summaries and explanations are off: they need a local language model. See the README."
)
GENERATION_FAILED = "The language model did not answer; see the server log."
UNEXPECTED_ERROR = "Something went wrong; see the server log."
BAD_THRESHOLDS = (
    "The Verify threshold must be at or below the Found threshold, both between 0 and 1."
)
SIGN_IN_FIRST = "Sign in first."
ROLE_NAMES = {ADMIN: "Administrator", "user": "User"}
TRANSLATED = (SIGN_IN_FIRST, *ROLE_NAMES.values())


class WorkspaceError(Exception):
    """A problem to tell the user about. The app shows the message as a notice."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Thresholds:
    """The Verify (low) and Found (high) confidence thresholds of one request."""

    low: float
    high: float

    def params(self) -> dict[str, str]:
        """Return the query parameters that carry the thresholds."""
        return {"low": f"{self.low:g}", "high": f"{self.high:g}"}


def thresholds_of(settings, low: float | None, high: float | None) -> Thresholds:
    """Return the thresholds of a request, the settings' defaults for a missing value."""
    chosen = Thresholds(
        settings.tau_low if low is None else low, settings.tau_high if high is None else high
    )
    if not 0 <= chosen.low <= chosen.high <= 1:
        raise WorkspaceError(BAD_THRESHOLDS)
    return chosen


Role = Literal["user", "assistant"]


@dataclass
class Message:
    """One bubble of a chat: who wrote it and its rendered HTML."""

    role: Role
    html: str
    # The answer behind an assistant message, for the feedback buttons.
    answer: Answer | None = None


@dataclass
class Conversation:
    """One user's chat about one contract; its messages are also kept in the database."""

    messages: list[Message] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    last: Answer | None = None

    def remember(self, turn: Turn) -> None:
        """Keep the turn for the router, forgetting the oldest beyond the limit."""
        self.turns = [*self.turns, turn][-MAX_REMEMBERED_TURNS:]

    @classmethod
    def restored(cls, messages: list[Message]) -> Conversation:
        """Rebuild a chat from its saved messages: the router's turns and the last answer."""
        conversation = cls(messages=list(messages))
        for message in messages:
            reply = getattr(message, "reply", None)
            if reply is None:
                continue
            if reply.turn is not None:
                conversation.remember(reply.turn)
            if reply.answer is not None:
                conversation.last = reply.answer
        return conversation


@dataclass(frozen=True)
class DocumentEntry:
    """A contract as the lists show it; `doc_id` is the contract id."""

    doc_id: str
    name: str
    kind: Literal["pdf", "txt"]
    size: str
    found: int
    check: int
    not_found: int = 0
    # Pages that were scans and were read with OCR.
    scanned: int = 0
    # The file name as kept, and the analysis of the text, shared by contracts of one text.
    filename: str = ""
    analysis_id: str = ""
    folder_id: str | None = None
    owner_id: str = ""
    owner_name: str = ""
    review_status: str = IN_REVIEW
    # Whether the reader owns it, rather than seeing it through a share or as an admin.
    mine: bool = True


@dataclass(frozen=True)
class _Counts:
    size: str
    found: int
    check: int
    not_found: int
    scanned: int


# A contract's decisions as (clause type, verdict, start, end), sorted: what its counts depend on.
Signature = tuple[tuple[str, str, int, int], ...]


class Workspace:
    """What every request shares: the services, the database and its stores, and the chats."""

    def __init__(self, services: Services, database: Database):
        self.services = services
        self.database = database
        settings = services.settings
        self.names = [clause_type.name for clause_type in services.clause_types]
        self.clause_by_name = {display_name(name): name for name in self.names}
        self.analyses = AnalysisStore(database, settings, settings.demo_cache_dir)
        self.blobs = BlobStore(database, settings.cache_dir / FILES_DIR)
        records = DbRecords(database)
        self.summaries = {
            ENGLISH: SummaryStore(records),
            SERBIAN: SummaryStore(records, SERBIAN),
        }
        self.term_sheets = TermSheetStore(records)
        self.obligations = ObligationStore(records)
        self.fulltexts = FullTextStore(records)
        self.feedback = FeedbackLog(database)
        self.chats = ChatLog(database)
        self.throttle = LoginThrottle()
        self.rules = services.rules or load_rules(known_types=self.names)
        self._records = records
        self._lock = threading.Lock()
        self._definitions: OrderedDict[str, list[Definition]] = OrderedDict()
        self._conversations: OrderedDict[tuple[str, str], Conversation] = OrderedDict()
        self._counts: OrderedDict[tuple[str, Signature], _Counts] = OrderedDict()

    @property
    def settings(self):
        return self.services.settings

    @property
    def gen_model(self) -> str | None:
        """The name of the generation model, or None when generation is off."""
        return self.services.gen.model if self.services.gen is not None else None

    def gen(self) -> LLMClient:
        """Return the generation client, or tell the user that generation is off."""
        if self.services.gen is None:
            raise WorkspaceError(SUMMARIES_OFF)
        return self.services.gen

    def for_user(self, actor: Actor, ip: str | None = None) -> UserWorkspace:
        """Return the workspace as a signed-in user sees it, for one request."""
        return UserWorkspace(self, actor, ip)

    def analysis(self, doc_id: str) -> DocumentCache:
        """Return the analysis of a text; 404 when there is none, a notice when it is damaged."""
        try:
            cache = self.analyses.load(doc_id)
        except (CacheIntegrityError, CacheMismatchError) as error:
            logger.warning("cannot load %s: %s", doc_id, error)
            raise WorkspaceError(DAMAGED_ANALYSIS) from error
        if cache is None:
            raise HTTPException(status_code=404, detail=NOT_ANALYSED)
        return cache

    def save_analysis(self, cache: DocumentCache) -> None:
        """Keep the analysis of a text, replacing the one before, and forget its counts."""
        self.analyses.save(cache)
        with self._lock:
            for key in [key for key in self._counts if key[0] == cache.doc_id]:
                del self._counts[key]

    def definitions(self, cache: DocumentCache) -> list[Definition]:
        """Return the defined terms of a contract, found once per server run."""
        with self._lock:
            found = self._definitions.get(cache.doc_id)
        if found is None:
            found = find_definitions(cache.text, cache.page_starts)
            with self._lock:
                self._definitions[cache.doc_id] = found
                while len(self._definitions) > MAX_CACHED_DEFINITIONS:
                    self._definitions.popitem(last=False)
        return found

    def flags(self, cache: DocumentCache, tau: Thresholds) -> list[Flag]:
        """Return the review flags of a contract at the given thresholds."""
        return review(cache, self.rules, tau.low, tau.high, self.services.unvalidated)

    def clause_type_of(self, label: str) -> str:
        """Return the clause type shown under a label; another label is returned as is."""
        return self.clause_by_name.get(label, label)

    def summary_store(self, language: str) -> SummaryStore:
        """Return the store of the summaries written in a language."""
        store = self.summaries.get(language)
        if store is None:
            store = SummaryStore(self._records, language)
            self.summaries[language] = store
        return store

    def counts(self, contract: ContractView, signature: Signature) -> _Counts:
        """Return the size and the counts of a contract's lists, worked out once."""
        key = (contract.doc_id, signature)
        with self._lock:
            known = self._counts.get(key)
            if known is not None:
                self._counts.move_to_end(key)
                return known
        known = self._count(contract, signature)
        with self._lock:
            self._counts[key] = known
            while len(self._counts) > MAX_COUNTS:
                self._counts.popitem(last=False)
        return known

    def _count(self, contract: ContractView, signature: Signature) -> _Counts:
        settings = self.settings
        try:
            cache = self.analyses.load(contract.doc_id)
        except (CacheIntegrityError, CacheMismatchError):
            cache = None
        if cache is None:
            return _Counts("", 0, 0, len(self.names), 0)
        decided = {
            name: Decision(clause_type=name, verdict=verdict, start=start, end=end, at="")
            for name, verdict, start, end in signature
        }
        cache = apply_decisions(cache, decided)
        size = (
            f"{cache.source.n_pages} pages"
            if cache.source.n_pages is not None
            else f"{len(cache.text.split()):,} words"
        )
        found = check = not_found = 0
        for name in self.names:
            candidates = cache.clauses.get(name, [])
            status = "not_found"
            if candidates:
                status = tier(candidates[0].conf, settings.tau_low, settings.tau_high)
                capped = name in self.services.unvalidated and not candidates[0].confirmed
                if capped and status == "found":
                    status = "check"
            found += status == "found"
            check += status == "check"
            not_found += status == "not_found"
        return _Counts(size, found, check, not_found, len(cache.source.ocr_pages))

    def conversation(self, user_id: str, contract_id: str) -> Conversation:
        """Return a user's chat about a contract, read back from the database when needed."""
        key = (user_id, contract_id)
        with self._lock:
            conversation = self._conversations.get(key)
            if conversation is not None:
                self._conversations.move_to_end(key)
                return conversation
        saved = [message for _, message in self.chats.load(contract_id, user_id)]
        with self._lock:
            conversation = self._conversations.get(key)
            if conversation is None:
                conversation = Conversation.restored(saved)
                self._conversations[key] = conversation
                while len(self._conversations) > MAX_CONVERSATIONS:
                    self._conversations.popitem(last=False)
            self._conversations.move_to_end(key)
            return conversation

    def scoped(self, contract: ContractView, cache: DocumentCache) -> DocumentCache:
        """Return an analysis as a contract's: the contract id as doc_id, its name as file."""
        source = cache.source.model_copy(update={"filename": contract.name})
        return cache.model_copy(update={"doc_id": contract.id, "source": source})

    def precompute_all(self) -> int:
        """Write ahead the summary, key terms and obligations of every contract missing them."""
        gen = self.services.gen
        if gen is None:
            return 0
        with self.database.session() as session:
            rows = session.execute(
                select(Contract.id, Contract.doc_id, Contract.name)
                .where(Contract.deleted_at.is_(None))
                .order_by(Contract.created_at.desc())
            ).all()
        done = 0
        settings, unvalidated = self.settings, self.services.unvalidated
        for contract_id, doc_id, name in rows:
            try:
                cache = self.analyses.load(doc_id)
                if cache is None:
                    continue
                with self.database.session() as session:
                    decided = files.decisions_of(session, contract_id)
                source = cache.source.model_copy(update={"filename": name})
                cache = cache.model_copy(update={"doc_id": contract_id, "source": source})
                cache = apply_decisions(cache, {k: v.decision for k, v in decided.items()})
                keys = (settings.tau_low, settings.tau_high, unvalidated)
                precompute(
                    cache, self.names, gen, self.summary_store(ENGLISH), self.term_sheets, *keys
                )
                precompute_obligations(cache, self.names, gen, self.obligations, *keys)
            except (LLMError, CacheIntegrityError, CacheMismatchError) as error:
                logger.warning("precompute of %s skipped: %s", contract_id, error)
                continue
            done += 1
        logger.info("precomputed %d contracts", done)
        return done


class UserWorkspace:
    """The workspace as one signed-in user may see it, for one request.

    Every contract it gives out was checked against the user's access; what it does not have
    itself (the settings, the stores, the models) it takes from the shared `Workspace`.
    """

    def __init__(self, workspace: Workspace, actor: Actor, ip: str | None = None):
        self.workspace = workspace
        self.actor = actor
        self.ip = ip
        self._contracts: dict[str, ContractView] = {}
        self._decisions: dict[str, dict[str, DecisionView]] = {}

    def __getattr__(self, name: str):
        return getattr(self.workspace, name)

    @property
    def role_name(self) -> str:
        """The name of the user's role, in English."""
        return ROLE_NAMES[self.actor.role]

    # Contracts

    def contract(self, contract_id: str, action: Action = Action.READ) -> ContractView:
        """Return a contract the user may act on; raise NotFound or Forbidden otherwise."""
        known = self._contracts.get(contract_id)
        if known is not None and allowed(known.level, action):
            return known
        with self.database.session() as session:
            found = files.contract(session, self.actor, contract_id, action)
        self._contracts[contract_id] = found
        return found

    def can(self, contract_id: str, action: Action) -> bool:
        """Tell whether the user may take an action on a contract."""
        try:
            return allowed(self.contract(contract_id).level, action)
        except AccessError:
            return False

    def can_decide(self, contract_id: str) -> bool:
        """Tell whether the user may confirm or reject passages of a contract now."""
        try:
            contract = self.contract(contract_id)
        except AccessError:
            return False
        return allowed(contract.level, Action.DECIDE) and decisions_open(contract.review_status)

    def analysis(self, contract_id: str) -> DocumentCache:
        """Return a contract's analysis as extracted, without the decisions, under its own id."""
        contract = self.contract(contract_id)
        return self.workspace.scoped(contract, self.workspace.analysis(contract.doc_id))

    def load(self, contract_id: str) -> DocumentCache:
        """Return a contract's analysis with its decisions applied; 404 when it is not shown."""
        return apply_decisions(self.analysis(contract_id), self.decided(contract_id))

    def documents(self) -> list[DocumentEntry]:
        """Return every contract the user can open, the newest first."""
        with self.database.session() as session:
            contracts = files.visible_contracts(session, self.actor)
        for contract in contracts:
            self._contracts.setdefault(contract.id, contract)
        return self.entries(contracts)

    def entries(self, contracts: list[ContractView]) -> list[DocumentEntry]:
        """Return the list entries of some contracts, in order, their decisions read at once."""
        with self.database.session() as session:
            signatures = _signatures(session, [contract.id for contract in contracts])
        return [self._entry(contract, signatures.get(contract.id, ())) for contract in contracts]

    def _entry(self, contract: ContractView, signature: Signature) -> DocumentEntry:
        counts = self.workspace.counts(contract, signature)
        return DocumentEntry(
            doc_id=contract.id,
            name=document_name(contract.name),
            kind=contract.kind,  # type: ignore[arg-type]
            size=counts.size,
            found=counts.found,
            check=counts.check,
            not_found=counts.not_found,
            scanned=counts.scanned,
            filename=contract.name,
            analysis_id=contract.doc_id,
            folder_id=contract.folder_id,
            owner_id=contract.owner_id,
            owner_name=contract.owner_name,
            review_status=contract.review_status,
            mine=contract.owner_id == self.actor.id,
        )

    def entry(self, contract_id: str) -> DocumentEntry | None:
        """Return the list entry of a contract, or None if the user cannot open it."""
        try:
            contract = self.contract(contract_id)
        except AccessError:
            return None
        with self.database.session() as session:
            signature = _signatures(session, [contract.id]).get(contract.id, ())
        return self._entry(contract, signature)

    def default_doc_id(self) -> str | None:
        """Return the id of the short sample contract, else of the newest contract, if any."""
        documents = self.documents()
        for entry in documents:
            if entry.filename == DEFAULT_DOCUMENT:
                return entry.doc_id
        return documents[0].doc_id if documents else None

    def find(self, analysis_id: str, folder_id: str | None) -> ContractView | None:
        """Return the user's contract of a text in a folder, to open it instead of a copy."""
        with self.database.session() as session:
            contracts = files.visible_contracts(session, self.actor)
        # At the top level only the user's own space counts, not another's top.
        for contract in contracts:
            same = contract.doc_id == analysis_id and contract.folder_id == folder_id
            if same and (folder_id is not None or contract.owner_id == self.actor.id):
                return contract
        return None

    def add_contract(
        self, folder_id: str | None, name: str, analysis_id: str, kind: str, data: bytes | None
    ) -> ContractView:
        """Add a contract of a stored analysis to a folder, keeping its file."""
        self.analyses.ensure(analysis_id)
        sha256 = self.blobs.put(data, MEDIA_TYPES[kind]) if data is not None else None
        with self.database.session() as session:
            contract = files.add_contract(
                session, self.actor, folder_id, name, analysis_id, kind, sha256, ip=self.ip
            )
        self._contracts[contract.id] = contract
        return contract

    def require_add(self, folder_id: str | None) -> None:
        """Raise unless the user may add contracts to a folder (None: their own top level)."""
        require_permission(self.actor, Permission.ADD_CONTRACTS)
        if folder_id is None:
            return
        with self.database.session() as session:
            listing = files.listing(session, self.actor, folder_id)
        require(listing.level, Action.ADD)

    def pdf_path(self, contract_id: str) -> Path | None:
        """Return a local copy of a contract's PDF, for the page view; None for a text file."""
        contract = self.contract(contract_id)
        if contract.kind != PDF or contract.file_sha256 is None:
            return None
        return self.blobs.local_path(contract.file_sha256, ".pdf")

    # Decisions

    def decision_views(self, contract_id: str) -> dict[str, DecisionView]:
        """Return the decisions on a contract's passages, with who made them."""
        known = self._decisions.get(contract_id)
        if known is None:
            self.contract(contract_id)
            with self.database.session() as session:
                known = files.decisions_of(session, contract_id)
            self._decisions[contract_id] = known
        return known

    def decided(self, contract_id: str) -> dict[str, Decision]:
        """Return the decisions on a contract by clause type."""
        return {name: view.decision for name, view in self.decision_views(contract_id).items()}

    def decided_by(self, contract_id: str, label: str) -> str | None:
        """Return who decided the passage of a clause type, named by its type or its label."""
        view = self.decision_views(contract_id).get(self.clause_type_of(label))
        return view.by_name if view is not None else None

    def decided_status(self, contract_id: str, clause_type: str, status: str) -> str:
        """Return the status to show: the reviewer's verdict when the top passage is theirs."""
        decision = self.decided(contract_id).get(clause_type)
        if decision is None:
            return status
        if decision.verdict == "confirmed" and status == "found":
            return "confirmed"
        return status

    def decide(
        self, contract_id: str, clause_type: str, verdict: str, start: int, end: int, quote: str
    ) -> None:
        """Confirm or reject a passage; the decision and its event are saved together."""
        with self.database.session() as session:
            files.decide(
                session, self.actor, contract_id, clause_type, verdict, start, end, quote, self.ip
            )
        self._decisions.pop(contract_id, None)

    def undo(self, contract_id: str, clause_type: str) -> None:
        """Drop the decision on a clause type of a contract."""
        with self.database.session() as session:
            files.undo_decision(session, self.actor, contract_id, clause_type, self.ip)
        self._decisions.pop(contract_id, None)

    # Chat, feedback and the log

    def conversation(self, contract_id: str) -> Conversation:
        """Return the user's chat about a contract."""
        return self.workspace.conversation(self.actor.id, contract_id)

    def keep_messages(self, contract_id: str, messages: list[Message]) -> None:
        """Save new chat messages; one that cannot be saved stays in memory only."""
        for message in messages:
            try:
                self.chats.append(contract_id, self.actor.id, message.role, _bare(message))
            except Exception:  # a chat that cannot be saved still works
                logger.warning("chat message of %s not saved", contract_id, exc_info=True)

    def replace_message(self, contract_id: str, index: int, message: Message) -> None:
        """Save a changed chat message in its place."""
        try:
            self.chats.replace(contract_id, self.actor.id, index, _bare(message))
        except Exception:  # the change stays in memory
            logger.warning("chat message of %s not updated", contract_id, exc_info=True)

    def clear_chat(self, contract_id: str) -> None:
        """Delete the user's saved chat about a contract and log it."""
        self.chats.clear(contract_id, self.actor.id)
        self.log(audit.CHAT_CLEARED, contract_id=contract_id)

    def record_feedback(self, contract_id: str, verdict: Feedback) -> None:
        """Keep a thumbs up or down on an answer and log it."""
        self.feedback.record(contract_id, self.actor.id, verdict)
        self.log(audit.FEEDBACK_GIVEN, contract_id=contract_id, details={"liked": verdict.liked})

    def log(
        self, action: str, *, contract_id: str | None = None, details: dict | None = None
    ) -> None:
        """Record an event done by the user, such as a download."""
        if contract_id is not None:
            name = self.contract(contract_id).name
            details = {"name": name, **(details or {})}
        with self.database.session() as session:
            audit.record(
                session,
                action,
                actor_id=self.actor.id,
                contract_id=contract_id,
                details=details,
                ip=self.ip,
            )


def _bare(message: Message) -> Message:
    """Return a chat message without its drawn HTML, which is drawn again when shown."""
    return dataclasses.replace(message, html="")


def _signatures(session, contract_ids: list[str]) -> dict[str, Signature]:
    """Return the decisions of some contracts, as what their counts depend on."""
    if not contract_ids:
        return {}
    rows = session.execute(
        select(
            DecisionRow.contract_id,
            DecisionRow.clause_type,
            DecisionRow.verdict,
            DecisionRow.start,
            DecisionRow.end,
        ).where(DecisionRow.contract_id.in_(contract_ids))
    ).all()
    found: dict[str, list[tuple[str, str, int, int]]] = {}
    for contract_id, *decision in rows:
        found.setdefault(contract_id, []).append(tuple(decision))  # type: ignore[arg-type]
    return {key: tuple(sorted(value)) for key, value in found.items()}


class ContractShelf:
    """The contracts a user can open, shaped as the store `portfolio.portfolio_rows` reads."""

    @dataclass(frozen=True)
    class Entry:
        doc_id: str
        source: object

    @dataclass(frozen=True)
    class Source:
        filename: str

    def __init__(self, ws: UserWorkspace):
        self.ws = ws

    def list(self) -> list[Entry]:
        return [
            self.Entry(entry.doc_id, self.Source(entry.filename)) for entry in self.ws.documents()
        ]

    def load(self, doc_id: str, settings) -> DocumentCache:
        return self.ws.load(doc_id)


def attachment(filename: str) -> dict[str, str]:
    """Return the header that offers a response as a download under a file name."""
    quoted = quote(filename)
    if quoted == filename:
        return {"Content-Disposition": f'attachment; filename="{filename}"'}
    return {"Content-Disposition": f"attachment; filename*=utf-8''{quoted}"}


def client_ip(request: Request) -> str | None:
    """Return the address a request came from."""
    return request.client.host if request.client else None


def current_workspace(request: Request) -> UserWorkspace | None:
    """Return the workspace of the signed-in user of a request, or None when signed out."""
    ws = getattr(request.state, "ws", None)
    if ws is not None:
        return ws
    actor = getattr(request.state, "actor", None)
    if actor is None:
        return None
    ws = request.app.state.workspace.for_user(actor, client_ip(request))
    request.state.ws = ws
    return ws


def get_workspace(request: Request) -> UserWorkspace:
    ws = current_workspace(request)
    if ws is None:
        raise HTTPException(status_code=401, detail=SIGN_IN_FIRST)
    return ws


def get_thresholds(
    request: Request, low: float | None = None, high: float | None = None
) -> Thresholds:
    return thresholds_of(request.app.state.workspace.settings, low, high)


Ws = Annotated[UserWorkspace, Depends(get_workspace)]
Tau = Annotated[Thresholds, Depends(get_thresholds)]


def view_url(
    doc_id: str,
    label: str,
    citation: Citation | None = None,
    status: str | None = None,
    mode: str | None = None,
) -> str:
    """Return the URL of the document pane showing the span of a citation under a label.

    The label is a clause type's display name, DEFINITION_LABEL, RELATED_LABEL or a section
    label. The status, when given, is a tier or "definition" or "related" and picks the badge.
    """
    params: dict[str, object] = {"label": label}
    if citation is not None:
        params.update(start=citation.start, end=citation.end, conf=f"{citation.conf:g}")
    if status:
        params["status"] = status
    if mode:
        params["mode"] = mode
    return f"/documents/{doc_id}/view?{urlencode(params)}"


def frame(
    html: str | None = None,
    *,
    oob: dict[str, str] | None = None,
    load: dict[str, str] | None = None,
    notice: str | None = None,
    redirect: str | None = None,
    activate: str | None = None,
    done: bool = False,
) -> str:
    """Return one line of a stream. The browser applies the keys in this order:

    - html: replaces the inner HTML of the stream's target;
    - oob: {element id: inner HTML} for other elements of the page;
    - load: {element id: URL} fetched with GET into those elements;
    - notice: a short message shown as a toast;
    - activate: the name of the insights tab to mark active;
    - redirect: a URL the page navigates to;
    - done: the stream is complete (sent on the last line).
    """
    payload = {
        "html": html,
        "oob": oob,
        "load": load,
        "notice": notice,
        "activate": activate,
        "redirect": redirect,
        "done": True if done else None,
    }
    return json.dumps({key: value for key, value in payload.items() if value is not None}) + "\n"


def stream(frames: Iterator[str]) -> StreamingResponse:
    """Send frames as they come. A problem becomes a notice; a closed connection ends it."""
    guarded = _guarded(frames)

    async def body():
        try:
            async for chunk in iterate_in_threadpool(guarded):
                yield chunk
        finally:
            _close(guarded)

    return StreamingResponse(body(), media_type=NDJSON, headers={"Cache-Control": "no-store"})


def _guarded(frames: Iterator[str]) -> Iterator[str]:
    try:
        yield from frames
    except WorkspaceError as error:
        yield frame(notice=error.message, done=True)
    except AccessError as error:
        yield frame(notice=error.message, done=True)
    except LLMError:
        logger.exception("generation failed")
        yield frame(notice=GENERATION_FAILED, done=True)
    except Exception:
        logger.exception("stream failed")
        yield frame(notice=UNEXPECTED_ERROR, done=True)


def _close(generator: Iterator[str]) -> None:
    try:
        generator.close()  # type: ignore[attr-defined]
    except (ValueError, AttributeError):
        # Still running in its thread; it ends when that step completes.
        pass


def shorten(text: str, limit: int = SHORT_CHARS) -> str:
    """Return the text on one line, cut with an ellipsis when it is longer than limit."""
    squashed = " ".join(text.split())
    if len(squashed) <= limit:
        return squashed
    return squashed[: limit - len(ELLIPSIS)] + ELLIPSIS


def _percent(conf: float) -> str:
    return f"{conf:.0%}"


BADGE_LABELS = copy.STATUS


@pass_context
def _status_label(context, status: str) -> str:
    """The badge text of a status in the language of the page being rendered."""
    return copy_for(context.get("lang", "en")).STATUS.get(status, status)


templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters.update(
    display_name=display_name,
    document_name=document_name,
    percent=_percent,
    status_label=_status_label,
    severity_label=lambda severity: SEVERITY_LABELS.get(severity, severity),
    shorten=shorten,
    squash=lambda text: " ".join(str(text).split()),
)
templates.env.globals.update(
    view_url=view_url,
    copy=copy,
    DEFINITION_LABEL=DEFINITION_LABEL,
    RELATED_LABEL=RELATED_LABEL,
)


def language(request: Request) -> str:
    """Return the interface language of a request."""
    lang = getattr(request.state, "lang", None)
    return lang or language_of(request)


def _language_context(request: Request, context: dict) -> dict:
    lang = language(request)
    context.setdefault("ws", current_workspace(request))
    context.setdefault("actor", getattr(request.state, "actor", None))
    context.setdefault("lang", lang)
    context.setdefault("copy", copy_for(lang))
    context.setdefault("_", lambda text: translate(text, lang))
    return context


def render(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    """Render a template as the response; the workspace is `ws`, the words `copy` and `_`."""
    context = _language_context(request, context)
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def render_string(request: Request, name: str, **context) -> str:
    """Render a template to a string, for a frame or an out-of-band update."""
    context = _language_context(request, context)
    context.setdefault("request", request)
    return templates.get_template(name).render(context)
