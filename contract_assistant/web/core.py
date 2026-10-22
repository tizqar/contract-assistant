"""Shared state and helpers of the web app: the workspace, thresholds, templates and streams.

The app is server-rendered. Every route returns HTML, a partial for htmx or a whole page,
except the streaming routes, which send newline-delimited JSON frames (see `frame`) that
`static/app.js` applies as they arrive. All URLs are free of the thresholds: the browser
appends `low` and `high` to every request from the hidden inputs of the page.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import iterate_in_threadpool

from contract_assistant.answer import Answer, Citation, Definition, tier
from contract_assistant.decisions import DECISIONS_DIR, Decision, DecisionStore
from contract_assistant.decisions import apply as apply_decisions
from contract_assistant.definitions import find_definitions
from contract_assistant.fulltext import FULLTEXT_DIR, FullTextStore
from contract_assistant.obligations import OBLIGATIONS_DIR, ObligationStore
from contract_assistant.review import Flag, load_rules, review
from contract_assistant.router.base import Turn
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.services import Services
from contract_assistant.store import (
    CacheEntry,
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    Store,
)
from contract_assistant.summary import SUMMARY_DIR, SummaryStore
from contract_assistant.term_sheet import TERM_SHEET_DIR, TermSheetStore
from contract_assistant.ui.views import SEVERITY_LABELS, display_name, document_name
from contract_assistant.web import copy

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
MAX_CONVERSATIONS = 8
KEEP_WARM_S = 1500  # Ollama unloads an idle model after 30 minutes; load it again before then.
SHORT_CHARS = 120
ELLIPSIS = "…"

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
    """The chat about one document, kept in memory while the server runs."""

    messages: list[Message] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    last: Answer | None = None

    def remember(self, turn: Turn) -> None:
        """Keep the turn for the router, forgetting the oldest beyond the limit."""
        self.turns = [*self.turns, turn][-MAX_REMEMBERED_TURNS:]


@dataclass(frozen=True)
class DocumentEntry:
    """A cached contract as the lists show it."""

    doc_id: str
    name: str
    kind: Literal["pdf", "txt"]
    size: str
    found: int
    check: int
    not_found: int = 0


class DecidedStore:
    """The store as the reader's decisions make it: every cache it loads has them applied."""

    def __init__(self, store: Store, decisions: DecisionStore):
        self.store = store
        self.decisions = decisions

    def list(self) -> list[CacheEntry]:
        return self.store.list()

    def find(self, doc_id: str) -> Path | None:
        return self.store.find(doc_id)

    def save(self, cache: DocumentCache) -> Path:
        return self.store.save(cache)

    def load(self, doc_id: str, settings) -> DocumentCache | None:
        cache = self.store.load(doc_id, settings)
        if cache is None:
            return None
        return apply_decisions(cache, self.decisions.load(cache.doc_id))

    def load_path(self, path: Path, settings) -> DocumentCache:
        cache = self.store.load_path(path, settings)
        return apply_decisions(cache, self.decisions.load(cache.doc_id))


class Workspace:
    """The state every route shares: the services, the saved generations and the chats."""

    def __init__(self, services: Services):
        self.services = services
        settings = services.settings
        self.names = [clause_type.name for clause_type in services.clause_types]
        self.clause_by_name = {display_name(name): name for name in self.names}
        self.decisions = DecisionStore(settings.cache_dir / DECISIONS_DIR)
        self.store = DecidedStore(services.store, self.decisions)
        self.summaries = {
            ENGLISH: services.summaries or SummaryStore(settings.cache_dir / SUMMARY_DIR),
            SERBIAN: SummaryStore(settings.cache_dir / SUMMARY_DIR, SERBIAN),
        }
        self.term_sheets = services.term_sheets or TermSheetStore(
            settings.cache_dir / TERM_SHEET_DIR
        )
        self.obligations = services.obligations or ObligationStore(
            settings.cache_dir / OBLIGATIONS_DIR
        )
        self.fulltexts = services.fulltexts or FullTextStore(settings.cache_dir / FULLTEXT_DIR)
        self.rules = services.rules or load_rules(known_types=self.names)
        self._definitions: OrderedDict[str, list[Definition]] = OrderedDict()
        self._conversations: OrderedDict[str, Conversation] = OrderedDict()
        self._entries: dict[tuple[Path, float], DocumentEntry] = {}

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

    def documents(self) -> list[DocumentEntry]:
        """Return every cached contract once, uploads first, then the demo contracts."""
        entries: dict[str, DocumentEntry] = {}
        for cached in self.store.list():
            entries.setdefault(cached.doc_id, self._entry(cached))
        return list(entries.values())

    def _entry(self, cached: CacheEntry) -> DocumentEntry:
        decided = self.decisions.path_of(cached.doc_id)
        decided_at = decided.stat().st_mtime if decided.is_file() else 0.0
        key = (cached.path, cached.path.stat().st_mtime + decided_at)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._build_entry(cached)
            self._entries[key] = entry
        return entry

    def _build_entry(self, cached: CacheEntry) -> DocumentEntry:
        settings = self.settings
        found = check = not_found = 0
        size = f"{cached.source.n_pages} pages"
        try:
            cache = self.store.load_path(cached.path, settings)
        except (CacheIntegrityError, CacheMismatchError):
            cache = None
        if cache is not None:
            if cached.source.n_pages is None:
                size = f"{len(cache.text.split()):,} words"
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
        return DocumentEntry(
            doc_id=cached.doc_id,
            name=document_name(cached.source.filename),
            kind=cached.source.kind,
            size=size,
            found=found,
            check=check,
            not_found=not_found,
        )

    def entry(self, doc_id: str) -> DocumentEntry | None:
        """Return the list entry of a document, or None if it is not cached."""
        return next((entry for entry in self.documents() if entry.doc_id == doc_id), None)

    def default_doc_id(self) -> str | None:
        """Return the id of the short demo contract, else of the first contract."""
        documents = self.documents()
        for entry in documents:
            if entry.name == document_name(DEFAULT_DOCUMENT):
                return entry.doc_id
        return documents[0].doc_id if documents else None

    def load(self, doc_id: str) -> DocumentCache:
        """Load a cached contract; 404 when unknown, a notice when its analysis is damaged."""
        try:
            cache = self.store.load(doc_id, self.settings)
        except (CacheIntegrityError, CacheMismatchError) as error:
            logger.warning("cannot load %s: %s", doc_id, error)
            raise WorkspaceError(DAMAGED_ANALYSIS) from error
        if cache is None:
            raise HTTPException(status_code=404, detail=NOT_ANALYSED)
        return cache

    def definitions(self, cache: DocumentCache) -> list[Definition]:
        """Return the defined terms of a contract, found once per server run."""
        found = self._definitions.get(cache.doc_id)
        if found is None:
            found = find_definitions(cache.text, cache.page_starts)
            self._definitions[cache.doc_id] = found
            while len(self._definitions) > MAX_CACHED_DEFINITIONS:
                self._definitions.popitem(last=False)
        return found

    def flags(self, cache: DocumentCache, tau: Thresholds) -> list[Flag]:
        """Return the review flags of a contract at the given thresholds."""
        return review(cache, self.rules, tau.low, tau.high, self.services.unvalidated)

    def conversation(self, doc_id: str) -> Conversation:
        """Return the chat about a document, starting one if there is none."""
        conversation = self._conversations.get(doc_id)
        if conversation is None:
            conversation = Conversation()
            self._conversations[doc_id] = conversation
            while len(self._conversations) > MAX_CONVERSATIONS:
                self._conversations.popitem(last=False)
        self._conversations.move_to_end(doc_id)
        return conversation

    def clause_type_of(self, label: str) -> str:
        """Return the clause type shown under a label; another label is returned as is."""
        return self.clause_by_name.get(label, label)

    def decided(self, doc_id: str) -> dict[str, Decision]:
        """Return the reader's decisions on a contract by clause type."""
        return self.decisions.load(doc_id)

    def decided_status(self, doc_id: str, clause_type: str, status: str) -> str:
        """Return the status to show: the reader's verdict when the top passage is theirs."""
        decision = self.decided(doc_id).get(clause_type)
        if decision is None:
            return status
        if decision.verdict == "confirmed" and status == "found":
            return "confirmed"
        return status

    def summary_store(self, language: str) -> SummaryStore:
        """Return the store of the summaries written in a language."""
        store = self.summaries.get(language)
        if store is None:
            store = SummaryStore(self.settings.cache_dir / SUMMARY_DIR, language)
            self.summaries[language] = store
        return store


def get_workspace(request: Request) -> Workspace:
    return request.app.state.workspace


def get_thresholds(
    request: Request, low: float | None = None, high: float | None = None
) -> Thresholds:
    return thresholds_of(request.app.state.workspace.settings, low, high)


Ws = Annotated[Workspace, Depends(get_workspace)]
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

templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.filters.update(
    display_name=display_name,
    document_name=document_name,
    percent=_percent,
    status_label=lambda status: BADGE_LABELS.get(status, status),
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


def render(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    """Render a template as the response; the workspace is in the context as `ws`."""
    context.setdefault("ws", get_workspace(request))
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def render_string(request: Request, name: str, **context) -> str:
    """Render a template to a string, for a frame or an out-of-band update."""
    context.setdefault("ws", get_workspace(request))
    context.setdefault("request", request)
    return templates.get_template(name).render(context)
