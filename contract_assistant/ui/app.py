"""Gradio app: layout and event wiring over the view functions."""

import functools
import inspect
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from contract_assistant.answer import (
    Answer,
    Definition,
    Passage,
    Status,
    answer,
    restrict_to_cache,
)
from contract_assistant.background import precompute, run_in_background, warm
from contract_assistant.clause_types import ClauseType
from contract_assistant.config import Settings
from contract_assistant.definitions import find_definitions, lookup
from contract_assistant.explain import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    explanation_from,
    explanation_request,
)
from contract_assistant.extract.extractor import Extractor
from contract_assistant.generation import Generation, Request, stream
from contract_assistant.intents import Intent, detect_intent
from contract_assistant.passage_answer import APP_ATTEMPTS as PASSAGE_ATTEMPTS
from contract_assistant.passage_answer import generated_answer_from, passage_request
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.router.base import Router, RouteResult, Turn
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.search import PassageSearch
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    Store,
    doc_id_of,
)
from contract_assistant.summary import (
    APP_ATTEMPTS,
    SUMMARY_DIR,
    Summary,
    SummaryStore,
    summary_from,
    summary_items,
    summary_request,
)
from contract_assistant.term_sheet import APP_ATTEMPTS as TERM_SHEET_ATTEMPTS
from contract_assistant.term_sheet import (
    TERM_SHEET_DIR,
    TermSheetStore,
    request_term_sheet,
    term_sheet_items,
)
from contract_assistant.ui.views import (
    CLAUSE_HEADERS,
    DEFINITION_HEADERS,
    DEFINITION_LABEL,
    DEFINITION_VALUE,
    PASSAGE_PREFIX,
    RELATED_LABEL,
    STATUS_LABELS,
    SUGGESTED_QUESTIONS,
    chat_reply,
    checklist_intro,
    clause_rows,
    definition_citation,
    definition_rows,
    display_name,
    document_name,
    document_title,
    document_view,
    explanation_markdown,
    highlighted_quote,
    passage_choices,
    passage_citation,
    sources_json,
    streaming_markdown,
    summary_markdown,
    technical_markdown,
    term_rows,
    top_citation,
)

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT = "short_supply_agreement.txt"
HIGHLIGHT_COLOR = "#fcd34d"
TABLE_MAX_HEIGHT = 460
DEFINITIONS_MAX_HEIGHT = 260
CHAT_HEIGHT = 400
UPLOAD_HEIGHT = 120
SLIDER_STEP = 0.01
KEEP_WARM_S = 1500  # Ollama unloads an idle model after 30 minutes; load it again before then.
WRITING_SUMMARY = "Writing the summary"
WRITING_EXPLANATION = "Writing the explanation"
WRITING_ANSWER = "Writing the answer from the passages"
TERM_SHEET_STATUS = "Extracting the key terms. This can take up to half a minute."
ASK_OTHER_OUTPUTS = 7
MAX_REMEMBERED_TURNS = 8
MAX_CACHED_DEFINITIONS = 4
UNEXPECTED_ERROR = "Something went wrong; see the server log."
DAMAGED_ANALYSIS = "The saved analysis of this contract is damaged or inconsistent."
NO_MODEL = (
    "This contract cannot be analysed on this computer yet: "
    "the analysis model is not installed. See the README."
)
UPLOAD_ERRORS = (UnsupportedFileError, EmptyDocumentError, CacheIntegrityError)

DOCUMENT_TAB = "tab-document"
SUMMARY_TAB = "tab-summary"
CHECKLIST_TAB = "tab-checklist"
SETTINGS_TAB = "tab-settings"
TECHNICAL_TAB = "tab-technical"
ALL_CLAUSES = "all"
STATUS_FILTER_CHOICES = [
    ("All", ALL_CLAUSES),
    *((label, status) for status, label in STATUS_LABELS.items()),
]

FONT_SANS = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"]
FONT_MONO = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]
THEME = gr.themes.Soft(
    primary_hue="indigo", neutral_hue="slate", font=FONT_SANS, font_mono=FONT_MONO
)
COLUMN_WIDTHS = ["14%", "19%", "11%", "15%", "8%", "33%"]
COLUMN_TYPES = ["str", "str", "html", "str", "str", "str"]
INTRO = "Ask questions about a contract. Every answer quotes the contract word for word, or says that the clause was not found."
DISCLAIMER = (
    "Passages are located automatically. Check each highlighted passage before relying on it; "
    "this is not legal advice."
)
CHAT_PLACEHOLDER = (
    "**Ask a question about this contract**\n\n"
    "Ask, for example, who the parties are, which law governs or whether either side may terminate.\n\n"
    "Each answer is marked Found, Verify (read the passage before relying on it) "
    "or Not found (the contract does not appear to contain the clause)."
)
SELECT_CONTRACT = "Select a contract first."
SUMMARIES_OFF = (
    "Summaries and explanations are off: they need a local language model. See the README."
)
TERMS_INTRO = (
    "### Key terms\n\n"
    "The values are copied word for word from the clauses found in this contract. "
    "A value the language model gave that is not in a clause is shown as not verified. "
    "A term the clauses do not state is shown as not stated."
)
NO_TERM_CLAUSES = "None of the clauses the key terms are read from was found in this contract."
TERM_SHEET_WITHHELD = "The term sheet is not shown because {reason}."
TERM_HEADERS = ["Term", "Value", "From clause", "Page"]
TERM_DOWNLOAD_LABEL = "Download the term sheet (JSON)"
GENERATION_FAILED = "The language model did not answer; see the server log."
NOTHING_TO_EXPLAIN = "Ask about a clause first, then say 'explain that'."
NOTHING_TO_SUMMARIZE = "No clause was found in this contract, so there is nothing to summarize."
NOTHING_HIGHLIGHTED = (
    "Nothing is highlighted. Ask a question or select a clause in the checklist first."
)
SUMMARY_INTRO_ON = (
    "A short overview written by a local language model from the clauses found in this "
    "contract. Text in double quotes is checked word for word against the contract; "
    "the rest is the model's own wording. Read the passages before relying on it."
)
SUMMARY_INTRO_OFF = "Summaries need a local language model. See the README."
LANGUAGE_CHOICES = [(name.split(",")[0], code) for code, name in LANGUAGES.items()]
ANALYSIS_UPDATED = "Analysis updated"
THRESHOLD_HELP = (
    "A passage at or above the first value is marked Found. Between the two values it is "
    "marked Verify. Below the second value the clause is reported as not found. "
    "The defaults were chosen on validation data."
)
SCROLL_TO_HIGHLIGHT = """() => {
  const token = document.querySelector('#view .token:not(:has(.unlabeled))');
  if (token) token.scrollIntoView({block: 'center'});
}"""

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto; }
#top { gap: 2px; }
#header, #disclaimer { padding: 0 !important; }
#header h1 { margin: 0 0 4px; }
#header p, #disclaimer p { margin: 0; }
#disclaimer p { color: var(--body-text-color-subdued); font-size: 0.85em; }
.tier {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid transparent;
  font-size: 0.85em;
  white-space: nowrap;
}
.tier-found { background: #1f7a4d; color: #ffffff; }
.tier-check { background: transparent; border-color: #b45309; color: #92400e; }
.tier-not_found { background: transparent; border-color: #9ca3af; color: #6b7280; }
.tier-related { background: #e5e7eb; color: #374151; }
.tier-generated { background: #dbeafe; color: #1e3a8a; }
.tier-definition { background: #ede9fe; color: #4c1d95; }
.dark .tier-found { background: #34a56f; color: #06210f; }
.dark .tier-check { border-color: #fbbf24; color: #fbbf24; }
.dark .tier-not_found { border-color: #6b7280; color: #9ca3af; }
.dark .tier-related { background: #374151; color: #e5e7eb; }
.dark .tier-generated { background: #1e3a8a; color: #dbeafe; }
.dark .tier-definition { background: #4c1d95; color: #ede9fe; }
#chat .message, #clause-table td, #clause-table td * {
  overflow-wrap: anywhere;
  white-space: normal;
}
#chat .message .tier { white-space: nowrap; }
#suggestions { gap: 6px; }
#suggestions button {
  flex: 0 0 auto;
  min-width: 0;
  width: auto;
  padding: 4px 12px;
  font-size: 0.8em;
  font-weight: 400;
  box-shadow: none;
  background: transparent;
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
}
#ask { height: 100%; }
#clause-table th, #clause-table th * { white-space: nowrap; word-break: keep-all; }
#upload svg.feather-upload { display: none; }
#clause-table, #clause-table * { font-family: var(--font); }
#definitions-table, #definitions-table * { font-family: var(--font); }
#view { max-height: 520px; overflow-y: auto; }
#view .textfield {
  display: block;
  line-height: 1.7;
  word-break: normal;
  overflow-wrap: break-word;
}
#view .token-container { display: inline; }
#view .line-break { display: block; }
#view .text, #view .token { word-break: normal; overflow-wrap: break-word; }
"""


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


def guarded(handler: Callable) -> Callable:
    """Turn an unexpected exception into a warning and an unchanged page."""
    if inspect.isgeneratorfunction(handler):

        @functools.wraps(handler)
        def stream_wrapper(*args, **kwargs):
            try:
                yield from handler(*args, **kwargs)
            except Exception:
                logger.exception("handler %s failed", handler.__name__)
                gr.Warning(UNEXPECTED_ERROR)
                yield gr.skip()

        return stream_wrapper

    @functools.wraps(handler)
    def wrapper(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except Exception:
            logger.exception("handler %s failed", handler.__name__)
            gr.Warning(UNEXPECTED_ERROR)
            return gr.skip()

    return wrapper


def _stream_checked(
    client: LLMClient, what: str, request: Request, attempts: int
) -> Iterator[str | Generation]:
    """Yield the status and growing text as Markdown, then the Generation the request ends with."""
    started = time.perf_counter()
    yield streaming_markdown(what, "", started)
    for item in stream(client, request, attempts):
        if isinstance(item, Generation):
            yield item
        else:
            yield streaming_markdown(what, item.text, started, item.previous_reason)


def document_choices(store: Store) -> list[tuple[str, str]]:
    """Return (display name, document id) for every cached document, once per id."""
    choices: dict[str, str] = {}
    for entry in store.list():
        choices.setdefault(entry.doc_id, document_name(entry.source.filename))
    return [(name, doc_id) for doc_id, name in choices.items()]


def default_document(choices: list[tuple[str, str]]) -> str | None:
    """Return the id of the short demo document, else the first one."""
    for name, doc_id in choices:
        if name == document_name(DEFAULT_DOCUMENT):
            return doc_id
    return choices[0][1] if choices else None


def status_of_filter(value: str | None) -> Status | None:
    """Map the value of the status filter to a status, or None for all clauses."""
    return None if value in (None, ALL_CLAUSES) else value


def show_tab(tab_id: str) -> gr.Tabs:
    return gr.Tabs(selected=tab_id)


class Handlers:
    """The event handlers of the app, one method per event."""

    def __init__(self, services: Services):
        self.services = services
        self.clause_by_name = {
            display_name(clause_type.name): clause_type.name
            for clause_type in services.clause_types
        }
        self.summaries = services.summaries or SummaryStore(
            services.settings.cache_dir / SUMMARY_DIR
        )
        self.term_sheets = services.term_sheets or TermSheetStore(
            services.settings.cache_dir / TERM_SHEET_DIR
        )
        self.definitions: OrderedDict[str, list[Definition]] = OrderedDict()

    @guarded
    def open_document(
        self, doc_id: str | None, tau_low: float, tau_high: float, status_filter: str
    ):
        """Load a document and reset the chat, the view and the checklist."""
        if not doc_id:
            return gr.skip()
        services = self.services
        try:
            cache = services.store.load(doc_id, services.settings)
        except (CacheIntegrityError, CacheMismatchError) as error:
            logger.warning("cannot load %s: %s", doc_id, error)
            gr.Warning(DAMAGED_ANALYSIS)
            return gr.skip()
        if cache is None:
            gr.Warning("This contract has not been analysed yet.")
            return gr.skip()
        label, segments = document_view(cache, None, None)
        path = services.store.find(doc_id)
        rows = clause_rows(
            cache, tau_low, tau_high, status_of_filter(status_filter), services.unvalidated
        )
        return (
            cache,
            None,
            [],
            document_title(cache),
            rows,
            checklist_intro(cache, services.unvalidated),
            label,
            segments,
            gr.Radio(choices=[], value=None, visible=False),
            {},
            gr.DownloadButton(value=str(path)),
            [],
            definition_rows(self._definitions(cache)),
        )

    @guarded
    def ask(
        self,
        question: str,
        history: list[dict] | None,
        cache: DocumentCache | None,
        tau_low: float,
        tau_high: float,
        turns: list[dict] | None = None,
        last: Answer | None = None,
    ):
        """Answer a definition or a chat command, else route a question and show the passage."""
        if not question or not question.strip():
            gr.Warning("Type a question first.")
            yield gr.skip()
            return
        if cache is None:
            gr.Warning(SELECT_CONTRACT)
            yield gr.skip()
            return
        services = self.services
        asked = [*(history or []), {"role": "user", "content": question}]
        definition = lookup(self._definitions(cache), question)
        if definition is not None:
            yield self._definition_reply(question, definition, asked, cache, tau_low, tau_high)
            return
        intent = detect_intent(question)
        if intent is not None:
            yield from self._command(intent, asked, cache, tau_low, tau_high, last)
            return
        earlier = [Turn.model_validate(turn) for turn in turns or []]
        route = restrict_to_cache(services.router.route(question, earlier), cache)
        turn = Turn(question=question, clause_types=route.clause_types)
        remembered = [*(turns or []), turn.model_dump()][-MAX_REMEMBERED_TURNS:]
        try:
            result = answer(question, route, cache, tau_low, tau_high, services.unvalidated)
        except ValueError as error:
            gr.Warning(str(error))
            yield gr.skip()
            return
        if not route.clause_types:
            passages = services.search.search(cache, question)
            generated = yield from self._stream_answer(question, passages, asked)
            result = result.model_copy(update={"passages": passages, "generated": generated})
        reply = {"role": "assistant", "content": chat_reply(result, services.unvalidated)}
        picker, label, segments = _citation_updates(result, cache)
        yield (
            "",
            [*asked, reply],
            result,
            picker,
            label,
            segments,
            sources_json(result, cache),
            show_tab(DOCUMENT_TAB),
            remembered,
        )

    @guarded
    def pick_citation(
        self, clause_type: str | None, last: Answer | None, cache: DocumentCache | None
    ):
        """Show the passage of the chosen clause type of the last answer."""
        if cache is None or last is None:
            return gr.skip()
        for item in last.items:
            if item.clause_type == clause_type and item.citation:
                return document_view(cache, clause_type, item.citation)
        if clause_type == DEFINITION_VALUE and last.definition is not None:
            citation = definition_citation(last.definition)
            return document_view(cache, DEFINITION_LABEL, citation)
        for index, passage in enumerate(last.passages):
            if clause_type == f"{PASSAGE_PREFIX}{index}":
                return document_view(cache, RELATED_LABEL, passage_citation(passage))
        return gr.skip()

    @guarded
    def filter_clauses(
        self, status_filter: str, tau_low: float, tau_high: float, cache: DocumentCache | None
    ):
        """Show only the checklist rows with the chosen status."""
        return _table(cache, tau_low, tau_high, status_filter, self.services.unvalidated)

    @guarded
    def release_low(
        self, tau_low: float, tau_high: float, cache: DocumentCache | None, status_filter: str
    ):
        """Keep the Verify threshold at or below the Found threshold and re-tier the checklist."""
        tau_low = min(tau_low, tau_high)
        return tau_low, _table(
            cache, tau_low, tau_high, status_filter, self.services.unvalidated
        )

    @guarded
    def release_high(
        self, tau_low: float, tau_high: float, cache: DocumentCache | None, status_filter: str
    ):
        """Keep the Found threshold at or above the Verify threshold and re-tier the checklist."""
        tau_high = max(tau_high, tau_low)
        return tau_high, _table(
            cache, tau_low, tau_high, status_filter, self.services.unvalidated
        )

    @guarded
    def reset_thresholds(self, cache: DocumentCache | None, status_filter: str):
        """Restore the default thresholds and re-tier the checklist."""
        services = self.services
        tau_low, tau_high = services.settings.tau_low, services.settings.tau_high
        return (
            tau_low,
            tau_high,
            _table(cache, tau_low, tau_high, status_filter, services.unvalidated),
        )

    @guarded
    def show_row(self, cache: DocumentCache | None, evt: gr.SelectData):
        """Show the top passage of the selected checklist row in the Document tab."""
        if cache is None:
            return gr.skip()
        name = evt.row_value[1]
        clause_type = self.clause_by_name.get(name)
        citation = top_citation(cache, clause_type) if clause_type else None
        if citation is None:
            gr.Info(f"No passage found for {name}.")
            return gr.skip()
        return *document_view(cache, clause_type, citation), show_tab(DOCUMENT_TAB)

    @guarded
    def show_definition(self, cache: DocumentCache | None, evt: gr.SelectData):
        """Show the sentence that defines the selected term in the Document tab."""
        if cache is None:
            return gr.skip()
        term = evt.row_value[0]
        for definition in self._definitions(cache):
            if definition.term == term:
                citation = definition_citation(definition)
                label, segments = document_view(cache, DEFINITION_LABEL, citation)
                return label, segments, show_tab(DOCUMENT_TAB)
        return gr.skip()

    @guarded
    def summarize(self, cache: DocumentCache | None, tau_low: float, tau_high: float):
        """Summarize the clauses found in the open contract, reusing a saved summary."""
        services = self.services
        gen = services.gen
        if cache is None:
            gr.Warning(SELECT_CONTRACT)
            yield gr.skip()
            return
        if gen is None:
            gr.Warning(SUMMARIES_OFF)
            yield gr.skip()
            return
        try:
            for step in self._summary_steps(cache, tau_low, tau_high):
                if isinstance(step, Summary):
                    yield summary_markdown(step, services.unvalidated)
                else:
                    yield step
        except ValueError as error:
            gr.Warning(str(error))
            yield gr.skip()
        except LLMError:
            logger.exception("summary of %s failed", cache.doc_id)
            gr.Warning(GENERATION_FAILED)
            yield gr.skip()

    @guarded
    def term_sheet(self, cache: DocumentCache | None, tau_low: float, tau_high: float):
        """Extract the key terms of the open contract, reusing a saved term sheet."""
        services = self.services
        gen = services.gen
        if cache is None:
            gr.Warning(SELECT_CONTRACT)
            yield gr.skip()
            return
        if gen is None:
            gr.Warning(SUMMARIES_OFF)
            yield gr.skip()
            return
        try:
            items = term_sheet_items(cache, tau_low, tau_high, services.unvalidated)
        except ValueError as error:
            gr.Warning(str(error))
            yield gr.skip()
            return
        if not items:
            gr.Info(NO_TERM_CLAUSES)
            yield gr.skip()
            return
        sheet = self.term_sheets.load(cache.doc_id, gen.model, tau_low, tau_high)
        if sheet is None:
            yield gr.skip(), gr.skip(), TERM_SHEET_STATUS
            try:
                sheet = request_term_sheet(
                    cache.doc_id, items, gen, tau_low, tau_high, attempts=TERM_SHEET_ATTEMPTS
                )
            except LLMError:
                logger.exception("term sheet of %s failed", cache.doc_id)
                gr.Warning(GENERATION_FAILED)
                yield gr.skip(), gr.skip(), ""
                return
            # A withheld term sheet is not saved, so the next click asks the model again.
            if sheet.reason is None:
                self.term_sheets.save(sheet)
        if sheet.reason is not None:
            gr.Warning(TERM_SHEET_WITHHELD.format(reason=sheet.reason))
            yield term_rows(sheet), gr.DownloadButton(visible=False), ""
            return
        path = self.term_sheets.path_of(sheet)
        yield term_rows(sheet), gr.DownloadButton(value=str(path), visible=True), ""

    @guarded
    def explain(self, segments: list | None, language: str, cache: DocumentCache | None):
        """Restate the highlighted passage in plain language."""
        gen = self.services.gen
        if cache is None:
            gr.Warning(SELECT_CONTRACT)
            yield gr.skip()
            return
        if gen is None:
            gr.Warning(SUMMARIES_OFF)
            yield gr.skip()
            return
        highlight = highlighted_quote(segments)
        if highlight is None:
            gr.Info(NOTHING_HIGHLIGHTED)
            yield gr.skip()
            return
        quote, label = highlight
        clause_type = self.clause_by_name.get(label, label)
        request = explanation_request(clause_type, quote, language)
        try:
            for step in _stream_checked(gen, WRITING_EXPLANATION, request, 1):
                if isinstance(step, Generation):
                    explanation = explanation_from(step, clause_type, quote, language, gen)
                else:
                    yield step
        except LLMError:
            logger.exception("explanation of %s failed", clause_type)
            gr.Warning(GENERATION_FAILED)
            yield gr.skip()
            return
        yield explanation_markdown(explanation)

    @guarded
    def extract_upload(self, path: str | None, progress: gr.Progress = gr.Progress()):  # noqa: B008
        """Analyse an uploaded file unless it was analysed before, then select it."""
        if not path:
            gr.Warning("Choose a PDF or TXT file first.")
            return gr.skip()
        services = self.services
        try:
            doc = load_document(Path(path))
            doc_id = doc_id_of(doc.text)
            cached = self._cached(doc_id)
        except UPLOAD_ERRORS as error:
            gr.Warning(str(error))
            return gr.skip()
        extractor = services.extractor
        report = _reporter(progress)
        if cached is not None and extractor is not None:
            extended = extractor.extend(cached, report)
            if extended is not cached:
                services.store.save(extended)
                self._precompute(extended)
                return self._selected(doc_id), ANALYSIS_UPDATED, doc_id
        if cached is not None:
            gr.Info(f"{document_name(doc.filename)}: already analysed")
            return self._selected(doc_id), "Already analysed", None
        if extractor is None:
            gr.Warning(NO_MODEL)
            return gr.skip()
        analysed = extractor.extract(doc, report)
        services.store.save(analysed)
        self._precompute(analysed)
        return self._selected(doc_id), "Analysis complete", None

    @guarded
    def reload_extended(
        self, doc_id: str | None, tau_low: float, tau_high: float, status_filter: str
    ):
        """Reopen a contract whose analysis was just extended; do nothing otherwise."""
        if not doc_id:
            return gr.skip()
        result = self.open_document(doc_id, tau_low, tau_high, status_filter)
        return (*result, None) if isinstance(result, tuple) else result

    @guarded
    def keep_warm(self):
        """Load the language models again so that Ollama does not unload them while idle."""
        services = self.services
        warm([services.gen, getattr(services.router, "client", None)])

    def _summary_steps(
        self, cache: DocumentCache, tau_low: float, tau_high: float
    ) -> Iterator[str | Summary]:
        services = self.services
        gen = services.gen
        names = [clause_type.name for clause_type in services.clause_types]
        items = summary_items(cache, names, tau_low, tau_high, services.unvalidated)
        if not items:
            yield NOTHING_TO_SUMMARIZE
            return
        summary = self.summaries.load(cache.doc_id, gen.model, tau_low, tau_high)
        if summary is None:
            request = summary_request(cache.doc_id, items)
            for step in _stream_checked(gen, WRITING_SUMMARY, request, APP_ATTEMPTS):
                if isinstance(step, Generation):
                    summary = summary_from(step, cache.doc_id, items, gen, tau_low, tau_high)
                else:
                    yield step
            # A withheld summary is not saved, so the next click asks the model again.
            if summary.text is not None:
                self.summaries.save(summary)
        yield summary

    def _definitions(self, cache: DocumentCache) -> list[Definition]:
        found = self.definitions.get(cache.doc_id)
        if found is None:
            found = find_definitions(cache.text, cache.page_starts)
            self.definitions[cache.doc_id] = found
            while len(self.definitions) > MAX_CACHED_DEFINITIONS:
                self.definitions.popitem(last=False)
        return found

    def _definition_reply(
        self,
        question: str,
        definition: Definition,
        asked: list[dict],
        cache: DocumentCache,
        tau_low: float,
        tau_high: float,
    ) -> tuple:
        route = RouteResult(clause_types=[], kind="definition", latency_ms=0.0)
        result = Answer(
            question=question,
            route=route,
            tau_low=tau_low,
            tau_high=tau_high,
            items=[],
            dropped_quotes=0,
            definition=definition,
        )
        reply = {"role": "assistant", "content": chat_reply(result, self.services.unvalidated)}
        picker, label, segments = _citation_updates(result, cache)
        return (
            "",
            [*asked, reply],
            result,
            picker,
            label,
            segments,
            sources_json(result, cache),
            show_tab(DOCUMENT_TAB),
            gr.skip(),
        )

    def _command(
        self,
        intent: Intent,
        asked: list[dict],
        cache: DocumentCache,
        tau_low: float,
        tau_high: float,
        last: Answer | None,
    ) -> Iterator[tuple]:
        if intent.kind == "explain":
            yield from self._explain_last(intent, asked, last)
        else:
            yield from self._summarize_in_chat(asked, cache, tau_low, tau_high)

    def _explain_last(
        self, intent: Intent, asked: list[dict], last: Answer | None
    ) -> Iterator[tuple]:
        gen = self.services.gen
        shown = _last_shown(last)
        if shown is None:
            yield _chat_only(asked, NOTHING_TO_EXPLAIN)
            return
        if gen is None:
            yield _chat_only(asked, SUMMARIES_OFF)
            return
        clause_type, quote = shown
        request = explanation_request(clause_type, quote, intent.language)
        try:
            for step in _stream_checked(gen, WRITING_EXPLANATION, request, 1):
                if isinstance(step, Generation):
                    explanation = explanation_from(
                        step, clause_type, quote, intent.language, gen
                    )
                else:
                    yield _chat_only(asked, step)
        except LLMError:
            logger.exception("explanation of %s failed", clause_type)
            yield _chat_only(asked, GENERATION_FAILED)
            return
        yield _chat_only(asked, explanation_markdown(explanation))

    def _summarize_in_chat(
        self, asked: list[dict], cache: DocumentCache, tau_low: float, tau_high: float
    ) -> Iterator[tuple]:
        if self.services.gen is None:
            yield _chat_only(asked, SUMMARIES_OFF)
            return
        try:
            for step in self._summary_steps(cache, tau_low, tau_high):
                if isinstance(step, Summary):
                    step = summary_markdown(step, self.services.unvalidated)
                yield _chat_only(asked, step)
        except ValueError as error:
            yield _chat_only(asked, str(error))
        except LLMError:
            logger.exception("summary of %s failed", cache.doc_id)
            yield _chat_only(asked, GENERATION_FAILED)

    def _stream_answer(
        self, question: str, passages: list[Passage], asked: list[dict]
    ) -> Iterator[tuple]:
        gen = self.services.gen
        if gen is None or not passages:
            return None
        request = passage_request(question, passages)
        generated = None
        try:
            for step in _stream_checked(gen, WRITING_ANSWER, request, PASSAGE_ATTEMPTS):
                if isinstance(step, Generation):
                    generated = generated_answer_from(step, passages, gen)
                else:
                    writing = {"role": "assistant", "content": step}
                    yield "", [*asked, writing], *(gr.skip() for _ in range(ASK_OTHER_OUTPUTS))
        except LLMError:
            logger.warning("answer from the passages failed", exc_info=True)
        return generated

    def _precompute(self, cache: DocumentCache) -> None:
        services = self.services
        if services.gen is None:
            return
        names = [clause_type.name for clause_type in services.clause_types]
        settings = services.settings
        run_in_background(
            precompute,
            cache,
            names,
            services.gen,
            self.summaries,
            self.term_sheets,
            settings.tau_low,
            settings.tau_high,
            services.unvalidated,
        )

    def _cached(self, doc_id: str) -> DocumentCache | None:
        try:
            return self.services.store.load(doc_id, self.services.settings)
        except CacheMismatchError:
            return None

    def _selected(self, doc_id: str) -> gr.Dropdown:
        return gr.Dropdown(choices=document_choices(self.services.store), value=doc_id)


def _reporter(progress: gr.Progress) -> Callable[[int, int, str], None]:
    started = time.perf_counter()

    def report(done: int, total: int, name: str) -> None:
        elapsed = time.perf_counter() - started
        progress((done, total), desc=f"Checking {display_name(name)} · {elapsed:.0f} s")

    return report


def _chat_only(asked: list[dict], content: str) -> tuple:
    message = {"role": "assistant", "content": content}
    return "", [*asked, message], *(gr.skip() for _ in range(ASK_OTHER_OUTPUTS))


def _last_shown(last: Answer | None) -> tuple[str, str] | None:
    if last is None:
        return None
    for item in last.items:
        if item.citation:
            return item.clause_type, item.citation.quote
    if last.definition is not None:
        return DEFINITION_LABEL, last.definition.quote
    if last.passages:
        return RELATED_LABEL, last.passages[0].quote
    return None


def _table(
    cache: DocumentCache | None,
    tau_low: float,
    tau_high: float,
    status_filter: str,
    unvalidated: frozenset[str],
):
    if cache is None:
        return gr.skip()
    return clause_rows(cache, tau_low, tau_high, status_of_filter(status_filter), unvalidated)


def _citation_updates(result: Answer, cache: DocumentCache):
    if result.definition is not None:
        definition = result.definition
        picker = gr.Radio(
            choices=[(f"{DEFINITION_LABEL}: {definition.term}", DEFINITION_VALUE)],
            value=DEFINITION_VALUE,
            visible=True,
        )
        label, segments = document_view(
            cache, DEFINITION_LABEL, definition_citation(definition)
        )
        return picker, label, segments
    if result.passages:
        first = passage_citation(result.passages[0])
        label, segments = document_view(cache, RELATED_LABEL, first)
        picker = gr.Radio(
            choices=passage_choices(result.passages),
            value=f"{PASSAGE_PREFIX}0",
            visible=True,
        )
        return picker, label, segments
    cited = [item for item in result.items if item.citation]
    picker = gr.Radio(
        choices=[(display_name(item.clause_type), item.clause_type) for item in cited],
        value=cited[0].clause_type if cited else None,
        visible=bool(cited),
    )
    if not cited:
        return picker, gr.skip(), gr.skip()
    label, segments = document_view(cache, cited[0].clause_type, cited[0].citation)
    return picker, label, segments


def build_app(services: Services) -> gr.Blocks:
    """Build the layout and connect the handlers."""
    handlers = Handlers(services)
    settings = services.settings
    choices = document_choices(services.store)
    color_map = {display_name(c.name): HIGHLIGHT_COLOR for c in services.clause_types}
    color_map[RELATED_LABEL] = HIGHLIGHT_COLOR
    color_map[DEFINITION_LABEL] = HIGHLIGHT_COLOR

    with gr.Blocks(title="Contract Assistant", fill_width=True) as demo:
        cache_state = gr.State(None)
        answer_state = gr.State(None)
        turns_state = gr.State([])
        refresh_state = gr.State(None)
        with gr.Column(elem_id="top"):
            gr.Markdown(f"# Contract Assistant\n\n{INTRO}", elem_id="header")
            gr.Markdown(DISCLAIMER, elem_id="disclaimer")
        with gr.Row():
            with gr.Column(scale=1):
                document = gr.Dropdown(
                    choices=choices,
                    value=default_document(choices),
                    label="Contract",
                    elem_id="contract",
                )
                with gr.Accordion("Add a contract", open=False):
                    upload = gr.File(
                        file_types=[".pdf", ".txt"],
                        label="PDF or TXT file",
                        height=UPLOAD_HEIGHT,
                        elem_id="upload",
                    )
                    extract = gr.Button("Analyze contract", elem_id="extract")
                    extract_status = gr.Markdown(elem_id="extract-status")
                chat = gr.Chatbot(
                    height=CHAT_HEIGHT,
                    elem_id="chat",
                    show_label=False,
                    placeholder=CHAT_PLACEHOLDER,
                    buttons=["copy"],
                )
                with gr.Row(equal_height=True):
                    question = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about this contract",
                        scale=4,
                        elem_id="question",
                    )
                    ask = gr.Button("Ask", variant="primary", scale=1, elem_id="ask")
                    stop = gr.Button("Stop", elem_id="stop", variant="stop", size="sm", scale=0)
                with gr.Row(elem_id="suggestions"):
                    suggestions = [
                        gr.Button(text, size="sm", variant="secondary")
                        for text in SUGGESTED_QUESTIONS
                    ]
            with gr.Column(scale=1), gr.Tabs(selected=DOCUMENT_TAB, elem_id="tabs") as tabs:
                with gr.Tab("Document", id=DOCUMENT_TAB):
                    title = gr.Markdown(elem_id="doc-title")
                    picker = gr.Radio(
                        choices=[],
                        visible=False,
                        label="Passage shown",
                        elem_id="citation-picker",
                    )
                    view_label = gr.Markdown(elem_id="view-label")
                    view = gr.HighlightedText(
                        color_map=color_map,
                        show_legend=False,
                        combine_adjacent=True,
                        show_label=False,
                        elem_id="view",
                    )
                    with gr.Accordion("Defined terms", open=False, elem_id="terms"):
                        definitions_table = gr.Dataframe(
                            headers=DEFINITION_HEADERS,
                            datatype=["str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            max_height=DEFINITIONS_MAX_HEIGHT,
                            elem_id="definitions-table",
                        )
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value=DEFAULT_LANGUAGE,
                            label="Explain in",
                            elem_id="explain-language",
                            scale=1,
                        )
                        explain = gr.Button(
                            "Explain the highlighted passage",
                            elem_id="explain",
                            scale=2,
                            interactive=services.gen is not None,
                        )
                    explanation = gr.Markdown(elem_id="explanation")
                with gr.Tab("Summary", id=SUMMARY_TAB):
                    gr.Markdown(
                        SUMMARY_INTRO_ON if services.gen else SUMMARY_INTRO_OFF,
                        elem_id="summary-intro",
                    )
                    summarize = gr.Button(
                        "Summarize contract",
                        variant="primary",
                        elem_id="summarize",
                        interactive=services.gen is not None,
                    )
                    summary_output = gr.Markdown(elem_id="summary")
                    gr.Markdown(TERMS_INTRO, elem_id="terms-intro")
                    term_sheet_button = gr.Button(
                        "Extract key terms",
                        elem_id="term-sheet",
                        interactive=services.gen is not None,
                    )
                    term_status = gr.Markdown(elem_id="term-status")
                    term_table = gr.Dataframe(
                        headers=TERM_HEADERS,
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        elem_id="term-table",
                    )
                    term_download = gr.DownloadButton(
                        TERM_DOWNLOAD_LABEL, elem_id="term-download", visible=False
                    )
                with gr.Tab("Clause checklist", id=CHECKLIST_TAB):
                    checklist_text = gr.Markdown(elem_id="checklist-intro")
                    status_filter = gr.Radio(
                        choices=STATUS_FILTER_CHOICES,
                        value=ALL_CLAUSES,
                        label="Show",
                        elem_id="status-filter",
                    )
                    table = gr.Dataframe(
                        headers=CLAUSE_HEADERS,
                        datatype=COLUMN_TYPES,
                        column_widths=COLUMN_WIDTHS,
                        interactive=False,
                        wrap=True,
                        max_height=TABLE_MAX_HEIGHT,
                        elem_id="clause-table",
                    )
                with gr.Tab("Settings", id=SETTINGS_TAB):
                    gr.Markdown(THRESHOLD_HELP)
                    tau_high = gr.Slider(
                        0,
                        1,
                        value=settings.tau_high,
                        step=SLIDER_STEP,
                        label="Found from",
                        elem_id="tau-high",
                    )
                    tau_low = gr.Slider(
                        0,
                        1,
                        value=settings.tau_low,
                        step=SLIDER_STEP,
                        label="Verify from",
                        elem_id="tau-low",
                    )
                    reset = gr.Button("Restore defaults", elem_id="reset")
                with gr.Tab("Technical details", id=TECHNICAL_TAB):
                    gr.Markdown(
                        technical_markdown(
                            settings,
                            services.router.kind,
                            services.unvalidated,
                            services.gen.model if services.gen else None,
                        )
                    )
                    sources = gr.JSON(label="Last answer", elem_id="sources")
                    download = gr.DownloadButton(
                        "Download the analysis (JSON)", elem_id="download"
                    )

        open_inputs = [document, tau_low, tau_high, status_filter]
        open_outputs = [
            cache_state,
            answer_state,
            chat,
            title,
            table,
            checklist_text,
            view_label,
            view,
            picker,
            sources,
            download,
            turns_state,
            definitions_table,
        ]
        gr.on(
            triggers=[demo.load, document.change],
            fn=handlers.open_document,
            inputs=open_inputs,
            outputs=open_outputs,
            api_name="load_document",
        )
        ask_outputs = [
            question,
            chat,
            answer_state,
            picker,
            view_label,
            view,
            sources,
            tabs,
            turns_state,
        ]
        ask_inputs = [chat, cache_state, tau_low, tau_high, turns_state, answer_state]
        running = [
            gr.on(
                triggers=[ask.click, question.submit],
                fn=handlers.ask,
                inputs=[question, *ask_inputs],
                outputs=ask_outputs,
                api_name="ask",
            )
        ]
        running.extend(
            suggestion.click(
                handlers.ask,
                [suggestion, *ask_inputs],
                ask_outputs,
                api_visibility="private",
            )
            for suggestion in suggestions
        )
        for event in running:
            event.then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        picker.input(
            handlers.pick_citation,
            [picker, answer_state, cache_state],
            [view_label, view],
            api_name="pick_citation",
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        status_filter.input(
            handlers.filter_clauses,
            [status_filter, tau_low, tau_high, cache_state],
            [table],
            api_name="filter_clauses",
        )
        tau_low.release(
            handlers.release_low,
            [tau_low, tau_high, cache_state, status_filter],
            [tau_low, table],
            api_name="release_tau_low",
        )
        tau_high.release(
            handlers.release_high,
            [tau_low, tau_high, cache_state, status_filter],
            [tau_high, table],
            api_name="release_tau_high",
        )
        reset.click(
            handlers.reset_thresholds,
            [cache_state, status_filter],
            [tau_low, tau_high, table],
            api_name="reset_thresholds",
        )
        table.select(
            handlers.show_row, [cache_state], [view_label, view, tabs], api_name="show_row"
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        definitions_table.select(
            handlers.show_definition,
            [cache_state],
            [view_label, view, tabs],
            api_name="show_definition",
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        running.append(
            summarize.click(
                handlers.summarize,
                [cache_state, tau_low, tau_high],
                [summary_output],
                api_name="summarize",
            )
        )
        running.append(
            term_sheet_button.click(
                handlers.term_sheet,
                [cache_state, tau_low, tau_high],
                [term_table, term_download, term_status],
                api_name="term_sheet",
            )
        )
        running.append(
            explain.click(
                handlers.explain,
                [view, language, cache_state],
                [explanation],
                api_name="explain",
            )
        )
        stop.click(None, None, None, cancels=running, api_visibility="private")
        keep_warm = gr.Timer(KEEP_WARM_S)
        keep_warm.tick(handlers.keep_warm, None, None, api_visibility="private")
        extract.click(
            handlers.extract_upload,
            [upload],
            [document, extract_status, refresh_state],
            api_name="extract",
        ).then(
            handlers.reload_extended,
            [refresh_state, tau_low, tau_high, status_filter],
            [*open_outputs, refresh_state],
            api_visibility="private",
        )
    return demo


def launch_app(
    demo: gr.Blocks, services: Services, share: bool = False, port: int | None = None
):
    """Launch the app with the shared theme and the folders it serves files from."""
    settings = services.settings
    demo.launch(
        css=CSS,
        theme=THEME,
        footer_links=[],
        share=share,
        server_port=port,
        allowed_paths=[str(settings.cache_dir), str(settings.demo_cache_dir)],
    )
