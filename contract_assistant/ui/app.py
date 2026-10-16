"""Gradio app: layout and event wiring over the view functions."""

import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from contract_assistant.answer import Answer, answer
from contract_assistant.clause_types import ClauseType
from contract_assistant.config import Settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.router.base import Router
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    Store,
    doc_id_of,
)
from contract_assistant.ui.views import (
    CLAUSE_HEADERS,
    chat_reply,
    clause_rows,
    document_title,
    document_view,
    header_markdown,
    sources_json,
    top_citation,
)

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENT = "short_supply_agreement.txt"
HIGHLIGHT_COLOR = "#fcd34d"
TABLE_MAX_HEIGHT = 420
CHAT_HEIGHT = 380
SLIDER_STEP = 0.01
UNEXPECTED_ERROR = "Something went wrong; see the server log."
DAMAGED_CACHE = "The cache file for this document is damaged or inconsistent."
UPLOAD_ERRORS = (UnsupportedFileError, EmptyDocumentError, CacheIntegrityError)

FONT_SANS = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"]
FONT_MONO = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]
THEME = gr.themes.Soft(
    primary_hue="indigo", neutral_hue="slate", font=FONT_SANS, font_mono=FONT_MONO
)
UPLOAD_HEIGHT = 120
COLUMN_WIDTHS = ["22%", "17%", "12%", "12%", "37%"]
CHAT_PLACEHOLDER = (
    "**Ask about this contract**\n\n"
    "- who are the parties and which law governs?\n"
    "- can either side terminate for convenience?\n"
    "- give me all the important dates"
)
SCROLL_TO_HIGHLIGHT = """() => {
  const token = document.querySelector('#view .token:not(:has(.unlabeled))');
  if (token) token.scrollIntoView({block: 'center'});
}"""

CSS = """
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
.tier-not_found { background: transparent; color: #6b7280; }
.dark .tier-found { background: #34a56f; color: #06210f; }
.dark .tier-check { border-color: #fbbf24; color: #fbbf24; }
.dark .tier-not_found { color: #9ca3af; }
#chat .message, #clause-table td, #clause-table td * {
  overflow-wrap: anywhere;
  white-space: normal;
}
#upload svg.feather-upload { display: none; }
#clause-table, #clause-table * { font-family: var(--font); }
#view { max-height: 460px; overflow-y: auto; }
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


def guarded(handler: Callable) -> Callable:
    """Turn an unexpected exception into a warning and an unchanged page."""

    @functools.wraps(handler)
    def wrapper(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except Exception:
            logger.exception("handler %s failed", handler.__name__)
            gr.Warning(UNEXPECTED_ERROR)
            return gr.skip()

    return wrapper


def document_choices(store: Store) -> list[tuple[str, str]]:
    """Return (file name, document id) for every cached document, once per id."""
    choices: dict[str, str] = {}
    for entry in store.list():
        choices.setdefault(entry.doc_id, entry.source.filename)
    return [(filename, doc_id) for doc_id, filename in choices.items()]


def default_document(choices: list[tuple[str, str]]) -> str | None:
    """Return the id of the short demo document, else the first one."""
    for filename, doc_id in choices:
        if filename == DEFAULT_DOCUMENT:
            return doc_id
    return choices[0][1] if choices else None


class Handlers:
    """The event handlers of the app, one method per event."""

    def __init__(self, services: Services):
        self.services = services

    @guarded
    def open_document(self, doc_id: str | None, tau_low: float, tau_high: float):
        """Load a document and reset the chat, the view and the table."""
        if not doc_id:
            return gr.skip()
        services = self.services
        try:
            cache = services.store.load(doc_id, services.settings)
        except (CacheIntegrityError, CacheMismatchError) as error:
            logger.warning("cannot load %s: %s", doc_id, error)
            gr.Warning(DAMAGED_CACHE)
            return gr.skip()
        if cache is None:
            gr.Warning("No cache for this document.")
            return gr.skip()
        label, segments = document_view(cache, None, None)
        path = services.store.find(doc_id)
        return (
            cache,
            None,
            [],
            document_title(cache),
            clause_rows(cache, tau_low, tau_high),
            label,
            segments,
            gr.Radio(choices=[], value=None, visible=False),
            {},
            gr.DownloadButton(value=str(path)),
        )

    @guarded
    def ask(
        self,
        question: str,
        history: list[dict] | None,
        cache: DocumentCache | None,
        tau_low: float,
        tau_high: float,
    ):
        """Route a question, answer it from the cache and show the first citation."""
        if not question or not question.strip():
            gr.Warning("Type a question first.")
            return gr.skip()
        if cache is None:
            gr.Warning("Open a document first.")
            return gr.skip()
        services = self.services
        route = services.router.route(question)
        try:
            result = answer(question, route, cache, tau_low, tau_high)
        except ValueError as error:
            gr.Warning(str(error))
            return gr.skip()
        names = [clause_type.name for clause_type in services.clause_types]
        reply = chat_reply(result, names)
        history = [
            *(history or []),
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]
        picker, label, segments = _citation_updates(result, cache)
        return "", history, result, picker, label, segments, sources_json(result, cache)

    @guarded
    def pick_citation(
        self, clause_type: str | None, last: Answer | None, cache: DocumentCache | None
    ):
        """Show the citation of the chosen clause type of the last answer."""
        if cache is None or last is None:
            return gr.skip()
        for item in last.items:
            if item.clause_type == clause_type and item.citation:
                return document_view(cache, clause_type, item.citation)
        return gr.skip()

    @guarded
    def release_low(self, tau_low: float, tau_high: float, cache: DocumentCache | None):
        """Keep tau_low at or below tau_high and re-tier the table."""
        tau_low = min(tau_low, tau_high)
        return tau_low, _table(cache, tau_low, tau_high)

    @guarded
    def release_high(self, tau_low: float, tau_high: float, cache: DocumentCache | None):
        """Keep tau_high at or above tau_low and re-tier the table."""
        tau_high = max(tau_high, tau_low)
        return tau_high, _table(cache, tau_low, tau_high)

    @guarded
    def reset_thresholds(self, cache: DocumentCache | None):
        """Restore the default thresholds and re-tier the table."""
        tau_low, tau_high = self.services.settings.tau_low, self.services.settings.tau_high
        return tau_low, tau_high, _table(cache, tau_low, tau_high)

    @guarded
    def show_row(self, cache: DocumentCache | None, evt: gr.SelectData):
        """Show the top candidate of the clicked table row."""
        if cache is None:
            return gr.skip()
        clause_type = evt.row_value[0]
        citation = top_citation(cache, clause_type)
        if citation is None:
            gr.Info(f"No candidate for {clause_type}.")
            return gr.skip()
        return document_view(cache, clause_type, citation)

    @guarded
    def extract_upload(self, path: str | None, progress: gr.Progress = gr.Progress()):  # noqa: B008
        """Extract an uploaded file unless its cache exists, then select it."""
        if not path:
            gr.Warning("Choose a PDF or TXT file first.")
            return gr.skip()
        services = self.services
        try:
            doc = load_document(Path(path))
            doc_id = doc_id_of(doc.text)
            hit = self._has_cache(doc_id)
        except UPLOAD_ERRORS as error:
            gr.Warning(str(error))
            return gr.skip()
        if hit:
            gr.Info(f"{doc.filename}: cache hit")
            return self._selected(doc_id), "Cache hit"
        extractor = services.extractor
        if extractor is None:
            gr.Warning(
                f"Extraction needs the model in {services.settings.extractor_path}; "
                "see IMPLEMENTATION.md, Step A."
            )
            return gr.skip()
        started = time.perf_counter()

        def report(done: int, total: int, name: str) -> None:
            elapsed = time.perf_counter() - started
            progress((done, total), desc=f"{name} · {elapsed:.0f} s")

        cache = extractor.extract(doc, report)
        services.store.save(cache)
        seconds = cache.timing["extract_s"]
        return self._selected(doc_id), f"Extracted in {seconds:.0f} s"

    def _has_cache(self, doc_id: str) -> bool:
        try:
            return self.services.store.load(doc_id, self.services.settings) is not None
        except CacheMismatchError:
            return False

    def _selected(self, doc_id: str) -> gr.Dropdown:
        return gr.Dropdown(choices=document_choices(self.services.store), value=doc_id)


def _table(cache: DocumentCache | None, tau_low: float, tau_high: float):
    if cache is None:
        return gr.skip()
    return clause_rows(cache, tau_low, tau_high)


def _citation_updates(result: Answer, cache: DocumentCache):
    cited = [item for item in result.items if item.citation]
    picker = gr.Radio(
        choices=[item.clause_type for item in cited],
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
    color_map = {clause_type.name: HIGHLIGHT_COLOR for clause_type in services.clause_types}

    with gr.Blocks(title="Contract assistant", fill_width=True) as demo:
        cache_state = gr.State(None)
        answer_state = gr.State(None)
        gr.Markdown(header_markdown(settings, services.router.kind), elem_id="header")
        with gr.Row():
            with gr.Column(scale=1):
                document = gr.Dropdown(
                    choices=choices,
                    value=default_document(choices),
                    label="Document",
                    elem_id="document",
                )
                with gr.Row():
                    upload = gr.File(
                        file_types=[".pdf", ".txt"],
                        label="Upload PDF or TXT",
                        height=UPLOAD_HEIGHT,
                        elem_id="upload",
                    )
                    extract = gr.Button("Extract", elem_id="extract")
                extract_status = gr.Markdown(elem_id="extract-status")
                chat = gr.Chatbot(
                    height=CHAT_HEIGHT,
                    elem_id="chat",
                    show_label=False,
                    placeholder=CHAT_PLACEHOLDER,
                    buttons=["copy"],
                )
                with gr.Row():
                    question = gr.Textbox(
                        placeholder="Ask about this contract",
                        show_label=False,
                        scale=4,
                        elem_id="question",
                    )
                    ask = gr.Button("Ask", variant="primary", scale=1, elem_id="ask")
                with gr.Accordion("Sources (JSON)", open=False):
                    sources = gr.JSON(elem_id="sources")
            with gr.Column(scale=1):
                title = gr.Markdown(elem_id="doc-title")
                picker = gr.Radio(
                    choices=[],
                    visible=False,
                    label="Citation",
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
                with gr.Row():
                    tau_low = gr.Slider(
                        0,
                        1,
                        value=settings.tau_low,
                        step=SLIDER_STEP,
                        label="τ low",
                        elem_id="tau-low",
                    )
                    tau_high = gr.Slider(
                        0,
                        1,
                        value=settings.tau_high,
                        step=SLIDER_STEP,
                        label="τ high",
                        elem_id="tau-high",
                    )
                    reset = gr.Button("Reset", elem_id="reset")
                table = gr.Dataframe(
                    headers=CLAUSE_HEADERS,
                    datatype=["str", "html", "str", "str", "str"],
                    column_widths=COLUMN_WIDTHS,
                    interactive=False,
                    wrap=True,
                    max_height=TABLE_MAX_HEIGHT,
                    elem_id="clause-table",
                )
                download = gr.DownloadButton("Download cache JSON", elem_id="download")

        gr.on(
            triggers=[demo.load, document.change],
            fn=handlers.open_document,
            inputs=[document, tau_low, tau_high],
            outputs=[
                cache_state,
                answer_state,
                chat,
                title,
                table,
                view_label,
                view,
                picker,
                sources,
                download,
            ],
            api_name="load_document",
        )
        gr.on(
            triggers=[ask.click, question.submit],
            fn=handlers.ask,
            inputs=[question, chat, cache_state, tau_low, tau_high],
            outputs=[question, chat, answer_state, picker, view_label, view, sources],
            api_name="ask",
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        picker.input(
            handlers.pick_citation,
            [picker, answer_state, cache_state],
            [view_label, view],
            api_name="pick_citation",
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        tau_low.release(
            handlers.release_low,
            [tau_low, tau_high, cache_state],
            [tau_low, table],
            api_name="release_tau_low",
        )
        tau_high.release(
            handlers.release_high,
            [tau_low, tau_high, cache_state],
            [tau_high, table],
            api_name="release_tau_high",
        )
        reset.click(
            handlers.reset_thresholds,
            [cache_state],
            [tau_low, tau_high, table],
            api_name="reset_thresholds",
        )
        table.select(
            handlers.show_row, [cache_state], [view_label, view], api_name="show_row"
        ).then(None, None, None, js=SCROLL_TO_HIGHLIGHT)
        extract.click(
            handlers.extract_upload,
            [upload],
            [document, extract_status],
            api_name="extract",
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
        share=share,
        server_port=port,
        allowed_paths=[str(settings.cache_dir), str(settings.demo_cache_dir)],
    )
