"""The contracts: first page, drawer, document pane and its search, uploads, samples, compare
and portfolio.

Every string these pages show is written here and in the templates in English. A string built
here is translated for the request with `_t` before it is formatted; the ones the Serbian
catalogue (`translations/sr_library.py`) must cover are listed in TRANSLATED.
"""

from __future__ import annotations

import queue
import re
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from markupsafe import Markup, escape

from contract_assistant import audit
from contract_assistant.answer import Citation, Status
from contract_assistant.background import precompute, precompute_obligations, run_in_background
from contract_assistant.compare import ChecklistRow, compare_checklists, compare_terms
from contract_assistant.config import ROOT
from contract_assistant.dates import key_dates
from contract_assistant.extract.extractor import ProgressCallback
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.pdf_view import PageError, page_image
from contract_assistant.portfolio import NOT_EXTRACTED, PortfolioRow, portfolio_rows, to_csv
from contract_assistant.review import Flag
from contract_assistant.router.clients import LLMError
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    doc_id_of,
)
from contract_assistant.term_sheet import (
    LABELS,
    NOT_STATED,
    NOT_VERIFIED,
    TermSheet,
    term_sheet_items,
)
from contract_assistant.textsearch import (
    MAX_MATCHES,
    SNIPPET_CHARS,
    Match,
    occurrences,
    pattern_of,
)
from contract_assistant.ui.views import (
    EXCERPT_LABEL,
    SUGGESTED_QUESTIONS,
    TEXT_START_LABEL,
    display_name,
    document_name,
    document_view,
    page_of,
    technical_markdown,
)
from contract_assistant.web.core import (
    DEFINITION_LABEL,
    ENGLISH,
    ContractShelf,
    DocumentEntry,
    Tau,
    Thresholds,
    UserWorkspace,
    WorkspaceError,
    Ws,
    attachment,
    frame,
    language,
    render,
    render_string,
    stream,
)
from contract_assistant.web.generate import term_sheet_steps
from contract_assistant.web.i18n import translate

router = APIRouter()

PAGE_MODE = "page"
TEXT_MODE = "text"
PORTFOLIO_CSV = "portfolio.csv"
PAGE_CACHE_HEADERS = {"Cache-Control": "private, max-age=3600"}
# The label of a highlighted span opened without one, so that the span is still marked.
PASSAGE_LABEL = "Passage"
UPLOAD_ERRORS = (UnsupportedFileError, EmptyDocumentError, CacheIntegrityError)
NOTICE_LABEL = "notice"
SHORT_PARTIES = 60
SEVERITIES = ("high", "medium", "low")
BOLD = re.compile(r"\*\*(.+?)\*\*")
# Page breaks of the demo texts are runs of blank lines; one blank line is enough on screen.
BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
# A line break inside a paragraph (PDF text and OCR text are wrapped by line); a blank line stays.
LINE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")
RENDERED_PAGES = 12
CODE = re.compile(r"`(.+?)`")
# The marks of the text view: the passage shown, and every other occurrence of a searched term.
HIGHLIGHT = "highlight"
OCCURRENCE = "occurrence"
ELLIPSIS = "…"
# What the reading errors of pdf.py say, to tell the reader why in plain words.
TESSERACT = "Tesseract"
NOT_UTF8_ERROR = "not a UTF-8"

CHOOSE_FILE = "Choose a PDF or TXT file first."
# The steps of reading a contract, shown one at a time; no clause names, no model names.
# Reading a scanned page with OCR takes longer, but it is still "Reading the file".
READING_FILE = "Reading the file"
FINDING = "Finding the clauses · {done} of {total} · {elapsed} s"
SAVING = "Saving"
ALREADY_ANALYSED = "{name} was read before; opening it"
ANALYSIS_UPDATED = "Read again for the clauses added since; opening it"
ANALYSIS_COMPLETE = "Done; opening the contract"
NO_MODEL = (
    "New contracts cannot be read on this computer yet: the model that reads them is not "
    "installed. The README says how to add it."
)
SAMPLES_NEED_MODEL = "The samples need that model too."
UNKNOWN_SAMPLE = "There is no sample contract by that name."
# Why a file could not be read.
NEEDS_TESSERACT = (
    "This PDF is a scan. Reading scans needs Tesseract with its language packs: "
    "brew install tesseract tesseract-lang."
)
NO_TEXT = "This PDF has no text the assistant can read."
UNSUPPORTED_FILE = "This file cannot be read: choose a PDF or a TXT file."
NOT_UTF8 = "This text file is not saved as UTF-8; save it as UTF-8 and add it again."
DAMAGED_FILE = "The saved analysis of this contract is damaged and cannot be opened."
NO_PAGE_IMAGE = "No page image: the PDF of this contract is not kept on this computer."
NO_SUCH_PAGE = "This page is not in the PDF."
# The document pane and the search in it.
PAGE_HEADING = "Page {page}"
PAGES_HEADING = "Pages {first}–{last}"
SCANNED_PAGE = (
    "This page is a scan read with OCR; the words are shown as read, and the page image "
    "cannot box them."
)
PAGE_ALT = "Page {page} of the contract"
PAGE_ALT_PASSAGE = "Page {page} of the contract with the {label} passage boxed"
PAGE_ALT_TERM = "Page {page} of the contract with each occurrence of “{term}” boxed"
TYPE_MORE = "Type at least two characters"
NO_OCCURRENCES = "No occurrences"
ONE_OCCURRENCE = "1 occurrence"
OCCURRENCES = "{count} occurrences"
ONE_ON_ONE_PAGE = "1 occurrence on 1 page"
ON_ONE_PAGE = "{count} occurrences on 1 page"
ON_PAGES = "{count} occurrences on {pages} pages"
OCCURRENCE_OF = "Occurrence {index} of {count}"
FIRST_SHOWN = "Showing the first {limit}."
# The comparison.
COMPARING = "Comparing the clauses"
EXTRACTING_TERMS = "Reading the key terms of {name}"
CHOOSE_OTHER = "Choose a contract to compare with."
CHOOSE_DIFFERENT = "Choose two different contracts: the same one is on both sides."
OTHER_UNAVAILABLE = "The other contract has not been analysed, or its analysis is damaged."
ADD_SECOND = "Add a second contract to compare it with the first."
TERMS_SKIPPED = "The key terms are not compared: {reason}."
TERMS_NEED_MODEL = "they need the local AI, which is off"
TERMS_NEED_CLAUSES = "no clause they are read from was found in {name}"
TERMS_FAILED = "the local AI did not answer"
TERMS_WITHHELD = "The key terms of {name} are not shown because {reason}."
ONLY_IN = "Only in {name}"
FOUND_AND_CHECK = "In both, found in one and to check in the other"
FIRST_SECOND = "{left} first, {right} second"
SAME_IN_BOTH = "Same in both"
# What was downloaded, as the activity log names it.
THE_ANALYSIS = "the analysis"
THE_PORTFOLIO = "the portfolio"
# The portfolio's next notice deadline.
DATE = "{day} {month} {year}"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Every string above the reader sees, for the check that the Serbian catalogue covers them.
TRANSLATED = (
    PASSAGE_LABEL,
    DEFINITION_LABEL,
    CHOOSE_FILE,
    READING_FILE,
    FINDING,
    SAVING,
    ALREADY_ANALYSED,
    ANALYSIS_UPDATED,
    ANALYSIS_COMPLETE,
    NO_MODEL,
    SAMPLES_NEED_MODEL,
    UNKNOWN_SAMPLE,
    NEEDS_TESSERACT,
    NO_TEXT,
    UNSUPPORTED_FILE,
    NOT_UTF8,
    DAMAGED_FILE,
    NO_PAGE_IMAGE,
    NO_SUCH_PAGE,
    EXCERPT_LABEL,
    TEXT_START_LABEL,
    PAGE_HEADING,
    PAGES_HEADING,
    SCANNED_PAGE,
    PAGE_ALT,
    PAGE_ALT_PASSAGE,
    PAGE_ALT_TERM,
    TYPE_MORE,
    NO_OCCURRENCES,
    ONE_OCCURRENCE,
    OCCURRENCES,
    ONE_ON_ONE_PAGE,
    ON_ONE_PAGE,
    ON_PAGES,
    OCCURRENCE_OF,
    FIRST_SHOWN,
    COMPARING,
    EXTRACTING_TERMS,
    CHOOSE_OTHER,
    CHOOSE_DIFFERENT,
    OTHER_UNAVAILABLE,
    ADD_SECOND,
    TERMS_SKIPPED,
    TERMS_NEED_MODEL,
    TERMS_NEED_CLAUSES,
    TERMS_FAILED,
    TERMS_WITHHELD,
    ONLY_IN,
    FOUND_AND_CHECK,
    FIRST_SECOND,
    SAME_IN_BOTH,
    NOT_STATED,
    NOT_EXTRACTED,
    NOT_VERIFIED,
    DATE,
    THE_ANALYSIS,
    THE_PORTFOLIO,
    *MONTHS,
    *LABELS.values(),
)

SAMPLES_DIR = ROOT / "demo"
# The sample contracts the first page and the drawer offer: the allow-list of the samples route.
SAMPLE_FILES = (
    "short_supply_agreement.pdf",
    "short_supply_agreement.txt",
    "medium_sponsorship_agreement.txt",
    "long_outsourcing_agreement.txt",
)
SAMPLE_ERRORS = (UnsupportedFileError, EmptyDocumentError)

ProgressItem = tuple[int, int, str]


def _flowed(text: str) -> str:
    """Return a piece of the text with its paragraphs flowing: one blank line between them."""
    return LINE_BREAK.sub(" ", BLANK_LINES.sub("\n\n", text))
Span = tuple[int, int]
# A piece of the text view: its text, its mark (None, HIGHLIGHT or OCCURRENCE) and its label.
Segment = tuple[str, str | None, str | None]


@dataclass(frozen=True)
class Sample:
    """A sample contract as the first page and the drawer offer it."""

    name: str
    title: str
    kind: str
    size: str
    doc_id: str


@dataclass(frozen=True)
class Hit:
    """One occurrence of a searched term as the results list shows it."""

    page: int | None
    before: str
    text: str
    after: str
    url: str


# Each sample file is read once per version, to know its id and its size.
_samples_read: dict[tuple[Path, int], Sample] = {}


def _t(request: Request, text: str) -> str:
    """Return a string of these pages in the language of the request."""
    return translate(text, language(request))


@router.get("/", response_class=HTMLResponse)
def home(request: Request, ws: Ws, tau: Tau):
    """Open the default contract, or the first-run page: an upload and the sample contracts."""
    doc_id = ws.default_doc_id()
    if doc_id is None:
        return render(
            request,
            "home.html",
            samples=_samples([]),
            can_read=ws.services.extractor is not None,
            no_model=_t(request, NO_MODEL),
            samples_need_model=_t(request, SAMPLES_NEED_MODEL),
            tau=tau,
        )
    return RedirectResponse(f"/documents/{doc_id}", status_code=303)


@router.get("/documents/{doc_id}", response_class=HTMLResponse)
def workspace_page(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The workspace of one contract: the contracts drawer, the overview or document, the chat."""
    cache = ws.load(doc_id)
    documents = ws.documents()
    entry = next((d for d in documents if d.doc_id == cache.doc_id), None)
    return render(
        request,
        "workspace.html",
        entry=entry,
        suggestions=SUGGESTED_QUESTIONS,
        tau=tau,
        **_rail(request, ws, cache, documents),
    )


@router.get("/documents/{doc_id}/rail", response_class=HTMLResponse)
def rail(request: Request, doc_id: str, ws: Ws):
    """The contracts drawer alone, to refresh its counts after a decision or an upload."""
    cache = ws.load(doc_id)
    return render(request, "library/rail.html", **_rail(request, ws, cache, ws.documents()))


@router.get("/documents/{doc_id}/view", response_class=HTMLResponse)
def document_pane(
    request: Request,
    doc_id: str,
    ws: Ws,
    tau: Tau,
    label: str | None = None,
    start: int | None = None,
    end: int | None = None,
    conf: float = 1.0,
    status: str | None = None,
    mode: str | None = None,
    page: int | None = None,
    q: str = "",
):
    """The document pane: the PDF page or the text around a span, with the passage card.

    With a term `q`, every occurrence of it is marked. A span without a label that is one of
    them, or the first one when neither a span nor a page is asked for, is the main highlight
    and has no passage card. A page read with OCR opens as text: its image cannot box words.
    """
    cache = ws.load(doc_id)
    lang = language(request)
    term, pattern = _term(q)
    spans = _spans(cache.text, pattern)
    citation = _citation(cache, start, end, conf)
    if citation is None and spans and page is None:
        citation = _citation(cache, *spans[0], conf)
    hit = None if label else _hit_index(spans, citation)
    clause_type = ws.clause_type_of(label) if label else PASSAGE_LABEL
    copy = _pdf_copy(ws, cache)
    pages = len(cache.page_starts)
    span_page = page_of(cache.page_starts, citation.start) if citation and pages else 1
    shown = min(max(page or span_page, 1), max(pages, 1))
    scanned = bool(pages) and shown in cache.source.ocr_pages
    # The mode the reader chose; without one, a scanned page opens as text, others as the page.
    chosen = mode if copy is not None and mode in (PAGE_MODE, TEXT_MODE) else None
    mode = chosen or (PAGE_MODE if copy is not None and not scanned else TEXT_MODE)
    if mode == TEXT_MODE and page is not None and pages:
        lo, hi = _page_bounds(cache, shown)
    else:
        lo, hi = _window(cache, clause_type, citation)
    heading, own_page = _heading(cache, lo, hi, citation, lang)
    segments = _segments(cache.text, lo, hi, citation, display_name(clause_type), pattern)

    def url(**changes) -> str:
        params = {
            "label": label,
            "start": citation.start if citation else None,
            "end": citation.end if citation else None,
            "conf": f"{citation.conf:g}" if citation else None,
            "status": status,
            "mode": chosen,
            "page": page,
            "q": term or None,
            **changes,
        }
        query = urlencode({key: value for key, value in params.items() if value is not None})
        return f"/documents/{cache.doc_id}/view?{query}"

    boxed = citation is not None and hit is None and shown == span_page
    image = None
    if mode == PAGE_MODE:
        image = _image_url(cache, shown, citation if boxed else None, term)
    if label or not term:
        clear_url = url(q=None)
    else:
        clear_url = url(q=None, start=None, end=None, conf=None, page=shown if pages else None)
    return render(
        request,
        "library/view.html",
        cache=cache,
        heading=heading,
        show_heading=mode == TEXT_MODE and own_page != shown,
        segments=[(_flowed(text), mark, name) for text, mark, name in segments],
        label=label or PASSAGE_LABEL,
        label_text=translate(label or PASSAGE_LABEL, lang),
        citation=citation,
        status=status,
        hit=hit is not None,
        mode=mode,
        has_copy=copy is not None,
        image=image,
        image_alt=_image_alt(shown, label or PASSAGE_LABEL, boxed, term, lang),
        scanned_note=translate(SCANNED_PAGE, lang) if scanned else None,
        page=shown,
        pages=pages,
        page_url=url(mode=PAGE_MODE),
        text_url=url(mode=TEXT_MODE),
        previous_url=url(page=shown - 1) if shown > 1 else None,
        next_url=url(page=shown + 1) if shown < pages else None,
        query=q,
        term=term,
        search_url=_search_url(cache, "", chosen),
        found=_found(spans, hit, lang) if term else None,
        list_url=_search_url(cache, term, chosen) if term else None,
        previous_hit=_hit_url(cache, spans[hit - 1], term, chosen) if hit else None,
        next_hit=(
            _hit_url(cache, spans[hit + 1], term, chosen)
            if hit is not None and hit + 1 < len(spans)
            else None
        ),
        clear_url=clear_url,
        tau=tau,
    )


@router.get("/documents/{doc_id}/search", response_class=HTMLResponse)
def search(request: Request, doc_id: str, ws: Ws, q: str = "", mode: str | None = None):
    """Every occurrence of a term in the contract, each a link that opens it in the pane.

    The mode is the one the reader chose in the pane, if any; an occurrence on a scanned page
    opens as text whatever it is.
    """
    cache = ws.load(doc_id)
    lang = language(request)
    term, pattern = _term(q)
    mode = mode if mode in (PAGE_MODE, TEXT_MODE) else None
    if pattern is None:
        summary = translate(TYPE_MORE, lang) if q.strip() else None
        return render(request, "library/search.html", summary=summary, hits=[])
    spans = _spans(cache.text, pattern)
    starts = cache.page_starts
    pages = len({page_of(starts, start) for start, _ in spans}) if starts else None
    found = occurrences(cache.text, term, starts, MAX_MATCHES)
    more = len(spans) > len(found)
    every_mode = _mode_at(cache, spans[0][0], mode) if spans else None
    return render(
        request,
        "library/search.html",
        summary=_occurrences(len(spans), pages, lang),
        hits=[_hit(cache, match, term, mode) for match in found],
        more=translate(FIRST_SHOWN, lang).format(limit=len(found)) if more else None,
        every_url=_view_of_term(cache, term, every_mode) if spans else None,
    )


@router.get("/documents/{doc_id}/page/{number}.png")
def page_png(
    request: Request,
    doc_id: str,
    number: int,
    ws: Ws,
    start: int | None = None,
    end: int | None = None,
    q: str = "",
):
    """One page of the contract's PDF as an image, the passage between start and end boxed,
    and every occurrence of the term q."""
    cache = ws.load(doc_id)
    copy = _pdf_copy(ws, cache)
    if copy is None:
        raise HTTPException(status_code=404, detail=_t(request, NO_PAGE_IMAGE))
    citation = _citation(cache, start, end, 1.0)
    term, _ = _term(q)
    key = (cache.doc_id, number, citation.quote if citation else None, term or None)
    # Paging back and forth asks for the same images again; the last few are kept.
    state = request.app.state
    rendered = getattr(state, "rendered_pages", None)
    if rendered is None:
        rendered = state.rendered_pages = OrderedDict()
    image = rendered.get(key)
    if image is None:
        try:
            image = page_image(copy, number, key[2], key[3])
        except PageError as error:
            raise HTTPException(status_code=404, detail=_t(request, NO_SUCH_PAGE)) from error
        rendered[key] = image
        while len(rendered) > RENDERED_PAGES:
            rendered.popitem(last=False)
    rendered.move_to_end(key)
    return Response(image, media_type="image/png", headers=PAGE_CACHE_HEADERS)


@router.get("/documents/{doc_id}/analysis.json")
def analysis_json(request: Request, doc_id: str, ws: Ws):
    """The saved analysis of a contract: the clause candidates the encoder located."""
    contract = ws.contract(doc_id)
    cache = ws.workspace.analysis(contract.doc_id)
    ws.log(audit.CONTRACT_EXPORTED, contract_id=contract.id, details={"what": THE_ANALYSIS})
    filename = f"{Path(contract.name).stem}.analysis.json"
    return Response(
        cache.model_dump_json(indent=1),
        media_type="application/json",
        headers=attachment(filename),
    )


@router.post("/upload")
def upload(
    request: Request,
    ws: Ws,
    file: Annotated[UploadFile | None, File()] = None,
    folder: Annotated[str, Form()] = "",
):
    """Add an uploaded contract to a folder, reading it unless its text was read before.

    Without a folder it goes to the top of the user's own space. Streams the progress.
    """
    ws.require_add(folder or None)
    name = Path(file.filename or "").name if file is not None else ""
    data = file.file.read() if file is not None and name else b""
    return stream(_upload_frames(request, ws, name, data, folder or None))


@router.post("/samples/{name}")
def sample(request: Request, name: str, ws: Ws, folder: Annotated[str, Form()] = ""):
    """Add a sample contract the way an upload is added."""
    if name not in SAMPLE_FILES:
        raise WorkspaceError(_t(request, UNKNOWN_SAMPLE), status_code=404)
    ws.require_add(folder or None)
    return stream(_sample_frames(request, ws, SAMPLES_DIR / name, folder or None))


@router.get("/compare", response_class=HTMLResponse)
def compare_page(
    request: Request, ws: Ws, tau: Tau, left: str | None = None, right: str | None = None
):
    """Two contracts side by side: the key terms, the review flags and the clause checklist."""
    documents = ws.documents()
    by_id = {entry.doc_id: entry for entry in documents}
    left_id = left if left in by_id else ws.default_doc_id()
    others = [entry.doc_id for entry in documents if entry.doc_id != left_id]
    right_id = right if right in others else (others[0] if others else None)
    total = len(ws.names)
    return render(
        request,
        "library/compare.html",
        documents=documents,
        left=by_id.get(left_id) if left_id else None,
        right=by_id.get(right_id) if right_id else None,
        total=total,
        notice=None if right_id else _t(request, ADD_SECOND),
        tau=tau,
        cache=None,
    )


@router.post("/compare/result")
def compare_result(
    request: Request,
    ws: Ws,
    tau: Tau,
    left: Annotated[str, Form()] = "",
    right: Annotated[str, Form()] = "",
):
    """Stream the comparison: a status while a term sheet is extracted, then the result."""
    return stream(_compare_frames(request, ws, tau, left, right))


@router.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(request: Request, ws: Ws, tau: Tau):
    """Every analysed contract in one table, with totals and the next notice deadline."""
    lang = language(request)
    rows = _portfolio(ws)
    entries = {entry.doc_id: entry for entry in ws.documents()}
    total = len(ws.names)
    contracts = [_portfolio_item(row, entries.get(row.doc_id), total, lang) for row in rows]
    pdfs = sum(item["kind"] == "pdf" for item in contracts)
    return render(
        request,
        "library/portfolio.html",
        contracts=contracts,
        total=total,
        found=sum(row.found for row in rows),
        check=sum(row.check for row in rows),
        flags_high=sum(row.flags_high for row in rows),
        flags_medium=sum(row.flags_medium for row in rows),
        flags_low=sum(row.flags_low for row in rows),
        pdfs=pdfs,
        texts=len(contracts) - pdfs,
        deadline=_next_notice(ws, rows, lang),
        tau=tau,
        cache=None,
    )


@router.get("/portfolio.csv")
def portfolio_csv(ws: Ws):
    """The portfolio table as a CSV file."""
    rows = _portfolio(ws)
    ws.log(audit.CONTRACT_EXPORTED, details={"what": THE_PORTFOLIO})
    return Response(to_csv(rows), media_type="text/csv", headers=attachment(PORTFOLIO_CSV))


@router.get("/technical", response_class=HTMLResponse)
def technical(request: Request, ws: Ws):
    """The technical facts behind the answers, for the settings menu."""
    lines = technical_markdown(
        ws.settings, ws.services.router.kind, ws.services.unvalidated, ws.gen_model
    ).splitlines()
    return render(request, "library/technical.html", items=[_markup(line) for line in lines])


def _citation(
    cache: DocumentCache, start: int | None, end: int | None, conf: float
) -> Citation | None:
    if start is None or end is None or not 0 <= start < end <= len(cache.text):
        return None
    page = page_of(cache.page_starts, start) if cache.page_starts else None
    return Citation(quote=cache.text[start:end], page=page, conf=conf, start=start, end=end)


def _pdf_copy(ws: UserWorkspace, cache: DocumentCache) -> Path | None:
    if cache.source.kind != "pdf":
        return None
    return ws.pdf_path(cache.doc_id)


def _term(q: str) -> tuple[str, re.Pattern[str] | None]:
    """Return a searched term on one line and its pattern; a term too short is ignored."""
    term = " ".join(q.split())
    pattern = pattern_of(term)
    return (term, pattern) if pattern is not None else ("", None)


def _spans(text: str, pattern: re.Pattern[str] | None) -> list[Span]:
    """Return the span of every occurrence of a pattern, in document order."""
    return [] if pattern is None else [found.span() for found in pattern.finditer(text)]


def _hit_index(spans: list[Span], citation: Citation | None) -> int | None:
    """Return which occurrence a citation is, or None when it is not one of them."""
    if citation is None:
        return None
    try:
        return spans.index((citation.start, citation.end))
    except ValueError:
        return None


def _page_bounds(cache: DocumentCache, number: int) -> Span:
    starts = cache.page_starts
    hi = starts[number] if number < len(starts) else len(cache.text)
    return starts[number - 1], hi


def _window(cache: DocumentCache, clause_type: str, citation: Citation | None) -> Span:
    """Return the part of the text shown around a citation: its pages, or an excerpt."""
    _, segments = document_view(cache, clause_type, citation)
    lo = 0
    if citation is not None:
        before = segments[0][0] if segments and segments[0][1] is None else ""
        lo = citation.start - len(before)
    return lo, lo + sum(len(text) for text, _ in segments)


def _heading(
    cache: DocumentCache, lo: int, hi: int, citation: Citation | None, lang: str
) -> tuple[str, int | None]:
    """Return the heading of the text shown, and its page when it is a single page."""
    starts = cache.page_starts
    if not starts:
        return translate(EXCERPT_LABEL if citation else TEXT_START_LABEL, lang), None
    first, last = page_of(starts, lo), page_of(starts, max(hi - 1, lo))
    if first == last:
        return translate(PAGE_HEADING, lang).format(page=first), first
    return translate(PAGES_HEADING, lang).format(first=first, last=last), None


def _segments(
    text: str,
    lo: int,
    hi: int,
    citation: Citation | None,
    label: str,
    pattern: re.Pattern[str] | None,
) -> list[Segment]:
    """Return the text between lo and hi in pieces: the citation marked as the highlight,
    every other occurrence of the pattern marked as an occurrence."""
    marks: list[tuple[int, int, str]] = []
    if citation is not None and citation.start < hi and citation.end > lo:
        marks.append((max(citation.start, lo), min(citation.end, hi), HIGHLIGHT))
    for found in pattern.finditer(text, lo, hi) if pattern is not None else ():
        start, end = found.span()
        if citation is None or end <= citation.start or start >= citation.end:
            marks.append((start, end, OCCURRENCE))
    segments: list[Segment] = []
    position = lo
    for start, end, mark in sorted(marks):
        if start > position:
            segments.append((text[position:start], None, None))
        segments.append((text[start:end], mark, label if mark == HIGHLIGHT else None))
        position = end
    if position < hi:
        segments.append((text[position:hi], None, None))
    return segments


def _image_url(
    cache: DocumentCache, number: int, citation: Citation | None, term: str
) -> str:
    params: dict[str, object] = {}
    if citation is not None:
        params.update(start=citation.start, end=citation.end)
    if term:
        params["q"] = term
    query = f"?{urlencode(params)}" if params else ""
    return f"/documents/{cache.doc_id}/page/{number}.png{query}"


def _image_alt(number: int, label: str, boxed: bool, term: str, lang: str) -> str:
    if boxed:
        label = translate(label, lang)
        return translate(PAGE_ALT_PASSAGE, lang).format(page=number, label=label)
    if term:
        return translate(PAGE_ALT_TERM, lang).format(page=number, term=term)
    return translate(PAGE_ALT, lang).format(page=number)


def _occurrences(count: int, pages: int | None, lang: str) -> str:
    """Return how many occurrences there are, and on how many pages when it has pages."""
    if count == 0:
        text = NO_OCCURRENCES
    elif pages is None:
        text = ONE_OCCURRENCE if count == 1 else OCCURRENCES
    elif pages == 1:
        text = ONE_ON_ONE_PAGE if count == 1 else ON_ONE_PAGE
    else:
        text = ON_PAGES
    return translate(text, lang).format(count=count, pages=pages)


def _found(spans: list[Span], hit: int | None, lang: str) -> str:
    """Return the pane's line about the term: which occurrence is shown, or how many."""
    if hit is None:
        return _occurrences(len(spans), None, lang)
    return translate(OCCURRENCE_OF, lang).format(index=hit + 1, count=len(spans))


def _search_url(cache: DocumentCache, term: str, mode: str | None) -> str:
    params = {key: value for key, value in (("q", term), ("mode", mode)) if value}
    query = f"?{urlencode(params)}" if params else ""
    return f"/documents/{cache.doc_id}/search{query}"


def _view_of_term(cache: DocumentCache, term: str, mode: str | None) -> str:
    """Return the URL of the pane with every occurrence of a term marked, from the first."""
    params = {key: value for key, value in (("q", term), ("mode", mode)) if value}
    return f"/documents/{cache.doc_id}/view?{urlencode(params)}"


def _mode_at(cache: DocumentCache, start: int, mode: str | None) -> str | None:
    """Return the mode to open an occurrence in: the image of a scanned page cannot box it,
    so the pane opens such a page as text whatever mode the reader chose."""
    page = page_of(cache.page_starts, start) if cache.page_starts else None
    return None if mode == PAGE_MODE and page in cache.source.ocr_pages else mode


def _hit_url(cache: DocumentCache, span: Span, term: str, mode: str | None) -> str:
    """Return the URL of the pane showing one occurrence, the others marked around it."""
    start, end = span
    mode = _mode_at(cache, start, mode)
    params: dict[str, object] = {"label": "", "start": start, "end": end, "q": term}
    if mode:
        params["mode"] = mode
    return f"/documents/{cache.doc_id}/view?{urlencode(params)}"


def _hit(cache: DocumentCache, match: Match, term: str, mode: str | None) -> Hit:
    """Return an occurrence for the results list: the words around it on one line."""
    text = cache.text
    before, after = match.before, match.after
    if before and text[match.start - 1].isspace():
        before += " "
    if match.start > SNIPPET_CHARS:
        before = ELLIPSIS + before
    if after and text[match.end].isspace():
        after = " " + after
    if match.end + SNIPPET_CHARS < len(text):
        after += ELLIPSIS
    url = _hit_url(cache, (match.start, match.end), term, mode)
    return Hit(match.page, before, " ".join(match.text.split()), after, url)


def _status(request: Request, text: str, spinning: bool = True) -> str:
    """Return a line of progress; the text is already in the reader's language."""
    return render_string(request, "library/status.html", text=text, spinning=spinning)


def _read_error(error: Exception) -> str:
    """Return why a file could not be read, in plain words."""
    message = str(error)
    if isinstance(error, EmptyDocumentError):
        return NEEDS_TESSERACT if TESSERACT in message else NO_TEXT
    if isinstance(error, UnsupportedFileError):
        return NOT_UTF8 if message.startswith(NOT_UTF8_ERROR) else UNSUPPORTED_FILE
    return DAMAGED_FILE


def _upload_frames(
    request: Request, ws: Ws, name: str, data: bytes, folder: str | None
) -> Iterator[str]:
    if not name:
        yield frame("", notice=_t(request, CHOOSE_FILE), done=True)
        return
    yield frame(_status(request, _t(request, READING_FILE)))
    # The file keeps its name: the contract is named after it.
    with tempfile.TemporaryDirectory(prefix="contract-upload-") as directory:
        path = Path(directory) / name
        path.write_bytes(data)
        yield from _analyse(request, ws, path, folder)


def _sample_frames(request: Request, ws: Ws, path: Path, folder: str | None) -> Iterator[str]:
    yield frame(_status(request, _t(request, READING_FILE)))
    yield from _analyse(request, ws, path, folder)


def _analyse(request: Request, ws: Ws, path: Path, folder: str | None) -> Iterator[str]:
    """Read a contract file, find its clauses unless its text was read before, then open it.

    The analysis is kept once per text; the contract is the user's file in the folder. The
    same text in the same folder opens the contract there instead of adding a copy. A scanned
    PDF page is read with OCR here; the pages read that way are saved with the analysis.
    """
    try:
        doc = load_document(path)
        doc_id = doc_id_of(doc.text)
        cached = _cached(ws, doc_id)
    except UPLOAD_ERRORS as error:
        yield frame("", notice=_t(request, _read_error(error)), done=True)
        return
    extractor = ws.services.extractor
    if cached is None:
        if extractor is None:
            yield frame("", notice=_t(request, NO_MODEL), done=True)
            return
        analysed = yield from _with_progress(request, lambda report: extractor.extract(doc, report))
        yield frame(_status(request, _t(request, SAVING)))
        ws.save_analysis(analysed)
        message = ANALYSIS_COMPLETE
    else:
        message = yield from _extend(request, ws, cached)
    existing = ws.find(doc_id, folder)
    if existing is None:
        # Whether someone else read this text before is not the reader's to know.
        contract_id = ws.add_contract(folder, path.name, doc_id, doc.kind, path.read_bytes()).id
        message = ANALYSIS_COMPLETE
    else:
        contract_id = existing.id
        if message != ANALYSIS_UPDATED:
            message = ALREADY_ANALYSED
    if existing is None or message == ANALYSIS_UPDATED:
        _precompute(ws, contract_id)
    text = _t(request, message).format(name=document_name(path.name))
    yield _redirect(request, contract_id, text)


def _extend(request: Request, ws: Ws, cached: DocumentCache) -> Generator[str, None, str]:
    """Find the clauses added since a text was read, if any; return what to tell the reader."""
    extractor = ws.services.extractor
    if extractor is None:
        return ALREADY_ANALYSED
    extended = yield from _with_progress(request, lambda report: extractor.extend(cached, report))
    if extended is cached:
        return ALREADY_ANALYSED
    yield frame(_status(request, _t(request, SAVING)))
    ws.save_analysis(extended)
    return ANALYSIS_UPDATED


def _with_progress(
    request: Request, work: Callable[[ProgressCallback], DocumentCache]
) -> Generator[str, None, DocumentCache]:
    """Run the extraction in a thread; yield a status frame per clause it looks for.

    The extractor reports synchronously from inside its loop, so the frames come from a
    queue the callback fills. Returns the extraction's result.
    """
    finding = _t(request, FINDING)
    items: queue.Queue[ProgressItem | None] = queue.Queue()
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(work, lambda done, total, name: items.put((done, total, name)))
    future.add_done_callback(lambda _: items.put(None))
    executor.shutdown(wait=False)
    while (item := items.get()) is not None:
        done, total, _ = item
        elapsed = f"{time.perf_counter() - started:.0f}"
        yield frame(_status(request, finding.format(done=done, total=total, elapsed=elapsed)))
    return future.result()


def _cached(ws: UserWorkspace, doc_id: str) -> DocumentCache | None:
    """Return the analysis of a text as saved, or None; one of other settings is read again."""
    try:
        return ws.analyses.load(doc_id)
    except CacheMismatchError:
        return None


def _precompute(ws: UserWorkspace, contract_id: str) -> None:
    """Write the summary, the key terms and the obligations ahead of time, when generating."""
    gen = ws.services.gen
    if gen is None:
        return
    cache = ws.load(contract_id)
    settings, unvalidated = ws.settings, ws.services.unvalidated
    run_in_background(
        precompute,
        cache,
        ws.names,
        gen,
        ws.summary_store(ENGLISH),
        ws.term_sheets,
        settings.tau_low,
        settings.tau_high,
        unvalidated,
    )
    run_in_background(
        precompute_obligations,
        cache,
        ws.names,
        gen,
        ws.obligations,
        settings.tau_low,
        settings.tau_high,
        unvalidated,
    )


def _redirect(request: Request, doc_id: str, text: str) -> str:
    html = _status(request, text, spinning=False)
    return frame(html, notice=text, redirect=f"/documents/{doc_id}", done=True)


def _compare_frames(
    request: Request, ws: Ws, tau: Thresholds, left_id: str, right_id: str
) -> Iterator[str]:
    try:
        yield from _comparison(request, ws, tau, left_id, right_id)
    except WorkspaceError as error:
        message = _t(request, error.message)
        notice = render_string(request, "library/status.html", text=message, notice=True)
        yield frame(notice, notice=message, done=True)


def _comparison(
    request: Request, ws: Ws, tau: Thresholds, left_id: str, right_id: str
) -> Iterator[str]:
    if not left_id or not right_id:
        raise WorkspaceError(CHOOSE_OTHER)
    if left_id == right_id:
        raise WorkspaceError(CHOOSE_DIFFERENT)
    entries = [ws.entry(left_id), ws.entry(right_id)]
    if None in entries:
        raise WorkspaceError(OTHER_UNAVAILABLE)
    lang = language(request)
    yield frame(_status(request, translate(COMPARING, lang)))
    caches = [ws.load(left_id), ws.load(right_id)]
    try:
        rows = compare_checklists(*caches, ws.names, tau.low, tau.high, ws.services.unvalidated)
    except ValueError as error:
        raise WorkspaceError(str(error)) from error
    sheets: list[TermSheet] = []
    reason = None
    for cache in caches:
        result = yield from _term_sheet(request, ws, cache, tau)
        if isinstance(result, str):
            reason = result
            break
        sheets.append(result)
    terms = compare_terms(*sheets) if reason is None else []
    withheld = [
        translate(TERMS_WITHHELD, lang).format(
            name=document_name(cache.source.filename), reason=translate(sheet.reason, lang)
        )
        for cache, sheet in zip(caches, sheets, strict=False)
        if sheet.reason
    ]
    sides = [
        _side(ws, entry, cache, rows, index, tau, lang)
        for index, (entry, cache) in enumerate(zip(entries, caches, strict=True))
    ]
    context = {
        "sides": sides,
        "terms": terms,
        "terms_note": translate(TERMS_SKIPPED, lang).format(reason=reason) if reason else None,
        "withheld": withheld,
        "groups": _groups(rows, *sides, lang),
        "differ": sum(row.differs for row in rows),
        "terms_differ": sum(row.differs for row in terms),
        "total": len(rows),
        "not_stated": NOT_STATED,
        "not_verified": NOT_VERIFIED,
        "tau": tau,
    }
    oob = {
        "compare-summary": render_string(request, "library/compare_tile.html", **context),
        **{
            f"{side['key']}-counts": render_string(request, "library/counts.html", **side)
            for side in sides
        },
    }
    html = render_string(request, "library/compare_result.html", **context)
    yield frame(html, oob=oob, done=True)


def _term_sheet(
    request: Request, ws: Ws, cache: DocumentCache, tau: Thresholds
) -> Generator[str, None, TermSheet | str]:
    """Yield status frames while the key terms are extracted; return the sheet or why not.

    The reason is in the reader's language.
    """
    lang = language(request)
    name = document_name(cache.source.filename)
    if ws.services.gen is None:
        return translate(TERMS_NEED_MODEL, lang)
    if not term_sheet_items(cache, tau.low, tau.high, ws.services.unvalidated):
        return translate(TERMS_NEED_CLAUSES, lang).format(name=name)
    try:
        for step in term_sheet_steps(ws, cache, tau):
            if isinstance(step, TermSheet):
                return step
            yield frame(_status(request, translate(EXTRACTING_TERMS, lang).format(name=name)))
    except LLMError:
        return translate(TERMS_FAILED, lang)
    except WorkspaceError as error:
        return translate(error.message, lang).rstrip(".")
    return translate(TERMS_FAILED, lang)


def _side(
    ws: Ws,
    entry: DocumentEntry,
    cache: DocumentCache,
    rows: list[ChecklistRow],
    index: int,
    tau: Thresholds,
    lang: str,
) -> dict:
    """Return what the page shows of one contract: its counts and its review flags.

    The flag titles come from the fixed rule list, so the catalogue has them.
    """
    statuses: list[Status] = [row.right if index else row.left for row in rows]
    flags = ws.flags(cache, tau)
    return {
        "key": "right" if index else "left",
        "entry": entry,
        "found": statuses.count("found"),
        "check": statuses.count("check"),
        "not_found": statuses.count("not_found"),
        "severities": _severities(flags),
        "high": [translate(flag.title, lang) for flag in flags if flag.severity == "high"],
        "titles": [translate(flag.title, lang) for flag in flags],
    }


def _severities(flags: list[Flag]) -> list[tuple[str, int]]:
    counts = [(level, sum(flag.severity == level for flag in flags)) for level in SEVERITIES]
    return [(level, count) for level, count in counts if count]


def _groups(rows: list[ChecklistRow], left: dict, right: dict, lang: str) -> list[dict]:
    """Return the clauses grouped by how the two contracts differ, differences first."""
    left_name, right_name = left["entry"].name, right["entry"].name
    only_in = translate(ONLY_IN, lang)
    groups = [
        ("diff", only_in.format(name=right_name), None,
         [row for row in rows if row.left == "not_found" and row.right != "not_found"]),
        ("diff", only_in.format(name=left_name), None,
         [row for row in rows if row.right == "not_found" and row.left != "not_found"]),
        ("diff", translate(FOUND_AND_CHECK, lang),
         translate(FIRST_SECOND, lang).format(left=left_name, right=right_name),
         [row for row in rows if row.differs and "not_found" not in (row.left, row.right)]),
        ("same", translate(SAME_IN_BOTH, lang), None, [row for row in rows if not row.differs]),
    ]
    return [
        {"kind": kind, "title": title, "note": note, "rows": members}
        for kind, title, note, members in groups
        if members
    ]


def _portfolio(ws: Ws) -> list[PortfolioRow]:
    """Tabulate every contract the user can open."""
    return portfolio_rows(
        ContractShelf(ws),
        ws.settings,
        ws.names,
        ws.rules,
        ws.term_sheets,
        ws.gen_model,
        ws.services.unvalidated,
    )


def _portfolio_item(
    row: PortfolioRow, entry: DocumentEntry | None, total: int, lang: str
) -> dict:
    """Return one row of the portfolio table as the template shows it."""
    terms = {
        field: (value, True) if _stated(value) else (_absent(value, lang), False)
        for field, value in row.terms.items()
    }
    parties = row.terms.get("parties", "")
    kind = entry.kind if entry else Path(row.name).suffix.lstrip(".").lower()
    name = document_name(row.name)
    governing = row.terms.get("governing_law", "")
    return {
        "row": row,
        "name": name,
        "kind": kind,
        "parties": _shorten(parties, SHORT_PARTIES) if _stated(parties) else None,
        "terms": terms,
        "found_width": round(100 * row.found / total) if total else 0,
        "check_width": round(100 * row.check / total) if total else 0,
        "search": f"{name} {parties} {governing}".lower(),
        "renews": _stated(row.terms.get("renewal_term", "")),
        "no_law": governing == NOT_STATED,
        "severities": [
            (level, count)
            for level, count in zip(
                SEVERITIES, (row.flags_high, row.flags_medium, row.flags_low), strict=True
            )
            if count
        ],
    }


def _stated(value: str) -> bool:
    """Tell whether a term sheet cell holds a value rather than why it is empty."""
    return bool(value) and not value.startswith("not ")


def _absent(value: str, lang: str) -> str:
    """Return why a term sheet cell is empty in the reader's language; the values it names
    after "not verified" stay as the contract has them."""
    if value.startswith(NOT_VERIFIED):
        return translate(NOT_VERIFIED, lang) + value.removeprefix(NOT_VERIFIED)
    return translate(value, lang)


def _shorten(text: str, limit: int) -> str:
    squashed = " ".join(text.split())
    return squashed if len(squashed) <= limit else squashed[: limit - 1] + "…"


def _next_notice(ws: Ws, rows: list[PortfolioRow], lang: str) -> dict | None:
    """Return the earliest notice deadline still ahead, from the saved term sheets."""
    if ws.gen_model is None:
        return None
    settings, today = ws.settings, datetime.now(tz=UTC).date()
    upcoming = []
    for row in rows:
        sheet = ws.term_sheets.load(row.doc_id, ws.gen_model, settings.tau_low, settings.tau_high)
        if sheet is None:
            continue
        for item in key_dates(sheet):
            if NOTICE_LABEL in item.label.lower() and item.date and item.date > today:
                upcoming.append((item.date, row, item))
    if not upcoming:
        return None
    when, row, item = min(upcoming, key=lambda found: found[0])
    return {
        "when": _date(when, lang),
        "name": document_name(row.name),
        "source": item.source,
        "doc_id": row.doc_id,
    }


def _date(day: date, lang: str) -> str:
    """Return a date as "1 Jul 2027", or the way the language writes it."""
    month = translate(MONTHS[day.month - 1], lang)
    return translate(DATE, lang).format(day=day.day, month=month, year=day.year)


def _markup(line: str) -> Markup:
    """Turn one line of the technical Markdown list into HTML: bold and code only."""
    html = str(escape(line.removeprefix("- ")))
    html = BOLD.sub(r"<strong>\1</strong>", html)
    return Markup(CODE.sub(r"<code>\1</code>", html))


def _rail(
    request: Request, ws: UserWorkspace, cache: DocumentCache, documents: list[DocumentEntry]
) -> dict:
    """Return what the drawer shows, the same in the workspace page and in its refresh."""
    can_read = ws.services.extractor is not None
    return {
        "cache": cache,
        "documents": documents,
        "can_read": can_read,
        "samples": _samples(documents) if can_read else [],
        "samples_need_model": _t(request, SAMPLES_NEED_MODEL),
    }


def _samples(documents: list[DocumentEntry]) -> list[Sample]:
    """Return the sample contracts that are not among the analysed ones."""
    analysed = {entry.analysis_id for entry in documents}
    samples = [_sample(name) for name in SAMPLE_FILES]
    return [item for item in samples if item is not None and item.doc_id not in analysed]


def _sample(name: str) -> Sample | None:
    """Describe a sample contract; None when its file is missing or unreadable."""
    path = SAMPLES_DIR / name
    try:
        key = (path, path.stat().st_mtime_ns)
    except OSError:
        return None
    known = _samples_read.get(key)
    if known is not None:
        return known
    try:
        doc = load_document(path)
    except SAMPLE_ERRORS:
        return None
    pdf = doc.kind == "pdf"
    size = f"{len(doc.page_starts)} pages" if pdf else f"{len(doc.text.split()):,} words"
    known = Sample(name, document_name(name), doc.kind, size, doc_id_of(doc.text))
    _samples_read[key] = known
    return known
