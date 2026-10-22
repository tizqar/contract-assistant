"""The contracts: first page, drawer, document pane, uploads, samples, compare and portfolio."""

from __future__ import annotations

import queue
import re
import tempfile
import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from markupsafe import Markup, escape

from contract_assistant.answer import Citation, Status
from contract_assistant.background import precompute, precompute_obligations, run_in_background
from contract_assistant.compare import ChecklistRow, compare_checklists, compare_terms
from contract_assistant.config import ROOT
from contract_assistant.dates import key_dates
from contract_assistant.decisions import apply as apply_decisions
from contract_assistant.extract.extractor import ProgressCallback
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.pdf_view import (
    DOCUMENTS_DIR,
    PageError,
    copy_path,
    keep_copy,
    page_image,
)
from contract_assistant.portfolio import PortfolioRow, portfolio_rows, to_csv
from contract_assistant.review import Flag
from contract_assistant.router.clients import LLMError
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    doc_id_of,
)
from contract_assistant.term_sheet import NOT_STATED, TermSheet, term_sheet_items
from contract_assistant.ui.views import (
    SUGGESTED_QUESTIONS,
    display_name,
    document_name,
    document_view,
    flag_title,
    page_of,
    technical_markdown,
)
from contract_assistant.web.core import (
    ENGLISH,
    DocumentEntry,
    Tau,
    Thresholds,
    Workspace,
    WorkspaceError,
    Ws,
    frame,
    render,
    render_string,
    stream,
)
from contract_assistant.web.generate import term_sheet_steps

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
CODE = re.compile(r"`(.+?)`")

CHOOSE_FILE = "Choose a PDF or TXT file first."
# The steps of reading a contract, shown one at a time; no clause names, no model names.
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
NO_PAGE_IMAGE = "No page image: the PDF of this contract is not kept on this computer."
NO_SUCH_PAGE = "This page is not in the PDF."
NO_ANALYSIS_FILE = "The analysis file of this contract was not found."
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


@dataclass(frozen=True)
class Sample:
    """A sample contract as the first page and the drawer offer it."""

    name: str
    title: str
    kind: str
    size: str
    doc_id: str


# Each sample file is read once per version, to know its id and its size.
_samples_read: dict[tuple[Path, int], Sample] = {}


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
            no_model=NO_MODEL,
            samples_need_model=SAMPLES_NEED_MODEL,
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
        **_rail(ws, cache, documents),
    )


@router.get("/documents/{doc_id}/rail", response_class=HTMLResponse)
def rail(request: Request, doc_id: str, ws: Ws):
    """The contracts drawer alone, to refresh its counts after a decision or an upload."""
    cache = ws.load(doc_id)
    return render(request, "library/rail.html", **_rail(ws, cache, ws.documents()))


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
):
    """The document pane: the PDF page or the text around a span, with the passage card."""
    cache = ws.load(doc_id)
    citation = _citation(cache, start, end, conf)
    clause_type = ws.clause_type_of(label) if label else PASSAGE_LABEL
    copy = _pdf_copy(ws, cache)
    mode = mode if copy is not None and mode in (PAGE_MODE, TEXT_MODE) else None
    mode = mode or (PAGE_MODE if copy is not None else TEXT_MODE)
    pages = len(cache.page_starts)
    span_page = page_of(cache.page_starts, citation.start) if citation and pages else 1
    shown = min(max(page or span_page, 1), max(pages, 1))
    if mode == TEXT_MODE and page is not None and pages:
        heading, segments = f"Page {shown}", _page_segments(cache, shown, clause_type, citation)
    else:
        heading, segments = document_view(cache, clause_type, citation)

    def url(**changes) -> str:
        params = {
            "label": label,
            "start": citation.start if citation else None,
            "end": citation.end if citation else None,
            "conf": f"{citation.conf:g}" if citation else None,
            "status": status,
            "mode": mode,
            "page": page,
            **changes,
        }
        query = urlencode({key: value for key, value in params.items() if value is not None})
        return f"/documents/{cache.doc_id}/view?{query}"

    image = None
    if mode == PAGE_MODE:
        span = {"start": citation.start, "end": citation.end} if citation else {}
        boxed = f"?{urlencode(span)}" if span and shown == span_page else ""
        image = f"/documents/{cache.doc_id}/page/{shown}.png{boxed}"
    return render(
        request,
        "library/view.html",
        cache=cache,
        heading=heading,
        segments=[(BLANK_LINES.sub("\n\n", text), mark) for text, mark in segments],
        label=label or PASSAGE_LABEL,
        citation=citation,
        status=status,
        mode=mode,
        has_copy=copy is not None,
        image=image,
        page=shown,
        pages=pages,
        page_url=url(mode=PAGE_MODE),
        text_url=url(mode=TEXT_MODE),
        previous_url=url(page=shown - 1) if shown > 1 else None,
        next_url=url(page=shown + 1) if shown < pages else None,
        tau=tau,
    )


@router.get("/documents/{doc_id}/page/{number}.png")
def page_png(
    request: Request,
    doc_id: str,
    number: int,
    ws: Ws,
    start: int | None = None,
    end: int | None = None,
):
    """One page of the contract's PDF as an image, the passage between start and end boxed."""
    cache = ws.load(doc_id)
    copy = _pdf_copy(ws, cache)
    if copy is None:
        raise HTTPException(status_code=404, detail=NO_PAGE_IMAGE)
    citation = _citation(cache, start, end, 1.0)
    key = (cache.doc_id, number, citation.quote if citation else None)
    # Paging back and forth asks for the same image again; the last one is kept.
    rendered = getattr(request.app.state, "rendered_page", None)
    if rendered is None or rendered[0] != key:
        try:
            rendered = (key, page_image(copy, number, key[2]))
        except PageError as error:
            raise HTTPException(status_code=404, detail=NO_SUCH_PAGE) from error
        request.app.state.rendered_page = rendered
    return Response(rendered[1], media_type="image/png", headers=PAGE_CACHE_HEADERS)


@router.get("/documents/{doc_id}/analysis.json")
def analysis_json(doc_id: str, ws: Ws):
    """The saved analysis of a contract: the clause candidates the encoder located."""
    cache = ws.load(doc_id)
    path = ws.store.find(cache.doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail=NO_ANALYSIS_FILE)
    filename = f"{Path(cache.source.filename).stem}.analysis.json"
    return FileResponse(path, media_type="application/json", filename=filename)


@router.post("/upload")
def upload(
    request: Request, ws: Ws, file: Annotated[UploadFile | None, File()] = None
):
    """Analyse an uploaded contract unless it was analysed before; stream the progress."""
    name = Path(file.filename or "").name if file is not None else ""
    data = file.file.read() if file is not None and name else b""
    return stream(_upload_frames(request, ws, name, data))


@router.post("/samples/{name}")
def sample(request: Request, name: str, ws: Ws):
    """Analyse a sample contract the way an upload is, unless it was analysed before."""
    if name not in SAMPLE_FILES:
        raise WorkspaceError(UNKNOWN_SAMPLE, status_code=404)
    return stream(_sample_frames(request, ws, SAMPLES_DIR / name))


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
        notice=None if right_id else ADD_SECOND,
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
    rows = _portfolio(ws)
    entries = {entry.doc_id: entry for entry in ws.documents()}
    total = len(ws.names)
    contracts = [_portfolio_item(row, entries.get(row.doc_id), total) for row in rows]
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
        deadline=_next_notice(ws, rows),
        tau=tau,
        cache=None,
    )


@router.get("/portfolio.csv")
def portfolio_csv(ws: Ws):
    """The portfolio table as a CSV file."""
    _portfolio(ws)
    return FileResponse(
        ws.settings.cache_dir / PORTFOLIO_CSV, media_type="text/csv", filename=PORTFOLIO_CSV
    )


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


def _pdf_copy(ws: Workspace, cache: DocumentCache) -> Path | None:
    if cache.source.kind != "pdf":
        return None
    copy = copy_path(ws.settings.cache_dir / DOCUMENTS_DIR, cache.doc_id)
    return copy if copy.exists() else None


def _page_segments(
    cache: DocumentCache, number: int, clause_type: str, citation: Citation | None
) -> list[tuple[str, str | None]]:
    """Return the text of one page, the part of the citation on it marked."""
    starts, text = cache.page_starts, cache.text
    lo = starts[number - 1]
    hi = starts[number] if number < len(starts) else len(text)
    if citation is None or citation.end <= lo or citation.start >= hi:
        return [(text[lo:hi], None)]
    mark_lo, mark_hi = max(citation.start, lo), min(citation.end, hi)
    parts = [
        (text[lo:mark_lo], None),
        (text[mark_lo:mark_hi], display_name(clause_type)),
        (text[mark_hi:hi], None),
    ]
    return [part for part in parts if part[0]]


def _status(request: Request, text: str, spinning: bool = True) -> str:
    return render_string(request, "library/status.html", text=text, spinning=spinning)


def _upload_frames(request: Request, ws: Ws, name: str, data: bytes) -> Iterator[str]:
    if not name:
        yield frame("", notice=CHOOSE_FILE, done=True)
        return
    yield frame(_status(request, READING_FILE))
    # The file keeps its name: the analysis records it and the lists show it.
    with tempfile.TemporaryDirectory(prefix="contract-upload-") as directory:
        path = Path(directory) / name
        path.write_bytes(data)
        yield from _analyse(request, ws, path)


def _sample_frames(request: Request, ws: Ws, path: Path) -> Iterator[str]:
    yield frame(_status(request, READING_FILE))
    yield from _analyse(request, ws, path)


def _analyse(request: Request, ws: Ws, path: Path) -> Iterator[str]:
    """Read a contract file, find its clauses unless that was done before, then open it."""
    try:
        doc = load_document(path)
        cached = _cached(ws, doc_id_of(doc.text))
    except UPLOAD_ERRORS as error:
        yield frame("", notice=str(error), done=True)
        return
    if cached is not None:
        yield from _reopen(request, ws, path, cached)
        return
    extractor = ws.services.extractor
    if extractor is None:
        yield frame("", notice=NO_MODEL, done=True)
        return
    analysed = yield from _with_progress(request, lambda report: extractor.extract(doc, report))
    yield frame(_status(request, SAVING))
    _save(ws, path, analysed)
    yield _redirect(request, analysed.doc_id, ANALYSIS_COMPLETE)


def _reopen(request: Request, ws: Ws, path: Path, cached: DocumentCache) -> Iterator[str]:
    """Open a contract analysed before, after finding the clauses added since, if any."""
    extractor = ws.services.extractor
    if extractor is not None:
        extended = yield from _with_progress(
            request, lambda report: extractor.extend(cached, report)
        )
        if extended is not cached:
            yield frame(_status(request, SAVING))
            _save(ws, path, extended)
            yield _redirect(request, cached.doc_id, ANALYSIS_UPDATED)
            return
    if not copy_path(ws.settings.cache_dir / DOCUMENTS_DIR, cached.doc_id).exists():
        _keep_pdf(ws, path, cached.doc_id)
    name = document_name(path.name)
    yield _redirect(request, cached.doc_id, ALREADY_ANALYSED.format(name=name))


def _save(ws: Workspace, path: Path, cache: DocumentCache) -> None:
    """Save an analysis, keep the PDF to show its pages, and start writing ahead of time."""
    ws.store.save(cache)
    _keep_pdf(ws, path, cache.doc_id)
    _precompute(ws, apply_decisions(cache, ws.decided(cache.doc_id)))


def _with_progress(
    request: Request, work: Callable[[ProgressCallback], DocumentCache]
) -> Generator[str, None, DocumentCache]:
    """Run the extraction in a thread; yield a status frame per clause it looks for.

    The extractor reports synchronously from inside its loop, so the frames come from a
    queue the callback fills. Returns the extraction's result.
    """
    items: queue.Queue[ProgressItem | None] = queue.Queue()
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(work, lambda done, total, name: items.put((done, total, name)))
    future.add_done_callback(lambda _: items.put(None))
    executor.shutdown(wait=False)
    while (item := items.get()) is not None:
        done, total, _ = item
        elapsed = f"{time.perf_counter() - started:.0f}"
        yield frame(_status(request, FINDING.format(done=done, total=total, elapsed=elapsed)))
    return future.result()


def _cached(ws: Workspace, doc_id: str) -> DocumentCache | None:
    # The analysis as saved, without the reader's decisions: an extension of it is saved back,
    # and the decisions stay in their own file so that Undo still works.
    try:
        return ws.services.store.load(doc_id, ws.settings)
    except CacheMismatchError:
        return None


def _keep_pdf(ws: Workspace, path: Path, doc_id: str) -> None:
    if path.suffix.lower() == ".pdf":
        keep_copy(path, ws.settings.cache_dir / DOCUMENTS_DIR, doc_id)


def _precompute(ws: Workspace, cache: DocumentCache) -> None:
    """Write the summary, the key terms and the obligations ahead of time, when generating."""
    gen = ws.services.gen
    if gen is None:
        return
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
        notice = render_string(request, "library/status.html", text=error.message, notice=True)
        yield frame(notice, notice=error.message, done=True)


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
    yield frame(_status(request, COMPARING))
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
        TERMS_WITHHELD.format(name=document_name(cache.source.filename), reason=sheet.reason)
        for cache, sheet in zip(caches, sheets, strict=False)
        if sheet.reason
    ]
    sides = [
        _side(ws, entry, cache, rows, index, tau)
        for index, (entry, cache) in enumerate(zip(entries, caches, strict=True))
    ]
    context = {
        "sides": sides,
        "terms": terms,
        "terms_note": TERMS_SKIPPED.format(reason=reason) if reason else None,
        "withheld": withheld,
        "groups": _groups(rows, *sides),
        "differ": sum(row.differs for row in rows),
        "terms_differ": sum(row.differs for row in terms),
        "total": len(rows),
        "not_stated": NOT_STATED,
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
    """Yield status frames while the key terms are extracted; return the sheet or why not."""
    name = document_name(cache.source.filename)
    if ws.services.gen is None:
        return TERMS_NEED_MODEL
    if not term_sheet_items(cache, tau.low, tau.high, ws.services.unvalidated):
        return TERMS_NEED_CLAUSES.format(name=name)
    try:
        for step in term_sheet_steps(ws, cache, tau):
            if isinstance(step, TermSheet):
                return step
            yield frame(_status(request, EXTRACTING_TERMS.format(name=name)))
    except LLMError:
        return TERMS_FAILED
    except WorkspaceError as error:
        return error.message.rstrip(".")
    return TERMS_FAILED


def _side(
    ws: Ws,
    entry: DocumentEntry,
    cache: DocumentCache,
    rows: list[ChecklistRow],
    index: int,
    tau: Thresholds,
) -> dict:
    """Return what the page shows of one contract: its counts and its review flags."""
    statuses: list[Status] = [row.right if index else row.left for row in rows]
    flags = ws.flags(cache, tau)
    return {
        "key": "right" if index else "left",
        "entry": entry,
        "found": statuses.count("found"),
        "check": statuses.count("check"),
        "not_found": statuses.count("not_found"),
        "severities": _severities(flags),
        "high": [flag_title(flag) for flag in flags if flag.severity == "high"],
        "titles": [flag_title(flag) for flag in flags],
    }


def _severities(flags: list[Flag]) -> list[tuple[str, int]]:
    counts = [(level, sum(flag.severity == level for flag in flags)) for level in SEVERITIES]
    return [(level, count) for level, count in counts if count]


def _groups(rows: list[ChecklistRow], left: dict, right: dict) -> list[dict]:
    """Return the clauses grouped by how the two contracts differ, differences first."""
    left_name, right_name = left["entry"].name, right["entry"].name
    groups = [
        ("diff", f"Only in {right_name}", None,
         [row for row in rows if row.left == "not_found" and row.right != "not_found"]),
        ("diff", f"Only in {left_name}", None,
         [row for row in rows if row.right == "not_found" and row.left != "not_found"]),
        ("diff", "In both, found in one and to check in the other",
         f"{left_name} first, {right_name} second",
         [row for row in rows if row.differs and "not_found" not in (row.left, row.right)]),
        ("same", "Same in both", None, [row for row in rows if not row.differs]),
    ]
    return [
        {"kind": kind, "title": title, "note": note, "rows": members}
        for kind, title, note, members in groups
        if members
    ]


def _portfolio(ws: Ws) -> list[PortfolioRow]:
    """Tabulate every analysed contract and write the table as a CSV file."""
    rows = portfolio_rows(
        ws.store,
        ws.settings,
        ws.names,
        ws.rules,
        ws.term_sheets,
        ws.gen_model,
        ws.services.unvalidated,
    )
    path = ws.settings.cache_dir / PORTFOLIO_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_csv(rows), encoding="utf-8")
    return rows


def _portfolio_item(row: PortfolioRow, entry: DocumentEntry | None, total: int) -> dict:
    """Return one row of the portfolio table as the template shows it."""
    terms = {field: (value, _stated(value)) for field, value in row.terms.items()}
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


def _shorten(text: str, limit: int) -> str:
    squashed = " ".join(text.split())
    return squashed if len(squashed) <= limit else squashed[: limit - 1] + "…"


def _next_notice(ws: Ws, rows: list[PortfolioRow]) -> dict | None:
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
        "when": f"{when.day} {when:%b %Y}",
        "name": document_name(row.name),
        "source": item.source,
        "doc_id": row.doc_id,
    }


def _markup(line: str) -> Markup:
    """Turn one line of the technical Markdown list into HTML: bold and code only."""
    html = str(escape(line.removeprefix("- ")))
    html = BOLD.sub(r"<strong>\1</strong>", html)
    return Markup(CODE.sub(r"<code>\1</code>", html))


def _rail(ws: Workspace, cache: DocumentCache, documents: list[DocumentEntry]) -> dict:
    """Return what the drawer shows, the same in the workspace page and in its refresh."""
    can_read = ws.services.extractor is not None
    return {
        "cache": cache,
        "documents": documents,
        "can_read": can_read,
        "samples": _samples(documents) if can_read else [],
    }


def _samples(documents: list[DocumentEntry]) -> list[Sample]:
    """Return the sample contracts that are not among the analysed ones."""
    analysed = {entry.doc_id for entry in documents}
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
    known = Sample(name, document_name(name), "PDF" if pdf else "Text", size, doc_id_of(doc.text))
    _samples_read[key] = known
    return known
