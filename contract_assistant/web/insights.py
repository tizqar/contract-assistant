"""The insight tabs of one contract: summary, key terms, review, obligations, clauses, terms."""

from __future__ import annotations

import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse

from contract_assistant.answer import Citation, Status, tier
from contract_assistant.dates import key_dates
from contract_assistant.explain import LANGUAGES
from contract_assistant.fulltext import APP_ATTEMPTS as FULLTEXT_ATTEMPTS
from contract_assistant.fulltext import FullTextSummary, Progress, write_fulltext
from contract_assistant.generation import MS_PER_S, Generation
from contract_assistant.obligations import APP_ATTEMPTS as OBLIGATION_ATTEMPTS
from contract_assistant.obligations import (
    ObligationSheet,
    by_party,
    obligation_items,
    request_obligations,
)
from contract_assistant.report import build_report, write_report
from contract_assistant.review import Flag, flag_request
from contract_assistant.store import DOC_ID_PREFIX, DocumentCache
from contract_assistant.summary import Summary, summary_items
from contract_assistant.term_sheet import FIELDS, NOT_STATED, TermSheet, term_sheet_items
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    FULLTEXT_META_TEXT,
    FULLTEXT_WITHHELD_TEXT,
    NO_PAGE,
    OTHER_TOPIC,
    SECTION_WITHHELD,
    STATUS_LABELS,
    SUMMARY_LANGUAGE_TEXT,
    SUMMARY_META_TEXT,
    SUMMARY_MISMATCH_TEXT,
    SUMMARY_WITHHELD_TEXT,
    UNVALIDATED_MARK,
    UNVALIDATED_TEXT,
    checklist_intro,
    date_rows,
    definition_citation,
    display_name,
    document_name,
    flag_title,
    term_rows,
    top_citation,
)
from contract_assistant.web.core import (
    DEFINITION_LABEL,
    ENGLISH,
    SECTION_LABEL,
    SUMMARIES_OFF,
    Tau,
    Thresholds,
    Workspace,
    WorkspaceError,
    Ws,
    frame,
    render,
    render_string,
    shorten,
    stream,
    view_url,
)
from contract_assistant.web.generate import (
    EXTRACTING_TERMS,
    NO_TERM_CLAUSES,
    NOTHING_TO_SUMMARIZE,
    WRITING_SUMMARY,
    Step,
    checked,
    saved_summary,
    saved_term_sheet,
    summary_steps,
    term_sheet_steps,
)

router = APIRouter()

STACK = "insights/stack.html"
LIVE = "insights/live.html"
SUMMARY = "insights/summary.html"
FULLTEXT = "insights/fulltext.html"
TERMS = "insights/terms.html"
REVIEW = "insights/review.html"
NOTE = "insights/flag_note.html"
OBLIGATIONS = "insights/obligations.html"
CHECKLIST = "insights/checklist.html"
DEFINITIONS = "insights/definitions.html"
REPORT = "insights/report.html"
# A tab that needs the model renders its saved record, or a form that app.js submits at once.
# The stream writes into a live element next to the Stop button, which a frame replacing the
# whole tab would remove; its last frame replaces the tab's panel instead. The panel ids are
# unique to the tab and the language, so a stream that ends after the user moved on is a no-op.
SUMMARY_PANEL = "summary-panel-{language}"
FULLTEXT_PANEL = "fulltext-block"
TERMS_PANEL = "terms-panel"
OBLIGATIONS_PANEL = "obligations-panel"
ALL_CLAUSES = "all"
STATUS_FILTERS = {ALL_CLAUSES: "All", **STATUS_LABELS}
LANGUAGE_CHOICES = [(code, name.split(",")[0]) for code, name in LANGUAGES.items()]
OVERVIEW_STAGE = "overview"
REPORTS_DIR = "reports"
MARKDOWN = ".md"
DOCX = ".docx"
REPORT_TYPES = {
    MARKDOWN: "text/markdown; charset=utf-8",
    DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
JSON_TYPE = "application/json"

WRITING_FLAG = "Writing the note"
WRITING_OBLIGATIONS = "Reading the obligations. This can take half a minute."
WRITING_FULLTEXT = "Writing the overview · {stage}"
MERGING_STAGE = "merging the sections"
UNKNOWN_LANGUAGE = "The summary is written in English or Serbian."
UNKNOWN_FILTER = "Show all clauses, or those that are Found, Verify or Not found."
FLAG_CHANGED = "The flags changed since the page was shown; open the Review tab again."
NO_PASSAGE_ATTACHED = "No passage is attached to this flag."
NO_FLAGS = "No review flags."
NO_SAVED_TERMS = "The key terms of this contract were not extracted at these thresholds."
NO_REPORT = "No report of this contract was built at these thresholds; build it first."
NO_ROWS = "No clause has this status."
NO_DEFINITIONS = "No defined terms were found."
NO_DATES = "No dates could be read from the key terms."
SECTIONS_LINE = "{count} {sections} · {seconds} s"
TERMS_INTRO = (
    "The values are copied word for word from the clauses found in this contract. "
    "A value the language model gave that is not in a clause is shown as not verified. "
    "A term the clauses do not state is shown as not stated."
)
TERM_SHEET_WITHHELD = "The term sheet is not shown because {reason}."
DATES_INTRO = (
    "The dates are read from the key terms. A computed date shows its formula. "
    "The dates are only as good as the key terms above."
)
FLAG_WITHHELD = "The language model's note is not shown because {reason}."
FLAG_META = (
    "Written by the language model {model} from the quotes of the clauses. "
    "It is not legal advice; read the passages."
)
REVIEW_INTRO = (
    "These flags come from a rule file checked against the clauses found in this contract. "
    "No language model is used. Each flag names the clauses it rests on. "
    "\u201cNot validated on test data\u201d marks a flag that rests on one of the extra clause "
    "types. Show passage opens a flag's passage in the document; check the passages yourself."
)
REPORT_BUILT = "Report built from the saved summary, key terms, obligations and overview."
REPORT_MISSING = (
    "Report built without {missing}. Make {them} in the Summary, Key terms and Obligations "
    "tabs, then build the report again."
)
OBLIGATIONS_INTRO = (
    "Who must, may or must not do what, read by the language model from the clauses found "
    "in this contract. The party and the words of each entry are copied word for word from "
    "the clause; entries that are not are dropped. Up to 16 clauses are read, the ones found "
    "first. Show in document highlights the words in the contract."
)
OBLIGATIONS_DROPPED = (
    "{count} entries of the language model were not in the clauses and were dropped."
)
OBLIGATIONS_WITHHELD = "The obligations are not shown because {reason}."
NO_OBLIGATIONS = "None of the clauses found states an obligation the language model could read."
NO_OBLIGATION_CLAUSES = (
    "No clause was found in this contract, so there are no obligations to read."
)
FULLTEXT_INTRO = (
    "An overview of the whole text, written section by section by the language model. "
    "Each section's sentences are checked against that section, and the overview against the "
    "section summaries. It covers what the clause types do not. A long contract takes a few "
    "minutes; the section summaries are saved for next time."
)


@dataclass(frozen=True)
class Source:
    """A clause a generated text rests on, with the link that shows its passage."""

    name: str
    status: Status
    unvalidated: bool
    url: str


@dataclass(frozen=True)
class SummaryView:
    """A summary as the tab shows it: its text and sources, or why it is withheld."""

    text: str | None
    meta: str = ""
    sources: tuple[Source, ...] = ()
    note: str | None = None
    withheld: str | None = None
    mismatch: str | None = None


@dataclass(frozen=True)
class SectionView:
    """One section of a full-text overview with the link that shows its span."""

    number: int
    page: int | None
    text: str | None
    withheld: str | None
    url: str


@dataclass(frozen=True)
class FullTextView:
    """A full-text overview: its text, or why it is withheld, and its sections."""

    text: str | None
    withheld: str | None
    line: str
    meta: str
    sections: tuple[SectionView, ...]


@dataclass(frozen=True)
class TermRow:
    """One key term; kind is "value", "not_verified" or "not_stated"."""

    label: str
    value: str
    kind: str
    clauses: str
    page: str | None


@dataclass(frozen=True)
class DateRow:
    """One key date: when, what, and where it comes from and how."""

    when: str
    label: str
    source: str
    how: str


@dataclass(frozen=True)
class FlagView:
    """A review flag as a card shows it."""

    rule: str
    severity: str
    title: str
    note: str
    clauses: str
    url: str | None
    can_note: bool


@dataclass(frozen=True)
class ObligationView:
    """One verified obligation with its modality, words and link."""

    modality: str
    css: str
    action: str
    clause: str
    page: int | None
    url: str


@dataclass(frozen=True)
class Party:
    """The obligations of one party, in the order the model gave them."""

    name: str
    entries: tuple[ObligationView, ...]


@dataclass(frozen=True)
class ClauseRow:
    """One clause type of the checklist."""

    name: str
    status: Status
    conf: float | None
    page: int | None
    passage: str
    url: str | None


@dataclass(frozen=True)
class Topic:
    """The checklist rows of one topic."""

    name: str
    rows: tuple[ClauseRow, ...]


@dataclass(frozen=True)
class DefinitionView:
    """A defined term with the link that shows its sentence."""

    term: str
    quote: str
    page: int | None
    url: str


# Summary and full-text overview


@router.get("/documents/{doc_id}/summary", response_class=HTMLResponse)
def summary_tab(request: Request, doc_id: str, ws: Ws, tau: Tau, language: str = ENGLISH):
    """The summary in a language, or the form that writes it, then the full-text overview."""
    _check_language(language)
    cache = ws.load(doc_id)
    context = _summary_context(ws, cache, tau, language, saved_summary(ws, cache, tau, language))
    return render(request, STACK, **context)


@router.post("/documents/{doc_id}/summary")
def write_summary(
    request: Request,
    doc_id: str,
    ws: Ws,
    tau: Tau,
    language: Annotated[str, Form()] = ENGLISH,
):
    """Stream the summary as the model writes it, then the tab as it shows a saved one."""
    _check_language(language)
    cache = ws.load(doc_id)
    return stream(_cleared(_summary_frames(request, ws, cache, tau, language)))


def _summary_frames(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds, language: str
) -> Iterator[str]:
    for step in summary_steps(ws, cache, tau, language):
        if isinstance(step, Step):
            yield _live(request, step)
        else:
            yield _final(request, _summary_context(ws, cache, tau, language, step))


def _summary_context(
    ws: Workspace,
    cache: DocumentCache,
    tau: Thresholds,
    language: str,
    summary: Summary | None,
) -> dict:
    context = {
        "cache": cache,
        "language": language,
        "languages": LANGUAGE_CHOICES,
        "panel_id": SUMMARY_PANEL.format(language=language),
        "body": SUMMARY,
    }
    if ws.gen_model is None:
        return {**context, "notice": SUMMARIES_OFF}
    if summary is not None:
        view = _summary_view(cache.doc_id, summary, ws.services.unvalidated)
        return {**context, "view": view, **_fulltext_context(ws, cache, tau)}
    if not summary_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated):
        return {**context, "notice": NOTHING_TO_SUMMARIZE, **_fulltext_context(ws, cache, tau)}
    return {**context, "pending": WRITING_SUMMARY}


def _summary_view(doc_id: str, summary: Summary, unvalidated: Collection[str]) -> SummaryView:
    if summary.text is None:
        mismatch = None
        if summary.unfaithful_spans:
            mismatch = SUMMARY_MISMATCH_TEXT.format(count=len(summary.unfaithful_spans))
        return SummaryView(
            text=None,
            withheld=SUMMARY_WITHHELD_TEXT.format(reason=summary.reason),
            mismatch=mismatch,
        )
    meta = SUMMARY_META_TEXT.format(
        model=summary.model,
        count=len(summary.items),
        seconds=round(summary.latency_ms / MS_PER_S),
    )
    if summary.language != ENGLISH:
        meta += SUMMARY_LANGUAGE_TEXT.format(language=dict(LANGUAGE_CHOICES)[summary.language])
    # A summary keeps only the items with a citation (summary.cited).
    sources = tuple(
        Source(
            name=display_name(item.clause_type),
            status=item.status,
            unvalidated=item.clause_type in unvalidated,
            url=view_url(doc_id, display_name(item.clause_type), item.citation, item.status),
        )
        for item in summary.items
    )
    note = None
    if any(source.unvalidated for source in sources):
        note = f"{UNVALIDATED_MARK} {UNVALIDATED_TEXT}"
    return SummaryView(text=summary.text, meta=meta, sources=sources, note=note)


def _check_language(language: str) -> None:
    if language not in LANGUAGES:
        raise WorkspaceError(UNKNOWN_LANGUAGE)


@router.get("/documents/{doc_id}/fulltext", response_class=HTMLResponse)
def fulltext_block(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The full-text overview block alone: the saved overview, or the button that writes it."""
    cache = ws.load(doc_id)
    context = _fulltext_context(ws, cache, tau)
    return render(request, STACK, cache=cache, panel_id=FULLTEXT_PANEL, body=FULLTEXT, **context)


@router.post("/documents/{doc_id}/fulltext")
def write_fulltext_overview(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Stream the progress through the sections, then the overview block."""
    cache = ws.load(doc_id)
    return stream(_cleared(_fulltext_frames(request, ws, cache, tau)))


def _fulltext_frames(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[str]:
    for step in fulltext_steps(ws, cache, tau):
        if isinstance(step, Step):
            yield _live(request, step)
        else:
            context = _fulltext_context(ws, cache, tau, step)
            yield _final(
                request, {"cache": cache, "panel_id": FULLTEXT_PANEL, "body": FULLTEXT, **context}
            )


def fulltext_steps(
    ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[Step | FullTextSummary]:
    """Write the overview section by section, reusing a saved one; a step per stage."""
    gen = ws.gen()
    summary = ws.fulltexts.load(cache.doc_id, gen.model, tau.low, tau.high)
    if summary is None:
        started = time.perf_counter()
        for step in write_fulltext(cache, gen, tau.low, tau.high, attempts=FULLTEXT_ATTEMPTS):
            if isinstance(step, Progress):
                stage = MERGING_STAGE if step.stage == OVERVIEW_STAGE else step.stage
                elapsed = round(time.perf_counter() - started)
                yield Step(WRITING_FULLTEXT.format(stage=stage), "", elapsed)
            else:
                summary = step
        # A withheld overview is not saved, so the next request asks the model again.
        if summary.text is not None:
            ws.fulltexts.save(summary)
    yield summary


def _fulltext_context(
    ws: Workspace,
    cache: DocumentCache,
    tau: Thresholds,
    summary: FullTextSummary | None = None,
) -> dict:
    context: dict = {"fulltext_panel": FULLTEXT_PANEL, "fulltext_intro": FULLTEXT_INTRO}
    if ws.gen_model is None:
        return {**context, "fulltext_notice": SUMMARIES_OFF}
    if summary is None:
        summary = ws.fulltexts.load(cache.doc_id, ws.gen_model, tau.low, tau.high)
    if summary is None:
        return {**context, "fulltext": None, "fulltext_saved": False}
    view = _fulltext_view(cache, summary)
    return {**context, "fulltext": view, "fulltext_saved": summary.text is not None}


def _fulltext_view(cache: DocumentCache, summary: FullTextSummary) -> FullTextView:
    count = len(summary.sections)
    seconds = round(summary.latency_ms / MS_PER_S)
    sections = []
    for number, section in enumerate(summary.sections, start=1):
        label = SECTION_LABEL.format(number=number)
        citation = Citation(
            quote=cache.text[section.start : section.end],
            page=section.page,
            conf=1.0,
            start=section.start,
            end=section.end,
        )
        withheld = None
        if section.text is None:
            withheld = SECTION_WITHHELD.format(reason=section.reason)
        sections.append(
            SectionView(
                number=number,
                page=section.page,
                text=section.text,
                withheld=withheld,
                url=view_url(cache.doc_id, label, citation),
            )
        )
    withheld = None
    if summary.text is None:
        withheld = FULLTEXT_WITHHELD_TEXT.format(reason=summary.reason)
    return FullTextView(
        text=summary.text,
        withheld=withheld,
        line=SECTIONS_LINE.format(
            count=count, sections="section" if count == 1 else "sections", seconds=seconds
        ),
        meta=FULLTEXT_META_TEXT.format(model=summary.model, count=count, seconds=seconds),
        sections=tuple(sections),
    )


# Key terms


@router.get("/documents/{doc_id}/terms", response_class=HTMLResponse)
def terms_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The key terms and key dates, or the form that extracts them."""
    cache = ws.load(doc_id)
    context = _terms_context(ws, cache, tau, saved_term_sheet(ws, cache, tau))
    return render(request, STACK, **context)


@router.post("/documents/{doc_id}/terms")
def extract_terms(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Stream a status while the model extracts the key terms, then the tab."""
    cache = ws.load(doc_id)
    return stream(_cleared(_terms_frames(request, ws, cache, tau)))


def _terms_frames(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[str]:
    for step in term_sheet_steps(ws, cache, tau):
        if isinstance(step, Step):
            yield _live(request, step)
        else:
            yield _final(request, _terms_context(ws, cache, tau, step))


@router.get("/documents/{doc_id}/terms.json")
def terms_json(doc_id: str, ws: Ws, tau: Tau):
    """The saved term sheet as a JSON file."""
    cache = ws.load(doc_id)
    sheet = saved_term_sheet(ws, cache, tau)
    if sheet is None:
        raise WorkspaceError(NO_SAVED_TERMS, status_code=404)
    filename = f"{Path(cache.source.filename).stem}-key-terms.json"
    return FileResponse(ws.term_sheets.path_of(sheet), media_type=JSON_TYPE, filename=filename)


def _terms_context(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, sheet: TermSheet | None
) -> dict:
    context = {"cache": cache, "intro": TERMS_INTRO, "panel_id": TERMS_PANEL, "body": TERMS}
    if ws.gen_model is None:
        return {**context, "notice": SUMMARIES_OFF}
    if sheet is None:
        if not term_sheet_items(cache, tau.low, tau.high, ws.services.unvalidated):
            return {**context, "notice": NO_TERM_CLAUSES}
        return {**context, "pending": EXTRACTING_TERMS}
    rows = _term_rows(sheet)
    if sheet.reason is not None:
        notice = TERM_SHEET_WITHHELD.format(reason=sheet.reason)
        return {**context, "notice": notice, "terms": rows}
    dates = [
        DateRow(when=when, label=label, source=source, how=how)
        for label, when, source, how in date_rows(key_dates(sheet))
    ]
    return {
        **context,
        "terms": rows,
        "download": True,
        "dates": dates,
        "dates_intro": DATES_INTRO,
        "no_dates": NO_DATES,
    }


def _term_rows(sheet: TermSheet) -> list[TermRow]:
    rows = []
    for field, (label, value, clauses, page) in zip(FIELDS, term_rows(sheet), strict=True):
        kind = "value"
        if not sheet.terms.get(field):
            kind = "not_stated" if value == NOT_STATED else "not_verified"
        rows.append(TermRow(label, value, kind, clauses, None if page == NO_PAGE else page))
    return rows


# Review


@router.get("/documents/{doc_id}/review", response_class=HTMLResponse)
def review_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The review flags at the thresholds of the request, most severe first."""
    cache = ws.load(doc_id)
    can_note = ws.gen_model is not None
    flags = [_flag_view(cache.doc_id, flag, can_note) for flag in ws.flags(cache, tau)]
    return render(
        request,
        REVIEW,
        cache=cache,
        flags=flags,
        intro=REVIEW_INTRO,
        no_flags=NO_FLAGS,
        no_passage=NO_PASSAGE_ATTACHED,
    )


def _flag_view(doc_id: str, flag: Flag, can_note: bool) -> FlagView:
    cited = next((item for item in flag.items if item.citation), None)
    url = None
    if cited is not None:
        url = view_url(doc_id, display_name(cited.clause_type), cited.citation, cited.status)
    clauses = ", ".join(
        f"{display_name(item.clause_type)} ({STATUS_LABELS[item.status].lower()})"
        for item in flag.items
    )
    return FlagView(
        rule=flag.rule,
        severity=flag.severity,
        title=flag_title(flag),
        note=flag.note,
        clauses=clauses,
        url=url,
        can_note=can_note and cited is not None,
    )


@router.post("/documents/{doc_id}/review/{index}/note")
def write_flag_note(
    request: Request,
    doc_id: str,
    index: int,
    ws: Ws,
    tau: Tau,
    rule: Annotated[str | None, Form()] = None,
):
    """Stream a short note on a flag, written from its quotes and checked against them."""
    cache = ws.load(doc_id)
    flags = ws.flags(cache, tau)
    if not 0 <= index < len(flags) or rule not in (None, flags[index].rule):
        raise WorkspaceError(FLAG_CHANGED)
    flag = flags[index]
    if not any(item.citation for item in flag.items):
        raise WorkspaceError(NO_PASSAGE_ATTACHED)
    return stream(_cleared(_note_frames(request, ws, flag)))


def _note_frames(request: Request, ws: Workspace, flag: Flag) -> Iterator[str]:
    gen = ws.gen()
    outcome: Generation | None = None
    for step in checked(gen, WRITING_FLAG, flag_request(flag), 1):
        if isinstance(step, Generation):
            outcome = step
        else:
            yield _live(request, step)
    if outcome.text is None:
        context = {"withheld": FLAG_WITHHELD.format(reason=outcome.reason)}
    else:
        context = {
            "title": flag.title,
            "text": outcome.text,
            "meta": FLAG_META.format(model=gen.model),
        }
    yield frame(render_string(request, NOTE, **context), done=True)


# Obligations


@router.get("/documents/{doc_id}/obligations", response_class=HTMLResponse)
def obligations_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The obligations by party, or the form that reads them."""
    cache = ws.load(doc_id)
    context = _obligations_context(ws, cache, tau, saved_obligations(ws, cache, tau))
    return render(request, STACK, **context)


@router.post("/documents/{doc_id}/obligations")
def read_obligations(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Stream a status while the model reads the obligations, then the tab."""
    cache = ws.load(doc_id)
    return stream(_cleared(_obligation_frames(request, ws, cache, tau)))


def _obligation_frames(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[str]:
    for step in obligation_steps(ws, cache, tau):
        if isinstance(step, Step):
            yield _live(request, step)
        else:
            yield _final(request, _obligations_context(ws, cache, tau, step))


def saved_obligations(
    ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> ObligationSheet | None:
    """Return the saved obligations of a contract, or None without generating."""
    if ws.gen_model is None:
        return None
    return ws.obligations.load(cache.doc_id, ws.gen_model, tau.low, tau.high)


def obligation_steps(
    ws: Workspace, cache: DocumentCache, tau: Thresholds
) -> Iterator[Step | ObligationSheet]:
    """Read the obligations, reusing a saved sheet; one step while the model works."""
    gen = ws.gen()
    items = obligation_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated)
    if not items:
        raise WorkspaceError(NO_OBLIGATION_CLAUSES)
    sheet = ws.obligations.load(cache.doc_id, gen.model, tau.low, tau.high)
    if sheet is None:
        yield Step(WRITING_OBLIGATIONS, "", 0)
        sheet = request_obligations(
            cache.doc_id, items, gen, tau.low, tau.high, attempts=OBLIGATION_ATTEMPTS
        )
        # A withheld sheet is not saved, so the next request asks the model again.
        if sheet.reason is None:
            ws.obligations.save(sheet)
    yield sheet


def _obligations_context(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, sheet: ObligationSheet | None
) -> dict:
    context = {
        "cache": cache,
        "intro": OBLIGATIONS_INTRO,
        "panel_id": OBLIGATIONS_PANEL,
        "body": OBLIGATIONS,
    }
    if ws.gen_model is None:
        return {**context, "notice": SUMMARIES_OFF}
    if sheet is None:
        items = obligation_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated)
        if not items:
            return {**context, "notice": NO_OBLIGATION_CLAUSES}
        return {**context, "pending": WRITING_OBLIGATIONS}
    if sheet.reason is not None:
        return {**context, "notice": OBLIGATIONS_WITHHELD.format(reason=sheet.reason)}
    dropped = OBLIGATIONS_DROPPED.format(count=len(sheet.rejected)) if sheet.rejected else None
    return {
        **context,
        "notice": None if sheet.obligations else NO_OBLIGATIONS,
        "parties": _parties(cache.doc_id, sheet),
        "dropped": dropped,
    }


def _parties(doc_id: str, sheet: ObligationSheet) -> list[Party]:
    statuses = {item.clause_type: item.status for item in sheet.items}
    parties = []
    for party, obligations in by_party(sheet).items():
        entries = []
        for obligation in obligations:
            name = display_name(obligation.clause_type)
            citation = Citation(
                quote=obligation.action,
                page=obligation.page,
                conf=1.0,
                start=obligation.start,
                end=obligation.end,
            )
            entries.append(
                ObligationView(
                    modality=obligation.modality,
                    css=obligation.modality.replace(" ", "_"),
                    action=" ".join(obligation.action.split()),
                    clause=name,
                    page=obligation.page,
                    url=view_url(doc_id, name, citation, statuses.get(obligation.clause_type)),
                )
            )
        parties.append(Party(party, tuple(entries)))
    return parties


# Clauses and definitions


@router.get("/documents/{doc_id}/checklist", response_class=HTMLResponse)
def checklist(request: Request, doc_id: str, ws: Ws, tau: Tau, status: str = ALL_CLAUSES):
    """Every clause type by topic with its status, optionally only those of one status."""
    if status not in STATUS_FILTERS:
        raise WorkspaceError(UNKNOWN_FILTER)
    cache = ws.load(doc_id)
    unvalidated = ws.services.unvalidated
    wanted = None if status == ALL_CLAUSES else status
    return render(
        request,
        CHECKLIST,
        cache=cache,
        intro=checklist_intro(cache, unvalidated),
        filters=STATUS_FILTERS,
        status=status,
        topics=_topics(cache, tau, wanted, unvalidated),
        no_rows=NO_ROWS,
    )


def _topics(
    cache: DocumentCache, tau: Thresholds, wanted: str | None, unvalidated: Collection[str]
) -> list[Topic]:
    groups = dict(CLAUSE_GROUPS)
    extra = tuple(name for name in cache.clauses if name in unvalidated)
    if extra:
        groups[OTHER_TOPIC] = extra
    topics = []
    for topic, names in groups.items():
        rows = [_clause_row(cache, tau, name, name in unvalidated) for name in names]
        rows = [row for row in rows if wanted is None or row.status == wanted]
        if rows:
            topics.append(Topic(topic, tuple(rows)))
    return topics


def _clause_row(cache: DocumentCache, tau: Thresholds, name: str, capped: bool) -> ClauseRow:
    citation = top_citation(cache, name)
    if citation is None:
        return ClauseRow(display_name(name), "not_found", None, None, "", None)
    status = tier(citation.conf, tau.low, tau.high)
    if capped and status == "found":
        status = "check"
    return ClauseRow(
        name=display_name(name),
        status=status,
        conf=citation.conf,
        page=citation.page,
        passage=shorten(citation.quote),
        url=view_url(cache.doc_id, display_name(name), citation, status),
    )


@router.get("/documents/{doc_id}/definitions", response_class=HTMLResponse)
def definitions(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The terms the contract defines, with the sentence that defines each."""
    cache = ws.load(doc_id)
    views = [
        DefinitionView(
            term=definition.term,
            quote=definition.quote,
            page=definition.page,
            url=view_url(
                cache.doc_id, DEFINITION_LABEL, definition_citation(definition), "definition"
            ),
        )
        for definition in ws.definitions(cache)
    ]
    return render(
        request, DEFINITIONS, cache=cache, definitions=views, no_definitions=NO_DEFINITIONS
    )


# Report


@router.post("/documents/{doc_id}/report", response_class=HTMLResponse)
def build_report_files(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Write the report from what is saved, generating nothing; return its download links."""
    cache = ws.load(doc_id)
    model = ws.gen_model
    summary = sheet = obligations = fulltext = None
    if model is not None:
        keys = (cache.doc_id, model, tau.low, tau.high)
        summary = ws.summary_store(ENGLISH).load(*keys)
        sheet = ws.term_sheets.load(*keys)
        obligations = ws.obligations.load(*keys)
        fulltext = ws.fulltexts.load(*keys)
    missing = []
    if summary is None or summary.text is None:
        missing.append("the summary")
    if sheet is None:
        missing.append("the key terms")
    if obligations is None:
        missing.append("the obligations")
    if fulltext is None or fulltext.text is None:
        missing.append("the full-text overview")
    report = build_report(
        cache,
        document_name(cache.source.filename),
        ws.settings,
        ws.names,
        ws.services.unvalidated,
        summary,
        sheet,
        ws.flags(cache, tau),
        ws.definitions(cache),
        tau_low=tau.low,
        tau_high=tau.high,
        model=model,
        obligations=obligations,
        fulltext=fulltext,
    )
    write_report(report, ws.settings.cache_dir / REPORTS_DIR, _report_stem(cache.doc_id, tau))
    status = REPORT_BUILT
    if missing:
        status = REPORT_MISSING.format(
            missing=_listed(missing), them="it" if len(missing) == 1 else "them"
        )
    return render(request, REPORT, cache=cache, status=status)


@router.get("/documents/{doc_id}/report.docx")
def report_docx(doc_id: str, ws: Ws, tau: Tau):
    """The last report built for the contract at these thresholds, as DOCX."""
    return _report_file(ws, doc_id, tau, DOCX)


@router.get("/documents/{doc_id}/report.md")
def report_markdown(doc_id: str, ws: Ws, tau: Tau):
    """The last report built for the contract at these thresholds, as Markdown."""
    return _report_file(ws, doc_id, tau, MARKDOWN)


def _report_file(ws: Workspace, doc_id: str, tau: Thresholds, suffix: str) -> FileResponse:
    cache = ws.load(doc_id)
    path = ws.settings.cache_dir / REPORTS_DIR / f"{_report_stem(cache.doc_id, tau)}{suffix}"
    if not path.is_file():
        raise WorkspaceError(NO_REPORT, status_code=404)
    filename = f"{Path(cache.source.filename).stem}-report{suffix}"
    return FileResponse(path, media_type=REPORT_TYPES[suffix], filename=filename)


def _report_stem(doc_id: str, tau: Thresholds) -> str:
    # The Gradio app names the files the same way, so both apps share the reports folder.
    return f"{doc_id.removeprefix(DOC_ID_PREFIX)}-{tau.high}-{tau.low}"


def _listed(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


# Frames


def _live(request: Request, step: Step) -> str:
    """Return the frame that shows the status and the text written so far."""
    return frame(render_string(request, LIVE, status=step.status, text=step.text))


def _final(request: Request, context: dict) -> str:
    """Return the last frame: the tab's panel as the GET renders it once saved."""
    html = render_string(request, context["body"], **context)
    return frame(oob={context["panel_id"]: html}, done=True)


def _cleared(frames: Iterator[str]) -> Iterator[str]:
    """Clear the status line when a stream fails; core.stream shows the notice."""
    try:
        yield from frames
    except Exception:
        yield frame("")
        raise
