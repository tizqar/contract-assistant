"""The overview of one contract: what happened, then six sections that each load on their own.

The sections, in the order of copy.SECTIONS, are the summary (with the overview of the whole
text), the key facts, the things to check, who must do what, all clauses and the defined terms.
The words are plain on purpose: no model names, timings, confidence values or thresholds.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse

from contract_assistant.answer import Citation, tier
from contract_assistant.dates import KeyDate, key_dates
from contract_assistant.decisions import Decision
from contract_assistant.explain import LANGUAGES
from contract_assistant.fulltext import APP_ATTEMPTS as FULLTEXT_ATTEMPTS
from contract_assistant.fulltext import FullTextSummary, Progress, write_fulltext
from contract_assistant.generation import Generation
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
from contract_assistant.term_sheet import (
    FIELDS,
    LABELS,
    NOT_STATED,
    TermSheet,
    term_sheet_items,
)
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    OTHER_TOPIC,
    definition_citation,
    display_name,
    document_name,
    top_citation,
)
from contract_assistant.web import copy
from contract_assistant.web.core import (
    DEFINITION_LABEL,
    ENGLISH,
    SECTION_LABEL,
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

OVERVIEW = "insights/overview.html"
HAPPENED = "insights/happened.html"
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
# The route each section of the overview loads, by the section ids of copy.SECTIONS.
SECTION_ROUTES = {
    "summary": "summary",
    "facts": "terms",
    "checks": "review",
    "duties": "obligations",
    "clauses": "checklist",
    "terms": "definitions",
}
# A section that needs the model renders its saved record, or a form that app.js submits at
# once. The stream writes into a live element next to the Stop button, which a frame replacing
# the whole section would remove; its last frame replaces the section's panel instead. The
# panel ids are unique to the section and the language, so a stream that ends after the reader
# moved on is a no-op.
SUMMARY_PANEL = "summary-panel-{language}"
FULLTEXT_PANEL = "fulltext-block"
TERMS_PANEL = "terms-panel"
OBLIGATIONS_PANEL = "obligations-panel"
ALL_CLAUSES = "all"
FOUND_STATUSES = ("found", "confirmed")
STATUS_FILTERS = {
    ALL_CLAUSES: "All",
    "found": copy.STATUS["found"],
    "check": copy.STATUS["check"],
    "not_found": copy.STATUS["not_found"],
}
SEVERITY_TEXT = {"high": "Important", "medium": "Worth a look", "low": "Minor"}
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

LOADING = "Loading"
CHECK_HINT = (
    "Passages marked Check this are likely but not certain: "
    "read them and press Confirm or Not this."
)
CONFIRMED_LINE = "You confirmed {count} {passages}."
SHOW_TO_CHECK = "Show the {count} {passages} to check"
AI_NEEDED = "This part is written by the local AI, which is not running on this computer."
RETRYING = "The first draft did not match the contract; writing it again"
WRITING_FLAG = "Writing the note"
WRITING_OBLIGATIONS = "Reading who must do what. This can take half a minute."
WRITING_FULLTEXT = "Writing the overview of the whole text: {stage}"
MERGING_STAGE = "putting the sections together"
UNKNOWN_LANGUAGE = "The summary is written in English or Serbian."
UNKNOWN_FILTER = "Show all clauses, or those that are Found, Check this or Not found."
FLAG_CHANGED = "The things to check changed since the page was shown; open the overview again."
NO_PASSAGE_ATTACHED = "This point has no passage in the contract to show."
NO_PASSAGE_SHOWN = "This is about something the contract does not say."
NO_FLAGS = "Nothing on the list of things to check applies to this contract."
NO_SAVED_TERMS = "The key facts of this contract have not been read yet."
NO_REPORT = "There is no report of this contract yet; press Report first."
NO_ROWS = "No clause has this status."
NO_DEFINITIONS = "The contract does not seem to define any terms."
NO_DATES = "No dates could be read from the key facts."
SUMMARY_MISMATCH = "{count} {quotes} did not match the contract word for word."
COPIED_FROM = "Copied from: {names}"
NOT_CONFIRMED = "not confirmed in the contract"
DATES_INTRO = "Dates read from the key facts above, or worked out from them."
DATE_UNREADABLE = "could not be read"
DATE_STATED = "Stated as “{text}”"
DATE_WORKED_OUT = "Worked out: {formula}"
LESS_PRACTISED = "The assistant has less practice with this kind of clause."
OTHER_TOPIC_NOTE = (
    "The assistant has less practice with these, so it never marks them Found on its own."
)
BASED_ON = "Based on: {clauses}"
REPORT_BUILT = "Your report is ready."
REPORT_MISSING = (
    "Your report is ready, but it leaves out {missing}. "
    "Once {them} in the overview, press Report again."
)
REPORT_NO_AI = "Your report is ready. It leaves out {missing}, which need the local AI."
REPORT_PARTS = {
    "summary": "the summary",
    "facts": "the key facts",
    "duties": "who must do what",
    "fulltext": "the overview of the whole text",
}
OBLIGATIONS_DROPPED = {
    1: "1 entry could not be matched to the contract and was left out.",
    None: "{count} entries could not be matched to the contract and were left out.",
}
NO_OBLIGATIONS = "None of the clauses found says who must do what."
NO_OBLIGATION_CLAUSES = "No clause was found in this contract, so there is nobody to list."
FULLTEXT_TITLE = "Overview of the whole text"
FULLTEXT_INTRO = (
    "A longer overview that reads every section of the contract, not only the clauses; "
    "a long contract takes a few minutes."
)
FULLTEXT_BUTTON = "Write the overview of the whole text"


@dataclass(frozen=True)
class OverviewSection:
    """One section of the overview: its anchor id, title, line and the URL of its body."""

    id: str
    title: str
    line: str
    url: str


@dataclass(frozen=True)
class Source:
    """A clause a summary rests on, with the link that shows its passage."""

    name: str
    status: str
    hint: str
    url: str


@dataclass(frozen=True)
class SummaryView:
    """A summary as the section shows it: its text and sources, or why it is withheld."""

    text: str | None
    sources: tuple[Source, ...] = ()
    withheld: str | None = None
    mismatch: str | None = None


@dataclass(frozen=True)
class SectionView:
    """One section of the overview of the whole text with the link that shows its span."""

    number: int
    page: int | None
    text: str | None
    withheld: str | None
    url: str


@dataclass(frozen=True)
class FullTextView:
    """The overview of the whole text: its text, or why it is withheld, and its sections."""

    text: str | None
    withheld: str | None
    sections: tuple[SectionView, ...]


@dataclass(frozen=True)
class TermRow:
    """One key fact; kind is "value", "not_verified" or "not_stated"."""

    label: str
    value: str
    kind: str
    source: str = ""
    page: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class DateRow:
    """One key date: what, when, how it was read and the clause it comes from."""

    label: str
    when: str
    readable: bool
    how: str
    source: str


@dataclass(frozen=True)
class FlagView:
    """A thing to check as a card shows it."""

    rule: str
    severity: str
    label: str
    title: str
    note: str
    basis: str
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
    """One clause type with what the assistant found and what the reader decided."""

    name: str
    status: str
    page: int | None
    passage: str
    url: str | None
    citation: Citation | None
    rejected: bool


@dataclass(frozen=True)
class Topic:
    """The clause rows of one topic."""

    name: str
    rows: tuple[ClauseRow, ...]
    note: str | None = None


@dataclass(frozen=True)
class DefinitionView:
    """A defined term with the link that shows its sentence."""

    term: str
    quote: str
    page: int | None
    url: str


# Overview


@router.get("/documents/{doc_id}/overview", response_class=HTMLResponse)
def overview(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The overview: the section chips, what happened, then each section loading on its own."""
    cache = ws.load(doc_id)
    sections = [
        OverviewSection(id, title, line, f"/documents/{cache.doc_id}/{SECTION_ROUTES[id]}")
        for id, title, line in copy.SECTIONS
    ]
    return render(
        request,
        OVERVIEW,
        cache=cache,
        sections=sections,
        loading=LOADING,
        tau=tau,
        **_happened(ws, cache, tau),
    )


@router.get("/documents/{doc_id}/overview/happened", response_class=HTMLResponse)
def happened(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The card at the top of the overview alone, for after a decision."""
    cache = ws.load(doc_id)
    return render(request, HAPPENED, cache=cache, **_happened(ws, cache, tau))


def _happened(ws: Workspace, cache: DocumentCache, tau: Thresholds) -> dict:
    """What the assistant did, counted from the same rows as All clauses at these thresholds."""
    topics = _topics(cache, tau, ws.services.unvalidated, ws.decided(cache.doc_id))
    counts = _counts(topics)
    total = counts["found"] + counts["check"] + counts["not_found"]
    confirmed = sum(row.status == "confirmed" for topic in topics for row in topic.rows)
    check = counts["check"]
    return {
        "happened": copy.WHAT_HAPPENED.format(total=total, **counts),
        "check_hint": CHECK_HINT if check else None,
        "confirmed_line": (
            CONFIRMED_LINE.format(count=confirmed, passages=_plural(confirmed, "passage"))
            if confirmed
            else None
        ),
        "show_to_check": (
            SHOW_TO_CHECK.format(count=check, passages=_plural(check, "passage"))
            if check
            else None
        ),
    }


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"


# Summary and the overview of the whole text


@router.get("/documents/{doc_id}/summary", response_class=HTMLResponse)
def summary_tab(request: Request, doc_id: str, ws: Ws, tau: Tau, language: str = ENGLISH):
    """The summary in a language, or the form that writes it, then the whole-text overview."""
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
    """Stream the summary as the model writes it, then the section as it shows a saved one."""
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
        return {**context, "notice": AI_NEEDED, "languages": ()}
    if summary is not None:
        view = _summary_view(ws, cache, summary)
        return {**context, "view": view, **_fulltext_context(ws, cache, tau)}
    if not summary_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated):
        return {**context, "notice": NOTHING_TO_SUMMARIZE, **_fulltext_context(ws, cache, tau)}
    return {**context, "pending": WRITING_SUMMARY}


def _summary_view(ws: Workspace, cache: DocumentCache, summary: Summary) -> SummaryView:
    if summary.text is None:
        mismatch = None
        if summary.unfaithful_spans:
            count = len(summary.unfaithful_spans)
            mismatch = SUMMARY_MISMATCH.format(count=count, quotes=_plural(count, "quote"))
        return SummaryView(
            text=None,
            withheld=copy.AI_WITHHELD.format(reason=summary.reason),
            mismatch=mismatch,
        )
    decided = ws.decided(cache.doc_id)
    sources = []
    # A summary keeps only the items with a citation (summary.cited).
    for item in summary.items:
        name = display_name(item.clause_type)
        status = item.status
        if _is_confirmed(decided.get(item.clause_type), item.citation):
            status = "confirmed"
        hint = copy.STATUS.get(status, status)
        if item.clause_type in ws.services.unvalidated:
            hint = f"{hint}. {LESS_PRACTISED}"
        sources.append(
            Source(
                name=name,
                status=status,
                hint=hint,
                url=view_url(cache.doc_id, name, item.citation, status),
            )
        )
    return SummaryView(text=summary.text, sources=tuple(sources))


def _check_language(language: str) -> None:
    if language not in LANGUAGES:
        raise WorkspaceError(UNKNOWN_LANGUAGE)


@router.get("/documents/{doc_id}/fulltext", response_class=HTMLResponse)
def fulltext_block(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The whole-text overview alone: the saved overview, or the button that writes it."""
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
    context: dict = {
        "fulltext_panel": FULLTEXT_PANEL,
        "fulltext_title": FULLTEXT_TITLE,
        "fulltext_intro": FULLTEXT_INTRO,
        "fulltext_button": FULLTEXT_BUTTON,
    }
    if ws.gen_model is None:
        return {**context, "fulltext_notice": AI_NEEDED}
    if summary is None:
        summary = ws.fulltexts.load(cache.doc_id, ws.gen_model, tau.low, tau.high)
    if summary is None:
        return {**context, "fulltext": None, "fulltext_saved": False}
    view = _fulltext_view(cache, summary)
    return {**context, "fulltext": view, "fulltext_saved": summary.text is not None}


def _fulltext_view(cache: DocumentCache, summary: FullTextSummary) -> FullTextView:
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
            withheld = copy.AI_WITHHELD.format(reason=section.reason)
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
        withheld = copy.AI_WITHHELD.format(reason=summary.reason)
    return FullTextView(text=summary.text, withheld=withheld, sections=tuple(sections))


# Key facts


@router.get("/documents/{doc_id}/terms", response_class=HTMLResponse)
def terms_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The key facts and key dates, or the form that reads them."""
    cache = ws.load(doc_id)
    context = _terms_context(ws, cache, tau, saved_term_sheet(ws, cache, tau))
    return render(request, STACK, **context)


@router.post("/documents/{doc_id}/terms")
def extract_terms(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Stream a status while the model reads the key facts, then the section."""
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
    """The saved key facts as a JSON file."""
    cache = ws.load(doc_id)
    sheet = saved_term_sheet(ws, cache, tau)
    if sheet is None:
        raise WorkspaceError(NO_SAVED_TERMS, status_code=404)
    filename = f"{Path(cache.source.filename).stem}-key-terms.json"
    return FileResponse(ws.term_sheets.path_of(sheet), media_type=JSON_TYPE, filename=filename)


def _terms_context(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, sheet: TermSheet | None
) -> dict:
    context = {
        "cache": cache,
        "panel_id": TERMS_PANEL,
        "body": TERMS,
        "not_confirmed": NOT_CONFIRMED,
    }
    if ws.gen_model is None:
        return {**context, "notice": AI_NEEDED}
    if sheet is None:
        if not term_sheet_items(cache, tau.low, tau.high, ws.services.unvalidated):
            return {**context, "notice": NO_TERM_CLAUSES}
        return {**context, "pending": EXTRACTING_TERMS}
    rows = _term_rows(cache.doc_id, sheet)
    if sheet.reason is not None:
        notice = copy.AI_WITHHELD.format(reason=sheet.reason)
        return {**context, "notice": notice, "terms": rows}
    return {
        **context,
        "terms": rows,
        "download": True,
        "dates": [_date_row(item) for item in key_dates(sheet)],
        "dates_intro": DATES_INTRO,
        "no_dates": NO_DATES,
    }


def _term_rows(doc_id: str, sheet: TermSheet) -> list[TermRow]:
    rows = []
    for field in FIELDS:
        label = LABELS[field]
        values = sheet.terms.get(field, [])
        rejected = sheet.rejected.get(field, [])
        if not values:
            if rejected:
                rows.append(TermRow(label, ", ".join(rejected), "not_verified"))
            else:
                rows.append(TermRow(label, NOT_STATED, "not_stated"))
            continue
        first = values[0]
        names = ", ".join(dict.fromkeys(display_name(value.clause_type) for value in values))
        citation = Citation(
            quote=first.value, page=first.page, conf=1.0, start=first.start, end=first.end
        )
        rows.append(
            TermRow(
                label=label,
                value="; ".join(" ".join(value.value.split()) for value in values),
                kind="value",
                source=COPIED_FROM.format(names=names),
                page=first.page,
                # No status: the value is a slice of a passage, not a passage to decide on.
                url=view_url(doc_id, display_name(first.clause_type), citation),
            )
        )
    return rows


def _date_row(item: KeyDate) -> DateRow:
    how = DATE_STATED.format(text=" ".join(item.source.split()))
    if item.formula:
        how = DATE_WORKED_OUT.format(formula=item.formula)
    source = ""
    if item.clause_type:
        source = COPIED_FROM.format(names=display_name(item.clause_type))
    return DateRow(
        label=item.label,
        when=_long_date(item.date) if item.date else DATE_UNREADABLE,
        readable=item.date is not None,
        how=how,
        source=source,
    )


def _long_date(day: date) -> str:
    return f"{day.day} {day:%B %Y}"


# Things to check


@router.get("/documents/{doc_id}/review", response_class=HTMLResponse)
def review_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The things to check at the thresholds of the request, most important first."""
    cache = ws.load(doc_id)
    can_note = ws.gen_model is not None
    unvalidated = ws.services.unvalidated
    flags = [_flag_view(cache.doc_id, flag, can_note, unvalidated) for flag in ws.flags(cache, tau)]
    return render(
        request,
        REVIEW,
        cache=cache,
        flags=flags,
        no_flags=NO_FLAGS,
        no_passage=NO_PASSAGE_SHOWN,
    )


def _flag_view(doc_id: str, flag: Flag, can_note: bool, unvalidated: Collection[str]) -> FlagView:
    cited = next((item for item in flag.items if item.citation), None)
    url = None
    if cited is not None:
        url = view_url(doc_id, display_name(cited.clause_type), cited.citation, cited.status)
    clauses = ", ".join(
        f"{display_name(item.clause_type)} ({copy.STATUS[item.status].lower()})"
        for item in flag.items
    )
    basis = BASED_ON.format(clauses=clauses)
    if flag.unvalidated or any(item.clause_type in unvalidated for item in flag.items):
        basis = f"{basis}. {LESS_PRACTISED}"
    return FlagView(
        rule=flag.rule,
        severity=flag.severity,
        label=SEVERITY_TEXT[flag.severity],
        title=flag.title,
        note=flag.note,
        basis=basis,
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
    """Stream a short note on a point, written from its quotes and checked against them."""
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
        context = {"withheld": copy.AI_WITHHELD.format(reason=outcome.reason)}
    else:
        context = {"text": outcome.text}
    yield frame(render_string(request, NOTE, **context), done=True)


# Who must do what


@router.get("/documents/{doc_id}/obligations", response_class=HTMLResponse)
def obligations_tab(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Who must do what, by party, or the form that reads it."""
    cache = ws.load(doc_id)
    context = _obligations_context(ws, cache, tau, saved_obligations(ws, cache, tau))
    return render(request, STACK, **context)


@router.post("/documents/{doc_id}/obligations")
def read_obligations(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """Stream a status while the model reads the obligations, then the section."""
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
    context = {"cache": cache, "panel_id": OBLIGATIONS_PANEL, "body": OBLIGATIONS}
    if ws.gen_model is None:
        return {**context, "notice": AI_NEEDED}
    if sheet is None:
        items = obligation_items(cache, ws.names, tau.low, tau.high, ws.services.unvalidated)
        if not items:
            return {**context, "notice": NO_OBLIGATION_CLAUSES}
        return {**context, "pending": WRITING_OBLIGATIONS}
    if sheet.reason is not None:
        return {**context, "notice": copy.AI_WITHHELD.format(reason=sheet.reason)}
    dropped = None
    if sheet.rejected:
        count = len(sheet.rejected)
        dropped = OBLIGATIONS_DROPPED.get(count, OBLIGATIONS_DROPPED[None]).format(count=count)
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
            # The words are a slice of the passage: a passage to check is decided on from
            # All clauses, not from a slice of it, so its status is left out here.
            status = statuses.get(obligation.clause_type)
            entries.append(
                ObligationView(
                    modality=obligation.modality,
                    css=obligation.modality.replace(" ", "_"),
                    action=" ".join(obligation.action.split()),
                    clause=name,
                    page=obligation.page,
                    url=view_url(doc_id, name, citation, None if status == "check" else status),
                )
            )
        parties.append(Party(party, tuple(entries)))
    return parties


# All clauses and defined terms


@router.get("/documents/{doc_id}/checklist", response_class=HTMLResponse)
def checklist(request: Request, doc_id: str, ws: Ws, tau: Tau, status: str = ALL_CLAUSES):
    """Every clause type by topic with what was found, optionally only those of one status."""
    if status not in STATUS_FILTERS:
        raise WorkspaceError(UNKNOWN_FILTER)
    cache = ws.load(doc_id)
    topics = _topics(cache, tau, ws.services.unvalidated, ws.decided(cache.doc_id))
    counts = _counts(topics)
    counts[ALL_CLAUSES] = sum(counts.values())
    filters = [(value, label, counts[value]) for value, label in STATUS_FILTERS.items()]
    return render(
        request,
        CHECKLIST,
        cache=cache,
        filters=filters,
        status=status,
        topics=_filtered(topics, None if status == ALL_CLAUSES else status),
        no_rows=NO_ROWS,
    )


def _topics(
    cache: DocumentCache,
    tau: Thresholds,
    unvalidated: Collection[str],
    decided: dict[str, Decision],
) -> list[Topic]:
    groups = dict(CLAUSE_GROUPS)
    extra = tuple(name for name in cache.clauses if name in unvalidated)
    if extra:
        groups[OTHER_TOPIC] = extra
    return [
        Topic(
            name=topic,
            rows=tuple(
                _clause_row(cache, tau, name, name in unvalidated, decided.get(name))
                for name in names
            ),
            note=OTHER_TOPIC_NOTE if topic == OTHER_TOPIC else None,
        )
        for topic, names in groups.items()
    ]


def _filtered(topics: list[Topic], wanted: str | None) -> list[Topic]:
    if wanted is None:
        return [topic for topic in topics if topic.rows]
    shown = []
    for topic in topics:
        rows = tuple(row for row in topic.rows if _filter_of(row.status) == wanted)
        if rows:
            shown.append(Topic(topic.name, rows, topic.note))
    return shown


def _filter_of(status: str) -> str:
    """The filter a status falls under: a passage the reader confirmed counts as found."""
    return "found" if status in FOUND_STATUSES else status


def _counts(topics: list[Topic]) -> dict[str, int]:
    counts = {"found": 0, "check": 0, "not_found": 0}
    for topic in topics:
        for row in topic.rows:
            counts[_filter_of(row.status)] += 1
    return counts


def _clause_row(
    cache: DocumentCache,
    tau: Thresholds,
    name: str,
    capped: bool,
    decision: Decision | None,
) -> ClauseRow:
    label = display_name(name)
    rejected = decision is not None and decision.verdict == "rejected"
    citation = top_citation(cache, name)
    if citation is None:
        return ClauseRow(label, "not_found", None, "", None, None, rejected)
    if _is_confirmed(decision, citation):
        # Confirmed by the reader: found, even for a clause type capped at Check this.
        status = "confirmed"
    else:
        status = tier(citation.conf, tau.low, tau.high)
        if capped and status == "found":
            status = "check"
    if status == "not_found":
        # The top candidate is below the Check this threshold: no passage to show.
        return ClauseRow(label, status, None, "", None, None, rejected)
    return ClauseRow(
        name=label,
        status=status,
        page=citation.page,
        passage=shorten(citation.quote),
        url=view_url(cache.doc_id, label, citation, status),
        citation=citation,
        rejected=rejected,
    )


def _is_confirmed(decision: Decision | None, citation: Citation | None) -> bool:
    """Whether the passage shown is the one the reader confirmed."""
    return (
        decision is not None
        and decision.verdict == "confirmed"
        and citation is not None
        and (citation.start, citation.end) == (decision.start, decision.end)
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
        missing.append(REPORT_PARTS["summary"])
    if sheet is None:
        missing.append(REPORT_PARTS["facts"])
    if obligations is None:
        missing.append(REPORT_PARTS["duties"])
    if fulltext is None or fulltext.text is None:
        missing.append(REPORT_PARTS["fulltext"])
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
    if missing and model is None:
        status = REPORT_NO_AI.format(missing=_listed(missing))
    elif missing:
        them = "it appears" if len(missing) == 1 else "they appear"
        status = REPORT_MISSING.format(missing=_listed(missing), them=them)
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
    """Return the frame that shows what is being written and the text so far, without timings."""
    status = RETRYING if step.previous_reason else step.what
    return frame(render_string(request, LIVE, status=status, text=step.text))


def _final(request: Request, context: dict) -> str:
    """Return the last frame: the section's panel as the GET renders it once saved."""
    html = render_string(request, context["body"], **context)
    return frame(oob={context["panel_id"]: html}, done=True)


def _cleared(frames: Iterator[str]) -> Iterator[str]:
    """Clear the status line when a stream fails; core.stream shows the notice."""
    try:
        yield from frames
    except Exception:
        yield frame("")
        raise
