"""Key terms of a contract as verbatim values the model picks from the clause quotes."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from contract_assistant.answer import AnswerItem
from contract_assistant.generation import (
    GeneratedRecord,
    GeneratedStore,
    RecordBackend,
    generate,
)
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DocumentCache
from contract_assistant.summary import summary_items

logger = logging.getLogger(__name__)

PROMPT_VERSION = 1
MAX_TOKENS = 400
# The app and the CLI ask again once when a value is rejected, naming it; the evaluation too.
APP_ATTEMPTS = 2
TERM_SHEET_DIR = "term_sheets"
KIND = "term-sheet"
SOURCE = "clauses"
# Each field and the clause types its value is read from, in order of preference.
FIELDS: dict[str, tuple[str, ...]] = {
    "parties": ("Parties",),
    "agreement_date": ("Agreement Date",),
    "effective_date": ("Effective Date",),
    "expiration_date": ("Expiration Date",),
    "term": ("Expiration Date", "Renewal Term"),
    "renewal_term": ("Renewal Term",),
    "notice_period": ("Notice Period To Terminate Renewal", "Termination For Convenience"),
    "governing_law": ("Governing Law",),
}
LABELS = {
    "parties": "Parties",
    "agreement_date": "Agreement date",
    "effective_date": "Effective date",
    "expiration_date": "Expiration date",
    "term": "Initial term",
    "renewal_term": "Renewal",
    "notice_period": "Notice period",
    "governing_law": "Governing law",
}
INVALID_REASON = "the reply is not the JSON object the term sheet needs"
REJECTED_REASON = "these values are not in the clauses word for word: {values}"
NOT_STATED = "not stated"
NOT_VERIFIED = "not verified"
FENCE = re.compile(r"^```[a-z]*\s*|\s*```$")


class TermSheetReply(BaseModel):
    """The JSON the model writes: short verbatim fragments, or null when a term is not stated."""

    model_config = ConfigDict(extra="ignore")

    parties: list[str] = Field(default_factory=list)
    agreement_date: str | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    term: str | None = None
    renewal_term: str | None = None
    notice_period: str | None = None
    governing_law: str | None = None

    @field_validator("parties", mode="before")
    @classmethod
    def _one_party_as_list(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value


TEMPLATE = json.dumps(
    {
        "parties": ["party name", "party name"],
        "agreement_date": "date or null",
        "effective_date": "date or null",
        "expiration_date": "date or null",
        "term": "initial duration or null",
        "renewal_term": "renewal duration or null",
        "notice_period": "notice period or null",
        "governing_law": "governing law or null",
    }
)
SYSTEM_PROMPT = (
    "Fill in the term sheet of a contract from the clause items given. Reply with one JSON "
    f"object of this shape and nothing else:\n{TEMPLATE}\n"
    "Every value is a short fragment copied word for word from the quote of the matching "
    "item: a party name, a date, a duration such as 'ten (10) years', a notice period or "
    "the governing law. Use null, or an empty list for the parties, when the items do not "
    "state a value. Do not write anything that is not in the quotes."
)


class TermValue(BaseModel):
    """One verified value: a verbatim slice of a clause quote with its offsets in the document."""

    value: str
    clause_type: str
    page: int | None
    start: int
    end: int


class TermSheet(GeneratedRecord):
    """The verified key terms of one document and the values the model gave that were rejected."""

    prompt_version: int = PROMPT_VERSION
    items: list[AnswerItem]
    terms: dict[str, list[TermValue]]
    rejected: dict[str, list[str]]
    reason: str | None = None


def term_sheet_items(
    cache: DocumentCache, tau_low: float, tau_high: float, unvalidated: Collection[str] = ()
) -> list[AnswerItem]:
    """Return the found and verify items of the clause types the term sheet reads."""
    names = list(dict.fromkeys(name for sources in FIELDS.values() for name in sources))
    return summary_items(cache, names, tau_low, tau_high, unvalidated)


def request_term_sheet(
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    attempts: int = 1,
) -> TermSheet:
    """Ask the model for the term sheet; values that are not in the quotes are rejected."""
    items = [item for item in items if item.citation is not None]
    if not items:
        raise ValueError("no items for a term sheet")

    def check(text: str) -> str | None:
        reply = parse_reply(text)
        if reply is None:
            return INVALID_REASON
        rejected = [value for values in verify(reply, items)[1].values() for value in values]
        if rejected:
            return REJECTED_REASON.format(values=", ".join(f"'{value}'" for value in rejected))
        return None

    result = generate(
        client,
        SYSTEM_PROMPT,
        _user_message(items),
        MAX_TOKENS,
        check,
        source=SOURCE,
        attempts=attempts,
        label=f"term sheet of {doc_id}",
        trim=False,
    )
    reply = parse_reply(result.reply)
    terms, rejected = verify(reply, items) if reply else ({field: [] for field in FIELDS}, {})
    return TermSheet(
        doc_id=doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        items=items,
        terms=terms,
        rejected=rejected,
        reason=None if reply else INVALID_REASON,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
    )


def parse_reply(text: str) -> TermSheetReply | None:
    """Read the model's JSON object, with or without code fences; None when it is not one."""
    stripped = FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return TermSheetReply.model_validate_json(stripped[start : end + 1])
    except ValidationError:
        return None


def verify(
    reply: TermSheetReply, items: Sequence[AnswerItem]
) -> tuple[dict[str, list[TermValue]], dict[str, list[str]]]:
    """Locate each value in the quotes; return the verified values and the rejected ones."""
    terms: dict[str, list[TermValue]] = {}
    rejected: dict[str, list[str]] = {}
    for field, sources in FIELDS.items():
        given = getattr(reply, field)
        values = given if isinstance(given, list) else [given]
        found: list[TermValue] = []
        for value in values:
            if value is None or not value.strip():
                continue
            located = _locate(value, _ordered(items, sources))
            if located is None:
                rejected.setdefault(field, []).append(value)
            elif all((v.start, v.end) != (located.start, located.end) for v in found):
                found.append(located)
        terms[field] = found
    return terms, rejected


def _ordered(items: Sequence[AnswerItem], sources: tuple[str, ...]) -> list[AnswerItem]:
    by_type = {item.clause_type: item for item in items}
    first = [by_type[name] for name in sources if name in by_type]
    return first + [item for item in items if item.clause_type not in sources]


def _locate(value: str, items: Sequence[AnswerItem]) -> TermValue | None:
    pattern = re.compile(r"\s+".join(re.escape(word) for word in value.split()), re.IGNORECASE)
    for item in items:
        citation = item.citation
        if citation is None:
            continue
        match = pattern.search(citation.quote)
        if match:
            return TermValue(
                value=match.group(),
                clause_type=item.clause_type,
                page=citation.page,
                start=citation.start + match.start(),
                end=citation.start + match.end(),
            )
    return None


def _user_message(items: Sequence[AnswerItem]) -> str:
    lines = []
    for item in items:
        quote = " ".join(item.citation.quote.split()) if item.citation else ""
        lines.append(f'- {item.clause_type}: "{quote}"')
    return "Items:\n" + "\n".join(lines)


def cell(sheet: TermSheet, field: str) -> str:
    """Return the values of a field as one line, naming a clause of another field, or why empty."""
    values = sheet.terms.get(field, [])
    if values:
        return "; ".join(_value_text(value, FIELDS[field]) for value in values)
    rejected = sheet.rejected.get(field, [])
    if rejected:
        return f"{NOT_VERIFIED} ({', '.join(repr(value) for value in rejected)})"
    return NOT_STATED


def _value_text(value: TermValue, sources: tuple[str, ...]) -> str:
    text = " ".join(value.value.split())
    if value.clause_type in sources:
        return text
    return f"{text} (from {value.clause_type})"


def render_markdown(sheets: Sequence[tuple[str, TermSheet]]) -> str:
    """Render one or more term sheets as a Markdown table, one column per document."""
    header = "| Term | " + " | ".join(name for name, _ in sheets) + " |"
    rule = "| --- |" + " --- |" * len(sheets)
    rows = [
        f"| {LABELS[field]} | " + " | ".join(cell(sheet, field) for _, sheet in sheets) + " |"
        for field in FIELDS
    ]
    notes = [
        f"{name}: term sheet withheld: {sheet.reason}"
        for name, sheet in sheets
        if sheet.reason
    ]
    return "\n".join([header, rule, *rows, *([""] + notes if notes else [])])


class TermSheetStore(GeneratedStore[TermSheet]):
    """Term sheets saved per document, model, thresholds and prompt version."""

    def __init__(self, location: Path | RecordBackend):
        super().__init__(location, KIND, PROMPT_VERSION, TermSheet)
