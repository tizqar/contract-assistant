"""Obligations by party: who must, may or must not do what, read by the model from the clauses."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contract_assistant.answer import AnswerItem
from contract_assistant.generation import (
    GeneratedRecord,
    GeneratedStore,
    Generation,
    Request,
    run,
)
from contract_assistant.router.clients import LLMClient
from contract_assistant.store import DocumentCache
from contract_assistant.summary import summary_items
from contract_assistant.term_sheet import FENCE

logger = logging.getLogger(__name__)

PROMPT_VERSION = 1
MAX_TOKENS = 900
APP_ATTEMPTS = 2
# Found items first, then verify items, up to this many: each costs the model about 30 tokens.
MAX_ITEMS = 16
OBLIGATIONS_DIR = "obligations"
KIND = "obligations"
SOURCE = "clauses"
Modality = Literal["must", "may", "must not"]
MODALITIES: tuple[Modality, ...] = ("must", "may", "must not")
# The words a model uses for the three modalities; checked longest first.
MODALITY_WORDS: dict[str, Modality] = {
    "must not": "must not", "shall not": "must not", "may not": "must not", "will not": "must not",
    "cannot": "must not", "can not": "must not", "is not permitted to": "must not",
    "is prohibited from": "must not", "agrees not to": "must not",
    "must": "must", "shall": "must", "will": "must", "agrees to": "must", "agree to": "must",
    "undertakes to": "must", "is obliged to": "must", "is required to": "must", "has to": "must",
    "is responsible for": "must", "grants": "must", "covenants": "must",
    "may": "may", "can": "may", "is entitled to": "may", "has the right to": "may",
    "is permitted to": "may", "is free to": "may", "retains the right to": "may",
}  # fmt: skip
INVALID_REASON = "the reply is not the JSON list the obligations need"
REJECTED_REASON = "these entries are not in the clauses word for word: {entries}"
TEMPLATE = json.dumps(
    [{"clause_type": "the item's clause type", "party": "party named in the clause",
      "modality": "must | may | must not", "action": "the words of the clause that say what"}]
)  # fmt: skip
SYSTEM_PROMPT = (
    "Read the clause items of a contract and list the obligations they state. Reply with one "
    f"JSON array of objects of this shape and nothing else:\n{TEMPLATE}\n"
    "For each item give at most two entries. The party is a name or role copied word for "
    "word from the clause (such as 'the Customer' or 'M&I'). The modality is must, may or "
    "must not. The action is a short fragment copied word for word from the clause that says "
    "what the party must, may or must not do. Skip an item that states no obligation. "
    "Do not write anything that is not in the quotes."
)


class ObligationReply(BaseModel):
    """One entry of the model's reply."""

    model_config = ConfigDict(extra="ignore")

    clause_type: str
    party: str
    modality: str
    action: str


class Obligation(BaseModel):
    """A verified obligation: the party and the action are verbatim slices of the clause."""

    party: str
    modality: Modality
    action: str
    clause_type: str
    page: int | None
    start: int
    end: int


class ObligationSheet(GeneratedRecord):
    """The verified obligations of one document and the entries the model gave that were rejected."""

    prompt_version: int = PROMPT_VERSION
    items: list[AnswerItem]
    obligations: list[Obligation]
    rejected: list[dict[str, str]] = Field(default_factory=list)
    reason: str | None = None


def obligation_items(
    cache: DocumentCache,
    clause_types: Sequence[str],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> list[AnswerItem]:
    """Return up to MAX_ITEMS found-first items of the cache with a verified quote."""
    items = summary_items(cache, clause_types, tau_low, tau_high, unvalidated)
    found = [item for item in items if item.status == "found"]
    check = [item for item in items if item.status != "found"]
    return (found + check)[:MAX_ITEMS]


def obligations_request(doc_id: str, items: Sequence[AnswerItem]) -> Request:
    """Return the request that lists the obligations of the items, with its check."""
    items = [item for item in items if item.citation is not None]
    if not items:
        raise ValueError("no items for the obligations")

    def check(text: str) -> str | None:
        replies = parse_reply(text)
        if replies is None:
            return INVALID_REASON
        kept, rejected = verify(replies, items)
        # Entries that fail are dropped; the model is asked again only when nothing survived.
        if rejected and not kept:
            names = ", ".join(f"'{entry['party']}: {entry['action']}'" for entry in rejected)
            return REJECTED_REASON.format(entries=names)
        return None

    return Request(
        SYSTEM_PROMPT,
        _user_message(items),
        MAX_TOKENS,
        check,
        SOURCE,
        f"obligations of {doc_id}",
        trim=False,
    )


def obligations_from(
    result: Generation,
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
) -> ObligationSheet:
    """Return the ObligationSheet that a generation outcome gives for the items."""
    items = [item for item in items if item.citation is not None]
    replies = parse_reply(result.reply)
    obligations, rejected = verify(replies, items) if replies is not None else ([], [])
    return ObligationSheet(
        doc_id=doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        items=items,
        obligations=obligations,
        rejected=rejected,
        reason=None if replies is not None else INVALID_REASON,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
    )


def request_obligations(
    doc_id: str,
    items: Sequence[AnswerItem],
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    attempts: int = 1,
) -> ObligationSheet:
    """Ask the model for the obligations; entries that are not in the clauses are rejected."""
    result = run(client, obligations_request(doc_id, items), attempts)
    return obligations_from(result, doc_id, items, client, tau_low, tau_high)


def parse_reply(text: str) -> list[ObligationReply] | None:
    """Read the model's JSON array, with or without code fences; None when it is not one."""
    stripped = FENCE.sub("", text.strip())
    start, end = stripped.find("["), stripped.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
        return [ObligationReply.model_validate(entry) for entry in data]
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def verify(
    replies: Sequence[ObligationReply], items: Sequence[AnswerItem]
) -> tuple[list[Obligation], list[dict[str, str]]]:
    """Keep the entries whose party and action are in their clause's quote; list the rest."""
    by_type = {item.clause_type: item for item in items if item.citation is not None}
    kept: list[Obligation] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for reply in replies:
        item = by_type.get(reply.clause_type)
        modality = _modality(reply.modality)
        located = _locate(reply.action, item) if item else None
        party = _locate(reply.party, item) if item else None
        if item is None or modality is None or located is None or party is None:
            rejected.append(reply.model_dump())
            continue
        key = (
            party[0].lower(),
            modality,
            item.citation.start + located[1],
            located[2] - located[1],
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(
            Obligation(
                party=party[0],
                modality=modality,
                action=located[0],
                clause_type=item.clause_type,
                page=item.citation.page,
                start=item.citation.start + located[1],
                end=item.citation.start + located[2],
            )
        )
    return kept, rejected


def _modality(text: str) -> Modality | None:
    words = " ".join(text.lower().split())
    for phrase in sorted(MODALITY_WORDS, key=len, reverse=True):
        if words == phrase or words.startswith(phrase + " ") or words.endswith(" " + phrase):
            return MODALITY_WORDS[phrase]
    return None


def _locate(value: str, item: AnswerItem | None) -> tuple[str, int, int] | None:
    if item is None or item.citation is None or not value.strip():
        return None
    pattern = re.compile(r"\s+".join(re.escape(word) for word in value.split()), re.IGNORECASE)
    match = pattern.search(item.citation.quote)
    return None if match is None else (match.group(), match.start(), match.end())


def _user_message(items: Sequence[AnswerItem]) -> str:
    lines = [
        f'- {item.clause_type}: "{" ".join(item.citation.quote.split())}"'
        for item in items
        if item.citation is not None
    ]
    return "Items:\n" + "\n".join(lines)


def by_party(sheet: ObligationSheet) -> dict[str, list[Obligation]]:
    """Group the obligations by party, in order of first appearance."""
    groups: dict[str, list[Obligation]] = {}
    for obligation in sheet.obligations:
        groups.setdefault(obligation.party, []).append(obligation)
    return groups


def render_markdown(sheet: ObligationSheet) -> str:
    """Render the obligations by party for the command line."""
    if sheet.reason:
        return f"Obligations withheld: {sheet.reason}."
    if not sheet.obligations:
        return "No obligation was read from the clauses found."
    lines = []
    for party, obligations in by_party(sheet).items():
        lines.append(f"**{party}**")
        lines.extend(
            f'- {o.modality}: "{" ".join(o.action.split())}" ({o.clause_type})'
            for o in obligations
        )
        lines.append("")
    if sheet.rejected:
        lines.append(
            f"{len(sheet.rejected)} entries of the model were not in the clauses and were dropped."
        )
    return "\n".join(lines).rstrip()


class ObligationStore(GeneratedStore[ObligationSheet]):
    """Obligation sheets saved per document, model, thresholds and prompt version."""

    def __init__(self, directory: Path):
        super().__init__(directory, KIND, PROMPT_VERSION, ObligationSheet)
