"""Review flags: a rule file checked against the clauses found, with an optional grounded note."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from contract_assistant.answer import STATUS_WORDS, AnswerItem, Status, compose, verify_quotes
from contract_assistant.config import ROOT
from contract_assistant.generation import Request, faithful_check
from contract_assistant.store import DocumentCache

RULES_PATH = ROOT / "data" / "review_rules.json"
Severity = Literal["high", "medium", "low"]
Kind = Literal["missing", "present", "all_present", "present_without"]
SEVERITY_ORDER: dict[Severity, int] = {"high": 0, "medium": 1, "low": 2}
MAX_TOKENS = 120
SOURCE = "clauses"
SYSTEM_PROMPT = (
    "A lawyer reviews a contract. In at most two sentences, say what the clause quoted below "
    "means for the review point named, in plain English. Quote the clause only word for word, "
    "in double quotes, and use double quotes for nothing else. Copy every number exactly as "
    "the clause states it and do not add facts the clause does not state. "
    "Reply with the sentences only."
)


class Rule(BaseModel):
    """One review rule: what to look for among the clause types and what to say about it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Kind
    clause_types: list[str]
    unless: list[str] = []
    severity: Severity
    title: str
    note: str


class Flag(BaseModel):
    """A rule that applies to a document, with the clause items that triggered it."""

    rule: str
    kind: Kind
    severity: Severity
    title: str
    note: str
    status: Status
    items: list[AnswerItem]
    unvalidated: bool


class RuleFileError(ValueError):
    """Raised when the rule file names clause types the app does not know."""


def load_rules(path: Path = RULES_PATH, known_types: Collection[str] | None = None) -> list[Rule]:
    """Read the rules in file order; check their clause types when the known ones are given."""
    rules = [Rule(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))["rules"]]
    if known_types is not None:
        unknown = sorted(
            {name for rule in rules for name in [*rule.clause_types, *rule.unless]}
            - set(known_types)
        )
        if unknown:
            raise RuleFileError(f"unknown clause types in {path.name}: {', '.join(unknown)}")
    return rules


def review(
    cache: DocumentCache,
    rules: Sequence[Rule],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> list[Flag]:
    """Return the flags the rules raise for the document, most severe first."""
    names = sorted({name for rule in rules for name in [*rule.clause_types, *rule.unless]})
    present = [name for name in names if name in cache.clauses]
    items, _ = verify_quotes(compose(cache, present, tau_low, tau_high, unvalidated), cache.text)
    by_name = {item.clause_type: item for item in items}
    flags = [flag for rule in rules if (flag := _apply(rule, by_name, unvalidated))]
    return sorted(flags, key=lambda flag: SEVERITY_ORDER[flag.severity])


def _apply(rule: Rule, by_name: dict[str, AnswerItem], unvalidated: Collection[str]) -> Flag | None:
    items = [by_name[name] for name in rule.clause_types if name in by_name]
    shown = [item for item in items if item.citation is not None]
    unless = [by_name[name] for name in rule.unless if name in by_name]
    unless_shown = [item for item in unless if item.citation is not None]
    if rule.kind == "missing":
        if shown:
            return None
        return _flag(rule, "not_found", items, unvalidated)
    if rule.kind == "present" and shown:
        return _flag(rule, _best(shown), shown, unvalidated)
    if rule.kind == "all_present" and shown and len(shown) == len(rule.clause_types):
        return _flag(rule, _best(shown), shown, unvalidated)
    if rule.kind == "present_without" and shown and not unless_shown:
        return _flag(rule, _best(shown), shown, unvalidated)
    return None


def _best(items: Sequence[AnswerItem]) -> Status:
    return "found" if any(item.status == "found" for item in items) else "check"


def _flag(
    rule: Rule, status: Status, items: list[AnswerItem], unvalidated: Collection[str]
) -> Flag:
    return Flag(
        rule=rule.id,
        kind=rule.kind,
        severity=rule.severity,
        title=rule.title,
        note=rule.note,
        status=status,
        items=items,
        unvalidated=any(name in unvalidated for name in rule.clause_types),
    )


def flag_request(flag: Flag) -> Request:
    """Return the request that explains a flag from its quotes; a flag without quotes has none."""
    quotes = [item.citation.quote for item in flag.items if item.citation is not None]
    if not quotes:
        raise ValueError("a flag without quotes needs no explanation")
    lines = [
        f'- {item.clause_type} ({STATUS_WORDS[item.status]}): "{" ".join(item.citation.quote.split())}"'
        for item in flag.items
        if item.citation is not None
    ]
    user = f"Review point: {flag.title}\n\nClauses:\n" + "\n".join(lines)
    return Request(
        SYSTEM_PROMPT, user, MAX_TOKENS, faithful_check(quotes, SOURCE), SOURCE, f"flag {flag.rule}"
    )


def render_markdown(flags: Sequence[Flag]) -> str:
    """Render the flags for the command line, most severe first."""
    if not flags:
        return "No review flags."
    lines = []
    for flag in flags:
        mark = " (not validated on test data)" if flag.unvalidated else ""
        lines.append(f"- [{flag.severity}] {flag.title}{mark}: {flag.note}")
        for item in flag.items:
            if item.citation is not None:
                quote = " ".join(item.citation.quote.split())
                lines.append(f'    {item.clause_type} ({STATUS_WORDS[item.status]}): "{quote}"')
    return "\n".join(lines)
