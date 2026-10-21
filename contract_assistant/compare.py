"""Two contracts side by side: the clause checklist and the key terms, with differences marked."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from pydantic import BaseModel

from contract_assistant.answer import STATUS_WORDS, Status, compose, verify_quotes
from contract_assistant.store import DocumentCache
from contract_assistant.term_sheet import FIELDS, LABELS, TermSheet, cell

DIFFERS = "≠"
SAME = ""


class ChecklistRow(BaseModel):
    """The status of one clause type in each of the two contracts."""

    clause_type: str
    left: Status
    right: Status
    differs: bool


class TermRow(BaseModel):
    """The value of one key term in each of the two contracts."""

    field: str
    label: str
    left: str
    right: str
    differs: bool


def statuses(
    cache: DocumentCache,
    clause_types: Sequence[str],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> dict[str, Status]:
    """Return the status of every clause type for the document; a type not extracted is not found."""
    present = [name for name in clause_types if name in cache.clauses]
    items, _ = verify_quotes(compose(cache, present, tau_low, tau_high, unvalidated), cache.text)
    by_name = {item.clause_type: item for item in items}
    result: dict[str, Status] = {}
    for name in clause_types:
        item = by_name.get(name)
        result[name] = "not_found" if item is None or item.citation is None else item.status
    return result


def compare_checklists(
    left: DocumentCache,
    right: DocumentCache,
    clause_types: Sequence[str],
    tau_low: float,
    tau_high: float,
    unvalidated: Collection[str] = (),
) -> list[ChecklistRow]:
    """Return one row per clause type with its status in each contract."""
    left_status = statuses(left, clause_types, tau_low, tau_high, unvalidated)
    right_status = statuses(right, clause_types, tau_low, tau_high, unvalidated)
    return [
        ChecklistRow(
            clause_type=name,
            left=left_status[name],
            right=right_status[name],
            differs=left_status[name] != right_status[name],
        )
        for name in clause_types
    ]


def compare_terms(left: TermSheet, right: TermSheet) -> list[TermRow]:
    """Return one row per key term with its value in each contract."""
    rows = []
    for field in FIELDS:
        a, b = cell(left, field), cell(right, field)
        rows.append(
            TermRow(
                field=field,
                label=LABELS[field],
                left=a,
                right=b,
                differs=_normalise(a) != _normalise(b),
            )
        )
    return rows


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def render_markdown(
    names: tuple[str, str], checklist: Sequence[ChecklistRow], terms: Sequence[TermRow]
) -> str:
    """Render the comparisons as Markdown tables; a differing row is marked in the last column.

    The key terms table is left out when there are no term rows.
    """
    left, right = names
    lines = []
    if terms:
        lines += [
            f"| Term | {left} | {right} | |",
            "| --- | --- | --- | --- |",
            *(
                f"| {row.label} | {row.left} | {row.right} | {DIFFERS if row.differs else SAME} |"
                for row in terms
            ),
            "",
        ]
    lines += [
        f"| Clause | {left} | {right} | |",
        "| --- | --- | --- | --- |",
        *(
            f"| {row.clause_type} | {STATUS_WORDS[row.left]} | {STATUS_WORDS[row.right]} | "
            f"{DIFFERS if row.differs else SAME} |"
            for row in checklist
        ),
    ]
    return "\n".join(lines)
