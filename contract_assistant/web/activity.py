"""The activity log as the pages show it: each event as a sentence and a time in the reader's
language, for the admin's log and a contract's history alike."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime

from contract_assistant import audit
from contract_assistant.audit import LoggedEvent
from contract_assistant.db.models import EDIT, VIEW
from contract_assistant.ui.views import display_name
from contract_assistant.web.core import ROLE_NAMES
from contract_assistant.web.i18n import translate
from contract_assistant.web.library import DATE, MONTHS

# How a share level and a review step read inside a sentence of the log.
SHARE_LEVELS = {VIEW: "can view", EDIT: "can edit"}
WHEN = "{date}, {time}"
TRANSLATED = (*SHARE_LEVELS.values(), WHEN)

# The groups of actions an activity filter offers, by the prefix of the action.
GROUPS = {
    "user": "Accounts and sign-ins",
    "folder": "Folders and sharing",
    "contract": "Contracts",
    "clause": "Passage decisions",
    "review": "Reviews and approval",
    "chat": "Chat",
}
TRANSLATED += tuple(GROUPS.values())


@dataclass(frozen=True)
class ActivityRow:
    """One event as a row of a list: when, the sentence, and the contract it links to."""

    id: int
    at: datetime
    when: str
    sentence: str
    action: str
    group: str
    actor_name: str | None
    contract_id: str | None
    ip: str | None


def sentence(event: LoggedEvent, lang: str) -> str:
    """Return what happened as one sentence, the names and values in the reader's language."""
    details = dict(event.details)
    if "role" in details:
        details["role"] = translate(ROLE_NAMES.get(details["role"], details["role"]), lang)
    if "level" in details:
        details["level"] = translate(SHARE_LEVELS.get(details["level"], details["level"]), lang)
    if "clause" in details:
        details["clause"] = translate(display_name(details["clause"]), lang)
    for key in ("what", "target"):
        if isinstance(details.get(key), str):
            details[key] = translate(details[key], lang)
    shown = dataclasses.replace(event, details=details)
    return audit.describe(shown, lambda text: translate(text, lang))


def when(at: datetime, lang: str) -> str:
    """Return a time as "28 Oct 2026, 14:09" in this computer's time zone, or as the language
    writes it."""
    local = at.astimezone()
    month = translate(MONTHS[local.month - 1], lang)
    date = translate(DATE, lang).format(day=local.day, month=month, year=local.year)
    return translate(WHEN, lang).format(date=date, time=local.strftime("%H:%M"))


def rows(events: list[LoggedEvent], lang: str) -> list[ActivityRow]:
    """Return events as rows of a list, in the order given."""
    return [
        ActivityRow(
            id=event.id,
            at=event.at,
            when=when(event.at, lang),
            sentence=sentence(event, lang),
            action=event.action,
            group=event.action.split(".", 1)[0],
            actor_name=event.actor_name,
            contract_id=event.contract_id,
            ip=event.ip,
        )
        for event in events
    ]


def actions_of(group: str) -> list[str]:
    """Return the actions of a group of the filter; none for an unknown group."""
    if group not in GROUPS:
        return []
    return [action for action in audit.DESCRIPTIONS if action.startswith(group + ".")]
