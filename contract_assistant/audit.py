"""The activity log: who did what and when, written in the same transaction as the change.

Every service that changes something records an event through `record`, inside the session
that makes the change, so the log and the data never disagree. Events are never changed or
removed. `events` reads them back newest first with the names of the people and contracts
they are about; `describe` turns one into a sentence for the pages.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, aliased

from contract_assistant.db.models import Contract, Event, Folder, User

# Accounts
LOGIN = "user.login"
LOGIN_FAILED = "user.login_failed"
LOGOUT = "user.logout"
PASSWORD_CHANGED = "user.password_changed"
USER_CREATED = "user.created"
USER_RENAMED = "user.renamed"
ROLE_CHANGED = "user.role_changed"
USER_DEACTIVATED = "user.deactivated"
USER_ACTIVATED = "user.activated"
PASSWORD_RESET = "user.password_reset"
DATA_EXPORTED = "user.exported"
# Folders
FOLDER_CREATED = "folder.created"
FOLDER_RENAMED = "folder.renamed"
FOLDER_MOVED = "folder.moved"
FOLDER_DELETED = "folder.deleted"
FOLDER_RESTORED = "folder.restored"
FOLDER_PURGED = "folder.purged"
FOLDER_SHARED = "folder.shared"
FOLDER_UNSHARED = "folder.unshared"
# Contracts
CONTRACT_ADDED = "contract.added"
CONTRACT_IMPORTED = "contract.imported"
CONTRACT_RENAMED = "contract.renamed"
CONTRACT_MOVED = "contract.moved"
CONTRACT_DELETED = "contract.deleted"
CONTRACT_RESTORED = "contract.restored"
CONTRACT_PURGED = "contract.purged"
CONTRACT_REANALYSED = "contract.reanalysed"
CONTRACT_EXPORTED = "contract.exported"
# Passages
CLAUSE_CONFIRMED = "clause.confirmed"
CLAUSE_REJECTED = "clause.rejected"
CLAUSE_UNDONE = "clause.undone"
# Reviews
REVIEW_SUBMITTED = "review.submitted"
REVIEW_APPROVED = "review.approved"
REVIEW_RETURNED = "review.returned"
REVIEW_REOPENED = "review.reopened"
# Chat
FEEDBACK_GIVEN = "chat.feedback"
CHAT_CLEARED = "chat.cleared"

# One sentence per action. The fields come from the event: {actor}, {subject} (the user an
# account event is about), {contract}, {folder}, and the keys of its details, such as
# {clause}, {name}, {old_name}, {role}, {level}, {note}. The web layer translates them.
DESCRIPTIONS: dict[str, str] = {
    LOGIN: "{actor} signed in",
    LOGIN_FAILED: "A sign-in as {email} failed",
    LOGOUT: "{actor} signed out",
    PASSWORD_CHANGED: "{actor} changed their password",
    USER_CREATED: "{actor} created the account of {subject} ({role})",
    USER_RENAMED: "{actor} renamed {old_name} to {subject}",
    ROLE_CHANGED: "{actor} made {subject} {role}",
    USER_DEACTIVATED: "{actor} deactivated the account of {subject}",
    USER_ACTIVATED: "{actor} activated the account of {subject}",
    PASSWORD_RESET: "{actor} reset the password of {subject}",
    DATA_EXPORTED: "{actor} downloaded {what}",
    FOLDER_CREATED: "{actor} created the folder {folder}",
    FOLDER_RENAMED: "{actor} renamed the folder {old_name} to {folder}",
    FOLDER_MOVED: "{actor} moved the folder {folder} to {target}",
    FOLDER_DELETED: "{actor} moved the folder {folder} to the trash",
    FOLDER_RESTORED: "{actor} restored the folder {folder}",
    FOLDER_PURGED: "{actor} deleted the folder {name} for good",
    FOLDER_SHARED: "{actor} shared the folder {folder} with {subject} ({level})",
    FOLDER_UNSHARED: "{actor} stopped sharing the folder {folder} with {subject}",
    CONTRACT_ADDED: "{actor} added {contract}",
    CONTRACT_IMPORTED: "{contract} was imported from the cache folder",
    CONTRACT_RENAMED: "{actor} renamed {old_name} to {contract}",
    CONTRACT_MOVED: "{actor} moved {contract} to {target}",
    CONTRACT_DELETED: "{actor} moved {contract} to the trash",
    CONTRACT_RESTORED: "{actor} restored {contract}",
    CONTRACT_PURGED: "{actor} deleted {name} for good",
    CONTRACT_REANALYSED: "{actor} had {contract} read again for new clauses",
    CONTRACT_EXPORTED: "{actor} downloaded {what} of {contract}",
    CLAUSE_CONFIRMED: "{actor} confirmed the {clause} passage of {contract}",
    CLAUSE_REJECTED: "{actor} rejected the {clause} passage of {contract}",
    CLAUSE_UNDONE: "{actor} undid the decision on {clause} in {contract}",
    REVIEW_SUBMITTED: "{actor} sent {contract} for approval",
    REVIEW_APPROVED: "{actor} approved {contract}",
    REVIEW_RETURNED: "{actor} sent {contract} back: {note}",
    REVIEW_REOPENED: "{actor} reopened the review of {contract}",
    FEEDBACK_GIVEN: "{actor} rated an answer about {contract}",
    CHAT_CLEARED: "{actor} cleared their chat about {contract}",
}
SYSTEM = "The system"
SOMEONE = "Someone"
TRANSLATED = (*DESCRIPTIONS.values(), SYSTEM, SOMEONE)

# Detail values longer than this are cut, so that no event carries a whole contract.
MAX_DETAIL_CHARS = 300
DEFAULT_LIMIT = 50


def record(
    session: Session,
    action: str,
    *,
    actor_id: str | None = None,
    contract_id: str | None = None,
    folder_id: str | None = None,
    subject_user_id: str | None = None,
    details: dict[str, Any] | None = None,
    ip: str | None = None,
) -> Event:
    """Add an event to the session; it is written when the session commits."""
    if action not in DESCRIPTIONS:
        raise ValueError(f"unknown action: {action}")
    event = Event(
        action=action,
        actor_id=actor_id,
        contract_id=contract_id,
        folder_id=folder_id,
        subject_user_id=subject_user_id,
        details={key: _short(value) for key, value in (details or {}).items()},
        ip=ip,
    )
    session.add(event)
    return event


def _short(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    # Postgres keeps no NUL character in JSONB; a quote from a PDF can hold one.
    value = value.replace("\x00", "")
    if len(value) > MAX_DETAIL_CHARS:
        return value[: MAX_DETAIL_CHARS - 1] + "…"
    return value


@dataclass(frozen=True)
class EventFilter:
    """Which events to read; every field left out matches all."""

    actor_id: str | None = None
    contract_id: str | None = None
    folder_id: str | None = None
    subject_user_id: str | None = None
    # Events about a contract, a folder or an account of this user, or done by them.
    involving_user_id: str | None = None
    actions: Collection[str] = ()
    since: datetime | None = None
    until: datetime | None = None
    # For the next page: only events older than this id.
    before_id: int | None = None


@dataclass(frozen=True)
class LoggedEvent:
    """An event with the names of the people, contract and folder it is about."""

    id: int
    at: datetime
    action: str
    actor_id: str | None
    actor_name: str | None
    actor_email: str | None
    contract_id: str | None
    contract_name: str | None
    folder_id: str | None
    folder_name: str | None
    subject_user_id: str | None
    subject_name: str | None
    ip: str | None
    details: dict[str, Any] = field(default_factory=dict)


def events(
    session: Session, where: EventFilter | None = None, limit: int = DEFAULT_LIMIT
) -> list[LoggedEvent]:
    """Return the events that match, newest first, at most limit of them."""
    where = where or EventFilter()
    actor, subject = aliased(User), aliased(User)
    query: Select = (
        select(Event, actor, subject, Contract.name, Folder.name)
        .outerjoin(actor, Event.actor_id == actor.id)
        .outerjoin(subject, Event.subject_user_id == subject.id)
        .outerjoin(Contract, Event.contract_id == Contract.id)
        .outerjoin(Folder, Event.folder_id == Folder.id)
    )
    for column, value in (
        (Event.actor_id, where.actor_id),
        (Event.contract_id, where.contract_id),
        (Event.folder_id, where.folder_id),
        (Event.subject_user_id, where.subject_user_id),
    ):
        if value is not None:
            query = query.where(column == value)
    if where.involving_user_id is not None:
        user_id = where.involving_user_id
        query = query.where(
            (Event.actor_id == user_id)
            | (Event.subject_user_id == user_id)
            | (Contract.owner_id == user_id)
            | (Folder.owner_id == user_id)
        )
    if where.actions:
        query = query.where(Event.action.in_(list(where.actions)))
    if where.since is not None:
        query = query.where(Event.at >= where.since)
    if where.until is not None:
        query = query.where(Event.at < where.until)
    if where.before_id is not None:
        query = query.where(Event.id < where.before_id)
    query = query.order_by(Event.id.desc()).limit(limit)
    return [
        LoggedEvent(
            id=event.id,
            at=event.at,
            action=event.action,
            actor_id=event.actor_id,
            actor_name=by.name if by else None,
            actor_email=by.email if by else None,
            contract_id=event.contract_id,
            contract_name=contract_name,
            folder_id=event.folder_id,
            folder_name=folder_name,
            subject_user_id=event.subject_user_id,
            subject_name=about.name if about else None,
            ip=event.ip,
            details=dict(event.details or {}),
        )
        for event, by, about, contract_name, folder_name in session.execute(query)
    ]


class _Fields(dict):
    """Format fields that leave an unknown name as it is instead of failing."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def describe(event: LoggedEvent, translate=lambda text: text) -> str:
    """Return the sentence of an event, its words translated with `translate`."""
    template = translate(DESCRIPTIONS.get(event.action, event.action))
    fields = _Fields(
        {key: value for key, value in event.details.items() if value is not None},
        actor=event.actor_name or translate(SYSTEM if event.actor_id is None else SOMEONE),
        subject=event.subject_name or event.details.get("email") or translate(SOMEONE),
        contract=event.contract_name or event.details.get("name") or "",
        folder=event.folder_name or event.details.get("name") or "",
    )
    return template.format_map(fields)
