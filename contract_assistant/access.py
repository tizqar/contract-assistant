"""Who may do what: the two roles, a user's level on a folder or contract, and the review steps.

A user works in their own space: the folders and contracts they add, plus the folders other
users share with them, to view or to edit. An admin can do all of that in every space, and is
also the one who manages accounts, reads the whole activity log, approves reviews and deletes
for good. A contract's review goes In review -> Sent for approval -> Approved; while it is
anywhere but In review its decisions are locked.

This module only decides. The services call `require` before they change anything; the pages
call `allowed` to offer only what will work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from contract_assistant.db.models import (
    ADMIN,
    APPROVED,
    EDIT,
    IN_REVIEW,
    ROLES,
    SUBMITTED,
    USER,
    VIEW,
)


class Permission(StrEnum):
    """What a role may do anywhere, whatever folder or contract it is about."""

    ADD_CONTRACTS = "add_contracts"
    MANAGE_USERS = "manage_users"
    READ_ALL_ACTIVITY = "read_all_activity"
    SEE_ALL_SPACES = "see_all_spaces"
    APPROVE_REVIEWS = "approve_reviews"
    PURGE = "purge"


ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    ADMIN: frozenset(Permission),
    USER: frozenset({Permission.ADD_CONTRACTS}),
}


class Level(IntEnum):
    """How much a user may do with one folder or contract; each level includes the ones below."""

    NONE = 0
    # Shared to view: read it, ask about it, have summaries written, download it.
    VIEW = 1
    # Shared to edit: also confirm or reject passages, rename, add contracts and folders inside,
    # and send a contract for approval.
    EDIT = 2
    # Their own: also move, delete to the trash and restore, and share folders.
    OWN = 3
    # An admin: also delete for good and approve, send back or reopen a review.
    ADMIN = 4


class Action(StrEnum):
    """What can be done to a folder or contract."""

    READ = "read"
    DECIDE = "decide"
    RENAME = "rename"
    ADD = "add"
    SUBMIT = "submit"
    MOVE = "move"
    DELETE = "delete"
    RESTORE = "restore"
    SHARE = "share"
    PURGE = "purge"
    APPROVE = "approve"


NEEDED: dict[Action, Level] = {
    Action.READ: Level.VIEW,
    Action.DECIDE: Level.EDIT,
    Action.RENAME: Level.EDIT,
    Action.ADD: Level.EDIT,
    Action.SUBMIT: Level.EDIT,
    Action.MOVE: Level.OWN,
    Action.DELETE: Level.OWN,
    Action.RESTORE: Level.OWN,
    Action.SHARE: Level.OWN,
    Action.PURGE: Level.ADMIN,
    Action.APPROVE: Level.ADMIN,
}

SHARE_LEVEL = {VIEW: Level.VIEW, EDIT: Level.EDIT}

# What the reader is told when something is refused; the web layer translates them.
NOT_ALLOWED = "You do not have permission to do that."
NOT_FOUND = "That folder or contract does not exist, or it is not shared with you."
LOCKED = "This contract is sent for approval or approved, so its passages cannot be changed."
WRONG_STEP = "This contract's review is not at a step where that can be done."
TRANSLATED = (NOT_ALLOWED, NOT_FOUND, LOCKED, WRONG_STEP)


class AccessError(Exception):
    """A refusal to tell the reader about, with the HTTP status that fits it."""

    status_code = 403

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Forbidden(AccessError):
    """The actor's level or role is too low for the action."""

    status_code = 403


class NotFound(AccessError):
    """The folder or contract does not exist, or the actor may not know it does."""

    status_code = 404


class Conflict(AccessError):
    """The action does not fit the current state, such as a decision on an approved contract."""

    status_code = 409


@dataclass(frozen=True)
class Actor:
    """The signed-in user a request acts for."""

    id: str
    email: str
    name: str
    role: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role: {self.role}")

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN

    def can(self, permission: Permission) -> bool:
        """Return whether the actor's role grants a permission."""
        return permission in ROLE_PERMISSIONS[self.role]


def level_of(actor: Actor, owner_id: str, shared: str | None = None) -> Level:
    """Return an actor's level on something owner_id owns, shared with them at `shared`.

    `shared` is the highest share level of the folder or any folder above it, or None.
    """
    if actor.is_admin:
        return Level.ADMIN
    if actor.id == owner_id:
        return Level.OWN
    return SHARE_LEVEL.get(shared or "", Level.NONE)


def allowed(level: Level, action: Action) -> bool:
    """Return whether a level is enough for an action."""
    return level >= NEEDED[action]


def require(level: Level, action: Action) -> None:
    """Raise NotFound when the actor may not see the thing, Forbidden when the level is short."""
    if level < Level.VIEW:
        raise NotFound(NOT_FOUND)
    if not allowed(level, action):
        raise Forbidden(NOT_ALLOWED)


def require_permission(actor: Actor, permission: Permission) -> None:
    """Raise Forbidden unless the actor's role grants the permission."""
    if not actor.can(permission):
        raise Forbidden(NOT_ALLOWED)


class ReviewStep(StrEnum):
    """A step of a contract's review."""

    SUBMIT = "submit"
    APPROVE = "approve"
    RETURN = "return"
    REOPEN = "reopen"


# Each step: the status it starts from, the status it leads to, and the action it needs.
REVIEW_STEPS: dict[ReviewStep, tuple[str, str, Action]] = {
    ReviewStep.SUBMIT: (IN_REVIEW, SUBMITTED, Action.SUBMIT),
    ReviewStep.APPROVE: (SUBMITTED, APPROVED, Action.APPROVE),
    ReviewStep.RETURN: (SUBMITTED, IN_REVIEW, Action.APPROVE),
    ReviewStep.REOPEN: (APPROVED, IN_REVIEW, Action.APPROVE),
}


def review_steps(level: Level, status: str) -> list[ReviewStep]:
    """Return the review steps an actor at a level can take from a status, in order."""
    return [
        step
        for step, (start, _, action) in REVIEW_STEPS.items()
        if start == status and allowed(level, action)
    ]


def next_status(level: Level, status: str, step: ReviewStep) -> str:
    """Return the status a step leads to; raise when the level or the status does not allow it."""
    start, end, action = REVIEW_STEPS[step]
    require(level, action)
    if status != start:
        raise Conflict(WRONG_STEP)
    return end


def decisions_open(status: str) -> bool:
    """Return whether passages of a contract in a review status can be confirmed or rejected."""
    return status == IN_REVIEW


def require_open(status: str) -> None:
    """Raise Conflict when a contract's decisions are locked by its review."""
    if not decisions_open(status):
        raise Conflict(LOCKED)
