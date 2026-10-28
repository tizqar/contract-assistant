"""Folders, contracts, shares, the trash, decisions on passages and the review of a contract.

Every user has a space: a tree of folders with contracts in them, and at its top the folders
and contracts that are in no folder. The owner can share a folder, with everything inside it,
to view or to edit; an admin can do everything in every space. Each function here takes the
actor, checks their level with `access` before it reads or changes anything, records each
change in the activity log in the same session, and returns frozen views for the pages.

Names: a folder's name is unique among its siblings, ignoring case. A name the reader types
(a new folder, a rename) that is taken is refused; a name that comes along with something
added, moved or restored gets " (2)", " (3)" ... when taken, before the extension of a
contract's name. Deleting puts a folder or contract in the trash, and everything inside a
folder in the trash counts as in the trash too; only an admin deletes for good, from the trash.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session, aliased

from contract_assistant import audit, decisions
from contract_assistant.access import (
    NOT_FOUND,
    SHARE_LEVEL,
    AccessError,
    Action,
    Actor,
    Conflict,
    Level,
    NotFound,
    Permission,
    ReviewStep,
    level_of,
    next_status,
    require,
    require_open,
    require_permission,
)
from contract_assistant.db.models import (
    CONFIRMED,
    KINDS,
    NAME_LENGTH,
    SHARE_LEVELS,
    SUBMITTED,
    VERDICTS,
    Analysis,
    Blob,
    ChatMessage,
    Contract,
    Event,
    Feedback,
    Folder,
    FolderShare,
    Generated,
    User,
    new_id,
    utcnow,
)
from contract_assistant.db.models import Decision as DecisionRow

MAX_NOTE_CHARS = 2000
# An extension longer than this, or with other than letters and digits, is part of the name.
MAX_EXTENSION_CHARS = 8

# What the reader is told when something is refused; the web layer translates them.
EMPTY_NAME = "Give it a name."
NAME_TOO_LONG = f"A name can be at most {NAME_LENGTH} characters long."
FOLDER_NAME_TAKEN = "There is already a folder with that name here."
CONTRACT_NAME_TAKEN = "There is already a contract with that name here."
INTO_ITSELF = "A folder cannot be moved into itself or into a folder inside it."
OTHER_SPACE = (
    "Folders and contracts can only be moved within the space of the person they belong to."
)
OWNER_ALREADY = "That folder already belongs to that person."
NO_SUCH_USER = "There is no active account with that email address."
UNKNOWN_LEVEL = "Choose whether they can view or edit."
NOT_IN_TRASH = "That folder or contract is not in the trash."
UNKNOWN_VERDICT = "A passage can only be confirmed or rejected."
UNKNOWN_CLAUSE = "That clause type is not known."
BAD_SPAN = "That passage is not a part of the text."
NOTE_NEEDED = "Write a note saying what should change."
NOTE_TOO_LONG = f"A note can be at most {MAX_NOTE_CHARS} characters long."
UNKNOWN_KIND = "Only PDF and text files can be added."
NO_ANALYSIS = "This contract has not been read yet."
NO_FILE = "The file of this contract is missing."
ID_TAKEN = "A contract with that id already exists."
# The target of a move to the top of a space, kept in the event's details.
TOP_LEVEL = "the top level"
TRANSLATED = (
    EMPTY_NAME,
    NAME_TOO_LONG,
    FOLDER_NAME_TAKEN,
    CONTRACT_NAME_TAKEN,
    INTO_ITSELF,
    OTHER_SPACE,
    OWNER_ALREADY,
    NO_SUCH_USER,
    UNKNOWN_LEVEL,
    NOT_IN_TRASH,
    UNKNOWN_VERDICT,
    UNKNOWN_CLAUSE,
    BAD_SPAN,
    NOTE_NEEDED,
    NOTE_TOO_LONG,
    UNKNOWN_KIND,
    NO_ANALYSIS,
    NO_FILE,
    ID_TAKEN,
    TOP_LEVEL,
)

REVIEW_EVENTS = {
    ReviewStep.SUBMIT: audit.REVIEW_SUBMITTED,
    ReviewStep.APPROVE: audit.REVIEW_APPROVED,
    ReviewStep.RETURN: audit.REVIEW_RETURNED,
    ReviewStep.REOPEN: audit.REVIEW_REOPENED,
}


class FilesError(AccessError):
    """Input that cannot be used, such as an empty name or a send-back without a note."""

    status_code = 400


@dataclass(frozen=True)
class FolderView:
    """A folder as the actor sees it."""

    id: str
    name: str
    owner_id: str
    owner_name: str
    parent_id: str | None
    level: Level
    # The actor's share level here, through this folder or one above it; None for their own.
    shared_with_me: str | None
    # Whether this folder itself is shared with anyone.
    has_shares: bool
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class ContractView:
    """A contract as the actor sees it; `level` is the actor's level on it."""

    id: str
    name: str
    owner_id: str
    owner_name: str
    folder_id: str | None
    doc_id: str
    kind: str
    file_sha256: str | None
    review_status: str
    review_note: str | None
    review_changed_at: datetime | None
    review_changed_by_name: str | None
    created_by: str
    created_by_name: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    level: Level


@dataclass(frozen=True)
class Listing:
    """What is in a folder, or at the top of a space when `folder` is None.

    `path` runs from the top of the space down to the folder, the folder included, and holds
    only the folders the actor can see; `level` is the actor's level in the folder.
    """

    owner_id: str
    owner_name: str
    folder: FolderView | None
    path: list[FolderView]
    folders: list[FolderView]
    contracts: list[ContractView]
    level: Level


@dataclass(frozen=True)
class TrashView:
    """The top-most folders and contracts in the trash of one space, the newest first."""

    owner_id: str
    owner_name: str
    folders: list[FolderView]
    contracts: list[ContractView]
    level: Level


@dataclass(frozen=True)
class ShareView:
    """One user a folder is shared with."""

    user_id: str
    name: str
    email: str
    level: str
    granted_by_name: str | None
    granted_at: datetime


@dataclass(frozen=True)
class DecisionView:
    """A decision on a passage, ready for `decisions.apply`, and who made it."""

    decision: decisions.Decision
    by_id: str | None
    by_name: str | None


class _Tree:
    """The folders of some spaces with their shares, loaded once for one call."""

    def __init__(self, folders: Iterable[Folder], shares: Iterable[tuple[str, str, str]]):
        self.folders: dict[str, Folder] = {}
        self.children: dict[str | None, list[Folder]] = {}
        for folder in folders:
            self.add(folder)
        self.shares: dict[str, dict[str, str]] = {}
        for folder_id, user_id, level in shares:
            self.shares.setdefault(folder_id, {})[user_id] = level

    def add(self, folder: Folder) -> None:
        """Take in a folder made during the call."""
        self.folders[folder.id] = folder
        self.children.setdefault(folder.parent_id, []).append(folder)

    def chain(self, folder_id: str | None) -> list[Folder]:
        """Return the folders from the top of the space down to folder_id, both included."""
        chain: list[Folder] = []
        # The bound only guards against a cycle, which moves never make.
        while folder_id in self.folders and len(chain) <= len(self.folders):
            folder = self.folders[folder_id]
            chain.append(folder)
            folder_id = folder.parent_id
        chain.reverse()
        return chain

    def in_trash(self, folder_id: str | None) -> bool:
        """Return whether a folder, or one above it, is in the trash."""
        return any(folder.deleted_at is not None for folder in self.chain(folder_id))

    def shared(self, user_id: str, folder_id: str | None) -> str | None:
        """Return the highest level a folder or one above it is shared with a user at."""
        found = [
            self.shares.get(folder.id, {}).get(user_id) for folder in self.chain(folder_id)
        ]
        return _highest(level for level in found if level)

    def level(
        self, actor: Actor, owner_id: str, folder_id: str | None, *, trashed: bool = False
    ) -> Level:
        """Return the actor's level in a folder of owner_id's space, or at its top for None.

        A folder in the trash is hidden (NONE), except from its owner and admins when
        `trashed` asks for the trash.
        """
        level = level_of(actor, owner_id, self.shared(actor.id, folder_id))
        if self.in_trash(folder_id) and (not trashed or level < Level.OWN):
            return Level.NONE
        return level

    def subtree(self, folder_id: str) -> list[Folder]:
        """Return a folder and every folder inside it, parents before children."""
        found = [self.folders[folder_id]]
        for folder in found:
            found.extend(self.children.get(folder.id, ()))
        return found

    def trashed(self) -> set[str]:
        """Return the ids of every folder that counts as in the trash."""
        found: set[str] = set()
        stack = [(f, False) for f in self.folders.values() if f.parent_id not in self.folders]
        while stack:
            folder, above = stack.pop()
            inside = above or folder.deleted_at is not None
            if inside:
                found.add(folder.id)
            stack.extend((child, inside) for child in self.children.get(folder.id, ()))
        return found

    def reachable(self, user_id: str) -> dict[str, str]:
        """Return the folders out of the trash a user reaches through shares, with the level."""
        reached: dict[str, str] = {}
        for start, users in self.shares.items():
            if user_id not in users or start not in self.folders or self.in_trash(start):
                continue
            stack = [(self.folders[start], self.shared(user_id, start))]
            while stack:
                folder, inherited = stack.pop()
                level = _highest((inherited, self.shares.get(folder.id, {}).get(user_id)))
                known = reached.get(folder.id)
                if (
                    level is None
                    or known is not None
                    and SHARE_LEVEL[known] >= SHARE_LEVEL[level]
                ):
                    continue
                reached[folder.id] = level
                stack.extend(
                    (child, level)
                    for child in self.children.get(folder.id, ())
                    if child.deleted_at is None
                )
        return reached


def _highest(levels: Iterable[str | None]) -> str | None:
    """Return the highest of some share levels, None when there are none."""
    found = [level for level in levels if level]
    return max(found, key=SHARE_LEVEL.__getitem__) if found else None


def _tree(session: Session, owner_ids: Collection[str] | None) -> _Tree:
    """Load the folders and shares of some spaces; None loads every space."""
    folders = select(Folder).execution_options(populate_existing=True)
    shares = select(FolderShare.folder_id, FolderShare.user_id, FolderShare.level).join(
        Folder, Folder.id == FolderShare.folder_id
    )
    if owner_ids is not None:
        if not owner_ids:
            return _Tree((), ())
        folders = folders.where(Folder.owner_id.in_(list(owner_ids)))
        shares = shares.where(Folder.owner_id.in_(list(owner_ids)))
    return _Tree(session.scalars(folders), session.execute(shares).tuples())


def _lock_space(session: Session, owner_id: str) -> None:
    """Hold the owner's row until the commit, so that two changes cannot pick one name.

    FOR NO KEY UPDATE on Postgres, which leaves alone the rows that only refer to the user;
    SQLite has no row locks and writes one transaction at a time anyway.
    """
    session.execute(select(User.id).where(User.id == owner_id).with_for_update(key_share=True))


def _names(session: Session, ids: Iterable[str | None]) -> dict[str, str]:
    """Return the names of some users by id."""
    wanted = {user_id for user_id in ids if user_id}
    if not wanted:
        return {}
    return dict(session.execute(select(User.id, User.name).where(User.id.in_(wanted))).all())


def _people(rows: Iterable[Contract]) -> set[str]:
    """Return the ids of the users the views of some contracts name."""
    found: set[str] = set()
    for row in rows:
        found.update(
            user for user in (row.owner_id, row.created_by, row.review_changed_by) if user
        )
    return found


def _space_owner(session: Session, actor: Actor, owner_id: str | None) -> User:
    """Return the owner of a space the actor may open: their own, or any for an admin."""
    owner_id = owner_id or actor.id
    owner = session.get(User, owner_id) if owner_id == actor.id or actor.is_admin else None
    if owner is None:
        raise NotFound(NOT_FOUND)
    return owner


def _resolve_folder(
    session: Session,
    actor: Actor,
    folder_id: str | None,
    action: Action,
    *,
    trashed: bool = False,
    lock: bool = False,
) -> tuple[Folder, _Tree, Level]:
    """Return a folder, its space and the actor's level; raise unless the action is allowed.

    `trashed` asks for a folder in the trash, for restore and purge; otherwise one in the
    trash is NotFound. `lock` holds the space for a change that checks names.
    """
    folder = session.get(Folder, folder_id) if folder_id else None
    if folder is None:
        raise NotFound(NOT_FOUND)
    if lock:
        _lock_space(session, folder.owner_id)
    tree = _tree(session, [folder.owner_id])
    level = tree.level(actor, folder.owner_id, folder.id, trashed=trashed)
    require(level, action)
    if trashed and not tree.in_trash(folder.id):
        raise Conflict(NOT_IN_TRASH)
    return folder, tree, level


def _resolve_contract(
    session: Session,
    actor: Actor,
    contract_id: str | None,
    action: Action,
    *,
    trashed: bool = False,
    lock: bool = False,
    with_tree: bool = False,
) -> tuple[Contract, _Tree, Level]:
    """Return a contract, its space and the actor's level; raise unless the action is allowed.

    The space's folders are loaded when the contract is in a folder or `with_tree` asks.
    """
    row = session.get(Contract, contract_id) if contract_id else None
    if row is None:
        raise NotFound(NOT_FOUND)
    if lock:
        _lock_space(session, row.owner_id)
        session.refresh(row)
    if row.folder_id is not None or with_tree:
        tree = _tree(session, [row.owner_id])
    else:
        tree = _Tree((), ())
    in_trash = row.deleted_at is not None or tree.in_trash(row.folder_id)
    level = level_of(actor, row.owner_id, tree.shared(actor.id, row.folder_id))
    if in_trash and (not trashed or level < Level.OWN):
        level = Level.NONE
    require(level, action)
    if trashed and not in_trash:
        raise Conflict(NOT_IN_TRASH)
    return row, tree, level


def _folder_view(
    folder: Folder, tree: _Tree, actor: Actor, names: dict[str, str]
) -> FolderView:
    return FolderView(
        id=folder.id,
        name=folder.name,
        owner_id=folder.owner_id,
        owner_name=names.get(folder.owner_id, ""),
        parent_id=folder.parent_id,
        level=tree.level(actor, folder.owner_id, folder.id, trashed=True),
        shared_with_me=None
        if actor.id == folder.owner_id
        else tree.shared(actor.id, folder.id),
        has_shares=bool(tree.shares.get(folder.id)),
        created_at=folder.created_at,
        updated_at=folder.updated_at,
        deleted_at=folder.deleted_at,
    )


def _contract_view(row: Contract, level: Level, names: dict[str, str]) -> ContractView:
    return ContractView(
        id=row.id,
        name=row.name,
        owner_id=row.owner_id,
        owner_name=names.get(row.owner_id, ""),
        folder_id=row.folder_id,
        doc_id=row.doc_id,
        kind=row.kind,
        file_sha256=row.file_sha256,
        review_status=row.review_status,
        review_note=row.review_note,
        review_changed_at=row.review_changed_at,
        review_changed_by_name=names.get(row.review_changed_by or ""),
        created_by=row.created_by,
        created_by_name=names.get(row.created_by, ""),
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        level=level,
    )


def _one_contract(session: Session, row: Contract, level: Level) -> ContractView:
    return _contract_view(row, level, _names(session, _people([row])))


def _one_folder(session: Session, folder: Folder, tree: _Tree, actor: Actor) -> FolderView:
    return _folder_view(folder, tree, actor, _names(session, [folder.owner_id]))


# Names


def _clean_name(name: str | None) -> str:
    """Return a name with its spaces tidied; raise FilesError when empty or too long."""
    cleaned = " ".join((name or "").split())
    if not cleaned:
        raise FilesError(EMPTY_NAME)
    if len(cleaned) > NAME_LENGTH:
        raise FilesError(NAME_TOO_LONG)
    return cleaned


def _key(name: str) -> str:
    return name.casefold()


def _split_extension(name: str) -> tuple[str, str]:
    """Return a file name's stem and its extension with the dot, or the name and ""."""
    stem, dot, extension = name.rpartition(".")
    if dot and stem and extension.isalnum() and len(extension) <= MAX_EXTENSION_CHARS:
        return stem, dot + extension
    return name, ""


def _numbered(name: str, taken: set[str], *, extension: bool) -> str:
    """Return name, or name with " (2)", " (3)" ... (before the extension) when it is taken."""
    if _key(name) not in taken:
        return name
    stem, suffix = _split_extension(name) if extension else (name, "")
    number = 2
    while True:
        ending = f" ({number}){suffix}"
        candidate = stem[: NAME_LENGTH - len(ending)] + ending
        if _key(candidate) not in taken:
            return candidate
        number += 1


def _folder_names(
    tree: _Tree, owner_id: str, parent_id: str | None, except_id: str | None = None
) -> set[str]:
    """Return the names, ignoring case, of the folders in a folder that are not deleted."""
    return {
        _key(folder.name)
        for folder in tree.children.get(parent_id, ())
        if folder.owner_id == owner_id and folder.deleted_at is None and folder.id != except_id
    }


def _contract_names(
    session: Session, owner_id: str, folder_id: str | None, except_id: str | None = None
) -> set[str]:
    """Return the names, ignoring case, of the contracts in a folder that are not deleted."""
    where = (
        Contract.folder_id.is_(None) if folder_id is None else Contract.folder_id == folder_id
    )
    query = select(Contract.id, Contract.name).where(
        Contract.owner_id == owner_id, Contract.deleted_at.is_(None), where
    )
    return {_key(name) for row_id, name in session.execute(query) if row_id != except_id}


def _target(
    session: Session, actor: Actor, tree: _Tree, owner_id: str, target_id: str | None
) -> str:
    """Check that the actor may move something into a folder of the same space; return its name.

    target_id None is the top of the space.
    """
    if target_id is not None and target_id not in tree.folders:
        # Not in this space: say it does not exist unless the actor could add there.
        _resolve_folder(session, actor, target_id, Action.ADD)
        raise FilesError(OTHER_SPACE)
    require(tree.level(actor, owner_id, target_id), Action.ADD)
    return TOP_LEVEL if target_id is None else tree.folders[target_id].name


# Listing


def listing(
    session: Session, actor: Actor, folder_id: str | None = None, owner_id: str | None = None
) -> Listing:
    """Return what is in a folder, or at the top of the actor's space (an admin: owner_id's)."""
    if folder_id is None:
        owner_id = _space_owner(session, actor, owner_id).id
        tree = _tree(session, [owner_id])
        folder = None
    else:
        folder, tree, _ = _resolve_folder(session, actor, folder_id, Action.READ)
        if owner_id is not None and owner_id != folder.owner_id:
            raise NotFound(NOT_FOUND)
        owner_id = folder.owner_id
    level = tree.level(actor, owner_id, folder_id)
    require(level, Action.READ)
    children = [
        child
        for child in tree.children.get(folder_id, ())
        if child.owner_id == owner_id and child.deleted_at is None
    ]
    where = (
        Contract.folder_id.is_(None) if folder_id is None else Contract.folder_id == folder_id
    )
    rows = session.scalars(
        select(Contract).where(
            Contract.owner_id == owner_id, Contract.deleted_at.is_(None), where
        )
    ).all()
    names = _names(session, {owner_id, *_people(rows)})
    path = [
        _folder_view(above, tree, actor, names)
        for above in tree.chain(folder_id)
        if tree.level(actor, owner_id, above.id) >= Level.VIEW
    ]
    return Listing(
        owner_id=owner_id,
        owner_name=names.get(owner_id, ""),
        folder=path[-1] if folder is not None else None,
        path=path,
        folders=sorted(
            (_folder_view(child, tree, actor, names) for child in children),
            key=lambda view: _key(view.name),
        ),
        contracts=sorted(
            (_contract_view(row, level, names) for row in rows),
            key=lambda view: _key(view.name),
        ),
        level=level,
    )


def shared_with_me(session: Session, actor: Actor) -> list[FolderView]:
    """Return the top-most folders other users share with the actor, out of the trash."""
    owners = session.scalars(
        select(Folder.owner_id)
        .join(FolderShare, FolderShare.folder_id == Folder.id)
        .where(FolderShare.user_id == actor.id, Folder.owner_id != actor.id)
        .distinct()
    ).all()
    tree = _tree(session, owners)

    def mine(folder: Folder) -> bool:
        return actor.id in tree.shares.get(folder.id, {})

    tops = [
        folder
        for folder in tree.folders.values()
        if mine(folder)
        and not tree.in_trash(folder.id)
        and not any(mine(above) for above in tree.chain(folder.parent_id))
    ]
    names = _names(session, owners)
    views = [_folder_view(folder, tree, actor, names) for folder in tops]
    return sorted(views, key=lambda view: (_key(view.owner_name), _key(view.name), view.id))


def spaces(session: Session, actor: Actor) -> list[tuple[str, str, int]]:
    """Return every user's (id, name, number of contracts out of the trash), for an admin."""
    require_permission(actor, Permission.SEE_ALL_SPACES)
    trashed = _tree(session, None).trashed()
    counts: dict[str, int] = {}
    query = (
        select(Contract.owner_id, Contract.folder_id, func.count())
        .where(Contract.deleted_at.is_(None))
        .group_by(Contract.owner_id, Contract.folder_id)
    )
    for owner_id, folder_id, count in session.execute(query).tuples():
        if folder_id is None or folder_id not in trashed:
            counts[owner_id] = counts.get(owner_id, 0) + count
    users = session.execute(select(User.id, User.name)).tuples()
    return sorted(
        ((user_id, name, counts.get(user_id, 0)) for user_id, name in users),
        key=lambda space: (_key(space[1]), space[0]),
    )


# Folders


def create_folder(
    session: Session,
    actor: Actor,
    parent_id: str | None,
    name: str,
    ip: str | None = None,
    *,
    owner_id: str | None = None,
    system: bool = False,
) -> FolderView:
    """Make a folder in a folder, or at the top of the actor's space (an admin: owner_id's).

    With system, the log names no one: the app made it for the actor, as an import does.
    """
    name = _clean_name(name)
    if parent_id is None:
        owner_id = _space_owner(session, actor, owner_id).id
        _lock_space(session, owner_id)
        tree = _tree(session, [owner_id])
        require(tree.level(actor, owner_id, None), Action.ADD)
    else:
        parent, tree, _ = _resolve_folder(session, actor, parent_id, Action.ADD, lock=True)
        owner_id = parent.owner_id
    if _key(name) in _folder_names(tree, owner_id, parent_id):
        raise Conflict(FOLDER_NAME_TAKEN)
    folder = Folder(owner_id=owner_id, parent_id=parent_id, name=name, created_by=actor.id)
    session.add(folder)
    session.flush()
    tree.add(folder)
    audit.record(
        session,
        audit.FOLDER_CREATED,
        actor_id=None if system else actor.id,
        folder_id=folder.id,
        details={"name": name},
        ip=ip,
    )
    return _one_folder(session, folder, tree, actor)


def rename_folder(
    session: Session, actor: Actor, folder_id: str, name: str, ip: str | None = None
) -> FolderView:
    """Rename a folder; a name another folder beside it has is refused."""
    name = _clean_name(name)
    folder, tree, _ = _resolve_folder(session, actor, folder_id, Action.RENAME, lock=True)
    old_name = folder.name
    if name != old_name:
        if _key(name) in _folder_names(tree, folder.owner_id, folder.parent_id, folder.id):
            raise Conflict(FOLDER_NAME_TAKEN)
        folder.name = name
        session.flush()
        audit.record(
            session,
            audit.FOLDER_RENAMED,
            actor_id=actor.id,
            folder_id=folder.id,
            details={"name": name, "old_name": old_name},
            ip=ip,
        )
    return _one_folder(session, folder, tree, actor)


def move_folder(
    session: Session,
    actor: Actor,
    folder_id: str,
    new_parent_id: str | None,
    ip: str | None = None,
) -> FolderView:
    """Move a folder into another folder of the same space, or to its top for None."""
    folder, tree, _ = _resolve_folder(session, actor, folder_id, Action.MOVE, lock=True)
    if new_parent_id == folder.parent_id:
        return _one_folder(session, folder, tree, actor)
    if new_parent_id in {inside.id for inside in tree.subtree(folder.id)}:
        raise FilesError(INTO_ITSELF)
    target = _target(session, actor, tree, folder.owner_id, new_parent_id)
    old_name = folder.name
    name = _numbered(
        old_name, _folder_names(tree, folder.owner_id, new_parent_id), extension=False
    )
    folder.parent_id, folder.name = new_parent_id, name
    session.flush()
    details = {"name": name, "target": target}
    if name != old_name:
        details["old_name"] = old_name
    audit.record(
        session,
        audit.FOLDER_MOVED,
        actor_id=actor.id,
        folder_id=folder.id,
        details=details,
        ip=ip,
    )
    return _one_folder(session, folder, tree, actor)


def delete_folder(
    session: Session, actor: Actor, folder_id: str, ip: str | None = None
) -> FolderView:
    """Put a folder, with everything inside it, in the trash."""
    folder, tree, _ = _resolve_folder(session, actor, folder_id, Action.DELETE)
    folder.deleted_at, folder.deleted_by = utcnow(), actor.id
    session.flush()
    audit.record(
        session,
        audit.FOLDER_DELETED,
        actor_id=actor.id,
        folder_id=folder.id,
        details={"name": folder.name},
        ip=ip,
    )
    return _one_folder(session, folder, tree, actor)


def restore_folder(
    session: Session, actor: Actor, folder_id: str, ip: str | None = None
) -> FolderView:
    """Bring a folder back from the trash; to the top of the space if its parent is in it."""
    folder, tree, _ = _resolve_folder(
        session, actor, folder_id, Action.RESTORE, trashed=True, lock=True
    )
    parent_id = None if tree.in_trash(folder.parent_id) else folder.parent_id
    taken = _folder_names(tree, folder.owner_id, parent_id, folder.id)
    old_name = folder.name
    name = _numbered(old_name, taken, extension=False)
    details: dict[str, object] = {"name": name}
    if name != old_name:
        details["old_name"] = old_name
    if parent_id != folder.parent_id:
        details["target"] = TOP_LEVEL
    folder.deleted_at = folder.deleted_by = None
    folder.parent_id, folder.name = parent_id, name
    session.flush()
    audit.record(
        session,
        audit.FOLDER_RESTORED,
        actor_id=actor.id,
        folder_id=folder.id,
        details=details,
        ip=ip,
    )
    return _one_folder(session, folder, tree, actor)


def purge_folder(session: Session, actor: Actor, folder_id: str, ip: str | None = None) -> None:
    """Delete for good a folder in the trash, with every folder and contract inside it."""
    folder, tree, _ = _resolve_folder(session, actor, folder_id, Action.PURGE, trashed=True)
    inside = tree.subtree(folder.id)
    ids = [each.id for each in inside]
    names = {each.id: each.name for each in inside}
    rows = session.scalars(select(Contract).where(Contract.folder_id.in_(ids))).all()
    rows = sorted(rows, key=lambda row: (_key(row.name), row.id))
    gone = [(row.id, row.name, names[row.folder_id]) for row in rows]
    folder_name, contracts = folder.name, len(rows)
    _purge_contracts(session, rows)
    session.execute(update(Event).where(Event.folder_id.in_(ids)).values(folder_id=None))
    session.execute(delete(FolderShare).where(FolderShare.folder_id.in_(ids)))
    session.execute(delete(Folder).where(Folder.id.in_(ids)))
    for contract_id, name, in_folder in gone:
        audit.record(
            session,
            audit.CONTRACT_PURGED,
            actor_id=actor.id,
            details={"name": name, "id": contract_id, "folder": in_folder},
            ip=ip,
        )
    audit.record(
        session,
        audit.FOLDER_PURGED,
        actor_id=actor.id,
        details={
            "name": folder_name,
            "id": folder_id,
            "folders": len(ids),
            "contracts": contracts,
        },
        ip=ip,
    )


def _purge_contracts(session: Session, rows: list[Contract]) -> None:
    """Delete contracts and what is theirs; their analysis and file when no other uses them.

    Feedback and events stay, without the contract; the events keep its name in their details.
    """
    if not rows:
        return
    ids = [row.id for row in rows]
    doc_ids = {row.doc_id for row in rows}
    files = {row.file_sha256 for row in rows if row.file_sha256}
    session.execute(delete(DecisionRow).where(DecisionRow.contract_id.in_(ids)))
    session.execute(delete(ChatMessage).where(ChatMessage.contract_id.in_(ids)))
    # Generated records are kept under the contract id (the web app's doc_id).
    session.execute(delete(Generated).where(Generated.doc_id.in_(ids)))
    session.execute(
        update(Feedback).where(Feedback.contract_id.in_(ids)).values(contract_id=None)
    )
    session.execute(update(Event).where(Event.contract_id.in_(ids)).values(contract_id=None))
    session.execute(delete(Contract).where(Contract.id.in_(ids)))
    used = set(session.scalars(select(Contract.doc_id).where(Contract.doc_id.in_(doc_ids))))
    if doc_ids - used:
        session.execute(delete(Analysis).where(Analysis.doc_id.in_(doc_ids - used)))
    if files:
        used = set(
            session.scalars(select(Contract.file_sha256).where(Contract.file_sha256.in_(files)))
        )
        if files - used:
            session.execute(delete(Blob).where(Blob.sha256.in_(files - used)))


# Shares


def share_folder(
    session: Session,
    actor: Actor,
    folder_id: str,
    email: str,
    level: str,
    ip: str | None = None,
) -> ShareView:
    """Share a folder, with everything inside it, with the user of an email, to view or edit."""
    folder, _, _ = _resolve_folder(session, actor, folder_id, Action.SHARE)
    if level not in SHARE_LEVELS:
        raise FilesError(UNKNOWN_LEVEL)
    user = session.scalar(select(User).where(User.email == (email or "").strip().lower()))
    if user is None or not user.active:
        raise NotFound(NO_SUCH_USER)
    if user.id == folder.owner_id:
        raise FilesError(OWNER_ALREADY)
    share = session.get(FolderShare, (folder.id, user.id))
    if share is None or share.level != level:
        if share is None:
            share = FolderShare(folder_id=folder.id, user_id=user.id)
            session.add(share)
        share.level, share.granted_by, share.granted_at = level, actor.id, utcnow()
        session.flush()
        audit.record(
            session,
            audit.FOLDER_SHARED,
            actor_id=actor.id,
            folder_id=folder.id,
            subject_user_id=user.id,
            details={"name": folder.name, "level": level},
            ip=ip,
        )
    return ShareView(
        user_id=user.id,
        name=user.name,
        email=user.email,
        level=share.level,
        granted_by_name=_names(session, [share.granted_by]).get(share.granted_by),
        granted_at=share.granted_at,
    )


def unshare_folder(
    session: Session, actor: Actor, folder_id: str, user_id: str, ip: str | None = None
) -> None:
    """Stop sharing a folder with a user; nothing happens when it was not shared with them."""
    folder, _, _ = _resolve_folder(session, actor, folder_id, Action.SHARE)
    share = session.get(FolderShare, (folder.id, user_id))
    if share is None:
        return
    level = share.level
    session.delete(share)
    session.flush()
    audit.record(
        session,
        audit.FOLDER_UNSHARED,
        actor_id=actor.id,
        folder_id=folder.id,
        subject_user_id=user_id,
        details={"name": folder.name, "level": level},
        ip=ip,
    )


def shares_of(session: Session, actor: Actor, folder_id: str) -> list[ShareView]:
    """Return the users a folder itself is shared with, for the ones who may share it."""
    folder, _, _ = _resolve_folder(session, actor, folder_id, Action.SHARE)
    granter = aliased(User)
    query = (
        select(FolderShare, User, granter.name)
        .join(User, User.id == FolderShare.user_id)
        .outerjoin(granter, granter.id == FolderShare.granted_by)
        .where(FolderShare.folder_id == folder.id)
    )
    views = [
        ShareView(
            user_id=user.id,
            name=user.name,
            email=user.email,
            level=share.level,
            granted_by_name=granted_by_name,
            granted_at=share.granted_at,
        )
        for share, user, granted_by_name in session.execute(query).tuples()
    ]
    return sorted(views, key=lambda view: (_key(view.name), view.email))


# Contracts


def add_contract(
    session: Session,
    actor: Actor,
    folder_id: str | None,
    name: str,
    doc_id: str,
    kind: str,
    file_sha256: str | None = None,
    ip: str | None = None,
    contract_id: str | None = None,
    *,
    owner_id: str | None = None,
    system: bool = False,
    details: dict | None = None,
) -> ContractView:
    """Add a contract whose analysis is stored to a folder, or to the top of the actor's space.

    owner_id picks another space's top, for an admin; contract_id picks the id, for the
    importer and the tests. A taken name gets " (2)", " (3)" ... before its extension. With
    system the contract is logged as imported by the app, with details added to the event.
    """
    require_permission(actor, Permission.ADD_CONTRACTS)
    name = _clean_name(name)
    if kind not in KINDS:
        raise FilesError(UNKNOWN_KIND)
    if folder_id is None:
        owner_id = _space_owner(session, actor, owner_id).id
        _lock_space(session, owner_id)
        level = level_of(actor, owner_id)
        require(level, Action.ADD)
    else:
        folder, _, level = _resolve_folder(session, actor, folder_id, Action.ADD, lock=True)
        owner_id = folder.owner_id
    if session.scalar(select(Analysis.doc_id).where(Analysis.doc_id == doc_id)) is None:
        raise FilesError(NO_ANALYSIS)
    if file_sha256 is not None and (
        session.scalar(select(Blob.sha256).where(Blob.sha256 == file_sha256)) is None
    ):
        raise FilesError(NO_FILE)
    if contract_id is not None and session.get(Contract, contract_id) is not None:
        raise Conflict(ID_TAKEN)
    name = _numbered(name, _contract_names(session, owner_id, folder_id), extension=True)
    row = Contract(
        id=contract_id or new_id(),
        owner_id=owner_id,
        folder_id=folder_id,
        name=name,
        doc_id=doc_id,
        file_sha256=file_sha256,
        kind=kind,
        created_by=actor.id,
    )
    session.add(row)
    session.flush()
    audit.record(
        session,
        audit.CONTRACT_IMPORTED if system else audit.CONTRACT_ADDED,
        actor_id=None if system else actor.id,
        contract_id=row.id,
        folder_id=folder_id,
        details={**(details or {}), "name": name},
        ip=ip,
    )
    return _one_contract(session, row, level)


def contract(
    session: Session, actor: Actor, contract_id: str, action: Action = Action.READ
) -> ContractView:
    """Return a contract the actor may act on; NotFound when unknown, in the trash or hidden."""
    row, _, level = _resolve_contract(session, actor, contract_id, action)
    return _one_contract(session, row, level)


def visible_contracts(session: Session, actor: Actor) -> list[ContractView]:
    """Return every contract out of the trash the actor can read, the newest first."""
    query = select(Contract).where(Contract.deleted_at.is_(None))
    if actor.is_admin:
        tree = _tree(session, None)
        reached: dict[str, str] = {}
    else:
        owners = session.scalars(
            select(Folder.owner_id)
            .join(FolderShare, FolderShare.folder_id == Folder.id)
            .where(FolderShare.user_id == actor.id)
            .distinct()
        ).all()
        tree = _tree(session, {actor.id, *owners})
        reached = tree.reachable(actor.id)
        others = [
            folder_id for folder_id in reached if tree.folders[folder_id].owner_id != actor.id
        ]
        query = query.where(or_(Contract.owner_id == actor.id, Contract.folder_id.in_(others)))
    trashed = tree.trashed()
    rows = session.scalars(query.order_by(Contract.created_at.desc(), Contract.id)).all()
    names = _names(session, _people(rows))
    views = []
    for row in rows:
        if row.folder_id is not None and row.folder_id in trashed:
            continue
        shared = None if row.owner_id == actor.id else reached.get(row.folder_id or "")
        views.append(_contract_view(row, level_of(actor, row.owner_id, shared), names))
    return views


def rename_contract(
    session: Session, actor: Actor, contract_id: str, name: str, ip: str | None = None
) -> ContractView:
    """Rename a contract; a name another contract in its folder has is refused."""
    name = _clean_name(name)
    row, _, level = _resolve_contract(session, actor, contract_id, Action.RENAME, lock=True)
    old_name = row.name
    if name != old_name:
        if _key(name) in _contract_names(session, row.owner_id, row.folder_id, row.id):
            raise Conflict(CONTRACT_NAME_TAKEN)
        row.name = name
        session.flush()
        audit.record(
            session,
            audit.CONTRACT_RENAMED,
            actor_id=actor.id,
            contract_id=row.id,
            details={"name": name, "old_name": old_name},
            ip=ip,
        )
    return _one_contract(session, row, level)


def move_contract(
    session: Session,
    actor: Actor,
    contract_id: str,
    folder_id: str | None,
    ip: str | None = None,
) -> ContractView:
    """Move a contract into a folder of the same space, or to its top for None."""
    row, tree, level = _resolve_contract(
        session, actor, contract_id, Action.MOVE, lock=True, with_tree=True
    )
    if folder_id == row.folder_id:
        return _one_contract(session, row, level)
    target = _target(session, actor, tree, row.owner_id, folder_id)
    old_name = row.name
    taken = _contract_names(session, row.owner_id, folder_id)
    row.folder_id, row.name = folder_id, _numbered(old_name, taken, extension=True)
    session.flush()
    details = {"name": row.name, "target": target}
    if row.name != old_name:
        details["old_name"] = old_name
    audit.record(
        session,
        audit.CONTRACT_MOVED,
        actor_id=actor.id,
        contract_id=row.id,
        folder_id=folder_id,
        details=details,
        ip=ip,
    )
    return _one_contract(session, row, level)


def delete_contract(
    session: Session, actor: Actor, contract_id: str, ip: str | None = None
) -> ContractView:
    """Put a contract in the trash."""
    row, _, level = _resolve_contract(session, actor, contract_id, Action.DELETE)
    row.deleted_at, row.deleted_by = utcnow(), actor.id
    session.flush()
    audit.record(
        session,
        audit.CONTRACT_DELETED,
        actor_id=actor.id,
        contract_id=row.id,
        details={"name": row.name},
        ip=ip,
    )
    return _one_contract(session, row, level)


def restore_contract(
    session: Session, actor: Actor, contract_id: str, ip: str | None = None
) -> ContractView:
    """Bring a contract back from the trash; to the top of the space if its folder is in it."""
    row, tree, level = _resolve_contract(
        session, actor, contract_id, Action.RESTORE, trashed=True, lock=True
    )
    folder_id = None if tree.in_trash(row.folder_id) else row.folder_id
    old_name = row.name
    name = _numbered(
        old_name, _contract_names(session, row.owner_id, folder_id), extension=True
    )
    details: dict[str, object] = {"name": name}
    if name != old_name:
        details["old_name"] = old_name
    if folder_id != row.folder_id:
        details["target"] = TOP_LEVEL
    row.deleted_at = row.deleted_by = None
    row.folder_id, row.name = folder_id, name
    session.flush()
    audit.record(
        session,
        audit.CONTRACT_RESTORED,
        actor_id=actor.id,
        contract_id=row.id,
        details=details,
        ip=ip,
    )
    return _one_contract(session, row, level)


def purge_contract(
    session: Session, actor: Actor, contract_id: str, ip: str | None = None
) -> None:
    """Delete for good a contract in the trash, with its decisions, chats and generations."""
    row, _, _ = _resolve_contract(session, actor, contract_id, Action.PURGE, trashed=True)
    details = {"name": row.name, "id": row.id}
    _purge_contracts(session, [row])
    audit.record(session, audit.CONTRACT_PURGED, actor_id=actor.id, details=details, ip=ip)


def trash(session: Session, actor: Actor, owner_id: str | None = None) -> TrashView:
    """Return the top-most folders and contracts in the trash of the actor's space.

    An admin can ask for the trash of another space with owner_id.
    """
    owner = _space_owner(session, actor, owner_id)
    tree = _tree(session, [owner.id])
    level = level_of(actor, owner.id)
    folders = [
        folder
        for folder in tree.folders.values()
        if folder.deleted_at is not None and not tree.in_trash(folder.parent_id)
    ]
    rows = [
        row
        for row in session.scalars(
            select(Contract).where(
                Contract.owner_id == owner.id, Contract.deleted_at.is_not(None)
            )
        )
        if not tree.in_trash(row.folder_id)
    ]
    names = _names(session, {owner.id, *_people(rows)})
    return TrashView(
        owner_id=owner.id,
        owner_name=owner.name,
        folders=sorted(
            (_folder_view(folder, tree, actor, names) for folder in folders),
            key=lambda view: (view.deleted_at, view.id),
            reverse=True,
        ),
        contracts=sorted(
            (_contract_view(row, level, names) for row in rows),
            key=lambda view: (view.deleted_at, view.id),
            reverse=True,
        ),
        level=level,
    )


# Decisions


def _decision(row: DecisionRow) -> decisions.Decision:
    return decisions.Decision(
        clause_type=row.clause_type,
        verdict=row.verdict,
        start=row.start,
        end=row.end,
        at=row.decided_at.isoformat(timespec="seconds"),
    )


def decisions_of(session: Session, contract_id: str) -> dict[str, DecisionView]:
    """Return the decisions on a contract's passages by clause type; callers check access."""
    query = (
        select(DecisionRow, User.name)
        .outerjoin(User, User.id == DecisionRow.decided_by)
        .where(DecisionRow.contract_id == contract_id)
        .order_by(DecisionRow.clause_type)
    )
    return {
        row.clause_type: DecisionView(
            decision=_decision(row), by_id=row.decided_by, by_name=name
        )
        for row, name in session.execute(query).tuples()
    }


def decide(
    session: Session,
    actor: Actor,
    contract_id: str,
    clause_type: str,
    verdict: str,
    start: int,
    end: int,
    quote: str,
    ip: str | None = None,
) -> DecisionView:
    """Confirm or reject the passage of a clause type, replacing an earlier decision on it.

    Whether the span fits the text is the caller's to check; here only 0 <= start < end.
    """
    row, _, _ = _resolve_contract(session, actor, contract_id, Action.DECIDE, lock=True)
    require_open(row.review_status)
    if verdict not in VERDICTS:
        raise FilesError(UNKNOWN_VERDICT)
    if not clause_type or len(clause_type) > NAME_LENGTH:
        raise FilesError(UNKNOWN_CLAUSE)
    if not 0 <= start < end:
        raise FilesError(BAD_SPAN)
    current = session.get(DecisionRow, (row.id, clause_type))
    if current is not None and (current.verdict, current.start, current.end) == (
        verdict,
        start,
        end,
    ):
        by_name = _names(session, [current.decided_by]).get(current.decided_by or "")
        return DecisionView(
            decision=_decision(current), by_id=current.decided_by, by_name=by_name
        )
    previous = current.verdict if current is not None else None
    if current is None:
        current = DecisionRow(contract_id=row.id, clause_type=clause_type)
        session.add(current)
    current.verdict, current.start, current.end = verdict, start, end
    current.decided_by, current.decided_at = actor.id, utcnow()
    session.flush()
    audit.record(
        session,
        audit.CLAUSE_CONFIRMED if verdict == CONFIRMED else audit.CLAUSE_REJECTED,
        actor_id=actor.id,
        contract_id=row.id,
        details={
            "clause": clause_type,
            "verdict": verdict,
            "start": start,
            "end": end,
            "quote": quote,
            "previous": previous,
            "name": row.name,
        },
        ip=ip,
    )
    return DecisionView(decision=_decision(current), by_id=actor.id, by_name=actor.name)


def undo_decision(
    session: Session, actor: Actor, contract_id: str, clause_type: str, ip: str | None = None
) -> None:
    """Drop the decision on a clause type, so the passage shows as the model found it."""
    row, _, _ = _resolve_contract(session, actor, contract_id, Action.DECIDE, lock=True)
    require_open(row.review_status)
    current = session.get(DecisionRow, (row.id, clause_type))
    if current is None:
        return
    details = {
        "clause": clause_type,
        "previous": current.verdict,
        "start": current.start,
        "end": current.end,
        "quote": _last_quote(session, row.id, clause_type),
        "name": row.name,
    }
    session.delete(current)
    session.flush()
    audit.record(
        session,
        audit.CLAUSE_UNDONE,
        actor_id=actor.id,
        contract_id=row.id,
        details=details,
        ip=ip,
    )


def _last_quote(session: Session, contract_id: str, clause_type: str) -> str | None:
    """Return the quote of the latest decision event on a clause type of a contract."""
    query = (
        select(Event.details)
        .where(
            Event.contract_id == contract_id,
            Event.action.in_((audit.CLAUSE_CONFIRMED, audit.CLAUSE_REJECTED)),
        )
        .order_by(Event.id.desc())
    )
    for details in session.scalars(query):
        if (details or {}).get("clause") == clause_type:
            return details.get("quote")
    return None


# Review


def review(
    session: Session,
    actor: Actor,
    contract_id: str,
    step: ReviewStep,
    note: str | None = None,
    ip: str | None = None,
) -> ContractView:
    """Take a step of a contract's review; sending it back needs a note on what to change."""
    row, _, level = _resolve_contract(session, actor, contract_id, Action.READ, lock=True)
    status = next_status(level, row.review_status, step)
    note = (note or "").strip() or None
    if step is ReviewStep.RETURN and note is None:
        raise FilesError(NOTE_NEEDED)
    if note is not None and len(note) > MAX_NOTE_CHARS:
        raise FilesError(NOTE_TOO_LONG)
    details: dict[str, object] = {"name": row.name, "from": row.review_status, "to": status}
    if note is not None:
        details["note"] = note
    row.review_status = status
    if step is ReviewStep.SUBMIT:
        row.review_note = None
    elif step is ReviewStep.RETURN:
        row.review_note = note
    row.review_changed_at, row.review_changed_by = utcnow(), actor.id
    session.flush()
    audit.record(
        session,
        REVIEW_EVENTS[step],
        actor_id=actor.id,
        contract_id=row.id,
        details=details,
        ip=ip,
    )
    return _one_contract(session, row, level)


def awaiting_approval(session: Session, actor: Actor) -> list[ContractView]:
    """Return the contracts sent for approval, out of the trash, the longest waiting first."""
    require_permission(actor, Permission.APPROVE_REVIEWS)
    rows = session.scalars(
        select(Contract).where(
            Contract.review_status == SUBMITTED, Contract.deleted_at.is_(None)
        )
    ).all()
    tree = _tree(session, {row.owner_id for row in rows if row.folder_id is not None})
    rows = sorted(
        (row for row in rows if not tree.in_trash(row.folder_id)),
        key=lambda row: (row.review_changed_at or row.created_at, row.id),
    )
    names = _names(session, _people(rows))
    return [_contract_view(row, level_of(actor, row.owner_id), names) for row in rows]
