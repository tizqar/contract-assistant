"""The file manager: a user's folders and contracts, the ones shared with them, and the trash.

Each page is one view: a place of a space (the top of one or a folder in it: `/files`,
`/files/folders/{id}`, and for an admin `/files/spaces/{owner}`), the folders shared with the
reader, a trash, or for an admin the list of every space. A row offers only the actions the
reader's level allows (`access.allowed`); `files` checks them again.

Dialogs are partials loaded into the page's one <dialog>. A change is a POST that answers 204,
or the share dialog again, with an `HX-Trigger` header naming `files-changed` and a short
message: `static/files.js` shows the message as a toast and closes the dialog, and the view
reloads itself on the event.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from contract_assistant import files
from contract_assistant.access import (
    NOT_FOUND,
    Action,
    Level,
    NotFound,
    Permission,
    allowed,
    require,
    require_permission,
)
from contract_assistant.db.models import VIEW, Folder
from contract_assistant.files import ContractView, FolderView
from contract_assistant.ui.views import document_name
from contract_assistant.web import activity
from contract_assistant.web.core import (
    DocumentEntry,
    Tau,
    UserWorkspace,
    Ws,
    language,
    render,
    render_string,
)
from contract_assistant.web.i18n import translate
from contract_assistant.web.library import (
    MINE,
    NO_MODEL,
    SAMPLES_NEED_MODEL,
    SHARED,
    place_of,
    samples_for,
)

router = APIRouter()

SEE_OTHER = 303
# The event a change triggers, and the form value that confirms deleting for good.
CHANGED = "files-changed"
CONFIRMED = "yes"
FOLDERS = "folders"
CONTRACTS = "contracts"
Kind = Literal["folders", "contracts"]
# The entries of the side list.
MY_SECTION = "mine"
SHARED_SECTION = "shared"
TRASH_SECTION = "trash"
SPACES_SECTION = "spaces"

# What a change tells the reader, as a toast, and the names of the top of a space.
FOLDER_CREATED = "Folder created: {name}."
RENAMED = "Renamed: {name}."
MOVED = "Moved to {target}."
TRASHED = "Moved to the trash: {name}."
RESTORED = "Restored: {name}."
PURGED = "Deleted for good: {name}."
SHARED_NOW = "{name} now has access to the folder."
ACCESS_CHANGED = "Access changed: {name}."
UNSHARED = "{name} no longer has access to the folder."
CONFIRM_FIRST = "Confirm that it should be deleted for good."
MY_FILES = "My files"
TOP_OF = "{name} (top level)"
TRANSLATED = (
    FOLDER_CREATED,
    RENAMED,
    MOVED,
    TRASHED,
    RESTORED,
    PURGED,
    SHARED_NOW,
    ACCESS_CHANGED,
    UNSHARED,
    CONFIRM_FIRST,
    MY_FILES,
    TOP_OF,
)

# What each kind of row does to its folder or contract, through `files`.
RENAME: dict[str, Callable] = {FOLDERS: files.rename_folder, CONTRACTS: files.rename_contract}
MOVE: dict[str, Callable] = {FOLDERS: files.move_folder, CONTRACTS: files.move_contract}
DELETE: dict[str, Callable] = {FOLDERS: files.delete_folder, CONTRACTS: files.delete_contract}
RESTORE: dict[str, Callable] = {
    FOLDERS: files.restore_folder,
    CONTRACTS: files.restore_contract,
}
PURGE: dict[str, Callable] = {FOLDERS: files.purge_folder, CONTRACTS: files.purge_contract}


@dataclass(frozen=True)
class ContractRow:
    """A contract as a row of the file manager: what `files` says and what the lists count."""

    view: ContractView
    entry: DocumentEntry


@dataclass(frozen=True)
class Item:
    """A folder or a contract a dialog is about."""

    kind: str
    id: str
    name: str
    shown: str
    owner_id: str
    owner_name: str
    # The folder it is in; None at the top of its space.
    parent_id: str | None


# Pages


@router.get("/files", response_class=HTMLResponse)
def my_files(request: Request, ws: Ws, tau: Tau):
    """The top of the reader's own space; the welcome while it is empty."""
    return _place_page(request, ws, tau, None, None)


@router.get("/files/shared", response_class=HTMLResponse)
def shared_page(request: Request, ws: Ws, tau: Tau):
    """The folders other users share with the reader, with their owner and the access."""
    with ws.database.session() as session:
        folders = files.shared_with_me(session, ws.actor)
    return _page(request, "files/shared.html", tau, SHARED_SECTION, folders=folders)


@router.get("/files/trash", response_class=HTMLResponse)
def trash_page(request: Request, ws: Ws, tau: Tau, owner: str = ""):
    """What is in the reader's trash, or in another user's for an admin."""
    with ws.database.session() as session:
        trash = files.trash(session, ws.actor, owner or None)
    mine = trash.owner_id == ws.actor.id
    section = TRASH_SECTION if mine else SPACES_SECTION
    return _page(request, "files/trash.html", tau, section, trash=trash, mine=mine)


@router.get("/files/spaces", response_class=HTMLResponse)
def spaces_page(request: Request, ws: Ws, tau: Tau):
    """Every user's space with the number of its contracts, for an admin."""
    with ws.database.session() as session:
        spaces = files.spaces(session, ws.actor)
    return _page(request, "files/spaces.html", tau, SPACES_SECTION, spaces=spaces)


@router.get("/files/spaces/{owner_id}", response_class=HTMLResponse)
def space_page(request: Request, owner_id: str, ws: Ws, tau: Tau):
    """The top of another user's space, for an admin; one's own is /files."""
    if owner_id == ws.actor.id:
        return RedirectResponse("/files", status_code=SEE_OTHER)
    require_permission(ws.actor, Permission.SEE_ALL_SPACES)
    return _place_page(request, ws, tau, None, owner_id)


@router.get("/files/folders/new", response_class=HTMLResponse)
def new_folder_dialog(request: Request, ws: Ws, parent: str = "", owner: str = ""):
    """The dialog that names a new folder in the place shown."""
    with ws.database.session() as session:
        listing = files.listing(session, ws.actor, parent or None, owner or None)
    require(listing.level, Action.ADD)
    return render(
        request,
        "files/dialog_name.html",
        mode="new",
        action="/files/folders",
        parent=parent,
        owner=owner,
        value="",
    )


@router.post("/files/folders")
def create_folder(
    request: Request,
    ws: Ws,
    name: Annotated[str, Form()] = "",
    parent: Annotated[str, Form()] = "",
    owner: Annotated[str, Form()] = "",
):
    """Make a folder in a folder, or at the top of the reader's space (an admin: owner's)."""
    with ws.database.session() as session:
        folder = files.create_folder(
            session, ws.actor, parent or None, name, ws.ip, owner_id=owner or None
        )
    return _changed(request, FOLDER_CREATED, name=folder.name)


@router.get("/files/folders/{folder_id}", response_class=HTMLResponse)
def folder_page(request: Request, folder_id: str, ws: Ws, tau: Tau):
    """A folder: the folders in it, then its contracts."""
    return _place_page(request, ws, tau, folder_id, None)


# Rename, move, delete, restore and delete for good: the same for folders and contracts.


@router.get("/files/{kind}/{item_id}/rename", response_class=HTMLResponse)
def rename_dialog(request: Request, kind: Kind, item_id: str, ws: Ws):
    """The dialog that renames a folder or a contract."""
    with ws.database.session() as session:
        item = _item(session, ws, kind, item_id, Action.RENAME)
    return render(
        request,
        "files/dialog_name.html",
        mode="rename",
        action=f"/files/{kind}/{item_id}/rename",
        label=item.shown,
        value=item.name,
    )


@router.post("/files/{kind}/{item_id}/rename")
def rename(
    request: Request, kind: Kind, item_id: str, ws: Ws, name: Annotated[str, Form()] = ""
):
    """Rename a folder or a contract; a name taken beside it is refused."""
    with ws.database.session() as session:
        view = RENAME[kind](session, ws.actor, item_id, name, ws.ip)
    return _changed(request, RENAMED, name=_shown(kind, view.name))


@router.get("/files/{kind}/{item_id}/move", response_class=HTMLResponse)
def move_dialog(request: Request, kind: Kind, item_id: str, ws: Ws):
    """The dialog that moves a folder or a contract: the folders of the same space."""
    with ws.database.session() as session:
        item = _item(session, ws, kind, item_id, Action.MOVE)
        skip = item.id if kind == FOLDERS else None
        targets = destinations(session, item.owner_id, skip)
    return render(
        request,
        "files/dialog_move.html",
        action=f"/files/{kind}/{item_id}/move",
        label=item.shown,
        current=item.parent_id,
        targets=targets,
        mine=item.owner_id == ws.actor.id,
        owner_name=item.owner_name,
    )


@router.post("/files/{kind}/{item_id}/move")
def move(
    request: Request, kind: Kind, item_id: str, ws: Ws, target: Annotated[str, Form()] = ""
):
    """Move a folder or a contract into a folder of the same space, or to its top."""
    with ws.database.session() as session:
        view = MOVE[kind](session, ws.actor, item_id, target or None, ws.ip)
        where = _where(request, session, ws, view.owner_id, view.owner_name, target or None)
    return _changed(request, MOVED, target=where)


@router.post("/files/{kind}/{item_id}/delete")
def delete(request: Request, kind: Kind, item_id: str, ws: Ws):
    """Put a folder, with everything inside it, or a contract in the trash."""
    with ws.database.session() as session:
        view = DELETE[kind](session, ws.actor, item_id, ws.ip)
    return _changed(request, TRASHED, name=_shown(kind, view.name))


@router.post("/files/{kind}/{item_id}/restore")
def restore(request: Request, kind: Kind, item_id: str, ws: Ws):
    """Bring a folder or a contract back from the trash."""
    with ws.database.session() as session:
        view = RESTORE[kind](session, ws.actor, item_id, ws.ip)
    return _changed(request, RESTORED, name=_shown(kind, view.name))


@router.get("/files/{kind}/{item_id}/purge", response_class=HTMLResponse)
def purge_dialog(request: Request, kind: Kind, item_id: str, ws: Ws, owner: str = ""):
    """The step that confirms deleting for good, for an admin, from a trash."""
    item = _in_trash(ws, kind, item_id, owner)
    return render(
        request,
        "files/dialog_purge.html",
        kind=kind,
        action=f"/files/{kind}/{item_id}/purge",
        label=_shown(kind, item.name),
        owner=owner,
    )


@router.post("/files/{kind}/{item_id}/purge")
def purge(
    request: Request,
    kind: Kind,
    item_id: str,
    ws: Ws,
    confirm: Annotated[str, Form()] = "",
    owner: Annotated[str, Form()] = "",
):
    """Delete for good a folder or a contract in a trash, once the admin confirmed it."""
    item = _in_trash(ws, kind, item_id, owner)
    if confirm != CONFIRMED:
        raise files.FilesError(CONFIRM_FIRST)
    with ws.database.session() as session:
        PURGE[kind](session, ws.actor, item_id, ws.ip)
    return _changed(request, PURGED, name=_shown(kind, item.name))


# Sharing


@router.get("/files/folders/{folder_id}/share", response_class=HTMLResponse)
def share_dialog(request: Request, folder_id: str, ws: Ws):
    """The dialog that shares a folder: who has access, and a form to add someone."""
    return render(request, "files/dialog_share.html", **_shares(ws, folder_id))


@router.post("/files/folders/{folder_id}/shares")
def share(
    request: Request,
    folder_id: str,
    ws: Ws,
    email: Annotated[str, Form()] = "",
    level: Annotated[str, Form()] = VIEW,
):
    """Share a folder with the user of an email, or change what they can do in it."""
    with ws.database.session() as session:
        before = {view.user_id for view in files.shares_of(session, ws.actor, folder_id)}
        shared = files.share_folder(session, ws.actor, folder_id, email, level, ws.ip)
    message = ACCESS_CHANGED if shared.user_id in before else SHARED_NOW
    html = render_string(request, "files/dialog_share.html", **_shares(ws, folder_id))
    return _changed(request, message, keep=True, html=html, name=shared.name)


@router.post("/files/folders/{folder_id}/shares/{user_id}/delete")
def unshare(request: Request, folder_id: str, user_id: str, ws: Ws):
    """Stop sharing a folder with a user."""
    with ws.database.session() as session:
        shares = files.shares_of(session, ws.actor, folder_id)
        names = {view.user_id: view.name for view in shares}
        files.unshare_folder(session, ws.actor, folder_id, user_id, ws.ip)
    html = render_string(request, "files/dialog_share.html", **_shares(ws, folder_id))
    if user_id not in names:
        return HTMLResponse(html)
    return _changed(request, UNSHARED, keep=True, html=html, name=names[user_id])


# Helpers


def destinations(
    session: Session, owner_id: str, skip: str | None
) -> list[tuple[str, str, int]]:
    """Return the folders of a space out of the trash as (id, name, depth), parents first.

    The folder `skip` and the folders inside it are left out: a folder cannot go into itself.
    Callers check that the reader may move things in this space, which lets them add to
    every folder of it.
    """
    rows = session.execute(
        select(Folder.id, Folder.name, Folder.parent_id, Folder.deleted_at).where(
            Folder.owner_id == owner_id
        )
    ).all()
    children: dict[str | None, list] = {}
    for row in rows:
        children.setdefault(row.parent_id, []).append(row)

    def ordered(parent_id: str | None) -> list:
        kept = [row for row in children.get(parent_id, ()) if row.deleted_at is None]
        return sorted(kept, key=lambda row: (row.name.casefold(), row.id), reverse=True)

    found: list[tuple[str, str, int]] = []
    stack = [(row, 0) for row in ordered(None)]
    while stack:
        row, depth = stack.pop()
        if row.id == skip:
            continue
        found.append((row.id, row.name, depth))
        stack.extend((child, depth + 1) for child in ordered(row.id))
    return found


def _place_page(
    request: Request, ws: UserWorkspace, tau, folder_id: str | None, owner_id: str | None
) -> HTMLResponse:
    """Render a place of a space: the top of one, or a folder in it."""
    actor = ws.actor
    with ws.database.session() as session:
        listing = files.listing(session, actor, folder_id, owner_id)
        empty = (
            listing.folder is None
            and listing.owner_id == actor.id
            and not listing.folders
            and not listing.contracts
        )
        shared = files.shared_with_me(session, actor) if empty else []
    place = place_of(ws, listing)
    rows = [
        ContractRow(view, entry)
        for view, entry in zip(listing.contracts, place.entries, strict=True)
    ]
    can_add = allowed(listing.level, Action.ADD)
    # An upload or a sample without a folder goes to the top of the reader's own space.
    can_upload = can_add and (place.folder_id is not None or place.kind == MINE)
    can_read = ws.services.extractor is not None
    new_folder = {"parent": place.folder_id or ""}
    if place.folder_id is None and place.kind != MINE:
        new_folder["owner"] = place.owner_id
    section = {MINE: MY_SECTION, SHARED: SHARED_SECTION}.get(place.kind, SPACES_SECTION)
    return _page(
        request,
        "files/place.html",
        tau,
        section,
        place=place,
        folders=listing.folders,
        contracts=rows,
        welcome=empty,
        shared_count=len(shared),
        can_new_folder=can_add,
        new_folder_url=f"/files/folders/new?{urlencode(new_folder)}",
        can_upload=can_upload,
        upload_folder=place.upload_folder,
        can_read=can_read,
        samples=samples_for(place.entries) if can_upload and (can_read or empty) else [],
        no_model=_t(request, NO_MODEL),
        samples_need_model=_t(request, SAMPLES_NEED_MODEL),
    )


def _page(request: Request, template: str, tau, section: str, **context) -> HTMLResponse:
    """Render a page of the file manager, with what every view needs."""
    lang = language(request)
    url = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return render(
        request,
        template,
        section=section,
        view_url=url,
        can=_can,
        when=lambda at: activity.when(at, lang),
        tau=tau,
        cache=None,
        **context,
    )


def _can(level: Level, action: str) -> bool:
    """Tell whether a level is enough for an action named as in `access.Action`."""
    return allowed(level, Action(action))


def _t(request: Request, text: str) -> str:
    return translate(text, language(request))


def _shown(kind: str, name: str) -> str:
    """Return the name of a folder as it is, and of a contract the way the lists show it."""
    return document_name(name) if kind == CONTRACTS else name


def _changed(
    request: Request, message: str, *, keep: bool = False, html: str | None = None, **values
) -> Response:
    """Answer a change: the view reloads and shows the message; the dialog closes unless kept.

    Without html the answer is 204, else the html for the dialog. The header is ASCII JSON,
    which htmx reads as an event with the message as its detail.
    """
    text = _t(request, message).format(**values)
    headers = {"HX-Trigger": json.dumps({CHANGED: {"message": text, "keep": keep}})}
    if html is None:
        return Response(status_code=204, headers=headers)
    return HTMLResponse(html, headers=headers)


def _item(session: Session, ws: UserWorkspace, kind: str, item_id: str, action: Action) -> Item:
    """Return the folder or contract a dialog is about; raise unless the reader may act."""
    if kind == CONTRACTS:
        view = files.contract(session, ws.actor, item_id, action)
        return Item(
            kind,
            view.id,
            view.name,
            document_name(view.name),
            view.owner_id,
            view.owner_name,
            view.folder_id,
        )
    listing = files.listing(session, ws.actor, item_id)
    require(listing.level, action)
    folder = listing.folder
    return Item(
        kind,
        folder.id,
        folder.name,
        folder.name,
        folder.owner_id,
        folder.owner_name,
        folder.parent_id,
    )


def _in_trash(
    ws: UserWorkspace, kind: str, item_id: str, owner: str
) -> FolderView | ContractView:
    """Return a folder or contract at the top of a trash, for an admin to delete for good."""
    require_permission(ws.actor, Permission.PURGE)
    with ws.database.session() as session:
        trash = files.trash(session, ws.actor, owner or None)
    found = trash.folders if kind == FOLDERS else trash.contracts
    for item in found:
        if item.id == item_id:
            return item
    raise NotFound(NOT_FOUND)


def _where(
    request: Request,
    session: Session,
    ws: UserWorkspace,
    owner_id: str,
    owner_name: str,
    folder_id: str | None,
) -> str:
    """Return the name of a folder, or of the top of a space, in the reader's language."""
    if folder_id is not None:
        return session.scalar(select(Folder.name).where(Folder.id == folder_id)) or ""
    if owner_id == ws.actor.id:
        return _t(request, MY_FILES)
    return _t(request, TOP_OF).format(name=owner_name)


def _shares(ws: UserWorkspace, folder_id: str) -> dict:
    """Return what the share dialog shows: the folder and who it is shared with."""
    with ws.database.session() as session:
        shares = files.shares_of(session, ws.actor, folder_id)
        folder = files.listing(session, ws.actor, folder_id).folder
    return {"folder": folder, "shares": shares}
