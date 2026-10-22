"""The reader's decisions from the web app: confirm or reject a passage, or undo that."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from contract_assistant.answer import Citation, tier
from contract_assistant.decisions import Decision, now
from contract_assistant.store import DocumentCache
from contract_assistant.ui.views import display_name
from contract_assistant.web.core import Tau, WorkspaceError, Ws, view_url

router = APIRouter()

VERDICTS = ("confirmed", "rejected")
UNKNOWN_VERDICT = "A passage is confirmed or rejected; nothing else."
NO_SUCH_SPAN = "That passage is not in the contract."
UNKNOWN_CLAUSE = "That clause is not one the assistant checks."
SEE_OTHER = 303


@router.post("/documents/{doc_id}/decide")
def decide(
    request: Request,
    doc_id: str,
    ws: Ws,
    tau: Tau,
    clause_type: Annotated[str, Form()],
    verdict: Annotated[str, Form()],
    start: Annotated[int, Form()],
    end: Annotated[int, Form()],
):
    """Keep the decision, then show the passage the clause type now has in the document."""
    cache = ws.load(doc_id)
    name = _clause_type(ws, clause_type)
    if verdict not in VERDICTS:
        raise WorkspaceError(UNKNOWN_VERDICT)
    if not 0 <= start < end <= len(cache.text):
        raise WorkspaceError(NO_SUCH_SPAN)
    decision = Decision(clause_type=name, verdict=verdict, start=start, end=end, at=now())
    ws.decisions.decide(cache.doc_id, decision)
    return RedirectResponse(_after(ws, ws.load(doc_id), name, tau), status_code=SEE_OTHER)


@router.post("/documents/{doc_id}/decide/undo")
def undo(request: Request, doc_id: str, ws: Ws, tau: Tau, clause_type: Annotated[str, Form()]):
    """Forget the decision on a clause type, then show its passage as the model found it."""
    cache = ws.load(doc_id)
    name = _clause_type(ws, clause_type)
    ws.decisions.forget(cache.doc_id, name)
    return RedirectResponse(_after(ws, ws.load(doc_id), name, tau), status_code=SEE_OTHER)


def _clause_type(ws, label: str) -> str:
    name = ws.clause_type_of(label)
    if name not in ws.names:
        raise WorkspaceError(UNKNOWN_CLAUSE)
    return name


def _after(ws, cache: DocumentCache, name: str, tau) -> str:
    """The document pane at the top passage of the clause type, with its status after deciding."""
    label = display_name(name)
    candidates = cache.clauses.get(name, [])
    if not candidates:
        return view_url(cache.doc_id, label, status=ws.decided_status(cache.doc_id, name, "not_found"))
    top = candidates[0]
    citation = Citation(quote=top.text, page=top.page, conf=top.conf, start=top.start, end=top.end)
    status = ws.decided_status(cache.doc_id, name, tier(top.conf, tau.low, tau.high))
    return view_url(cache.doc_id, label, citation, status)
