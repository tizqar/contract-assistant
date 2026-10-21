"""The insight tabs of one contract: summary, key terms, review, obligations, clauses, terms."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from contract_assistant.web.core import Tau, Ws, render

router = APIRouter()


@router.get("/documents/{doc_id}/definitions", response_class=HTMLResponse)
def definitions(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The terms the contract defines, with the sentence that defines each."""
    cache = ws.load(doc_id)
    return render(
        request,
        "insights/definitions.html",
        cache=cache,
        definitions=ws.definitions(cache),
        tau=tau,
    )
