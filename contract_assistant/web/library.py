"""The contracts: the workspace page, the document pane, uploads, comparison and portfolio."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from contract_assistant.answer import Citation
from contract_assistant.ui.views import SUGGESTED_QUESTIONS, document_view
from contract_assistant.web.core import Tau, Ws, render

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request, ws: Ws, tau: Tau):
    """Open the default contract, or the page that asks for the first one."""
    doc_id = ws.default_doc_id()
    if doc_id is None:
        return render(request, "home.html", documents=[], tau=tau)
    return RedirectResponse(f"/documents/{doc_id}", status_code=303)


@router.get("/documents/{doc_id}", response_class=HTMLResponse)
def workspace_page(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The workspace of one contract: the contracts rail, the document and the insights."""
    cache = ws.load(doc_id)
    documents = ws.documents()
    entry = next((d for d in documents if d.doc_id == cache.doc_id), None)
    return render(
        request,
        "workspace.html",
        cache=cache,
        entry=entry,
        documents=documents,
        suggestions=SUGGESTED_QUESTIONS,
        tau=tau,
    )


@router.get("/documents/{doc_id}/view", response_class=HTMLResponse)
def document_pane(
    request: Request,
    doc_id: str,
    ws: Ws,
    tau: Tau,
    label: str | None = None,
    start: int | None = None,
    end: int | None = None,
    conf: float = 1.0,
    status: str | None = None,
):
    """The document pane: the text around a span with the span highlighted."""
    cache = ws.load(doc_id)
    citation = None
    if start is not None and end is not None:
        citation = Citation(
            quote=cache.text[start:end], page=None, conf=conf, start=start, end=end
        )
        if cache.page_starts:
            citation.page = max(
                sum(1 for page_start in cache.page_starts if page_start <= start), 1
            )
    clause_type = ws.clause_type_of(label) if label else None
    heading, segments = document_view(cache, clause_type, citation)
    return render(
        request,
        "library/view.html",
        cache=cache,
        heading=heading,
        segments=segments,
        label=label,
        citation=citation,
        status=status,
        tau=tau,
    )
