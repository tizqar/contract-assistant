"""The chat about one contract: questions, definitions, chat commands, explanations, feedback."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from contract_assistant.web.core import Tau, Ws, render

router = APIRouter()


@router.get("/documents/{doc_id}/chat", response_class=HTMLResponse)
def chat_body(request: Request, doc_id: str, ws: Ws, tau: Tau):
    """The messages of the chat about a contract, oldest first."""
    cache = ws.load(doc_id)
    conversation = ws.conversation(cache.doc_id)
    return render(request, "chat/body.html", cache=cache, conversation=conversation, tau=tau)
