"""A contract's review in the workspace: the bar with its status and steps, and its history.

The bar sits above the overview: the review's status and what it means, who changed it last
and when, the note of a contract sent back, and only the steps the reader may take now. A step
posts here; `files.review` checks it again and logs it, and the contract's page reloads so the
decision buttons, the badges and the bar all show the new state. The history lists everything
logged about the contract, the newest first, for anyone who can read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from jinja2 import pass_context

from contract_assistant import audit, files
from contract_assistant.access import ReviewStep, review_steps
from contract_assistant.db.models import IN_REVIEW
from contract_assistant.web import activity
from contract_assistant.web.auth import HTMX_HEADER, HTMX_REDIRECT, safe_next
from contract_assistant.web.core import (
    Tau,
    UserWorkspace,
    Ws,
    language,
    render,
    templates,
)
from contract_assistant.web.i18n import copy_for, translate

router = APIRouter()

SEE_OTHER = 303
HISTORY_PAGE = 30
# The htmx event that carries what a step did to the reloaded page (static/review.js).
DONE_EVENT = "review-done"

# Who changed the review last and when, by the way the status was reached.
RETURNED = "returned"
REOPENED = "reopened"
CHANGED_BY = {
    "submitted": "Sent for approval by {name}, {when}",
    "approved": "Approved by {name}, {when}",
    RETURNED: "Sent back by {name}, {when}",
    REOPENED: "Reopened by {name}, {when}",
}
# What the reader is told on the reloaded page after a step.
DONE = {
    ReviewStep.SUBMIT: (
        "Sent for approval. The decisions are locked until an administrator answers."
    ),
    ReviewStep.APPROVE: "Approved. The decisions stay locked.",
    ReviewStep.RETURN: "Sent back with your note.",
    ReviewStep.REOPEN: "Reopened. Passages can be confirmed or rejected again.",
}
TRANSLATED = (*CHANGED_BY.values(), *DONE.values())


@dataclass(frozen=True)
class ReviewBar:
    """What the bar shows of a contract's review, its words in the reader's language."""

    contract_id: str
    status: str
    label: str
    help: str
    # "Sent for approval by Marko, 28 Oct 2026, 14:09"; None while nobody has taken a step.
    changed: str | None
    # What to change, while a contract sent back is in review again.
    note: str | None
    steps: tuple[str, ...]


def bar(ws: UserWorkspace, contract_id: str, lang: str) -> ReviewBar:
    """Return the review bar of a contract the reader can read; raise when they cannot."""
    contract = ws.contract(contract_id)
    status = contract.review_status
    note = contract.review_note if status == IN_REVIEW else None
    changed = None
    if contract.review_changed_at is not None:
        how = status if status != IN_REVIEW else RETURNED if note else REOPENED
        name = contract.review_changed_by_name or translate(audit.SOMEONE, lang)
        changed = translate(CHANGED_BY[how], lang).format(
            name=name, when=activity.when(contract.review_changed_at, lang)
        )
    words = copy_for(lang)
    return ReviewBar(
        contract_id=contract.id,
        status=status,
        label=words.REVIEW_STATUS.get(status, status),
        help=words.REVIEW_HELP.get(status, ""),
        changed=changed,
        note=note,
        steps=tuple(step.value for step in review_steps(contract.level, status)),
    )


@pass_context
def _bar_in_page(context, contract_id: str) -> ReviewBar:
    """The review bar of a contract, for a template that includes review/bar.html."""
    return bar(context["ws"], contract_id, context["lang"])


templates.env.globals["review_bar"] = _bar_in_page


# /documents/{id}/review is the overview's things to check (web/insights.py).
@router.get("/documents/{contract_id}/review/bar", response_class=HTMLResponse)
def bar_partial(request: Request, contract_id: str, ws: Ws):
    """The review bar alone."""
    review = bar(ws, contract_id, language(request))
    return render(request, "review/bar.html", review=review)


@router.post("/documents/{contract_id}/review/{step}")
def take_step(
    request: Request,
    contract_id: str,
    step: str,
    ws: Ws,
    tau: Tau,
    note: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "",
):
    """Take a step of the review, then go to `next` or reload the contract's page."""
    chosen = _step(step)
    with ws.database.session() as session:
        files.review(session, ws.actor, contract_id, chosen, note, ws.ip)
    if next and safe_next(next) == next:
        return RedirectResponse(next, status_code=SEE_OTHER)
    url = _contract_url(ws, contract_id, tau)
    if not request.headers.get(HTMX_HEADER):
        return RedirectResponse(url, status_code=SEE_OTHER)
    done = translate(DONE[chosen], language(request))
    # JSON escapes every non-ASCII letter, so the header stays plain ASCII.
    return Response(headers={HTMX_REDIRECT: url, "HX-Trigger": json.dumps({DONE_EVENT: done})})


@router.get("/documents/{contract_id}/history", response_class=HTMLResponse)
def history(request: Request, contract_id: str, ws: Ws, before: int | None = None):
    """The contract's history, the newest first, a page at a time; `before` asks for older."""
    contract = ws.contract(contract_id)
    where = audit.EventFilter(contract_id=contract.id, before_id=before)
    with ws.database.session() as session:
        events = audit.events(session, where, limit=HISTORY_PAGE + 1)
    rows = activity.rows(events[:HISTORY_PAGE], language(request))
    older = rows[-1].id if len(events) > HISTORY_PAGE else None
    return render(
        request,
        "review/history.html",
        contract_id=contract.id,
        rows=rows,
        older=older,
        first=before is None,
    )


def _step(step: str) -> ReviewStep:
    try:
        return ReviewStep(step)
    except ValueError:
        raise HTTPException(status_code=404) from None


def _contract_url(ws: UserWorkspace, contract_id: str, tau) -> str:
    """The contract's page, with the thresholds when they are not the defaults."""
    url = f"/documents/{contract_id}"
    settings = ws.settings
    if (tau.low, tau.high) == (settings.tau_low, settings.tau_high):
        return url
    return f"{url}?{urlencode(tau.params())}"
