"""The chat about one contract: questions, definitions, chat commands, explanations, feedback."""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from contract_assistant.answer import (
    ELLIPSIS,
    MAX_CHAT_QUOTE_CHARS,
    Answer,
    AnswerItem,
    Citation,
    Definition,
    GeneratedAnswer,
    Passage,
    answer,
    restrict_to_cache,
)
from contract_assistant.definitions import lookup
from contract_assistant.explain import DEFAULT_LANGUAGE, LANGUAGES, Explanation
from contract_assistant.feedback import FEEDBACK_FILE, Feedback, now, record_feedback
from contract_assistant.generation import Generation
from contract_assistant.intents import detect_intent
from contract_assistant.passage_answer import APP_ATTEMPTS as PASSAGE_ATTEMPTS
from contract_assistant.passage_answer import generated_answer_from, passage_request
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.router.clients import LLMError
from contract_assistant.store import DocumentCache
from contract_assistant.summary import MS_PER_S, Summary
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    DEFINITION_NOTE,
    EMPTY_ROUTE_TEXT,
    EXAMPLE_QUESTIONS,
    EXPLANATION_META_TEXT,
    EXPLANATION_WITHHELD_TEXT,
    GENERATED_ANSWER_META,
    GENERATED_ANSWER_WITHHELD,
    NOT_FOUND_TEXT,
    OTHER_TOPIC,
    PASSAGES_TEXT,
    STATUS_LABELS,
    SUGGESTED_QUESTIONS,
    SUMMARY_BASED_ON,
    SUMMARY_LANGUAGE_TEXT,
    SUMMARY_META_TEXT,
    SUMMARY_MISMATCH_TEXT,
    SUMMARY_WITHHELD_TEXT,
    UNVALIDATED_MARK,
    UNVALIDATED_TEXT,
    WITHHELD_TEXT,
    definition_citation,
    display_name,
    passage_citation,
)
from contract_assistant.web.core import (
    DEFINITION_LABEL,
    GENERATION_FAILED,
    RELATED_LABEL,
    UNEXPECTED_ERROR,
    Conversation,
    Message,
    Tau,
    Thresholds,
    Workspace,
    WorkspaceError,
    Ws,
    frame,
    render,
    render_string,
    stream,
    view_url,
)
from contract_assistant.web.generate import Step, checked, explanation_steps, summary_steps

logger = logging.getLogger(__name__)

TYPE_A_QUESTION = "Type a question first."
THINKING = "Reading the contract"
WRITING_ANSWER = "Writing the answer from the passages"
NOTHING_TO_EXPLAIN = "Ask about a clause first, then say 'explain that'."
NO_QUOTE = "There is no passage to explain."
UNKNOWN_LANGUAGE = "Explanations are written in {languages}."
EXPLAIN_ASKED = "Explain: {label}"
EXPLAIN_ASKED_IN = "Explain in {language}: {label}"
FEEDBACK_THANKS = "Noted, thank you."
ANSWER_GONE = "This answer is no longer in the chat."
HIGHLIGHT_ONE = "The passage is highlighted in the document."
HIGHLIGHT_MANY = (
    "The first passage is highlighted in the document; Show in document opens the others."
)
PAGE_TEXT = "Page {page}"
CONFIDENCE_TEXT = "{conf:.0%} confidence"
PASSAGE_TITLE = "Passage {number}"
RELATED_TAG = "Related"

CHAT_TAB = "chat"
DOCUMENT_PANE = "document-pane"
PENDING_ID = "chat-pending"
SHOWN_STATUSES = frozenset({"found", "check", "definition", "related"})

router = APIRouter()

# Streams run in worker threads: a reply is added, and its feedback index fixed, under the lock.
_lock = threading.Lock()


@dataclass(frozen=True)
class Shown:
    """A passage the chat quotes: its label, the clause type behind it, its span and badge."""

    label: str
    clause_type: str
    quote: str
    citation: Citation | None = None
    status: str | None = None

    def view(self, doc_id: str) -> str | None:
        """Return the URL of the document pane at the passage, or None if its span is unknown."""
        if self.citation is None:
            return None
        return view_url(doc_id, self.label, self.citation, self.status)


@dataclass(frozen=True)
class Card:
    """One card of an answer: a badge and a title, then a quote or a short text."""

    kind: str
    tag: str
    title: str
    quote: str | None = None
    text: str | None = None
    meta: str | None = None
    plain: bool = False
    shown: Shown | None = None


@dataclass(frozen=True)
class Reply:
    """The assistant's bubble: the template that draws it and what the chat keeps of it."""

    template: str
    context: dict = field(default_factory=dict)
    answer: Answer | None = None
    turn: Turn | None = None
    shown: Shown | None = None


class Dropped(Exception):
    """Ends an exchange without a reply; the question is dropped and the message shown."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ExplainForm(BaseModel):
    """A passage to explain: its label and quote, the language, and its span when known."""

    label: str
    quote: str
    language: str = DEFAULT_LANGUAGE
    start: int | None = None
    end: int | None = None
    conf: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str | None = None
    # Sent by the buttons inside the chat, which stay in place while the reply is written.
    inline: bool = False


@router.get("/documents/{doc_id}/chat", response_class=HTMLResponse)
def chat_body(request: Request, doc_id: str, ws: Ws):
    """The messages of the chat about a contract, oldest first."""
    cache = ws.load(doc_id)
    conversation = ws.conversation(cache.doc_id)
    return render(request, "chat/body.html", cache=cache, conversation=conversation, pending=None)


@router.post("/documents/{doc_id}/ask")
def ask(
    request: Request, doc_id: str, ws: Ws, tau: Tau, question: Annotated[str, Form()] = ""
):
    """Answer a defined term, a chat command or a question about the contract, streaming."""
    cache = ws.load(doc_id)
    return stream(_ask(request, ws, cache, tau, question.strip()))


@router.post("/documents/{doc_id}/explain")
def explain(request: Request, doc_id: str, ws: Ws, form: Annotated[ExplainForm, Form()]):
    """Restate a quoted passage in plain language in the chat, streaming."""
    cache = ws.load(doc_id)
    return stream(_explain(request, ws, cache, form))


@router.post("/documents/{doc_id}/feedback", response_class=HTMLResponse)
def feedback(
    request: Request,
    doc_id: str,
    ws: Ws,
    index: Annotated[int, Form()],
    liked: Annotated[bool, Form()],
):
    """Keep a thumbs up or down on an answer of the chat in the local feedback file."""
    cache = ws.load(doc_id)
    conversation = ws.conversation(cache.doc_id)
    thanks = render_string(request, "chat/thanks.html", text=FEEDBACK_THANKS)
    with _lock:
        messages = conversation.messages
        message = messages[index] if 0 <= index < len(messages) else None
        if message is None or message.answer is None:
            raise WorkspaceError(ANSWER_GONE)
        result = message.answer
        verdict = Feedback(
            doc_id=cache.doc_id,
            question=result.question,
            route_kind=result.route.kind,
            clause_types=[item.clause_type for item in result.items],
            liked=liked,
            at=now(),
        )
        record_feedback(ws.settings.cache_dir / FEEDBACK_FILE, verdict)
        # The buttons render the same every time, so the chat drawn again shows the thanks.
        buttons = render_string(request, "chat/feedback.html", cache=cache, index=index)
        message.html = message.html.replace(buttons, thanks)
    return HTMLResponse(thanks)


@router.post("/documents/{doc_id}/chat/clear", response_class=HTMLResponse)
def clear_chat(request: Request, doc_id: str, ws: Ws):
    """Forget the chat about a contract: its messages, the turns and the last answer."""
    cache = ws.load(doc_id)
    conversation = ws.conversation(cache.doc_id)
    with _lock:
        conversation.messages = []
        conversation.turns = []
        conversation.last = None
    return render(request, "chat/body.html", cache=cache, conversation=conversation, pending=None)


def _ask(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds, question: str
) -> Iterator[str]:
    if not question:
        yield frame(notice=TYPE_A_QUESTION, done=True)
        return
    yield from _converse(request, ws, cache, question, _replies(ws, cache, tau, question))


def _explain(
    request: Request, ws: Workspace, cache: DocumentCache, form: ExplainForm
) -> Iterator[str]:
    if form.language not in LANGUAGES:
        languages = " or ".join(_language_name(code) for code in LANGUAGES)
        yield frame(notice=UNKNOWN_LANGUAGE.format(languages=languages), done=True)
        return
    if not form.quote.strip():
        yield frame(notice=NO_QUOTE, done=True)
        return
    asked = EXPLAIN_ASKED.format(label=form.label)
    if form.language != DEFAULT_LANGUAGE:
        asked = EXPLAIN_ASKED_IN.format(language=_language_name(form.language), label=form.label)
    replies = _explanation(ws, _posted(ws, cache, form), form.language)
    yield from _converse(request, ws, cache, asked, replies, inline=form.inline)


def _converse(
    request: Request,
    ws: Workspace,
    cache: DocumentCache,
    asked: str,
    replies: Iterator[Step | Reply],
    inline: bool = False,
) -> Iterator[str]:
    """Show the question with a status at once, the model's text as it comes, then the reply.

    The question and its reply are added to the chat together at the end, so a stopped
    exchange leaves nothing behind. Inline, the first frame only fills the pending place, so
    the button that asked stays on the page while the reply is written.
    """
    conversation = ws.conversation(cache.doc_id)
    user = render_string(request, "chat/user.html", text=asked)

    def pending(status: str, text: str = "") -> str:
        return render_string(request, "chat/pending.html", user=user, status=status, text=text)

    if inline:
        yield frame(oob={PENDING_ID: pending(THINKING)}, activate=CHAT_TAB)
    else:
        yield frame(_body(request, cache, conversation, pending(THINKING)), activate=CHAT_TAB)
    reply = _text(GENERATION_FAILED)
    try:
        for item in replies:
            if isinstance(item, Reply):
                reply = item
            else:
                yield frame(oob={PENDING_ID: pending(item.status, item.text)})
    except Dropped as dropped:
        body = _body(request, cache, conversation)
        yield frame(body, notice=dropped.message, activate=CHAT_TAB, done=True)
        return
    except WorkspaceError as error:
        reply = _text(error.message)
    except LLMError:
        logger.exception("generation failed in the chat about %s", cache.doc_id)
        reply = _text(GENERATION_FAILED)
    except Exception:
        logger.exception("chat reply failed for %s", cache.doc_id)
        reply = _text(UNEXPECTED_ERROR)
    try:
        _add(request, cache, conversation, user, reply)
    except Exception:
        logger.exception("cannot show the reply in the chat about %s", cache.doc_id)
        reply = _text(UNEXPECTED_ERROR)
        _add(request, cache, conversation, user, reply)
    load = None
    if reply.shown is not None and reply.shown.citation is not None:
        load = {DOCUMENT_PANE: reply.shown.view(cache.doc_id)}
    body = _body(request, cache, conversation)
    yield frame(body, load=load, activate=CHAT_TAB, done=True)


def _add(
    request: Request, cache: DocumentCache, conversation: Conversation, user: str, reply: Reply
) -> None:
    with _lock:
        index = len(conversation.messages) + 1
        html = render_string(request, reply.template, cache=cache, index=index, **reply.context)
        conversation.messages.extend(
            [
                Message(role="user", html=user),
                Message(role="assistant", html=html, answer=reply.answer),
            ]
        )
        if reply.turn is not None:
            conversation.remember(reply.turn)
        if reply.answer is not None:
            conversation.last = reply.answer


def _body(
    request: Request, cache: DocumentCache, conversation: Conversation, pending: str | None = None
) -> str:
    return render_string(
        request, "chat/body.html", cache=cache, conversation=conversation, pending=pending
    )


def _replies(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, question: str
) -> Iterator[Step | Reply]:
    """Answer a defined term, else a chat command, else route the question to clause types."""
    conversation = ws.conversation(cache.doc_id)
    definition = lookup(ws.definitions(cache), question)
    if definition is not None:
        yield _answer_reply(ws, _definition_answer(question, definition, tau))
        return
    intent = detect_intent(question)
    if intent is not None and intent.kind == "explain":
        shown = _first_shown(ws, conversation.last)
        if shown is None:
            yield _text(NOTHING_TO_EXPLAIN)
            return
        yield from _explanation(ws, shown, intent.language)
        return
    if intent is not None:
        yield from _summary(ws, cache, tau, intent.language)
        return
    earlier = list(conversation.turns)
    route = restrict_to_cache(ws.services.router.route(question, earlier), cache)
    try:
        result = answer(question, route, cache, tau.low, tau.high, ws.services.unvalidated)
    except ValueError as error:
        raise Dropped(str(error)) from error
    if not route.clause_types:
        passages = ws.services.search.search(cache, question)
        generated = yield from _passage_answer(ws, question, passages)
        result = result.model_copy(update={"passages": passages, "generated": generated})
    yield _answer_reply(ws, result, Turn(question=question, clause_types=route.clause_types))


def _passage_answer(
    ws: Workspace, question: str, passages: list[Passage]
) -> Generator[Step, None, GeneratedAnswer | None]:
    """Write an answer from the passages; without a model or when it fails, there is none."""
    gen = ws.services.gen
    if gen is None or not passages:
        return None
    request = passage_request(question, passages)
    try:
        for step in checked(gen, WRITING_ANSWER, request, PASSAGE_ATTEMPTS):
            if isinstance(step, Generation):
                return generated_answer_from(step, passages, gen)
            yield step
    except LLMError:
        logger.warning("answer from the passages failed", exc_info=True)
    return None


def _explanation(ws: Workspace, shown: Shown, language: str) -> Iterator[Step | Reply]:
    for step in explanation_steps(ws, shown.clause_type, shown.quote, language):
        yield _explanation_reply(step, shown) if isinstance(step, Explanation) else step


def _summary(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, language: str
) -> Iterator[Step | Reply]:
    for step in summary_steps(ws, cache, tau, language):
        if isinstance(step, Summary):
            yield _summary_reply(step, ws.services.unvalidated)
        else:
            yield step


def _definition_answer(question: str, definition: Definition, tau: Thresholds) -> Answer:
    route = RouteResult(clause_types=[], kind="definition", latency_ms=0.0)
    return Answer(
        question=question,
        route=route,
        tau_low=tau.low,
        tau_high=tau.high,
        items=[],
        dropped_quotes=0,
        definition=definition,
    )


def _posted(ws: Workspace, cache: DocumentCache, form: ExplainForm) -> Shown:
    """Return the passage a form names; its span is kept only when it holds the quote."""
    # Browsers send the line breaks of form fields as CRLF.
    quote = form.quote.replace("\r\n", "\n")
    citation = None
    start, end = form.start, form.end
    located = start is not None and end is not None and 0 <= start < end
    if located and cache.text[start:end] == quote:
        citation = Citation(quote=quote, page=None, conf=form.conf, start=start, end=end)
    status = form.status if form.status in SHOWN_STATUSES else None
    return Shown(form.label, ws.clause_type_of(form.label), quote, citation, status)


def _first_shown(ws: Workspace, result: Answer | None) -> Shown | None:
    """Return the passage an answer showed first: its first citation, definition or passage."""
    if result is None:
        return None
    cards = _cards(result, ws.services.unvalidated)
    return next((card.shown for card in cards if card.shown is not None), None)


def _cards(result: Answer, unvalidated: frozenset[str]) -> list[Card]:
    if result.definition is not None:
        return [_definition_card(result.definition)]
    if result.items:
        return [_item_card(item, item.clause_type in unvalidated) for item in result.items]
    return [_passage_card(number, p) for number, p in enumerate(result.passages, start=1)]


def _item_card(item: AnswerItem, unvalidated: bool) -> Card:
    name = display_name(item.clause_type)
    tag = STATUS_LABELS[item.status]
    if item.withheld:
        return Card(item.status, tag, name, text=WITHHELD_TEXT)
    citation = item.citation
    if citation is None:
        return Card(item.status, tag, name, text=NOT_FOUND_TEXT)
    meta = CONFIDENCE_TEXT.format(conf=citation.conf)
    if unvalidated:
        meta = f"{UNVALIDATED_TEXT} · {meta}"
    if citation.page is not None:
        meta = f"{PAGE_TEXT.format(page=citation.page)} · {meta}"
    shown = Shown(name, item.clause_type, citation.quote, citation, item.status)
    return Card(item.status, tag, name, quote=_cut(citation.quote), meta=meta, shown=shown)


def _definition_card(definition: Definition) -> Card:
    meta = DEFINITION_NOTE
    if definition.page is not None:
        meta = f"{meta} · {PAGE_TEXT.format(page=definition.page)}"
    citation = definition_citation(definition)
    shown = Shown(DEFINITION_LABEL, DEFINITION_LABEL, definition.quote, citation, "definition")
    return Card(
        "definition",
        DEFINITION_LABEL,
        definition.term,
        quote=_cut(definition.quote),
        meta=meta,
        plain=True,
        shown=shown,
    )


def _passage_card(number: int, passage: Passage) -> Card:
    meta = None if passage.page is None else PAGE_TEXT.format(page=passage.page)
    citation = passage_citation(passage)
    shown = Shown(RELATED_LABEL, RELATED_LABEL, passage.quote, citation, "related")
    title = PASSAGE_TITLE.format(number=number)
    return Card("related", RELATED_TAG, title, quote=_cut(passage.quote), meta=meta, shown=shown)


def _cut(quote: str) -> str:
    """Return a quote cut to the chat limit, as the Gradio chat cuts it."""
    if len(quote) <= MAX_CHAT_QUOTE_CHARS:
        return quote
    return quote[: MAX_CHAT_QUOTE_CHARS - len(ELLIPSIS)] + ELLIPSIS


def _answer_reply(ws: Workspace, result: Answer, turn: Turn | None = None) -> Reply:
    """Return the reply that shows an answer: its cards, the passages or the clause list."""
    cards = _cards(result, ws.services.unvalidated)
    located = [card.shown for card in cards if card.shown is not None]
    context: dict[str, object] = {"cards": cards, "note": _highlight_note(result, len(located))}
    generated = result.generated
    if generated is not None:
        context["generated"] = generated
        context["generated_note"] = (
            GENERATED_ANSWER_META.format(model=generated.model)
            if generated.text is not None
            else GENERATED_ANSWER_WITHHELD.format(reason=generated.reason)
        )
    if result.passages and not result.items:
        context["intro"] = PASSAGES_TEXT
    if not cards:
        context["empty_text"] = EMPTY_ROUTE_TEXT
        context["topics"] = _topics(ws.services.unvalidated)
        context["examples"] = SUGGESTED_QUESTIONS[:EXAMPLE_QUESTIONS]
    shown = located[0] if located else None
    return Reply("chat/answer.html", context, answer=result, turn=turn, shown=shown)


def _highlight_note(result: Answer, located: int) -> str | None:
    if result.definition is not None or not located:
        return None
    return HIGHLIGHT_ONE if located == 1 else HIGHLIGHT_MANY


def _topics(unvalidated: frozenset[str]) -> list[tuple[str, str]]:
    topics = dict(CLAUSE_GROUPS)
    if unvalidated:
        topics[OTHER_TOPIC] = tuple(sorted(unvalidated))
    return [
        (topic, ", ".join(display_name(name) for name in names))
        for topic, names in topics.items()
    ]


def _explanation_reply(explanation: Explanation, shown: Shown) -> Reply:
    context: dict[str, object] = {"shown": shown}
    if explanation.text is None:
        context["withheld"] = EXPLANATION_WITHHELD_TEXT.format(reason=explanation.reason)
    else:
        context.update(
            title=shown.label,
            language=_language_name(explanation.language),
            text=explanation.text,
            meta=EXPLANATION_META_TEXT.format(model=explanation.model),
        )
    return Reply("chat/explanation.html", context, shown=shown)


def _summary_reply(summary: Summary, unvalidated: frozenset[str]) -> Reply:
    if summary.text is None:
        spans = summary.unfaithful_spans
        mismatch = SUMMARY_MISMATCH_TEXT.format(count=len(spans)) if spans else None
        withheld = SUMMARY_WITHHELD_TEXT.format(reason=summary.reason)
        return Reply("chat/summary.html", {"withheld": withheld, "mismatch": mismatch})
    meta = SUMMARY_META_TEXT.format(
        model=summary.model,
        count=len(summary.items),
        seconds=round(summary.latency_ms / MS_PER_S),
    )
    if summary.language != DEFAULT_LANGUAGE:
        meta += SUMMARY_LANGUAGE_TEXT.format(language=_language_name(summary.language))
    based_on = [
        (display_name(item.clause_type), item.status, item.clause_type in unvalidated)
        for item in summary.items
    ]
    context = {
        "text": summary.text,
        "meta": meta,
        "based_on_text": SUMMARY_BASED_ON,
        "based_on": based_on,
        "mark": UNVALIDATED_MARK,
        "unvalidated_note": UNVALIDATED_TEXT if any(flag for *_, flag in based_on) else None,
    }
    return Reply("chat/summary.html", context)


def _text(text: str) -> Reply:
    return Reply("chat/text.html", {"text": text})


def _language_name(code: str) -> str:
    return LANGUAGES.get(code, code).split(",")[0]
