"""The chat about one contract: questions, definitions, chat commands, explanations, feedback.

Every reply is kept with what drew it and is drawn again whenever the chat is shown, so a card
follows the reader's later decisions (Confirmed by you, Rejected by you) and the thumbs follow
the feedback given. The chat always renders into the element `#chat-body`, whichever form asked.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Generator, Iterator
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
from contract_assistant.decisions import Decision
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
from contract_assistant.summary import Summary
from contract_assistant.ui.views import (
    CLAUSE_GROUPS,
    EXAMPLE_QUESTIONS,
    NOT_FOUND_TEXT,
    OTHER_TOPIC,
    SUGGESTED_QUESTIONS,
    WITHHELD_TEXT,
    definition_citation,
    display_name,
    passage_citation,
)
from contract_assistant.web.core import (
    DEFINITION_LABEL,
    RELATED_LABEL,
    SUMMARIES_OFF,
    Conversation,
    Message,
    Tau,
    Thresholds,
    Workspace,
    WorkspaceError,
    Ws,
    frame,
    language,
    render_string,
    stream,
    view_url,
)
from contract_assistant.web.generate import (
    NOTHING_TO_SUMMARIZE,
    WRITING_EXPLANATION,
    WRITING_SUMMARY,
    Step,
    checked,
    explanation_steps,
    summary_steps,
)
from contract_assistant.web.i18n import ENGLISH, translate

logger = logging.getLogger(__name__)

TYPE_A_QUESTION = "Type a question first."
THINKING = "Reading the contract"
WRITING_ANSWER = "Writing the answer"
NOTHING_TO_EXPLAIN = "Ask about a clause first, then say 'explain that'."
NO_QUOTE = "There is no passage to explain."
UNKNOWN_LANGUAGE = "Explanations are written in English or Serbian."
# The reader's bubble for an Explain button; the language is named when it is not the page's.
EXPLAIN_ASKED = "Explain: {label}"
EXPLAIN_ASKED_IN = {"en": "Explain in English: {label}", "sr": "Explain in Serbian: {label}"}
EXPLANATION_HEADING = {"en": "{label} in plain English", "sr": "{label} in plain Serbian"}
# A summary in the language of the page is the Summary; another one names its language.
SUMMARY_HEADING = "Summary"
SUMMARY_HEADING_IN = {"en": "Summary in English", "sr": "Summary in Serbian"}
# English named in a chat command; detect_intent recognises Serbian, and says English otherwise.
ENGLISH_NAMED = re.compile(r"\b(english|engleski|engleskog|engleskom|engleskome)\b", re.IGNORECASE)
BASED_ON = "Based on"
BASED_ON_MORE = "and {count} more"
# The clauses named under a summary; the overview lists all of them.
BASED_ON_SHOWN = 6
PASSAGES_INTRO = (
    "This is not one of the clauses the assistant checks, "
    "so here are the closest passages of the contract."
)
NO_MATCH = (
    "This question is not about one of the clauses the assistant checks, "
    "and no passage of the contract comes close to it."
)
TOPICS = "You can ask about:"
EXAMPLES = "For example:"
EMPTY_TITLE = "Ask anything about this contract: who signed it, when it ends, how to cancel it."
EMPTY_TEXT = "Every answer quotes the contract word for word, so you can check it."
FEEDBACK_THANKS = "Noted, thank you."
ANSWER_GONE = "This answer is no longer in the chat."
# Plain words for the errors the chat shows as a reply.
AI_OFF = "Summaries and explanations need the local AI, which is off on this computer."
NO_ANSWER = "The assistant did not answer. Please try again."
SOMETHING_WRONG = "Something went wrong. Please try again."
PLAIN_ERRORS = {SUMMARIES_OFF: AI_OFF}

CHAT_BODY = "chat-body"
PENDING_ID = "chat-pending"
SHOWN_STATUSES = frozenset({"found", "check", "definition", "related"})

router = APIRouter()

# Streams run in worker threads: a reply is added, drawn and given its feedback index under it.
_lock = threading.Lock()

@dataclass(frozen=True)
class Shown:
    """A passage the chat quotes: its label, the clause type behind it, its span and badge."""

    label: str
    clause_type: str
    quote: str
    citation: Citation | None = None
    status: str | None = None

    def view(self, doc_id: str, status: str | None = None) -> str | None:
        """Return the URL of the document pane at the passage, or None if its span is unknown."""
        if self.citation is None:
            return None
        return view_url(doc_id, self.label, self.citation, status or self.status)


# What a passage's badge shows now, given the passage and the status it was answered with.
StatusOf = Callable[[Shown | None, str | None], str | None]


@dataclass(frozen=True)
class Card:
    """One card of an answer: its status, a title, a quote or a short text, and the page."""

    kind: str
    title: str | None = None
    quote: str | None = None
    text: str | None = None
    page: int | None = None
    shown: Shown | None = None


@dataclass(frozen=True)
class Reply:
    """The assistant's bubble: the template that draws it and what the chat keeps of it."""

    template: str
    context: dict = field(default_factory=dict)
    answer: Answer | None = None
    turn: Turn | None = None
    # The passage the document pane moves to once the reply is shown.
    shown: Shown | None = None


@dataclass(frozen=True)
class Asked:
    """What the reader asked: their own words, or the passage and language of an Explain button."""

    text: str = ""
    label: str | None = None
    language: str | None = None

    def words(self, lang: str) -> str:
        """Return the reader's bubble in the language of the page; typed words stay as typed."""
        if self.label is None:
            return self.text
        wording = EXPLAIN_ASKED
        if self.language != lang:
            wording = EXPLAIN_ASKED_IN.get(self.language or "", EXPLAIN_ASKED)
        return translate(wording, lang).format(label=translate(self.label, lang))


@dataclass
class Said(Message):
    """A user message with what was asked, drawn again in the language of the page."""

    asked: Asked = field(default_factory=Asked)


@dataclass
class Bubble(Message):
    """An assistant message with the reply behind it, drawn again whenever the chat is shown."""

    reply: Reply | None = None
    index: int = 0
    thanked: bool = False


class Dropped(Exception):
    """Ends an exchange without a reply; the question is dropped and the message shown."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ExplainForm(BaseModel):
    """A passage to explain: its label and quote, the language, and its span when known."""

    label: str
    quote: str
    # The language of the page when the form names none.
    language: str | None = None
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
    return HTMLResponse(_body(request, ws, cache))


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
    with _lock:
        messages = conversation.messages
        message = messages[index] if 0 <= index < len(messages) else None
        if not isinstance(message, Bubble) or message.answer is None:
            raise WorkspaceError(translate(ANSWER_GONE, language(request)))
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
        message.thanked = True
    return HTMLResponse(render_string(request, "chat/thanks.html", thanks=FEEDBACK_THANKS))


@router.post("/documents/{doc_id}/chat/clear", response_class=HTMLResponse)
def clear_chat(request: Request, doc_id: str, ws: Ws):
    """Forget the chat about a contract: its messages, the turns and the last answer."""
    cache = ws.load(doc_id)
    conversation = ws.conversation(cache.doc_id)
    with _lock:
        conversation.messages = []
        conversation.turns = []
        conversation.last = None
    return HTMLResponse(_body(request, ws, cache))


def _ask(
    request: Request, ws: Workspace, cache: DocumentCache, tau: Thresholds, question: str
) -> Iterator[str]:
    lang = language(request)
    if not question:
        yield frame(notice=translate(TYPE_A_QUESTION, lang), done=True)
        return
    replies = _replies(ws, cache, tau, question, lang)
    yield from _converse(request, ws, cache, Asked(question), replies)


def _explain(
    request: Request, ws: Workspace, cache: DocumentCache, form: ExplainForm
) -> Iterator[str]:
    lang = language(request)
    chosen = form.language or lang
    if chosen not in LANGUAGES:
        yield frame(notice=translate(UNKNOWN_LANGUAGE, lang), done=True)
        return
    if not form.quote.strip():
        yield frame(notice=translate(NO_QUOTE, lang), done=True)
        return
    asked = Asked(label=form.label, language=chosen)
    replies = _explanation(ws, _posted(ws, cache, form), chosen)
    yield from _converse(request, ws, cache, asked, replies, inline=form.inline)


def _converse(
    request: Request,
    ws: Workspace,
    cache: DocumentCache,
    asked: Asked,
    replies: Iterator[Step | Reply],
    inline: bool = False,
) -> Iterator[str]:
    """Show the question with a status at once, the model's text as it comes, then the reply.

    The question and its reply are added to the chat together at the end, so a stopped
    exchange leaves nothing behind. Inline, the first frame only fills the pending place, so
    the button that asked stays on the page while the reply is written.
    """
    conversation = ws.conversation(cache.doc_id)
    said = Said("user", _said(request, asked), asked=asked)

    def pending(status: str, text: str = "") -> str:
        return render_string(
            request, "chat/pending.html", user=said.html, status=status, text=text
        )

    if inline:
        yield frame(oob={PENDING_ID: pending(THINKING)})
    else:
        yield frame(oob={CHAT_BODY: _body(request, ws, cache, pending(THINKING))})
    reply = _text(NO_ANSWER)
    try:
        for item in replies:
            if isinstance(item, Reply):
                reply = item
            else:
                yield frame(oob={PENDING_ID: pending(item.what, item.text)})
    except Dropped as dropped:
        body = _body(request, ws, cache)
        yield frame(oob={CHAT_BODY: body}, notice=dropped.message, done=True)
        return
    except WorkspaceError as error:
        reply = _text(PLAIN_ERRORS.get(error.message, error.message))
    except LLMError:
        logger.exception("generation failed in the chat about %s", cache.doc_id)
        reply = _text(NO_ANSWER)
    except Exception:
        logger.exception("chat reply failed for %s", cache.doc_id)
        reply = _text(SOMETHING_WRONG)
    try:
        _add(request, ws, cache, conversation, said, reply)
    except Exception:
        logger.exception("cannot show the reply in the chat about %s", cache.doc_id)
        reply = _text(SOMETHING_WRONG)
        _add(request, ws, cache, conversation, said, reply)
    follow = None
    if reply.shown is not None:
        status_of = _status_of(ws.decided(cache.doc_id))
        follow = reply.shown.view(cache.doc_id, status_of(reply.shown, reply.shown.status))
    yield frame(oob={CHAT_BODY: _body(request, ws, cache, follow=follow)}, done=True)


def _add(
    request: Request,
    ws: Workspace,
    cache: DocumentCache,
    conversation: Conversation,
    said: Said,
    reply: Reply,
) -> None:
    status_of = _status_of(ws.decided(cache.doc_id))
    with _lock:
        index = len(conversation.messages) + 1
        bubble = Bubble("assistant", "", answer=reply.answer, reply=reply, index=index)
        bubble.html = _draw(request, cache, bubble, status_of)
        conversation.messages.extend([said, bubble])
        if reply.turn is not None:
            conversation.remember(reply.turn)
        if reply.answer is not None:
            conversation.last = reply.answer


def _body(
    request: Request,
    ws: Workspace,
    cache: DocumentCache,
    pending: str | None = None,
    follow: str | None = None,
) -> str:
    """Draw the chat: its messages, the reply being written, and the passage to move to."""
    conversation = ws.conversation(cache.doc_id)
    status_of = _status_of(ws.decided(cache.doc_id))
    with _lock:
        for message in conversation.messages:
            if isinstance(message, Bubble):
                message.html = _draw(request, cache, message, status_of)
            elif isinstance(message, Said):
                message.html = _said(request, message.asked)
        messages = list(conversation.messages)
    return render_string(
        request,
        "chat/body.html",
        cache=cache,
        messages=messages,
        pending=pending,
        follow=follow,
        empty_title=EMPTY_TITLE,
        empty_text=EMPTY_TEXT,
    )


def _said(request: Request, asked: Asked) -> str:
    return render_string(request, "chat/user.html", asked=asked)


def _draw(request: Request, cache: DocumentCache, bubble: Bubble, status_of: StatusOf) -> str:
    reply = bubble.reply or _text(SOMETHING_WRONG)
    return render_string(
        request,
        reply.template,
        cache=cache,
        index=bubble.index,
        thanked=bubble.thanked,
        thanks=FEEDBACK_THANKS,
        status_of=status_of,
        **reply.context,
    )


def _status_of(decisions: dict[str, Decision]) -> StatusOf:
    """Return what a badge shows: the reader's verdict on its passage, else the status."""

    def status_of(shown: Shown | None, status: str | None) -> str | None:
        if shown is None or shown.citation is None:
            return status
        decision = decisions.get(shown.clause_type)
        if decision is None:
            return status
        if (decision.start, decision.end) != (shown.citation.start, shown.citation.end):
            return status
        return decision.verdict

    return status_of


def _replies(
    ws: Workspace, cache: DocumentCache, tau: Thresholds, question: str, lang: str
) -> Iterator[Step | Reply]:
    """Answer a defined term, else a chat command, else route the question to clause types.

    A command is answered in the language it names, else in the language of the page.
    """
    conversation = ws.conversation(cache.doc_id)
    definition = lookup(ws.definitions(cache), question)
    if definition is not None:
        yield _answer_reply(ws, _definition_answer(question, definition, tau))
        return
    intent = detect_intent(question)
    if intent is not None and intent.kind == "explain":
        shown = _first_shown(conversation.last)
        if shown is None:
            yield _text(NOTHING_TO_EXPLAIN)
            return
        yield from _explanation(ws, shown, _named_language(question, intent.language, lang))
        return
    if intent is not None:
        yield from _summary(ws, cache, tau, _named_language(question, intent.language, lang))
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


def _named_language(question: str, detected: str, lang: str) -> str:
    """Return the language a command names, else the language of the page."""
    if detected != DEFAULT_LANGUAGE:
        return detected
    return ENGLISH if ENGLISH_NAMED.search(question) else lang


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
        yield _summary_reply(step) if isinstance(step, Summary) else step


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


def _first_shown(result: Answer | None) -> Shown | None:
    """Return the passage an answer showed first: its first citation, definition or passage."""
    if result is None:
        return None
    return next((card.shown for card in _cards(result) if card.shown is not None), None)


def _cards(result: Answer) -> list[Card]:
    if result.definition is not None:
        return [_definition_card(result.definition)]
    if result.items:
        return [_item_card(item) for item in result.items]
    return [_passage_card(passage) for passage in result.passages]


def _item_card(item: AnswerItem) -> Card:
    name = display_name(item.clause_type)
    if item.withheld:
        return Card(item.status, name, text=WITHHELD_TEXT)
    citation = item.citation
    if citation is None:
        return Card(item.status, name, text=NOT_FOUND_TEXT)
    shown = Shown(name, item.clause_type, citation.quote, citation, item.status)
    return Card(item.status, name, quote=_cut(citation.quote), page=citation.page, shown=shown)


def _definition_card(definition: Definition) -> Card:
    citation = definition_citation(definition)
    shown = Shown(DEFINITION_LABEL, DEFINITION_LABEL, definition.quote, citation, "definition")
    quote = _cut(definition.quote)
    return Card("definition", definition.term, quote=quote, page=definition.page, shown=shown)


def _passage_card(passage: Passage) -> Card:
    citation = passage_citation(passage)
    shown = Shown(RELATED_LABEL, RELATED_LABEL, passage.quote, citation, "related")
    return Card("related", quote=_cut(passage.quote), page=passage.page, shown=shown)


def _cut(quote: str) -> str:
    """Return a quote cut to the chat limit, as the Gradio chat cuts it."""
    if len(quote) <= MAX_CHAT_QUOTE_CHARS:
        return quote
    return quote[: MAX_CHAT_QUOTE_CHARS - len(ELLIPSIS)] + ELLIPSIS


def _answer_reply(ws: Workspace, result: Answer, turn: Turn | None = None) -> Reply:
    """Return the reply that shows an answer: its cards, the passages or the topics."""
    cards = _cards(result)
    context: dict[str, object] = {"cards": cards}
    generated = result.generated
    if generated is not None and generated.text is not None:
        context["generated"] = generated.text.strip()
    elif generated is not None:
        context["reason"] = generated.reason
    if result.passages and not result.items:
        context["intro"] = PASSAGES_INTRO
    if not cards:
        context.update(
            no_match=NO_MATCH,
            topics_text=TOPICS,
            topics=_topics(ws.services.unvalidated),
            examples_text=EXAMPLES,
            examples=SUGGESTED_QUESTIONS[:EXAMPLE_QUESTIONS],
        )
    shown = next((card.shown for card in cards if card.shown is not None), None)
    return Reply("chat/answer.html", context, answer=result, turn=turn, shown=shown)


def _topics(unvalidated: frozenset[str]) -> list[tuple[str, list[str]]]:
    topics = dict(CLAUSE_GROUPS)
    if unvalidated:
        topics[OTHER_TOPIC] = tuple(sorted(unvalidated))
    return [(topic, [display_name(name) for name in names]) for topic, names in topics.items()]


# The replies keep the English text; the templates put it in the language of the page, so a
# reply drawn again after the reader switches languages follows the switch.


def _explanation_reply(explanation: Explanation, shown: Shown) -> Reply:
    context: dict[str, object] = {"shown": shown}
    if explanation.text is None:
        context["reason"] = explanation.reason
    else:
        heading = EXPLANATION_HEADING[explanation.language]
        context.update(heading=heading, text=explanation.text)
    return Reply("chat/explanation.html", context, shown=shown)


def _summary_reply(summary: Summary) -> Reply:
    if summary.text is None:
        return Reply("chat/summary.html", {"reason": summary.reason})
    based_on = [(display_name(item.clause_type), item.status) for item in summary.items]
    context = {
        "language": summary.language,
        "heading": SUMMARY_HEADING,
        "heading_in": SUMMARY_HEADING_IN[summary.language],
        "text": summary.text,
        "based_on_text": BASED_ON,
        "based_on": based_on[:BASED_ON_SHOWN],
        "hidden": max(len(based_on) - BASED_ON_SHOWN, 0),
        "based_on_more": BASED_ON_MORE,
    }
    return Reply("chat/summary.html", context)


def _text(text: str) -> Reply:
    return Reply("chat/text.html", {"text": text})


# The English texts this module shows, for the check that the Serbian catalogue covers them.
TRANSLATED = (
    TYPE_A_QUESTION,
    THINKING,
    WRITING_ANSWER,
    WRITING_SUMMARY,
    WRITING_EXPLANATION,
    NOTHING_TO_EXPLAIN,
    NOTHING_TO_SUMMARIZE,
    NO_QUOTE,
    UNKNOWN_LANGUAGE,
    EXPLAIN_ASKED,
    *EXPLAIN_ASKED_IN.values(),
    *EXPLANATION_HEADING.values(),
    SUMMARY_HEADING,
    *SUMMARY_HEADING_IN.values(),
    BASED_ON,
    BASED_ON_MORE,
    PASSAGES_INTRO,
    NO_MATCH,
    TOPICS,
    EXAMPLES,
    *CLAUSE_GROUPS,
    OTHER_TOPIC,
    *SUGGESTED_QUESTIONS[:EXAMPLE_QUESTIONS],
    NOT_FOUND_TEXT,
    WITHHELD_TEXT,
    DEFINITION_LABEL,
    RELATED_LABEL,
    EMPTY_TITLE,
    EMPTY_TEXT,
    FEEDBACK_THANKS,
    ANSWER_GONE,
    AI_OFF,
    NO_ANSWER,
    SOMETHING_WRONG,
)
