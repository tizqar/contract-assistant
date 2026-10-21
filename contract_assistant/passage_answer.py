"""An answer the model writes from the related passages of a contract, checked against them."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from contract_assistant.answer import GeneratedAnswer, Passage
from contract_assistant.generation import Generation, Request, faithful_check, run
from contract_assistant.phrase import unfaithful_spans
from contract_assistant.router.clients import LLMClient

logger = logging.getLogger(__name__)

PROMPT_VERSION = 1
MAX_TOKENS = 200
MAX_SENTENCES = 3
# The app and the CLI ask again once when a reply is withheld, naming the reason.
APP_ATTEMPTS = 2
SOURCE = "passages"
SYSTEM_PROMPT = (
    "Answer the question from the passages of a contract given, in at most "
    f"{MAX_SENTENCES} sentences. If the passages do not answer it, say so plainly. "
    "Quote the passages only word for word, in double quotes, and use double quotes for "
    "nothing else. Copy every number, date and amount exactly as the passages state it, "
    "and do not add facts the passages do not state. Reply with the answer only."
)


def passage_request(question: str, passages: Sequence[Passage]) -> Request:
    """Return the request that answers the question from the passages, with its check."""
    if not passages:
        raise ValueError("no passages to answer from")
    return Request(
        SYSTEM_PROMPT,
        _user_message(question, passages),
        MAX_TOKENS,
        faithful_check(_quotes(passages), SOURCE),
        SOURCE,
        f"answer to {question!r}",
    )


def generated_answer_from(
    result: Generation, passages: Sequence[Passage], client: LLMClient
) -> GeneratedAnswer:
    """Return the GeneratedAnswer that a generation outcome gives over the passages."""
    return GeneratedAnswer(
        model=client.model,
        text=result.text,
        reason=result.reason,
        unfaithful_spans=unfaithful_spans(result.reply, _quotes(passages)),
        attempts=result.attempts,
        latency_ms=result.latency_ms,
    )


def request_passage_answer(
    question: str, passages: Sequence[Passage], client: LLMClient, attempts: int = 1
) -> GeneratedAnswer:
    """Ask the model to answer from the passages; the text is withheld when it adds to them."""
    result = run(client, passage_request(question, passages), attempts)
    return generated_answer_from(result, passages, client)


def _quotes(passages: Sequence[Passage]) -> list[str]:
    return [passage.quote for passage in passages]


def _user_message(question: str, passages: Sequence[Passage]) -> str:
    lines = [f'- "{" ".join(passage.quote.split())}"' for passage in passages]
    return f"Question: {question}\n\nPassages:\n" + "\n".join(lines)
