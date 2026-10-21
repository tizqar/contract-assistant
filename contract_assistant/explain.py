"""A plain-language restatement of one clause quote, and the checks that keep it to the clause."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from contract_assistant import generation
from contract_assistant.generation import Generation, Request, faithful_check, run
from contract_assistant.phrase import unsupported_numbers as _unsupported_numbers
from contract_assistant.router.clients import LLMClient

logger = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "sr": "Serbian, in Latin script"}
DEFAULT_LANGUAGE = "en"
MAX_TOKENS = 160
MAX_SENTENCES = 2
SOURCE = "clause"
MISQUOTE_REASON = generation.MISQUOTE_REASON.format(source=SOURCE)
NUMBERS_REASON = generation.NUMBERS_REASON.format(source=SOURCE, numbers="{numbers}")
EMPTY_REASON = generation.EMPTY_REASON
SYSTEM_PROMPT = (
    "Restate the contract clause given by the user in plain {language}, in at most "
    f"{MAX_SENTENCES} sentences, for a reader who is not a lawyer. "
    "Say only what the clause itself states, in other words. Do not add parties, duties, "
    "consequences, numbers or dates, do not say what the clause means for anyone, "
    "and do not use double quotes. Reply with the restatement only."
)


class UnknownLanguageError(ValueError):
    """Raised when the language is not one the explanation supports."""


class Explanation(BaseModel):
    """One clause quote restated in plain language, or withheld with the reason."""

    clause_type: str
    quote: str
    language: str
    model: str
    text: str | None
    reason: str | None = None
    latency_ms: float


def unsupported_numbers(text: str, quote: str) -> list[str]:
    """Return the numbers written in text that the quote does not contain, in order."""
    return _unsupported_numbers(text, [quote])


def explanation_request(clause_type: str, quote: str, language: str) -> Request:
    """Return the request that restates the quote in the language, with its check."""
    if language not in LANGUAGES:
        raise UnknownLanguageError(
            f"unknown language {language!r}; use one of {', '.join(LANGUAGES)}"
        )
    return Request(
        SYSTEM_PROMPT.format(language=LANGUAGES[language]),
        f'Clause type: {clause_type}\nClause: "{" ".join(quote.split())}"',
        MAX_TOKENS,
        faithful_check([quote], SOURCE),
        SOURCE,
        f"explanation of {clause_type}",
    )


def explanation_from(
    result: Generation, clause_type: str, quote: str, language: str, client: LLMClient
) -> Explanation:
    """Return the Explanation that a generation outcome gives for the quote."""
    return Explanation(
        clause_type=clause_type,
        quote=quote,
        language=language,
        model=client.model,
        text=result.text,
        reason=result.reason,
        latency_ms=result.latency_ms,
    )


def request_explanation(
    clause_type: str, quote: str, language: str, client: LLMClient
) -> Explanation:
    """Ask the model to restate the quote; the text is withheld when it adds to the clause."""
    result = run(client, explanation_request(clause_type, quote, language))
    return explanation_from(result, clause_type, quote, language, client)
