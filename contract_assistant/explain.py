"""A plain-language restatement of one clause quote, and the checks that keep it to the clause."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel

from contract_assistant.phrase import unfaithful_spans
from contract_assistant.phrase import unsupported_numbers as _unsupported_numbers
from contract_assistant.router.clients import LLMClient

logger = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "sr": "Serbian, in Latin script"}
DEFAULT_LANGUAGE = "en"
MAX_TOKENS = 160
MAX_SENTENCES = 2
MS_PER_S = 1000
MISQUOTE_REASON = "it quotes text that is not in the clause"
NUMBERS_REASON = "it names numbers the clause does not: {numbers}"
EMPTY_REASON = "the model gave an empty reply"
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


def request_explanation(
    clause_type: str, quote: str, language: str, client: LLMClient
) -> Explanation:
    """Ask the model to restate the quote; the text is withheld when it adds to the clause."""
    if language not in LANGUAGES:
        raise UnknownLanguageError(
            f"unknown language {language!r}; use one of {', '.join(LANGUAGES)}"
        )
    system = SYSTEM_PROMPT.format(language=LANGUAGES[language])
    user = f'Clause type: {clause_type}\nClause: "{" ".join(quote.split())}"'
    start = time.perf_counter()
    text = client.complete_text(system, user, MAX_TOKENS).strip()
    latency_ms = (time.perf_counter() - start) * MS_PER_S
    reason = _reason(text, quote)
    if reason:
        logger.warning("explanation of %s by %s withheld: %s", clause_type, client.model, reason)
    return Explanation(
        clause_type=clause_type,
        quote=quote,
        language=language,
        model=client.model,
        text=None if reason else text,
        reason=reason,
        latency_ms=latency_ms,
    )


def _reason(text: str, quote: str) -> str | None:
    if not text:
        return EMPTY_REASON
    if unfaithful_spans(text, [quote]):
        return MISQUOTE_REASON
    numbers = unsupported_numbers(text, quote)
    if numbers:
        return NUMBERS_REASON.format(numbers=", ".join(numbers))
    return None
