"""Chat commands recognised in a question: explain the last passage, or summarize the contract."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from contract_assistant.explain import DEFAULT_LANGUAGE

IntentKind = Literal["explain", "summary"]

EXPLAIN = re.compile(
    r"\b(explain|clarify|in other words|plain (english|language|serbian)|objasni|pojasni)\b",
    re.IGNORECASE,
)
SERBIAN = re.compile(r"\b(serbian|srpski|srpskom|srpskome|na srpskom)\b", re.IGNORECASE)
REFERENCE = re.compile(
    r"\b(that|this|it|these|those|above|last|previous|the (passage|clause|answer|quote))\b",
    re.IGNORECASE,
)
SUMMARY = re.compile(
    r"\b(summar(y|ize|ise|ise|ies)|overview|sum up|sa[žz]etak|rezime|ukratko)\b", re.IGNORECASE
)
WHOLE = re.compile(
    r"\b(contract|agreement|document|deal|whole|everything|all|it|this|ugovor)\b", re.IGNORECASE
)
# A short question with the command word refers back to what was shown; a summary command
# must be shorter still or name the whole contract, so "summarize the termination clause" routes.
SHORT_QUESTION_WORDS = 6
SHORT_SUMMARY_WORDS = 3
SERBIAN_CODE = "sr"


class Intent(BaseModel):
    """A command found in a question and the language it asks for."""

    kind: IntentKind
    language: str = DEFAULT_LANGUAGE


def detect_intent(question: str) -> Intent | None:
    """Return the command a question gives, or None when it is a question to route."""
    short = len(question.split()) <= SHORT_QUESTION_WORDS
    language = SERBIAN_CODE if SERBIAN.search(question) else DEFAULT_LANGUAGE
    if EXPLAIN.search(question) and (short or REFERENCE.search(question)):
        return Intent(kind="explain", language=language)
    very_short = len(question.split()) <= SHORT_SUMMARY_WORDS
    if SUMMARY.search(question) and (very_short or WHOLE.search(question)):
        return Intent(kind="summary", language=language)
    return None
