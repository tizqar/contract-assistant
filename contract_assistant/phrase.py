"""The phrasing request to an LLM and the check that its quoted spans are verified quotes."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from contract_assistant.answer import STATUS_WORDS, Answer, AnswerItem
from contract_assistant.router.clients import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Write one short paragraph that answers the question from the items given. "
    "Quote the contract only with the quotes given, word for word, in double quotes, "
    "and use double quotes for nothing else. "
    "Do not add facts. Say plainly when a clause type was not found."
)
PHRASE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paragraph"],
    "properties": {"paragraph": {"type": "string"}},
}
QUOTED_SPAN = re.compile(r'"([^"]*)"|“([^”]*)”')
# A number written in words with its digits, "six (6)", counts as one number: contracts write
# numbers that way and a model mixes the pairs up ("six (10)") more often than it invents digits.
NUMBER = re.compile(
    r"[^\W\d_][\w-]*(?: (?:and|i) [^\W\d_][\w-]*)? \(\d+(?:[.,]\d+)*\)|\d+(?:[.,]\d+)*"
)
PAIR = re.compile(r"^([^\W\d_][\w-]*(?: (?:and|i) [^\W\d_][\w-]*)?) \((\d+(?:[.,]\d+)*)\)$")
UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
HUNDRED = "hundred"
# Serbian number words in Latin script, with the inflections a summary uses.
SERBIAN_UNITS = {
    "nula": 0, "jedan": 1, "jedna": 1, "jedno": 1, "jednog": 1, "jedne": 1, "dva": 2, "dve": 2,
    "dvije": 2, "tri": 3, "četiri": 4, "cetiri": 4, "pet": 5, "šest": 6, "sest": 6, "sedam": 7,
    "osam": 8, "devet": 9, "deset": 10, "jedanaest": 11, "dvanaest": 12, "trinaest": 13,
    "četrnaest": 14, "cetrnaest": 14, "petnaest": 15, "šesnaest": 16, "sesnaest": 16,
    "sedamnaest": 17, "osamnaest": 18, "devetnaest": 19,
}  # fmt: skip
SERBIAN_TENS = {
    "dvadeset": 20, "trideset": 30, "četrdeset": 40, "cetrdeset": 40, "pedeset": 50,
    "šezdeset": 60, "sezdeset": 60, "sedamdeset": 70, "osamdeset": 80, "devedeset": 90,
}  # fmt: skip
SERBIAN_HUNDRED = "sto"
CONJUNCTIONS = frozenset({"and", "i"})


class PhraseReply(BaseModel):
    """The model's reply: one non-blank paragraph."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paragraph: str = Field(min_length=1)


def cited_quotes(answer: Answer) -> list[str]:
    """Return the verified quotes of the items that kept their citation."""
    return [item.citation.quote for item in answer.items if item.citation is not None]


def quoted_spans(paragraph: str) -> list[str]:
    """Return the non-blank text between double quotes, straight or curly."""
    spans = (straight or curly for straight, curly in QUOTED_SPAN.findall(paragraph))
    return [span for span in spans if span.strip()]


def unfaithful_spans(paragraph: str, quotes: Sequence[str]) -> list[str]:
    """Return the quoted spans that are not word for word part of one of the quotes."""
    collapsed = [_collapse(quote) for quote in quotes]
    return [span for span in quoted_spans(paragraph) if not _is_part_of(span, collapsed)]


def unsupported_numbers(text: str, quotes: Sequence[str]) -> list[str]:
    """Return the numbers written in text that the quotes do not support, in order.

    A pair of a word and digits is supported when the digits are in a quote and the word,
    if it is an English number, says the same number; "six (10)" is never supported.
    """
    lowered = [_collapse(quote).lower() for quote in quotes]
    return [number for number in NUMBER.findall(text) if not _supported(number, lowered)]


def _supported(number: str, quotes: Sequence[str]) -> bool:
    if any(number.lower() in quote for quote in quotes):
        return True
    pair = PAIR.match(number)
    if pair is None:
        return False
    word, digits = pair.group(1), pair.group(2)
    spelled = spelled_number(word)
    if spelled is not None and spelled != _as_number(digits):
        return False
    return any(digits in quote for quote in quotes)


def spelled_number(words: str) -> int | None:
    """Return the number an English or Serbian number phrase names, or None if it is not one."""
    total = 0
    for word in re.split(r"[\s-]+", words.lower().strip()):
        if word in CONJUNCTIONS:
            continue
        if word in UNITS or word in SERBIAN_UNITS:
            total += UNITS.get(word, SERBIAN_UNITS.get(word, 0))
        elif word in TENS or word in SERBIAN_TENS:
            total += TENS.get(word, SERBIAN_TENS.get(word, 0))
        elif word in (HUNDRED, SERBIAN_HUNDRED):
            total = (total or 1) * 100
        else:
            return None
    return total


def english_number(words: str) -> int | None:
    """Return the number an English number phrase names; kept for callers of the old name."""
    return spelled_number(words)


def _as_number(digits: str) -> float | None:
    try:
        return float(digits.replace(",", ""))
    except ValueError:
        return None


def _is_part_of(span: str, collapsed_quotes: Sequence[str]) -> bool:
    needle = _collapse(span)
    return any(needle in quote for quote in collapsed_quotes)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _describe(item: AnswerItem) -> str:
    status = STATUS_WORDS[item.status]
    if item.withheld:
        return f"- {item.clause_type} ({status}, quote withheld)"
    if item.citation is None:
        return f"- {item.clause_type} ({status})"
    return f'- {item.clause_type} ({status}): "{item.citation.quote}"'


def _user_message(answer: Answer) -> str:
    items = "\n".join(_describe(item) for item in answer.items)
    return f"Question: {answer.question}\n\nItems:\n{items}"


def request_paragraph(answer: Answer, client: LLMClient) -> str | None:
    """Ask the model for a paragraph; None when its reply does not follow the schema."""
    if not answer.items:
        raise ValueError("answer has no items to phrase")
    raw = client.complete_json(SYSTEM_PROMPT, _user_message(answer), PHRASE_SCHEMA)
    try:
        return PhraseReply.model_validate_json(raw).paragraph
    except ValidationError:
        logger.warning("phrasing reply from %s does not follow the schema", client.model)
        return None
