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
