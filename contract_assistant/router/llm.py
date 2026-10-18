"""LLM router: a language model picks clause types from the closed list."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from contract_assistant.clause_types import ClauseType
from contract_assistant.router.base import Router, RouteResult
from contract_assistant.router.clients import LLMClient, LLMError

logger = logging.getLogger(__name__)

MS_PER_S = 1000
TASK = (
    "Map the user's question about a contract to the clause types it asks about, "
    "choosing only from the list below. "
    "Return an empty list when none applies, and never answer the question itself."
)
EXAMPLES = (
    ("Is there a minimum purchase obligation?", ["Minimum Commitment"]),
    (
        "Who owns the IP developed under the contract, and is any of it owned jointly?",
        ["IP Ownership Assignment", "Joint IP Ownership"],
    ),
    ("How many pages does this contract have?", []),
)


def route_schema(names: list[str]) -> dict:
    """JSON schema of a reply: an object holding a list of clause type names."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["clause_types"],
        "properties": {
            "clause_types": {"type": "array", "items": {"type": "string", "enum": names}},
        },
    }


def _build_reply_model(names: list[str]) -> type[BaseModel]:
    return create_model(
        "RouterReply",
        __config__=ConfigDict(extra="forbid"),
        clause_types=(list[Literal[*names]], ...),
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _example(question: str, names: list[str]) -> str:
    return f"Question: {question}\nOutput: {json.dumps({'clause_types': names})}"


def _system_prompt(clause_types: list[ClauseType]) -> str:
    listing = "\n".join(f"- {c.name}: {_one_line(c.description)}" for c in clause_types)
    examples = "\n\n".join(_example(question, names) for question, names in EXAMPLES)
    return f"{TASK}\n\nClause types:\n{listing}\n\nExamples:\n\n{examples}"


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * MS_PER_S


class LLMRouter:
    """Routes a question with a language model and falls back to another router on failure."""

    kind = "llm"

    def __init__(self, client: LLMClient, fallback: Router, clause_types: list[ClauseType]):
        names = [clause_type.name for clause_type in clause_types]
        self.client = client
        self.fallback = fallback
        self.system = _system_prompt(clause_types)
        self.schema = route_schema(names)
        self._reply_model = _build_reply_model(names)

    def route(self, question: str) -> RouteResult:
        """Return the model's clause types, or the fallback's route if the reply fails."""
        start = time.perf_counter()
        try:
            raw = self.client.complete_json(self.system, question, self.schema)
            reply = self._reply_model.model_validate_json(raw)
        except (LLMError, ValidationError) as error:
            return self._fall_back(question, error, start)
        # Ollama's grammar does not enforce uniqueItems.
        clause_types = list(dict.fromkeys(reply.clause_types))
        return RouteResult(
            clause_types=clause_types, kind="llm", latency_ms=_elapsed_ms(start), raw=raw
        )

    def _fall_back(self, question: str, error: Exception, start: float) -> RouteResult:
        error_class = type(error).__name__
        logger.warning("llm router fell back: model %s, %s", self.client.model, error_class)
        route = self.fallback.route(question)
        return RouteResult(
            clause_types=route.clause_types,
            kind="llm_fallback",
            latency_ms=_elapsed_ms(start),
            raw=error_class,
        )
