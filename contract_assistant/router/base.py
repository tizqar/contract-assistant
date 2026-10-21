"""Router result and the interface every router implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel

RouteKind = Literal["explicit", "oracle", "embedding", "llm", "llm_fallback", "definition"]


class Turn(BaseModel):
    """An earlier question of the conversation and the clause types it was routed to."""

    question: str
    clause_types: list[str]


class RouteResult(BaseModel):
    """The clause types chosen for a question and how they were chosen."""

    clause_types: list[str]
    kind: RouteKind
    latency_ms: float
    raw: str | None = None


class Router(Protocol):
    """Maps a question to clause type names, with the earlier turns of the conversation."""

    kind: RouteKind

    def route(self, question: str, turns: Sequence[Turn] = ()) -> RouteResult: ...
