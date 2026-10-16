"""Router result and the interface every router implements."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

RouteKind = Literal["explicit", "oracle", "embedding", "llm", "llm_fallback"]


class RouteResult(BaseModel):
    """The clause types chosen for a question and how they were chosen."""

    clause_types: list[str]
    kind: RouteKind
    latency_ms: float
    raw: str | None = None


class Router(Protocol):
    """Maps a question to clause type names."""

    kind: RouteKind

    def route(self, question: str) -> RouteResult: ...
