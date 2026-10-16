"""Builds the router selected by the settings."""

from __future__ import annotations

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router.base import Router
from contract_assistant.router.embedding import (
    EMBEDDING_MAX_TYPES,
    EmbeddingRouter,
    SentenceTransformerEncoder,
    load_descriptions,
)


def build_router(settings: Settings) -> Router:
    """Build the embedding router from the settings."""
    clause_types = load_clause_types(settings.clause_types_path)
    descriptions = load_descriptions(clause_types, settings.router_descriptions_path)
    return EmbeddingRouter(
        SentenceTransformerEncoder(settings.embedding_model_id),
        descriptions,
        settings.embedding_threshold,
        EMBEDDING_MAX_TYPES,
    )
