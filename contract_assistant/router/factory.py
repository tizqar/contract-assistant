"""Builds the router selected by the settings."""

from __future__ import annotations

import logging

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router.base import Router
from contract_assistant.router.clients import LLMClient, OllamaClient, OpenAICompatClient
from contract_assistant.router.embedding import (
    EMBEDDING_MAX_TYPES,
    EmbeddingRouter,
    SentenceTransformerEncoder,
    load_descriptions,
)
from contract_assistant.router.llm import LLMRouter

logger = logging.getLogger(__name__)

BACKEND_OLLAMA = "ollama"
BACKEND_OPENAI = "openai"
BACKEND_NONE = "none"
BACKENDS = (BACKEND_OLLAMA, BACKEND_OPENAI, BACKEND_NONE)


class UnknownBackendError(ValueError):
    """Raised when LLM_BACKEND is not a supported backend."""


def build_embedding_router(settings: Settings) -> EmbeddingRouter:
    """Build the embedding router from the settings."""
    clause_types = load_clause_types(settings.clause_types_path)
    descriptions = load_descriptions(clause_types, settings.router_descriptions_path)
    return EmbeddingRouter(
        SentenceTransformerEncoder(settings.embedding_model_id),
        descriptions,
        settings.embedding_threshold,
        EMBEDDING_MAX_TYPES,
    )


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Build the client of LLM_BACKEND; None without a backend or when Ollama lacks the model."""
    backend = settings.llm_backend
    if backend == BACKEND_OPENAI:
        return OpenAICompatClient.from_settings(settings)
    if backend == BACKEND_OLLAMA:
        return _ollama_client(settings)
    if backend == BACKEND_NONE:
        return None
    raise UnknownBackendError(
        f"unknown LLM_BACKEND {backend!r}; use one of {', '.join(BACKENDS)}"
    )


def build_router(settings: Settings) -> Router:
    """Build the router that LLM_BACKEND selects, with the embedding router as fallback."""
    client = build_llm_client(settings)
    embedding = build_embedding_router(settings)
    if client is None:
        return embedding
    clause_types = load_clause_types(settings.clause_types_path)
    return LLMRouter(client, embedding, clause_types)


def _ollama_client(settings: Settings) -> OllamaClient | None:
    client = OllamaClient.from_settings(settings)
    if client.available():
        return client
    logger.warning("ollama model %s is not available; using the embedding router", client.model)
    return None
