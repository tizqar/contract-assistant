"""Builds the router selected by the settings."""

from __future__ import annotations

import logging
from functools import cache

from contract_assistant.clause_types import ClauseType, load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router.base import Router
from contract_assistant.router.clients import (
    LLMClient,
    LLMError,
    OllamaClient,
    OpenAICompatClient,
)
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


@cache
def load_encoder(model_id: str) -> SentenceTransformerEncoder:
    """Load the embedding model once per model id."""
    return SentenceTransformerEncoder(model_id)


def build_embedding_router(
    settings: Settings, clause_types: list[ClauseType]
) -> EmbeddingRouter:
    """Build the embedding router over the given clause types."""
    return EmbeddingRouter(
        load_encoder(settings.embedding_model_id),
        _descriptions(settings, clause_types),
        settings.embedding_threshold,
        EMBEDDING_MAX_TYPES,
    )


def _descriptions(settings: Settings, clause_types: list[ClauseType]) -> dict[str, list[str]]:
    cuad_names = {
        clause_type.name for clause_type in load_clause_types(settings.clause_types_path)
    }
    cuad = [clause_type for clause_type in clause_types if clause_type.name in cuad_names]
    extra = [clause_type for clause_type in clause_types if clause_type.name not in cuad_names]
    by_name = load_descriptions(cuad, settings.router_descriptions_path)
    if extra:
        by_name |= load_descriptions(extra, settings.extra_router_descriptions_path)
    return {clause_type.name: by_name[clause_type.name] for clause_type in clause_types}


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


def build_router(settings: Settings, clause_types: list[ClauseType]) -> Router:
    """Build the router that LLM_BACKEND selects, with the embedding router as fallback."""
    client = build_llm_client(settings)
    embedding = build_embedding_router(settings, clause_types)
    if client is None:
        return embedding
    return LLMRouter(client, embedding, clause_types)


def _ollama_client(settings: Settings) -> OllamaClient | None:
    client = OllamaClient.from_settings(settings)
    if not client.available():
        logger.warning(
            "ollama model %s is not available; using the embedding router", client.model
        )
        return None
    try:
        client.load()
    except LLMError as error:
        logger.warning("%s; using the embedding router", error)
        return None
    return client
