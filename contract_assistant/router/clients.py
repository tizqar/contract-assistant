"""Clients for the language models behind the LLM router."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import httpx
import ollama

from contract_assistant.config import Settings

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)

TEMPERATURE = 0
# A failed call is not retried; the router falls back instead.
MAX_RETRIES = 0
SCHEMA_NAME = "response"
DEFAULT_TAG = "latest"
KEEP_ALIVE = "30m"
# A cold load takes longer than the per-question timeout.
LOAD_TIMEOUT_S = 120
# ollama raises the builtin ConnectionError when the server cannot be reached.
OLLAMA_ERRORS = (ollama.ResponseError, httpx.HTTPError, ConnectionError)


class LLMError(RuntimeError):
    """Raised for any transport, timeout or HTTP failure of an LLM call."""


class LLMClient(Protocol):
    """A chat model that answers with JSON constrained by a schema."""

    model: str

    def complete_json(self, system: str, user: str, schema: dict) -> str: ...


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OllamaClient:
    """A model served by an Ollama server."""

    def __init__(
        self, client: ollama.Client, model: str, load_client: ollama.Client | None = None
    ):
        self.client = client
        self.model = model
        self.load_client = load_client or client

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        """Build the client from the settings; OLLAMA_HOST selects the server."""
        return cls(
            ollama.Client(timeout=settings.llm_timeout_s),
            settings.llm_model,
            ollama.Client(timeout=LOAD_TIMEOUT_S),
        )

    def load(self) -> None:
        """Load the model into memory and keep it there."""
        try:
            self.load_client.generate(model=self.model, prompt="", keep_alive=KEEP_ALIVE)
        except OLLAMA_ERRORS as error:
            raise LLMError(f"ollama load of {self.model} failed: {error!r}") from error

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return the model's reply as JSON text that follows the schema."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=_messages(system, user),
                format=schema,
                think=False,
                options={"temperature": TEMPERATURE},
                keep_alive=KEEP_ALIVE,
            )
        except OLLAMA_ERRORS as error:
            raise LLMError(f"ollama request to {self.model} failed: {error!r}") from error
        content = response.message.content
        if not content:
            raise LLMError(f"ollama reply from {self.model} has no content")
        return content

    def available(self) -> bool:
        """Whether the server answers and lists the model."""
        try:
            listed = {entry.model for entry in self.client.list().models}
        except OLLAMA_ERRORS as error:
            logger.warning("cannot list the ollama models: %r", error)
            return False
        return self.model in listed or f"{self.model}:{DEFAULT_TAG}" in listed


class OpenAICompatClient:
    """A model behind an OpenAI-compatible chat completions endpoint."""

    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatClient:
        """Build the client from the settings."""
        import openai

        client = openai.OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=MAX_RETRIES,
        )
        return cls(client, settings.llm_model)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return the model's reply as JSON text that follows the schema."""
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_messages(system, user),
                temperature=TEMPERATURE,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": SCHEMA_NAME, "schema": schema, "strict": True},
                },
            )
        except openai.APIError as error:
            raise LLMError(f"openai request to {self.model} failed: {error!r}") from error
        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"openai reply from {self.model} has no content")
        return content
