"""Clients for the language models behind the LLM router, the summaries and the explanations."""

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
    """A chat model that answers with JSON constrained by a schema, or with free text."""

    model: str

    def complete_json(self, system: str, user: str, schema: dict) -> str: ...

    def complete_text(self, system: str, user: str, max_tokens: int) -> str: ...


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OllamaClient:
    """A model served by an Ollama server."""

    def __init__(
        self,
        client: ollama.Client,
        model: str,
        load_client: ollama.Client | None = None,
        num_ctx: int | None = None,
    ):
        self.client = client
        self.model = model
        self.load_client = load_client or client
        self.num_ctx = num_ctx

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        """Build the router's client from the settings; OLLAMA_HOST selects the server."""
        return cls(
            ollama.Client(timeout=settings.llm_timeout_s),
            settings.llm_model,
            ollama.Client(timeout=LOAD_TIMEOUT_S),
        )

    @classmethod
    def for_generation(cls, settings: Settings) -> OllamaClient:
        """Build the generation client: its own model, a longer timeout and a set context."""
        return cls(
            ollama.Client(timeout=settings.gen_timeout_s),
            settings.gen_model,
            ollama.Client(timeout=LOAD_TIMEOUT_S),
            num_ctx=settings.gen_num_ctx,
        )

    def load(self) -> None:
        """Load the model into memory, with its context size when set, and keep it there."""
        arguments = {"options": {"num_ctx": self.num_ctx}} if self.num_ctx is not None else {}
        try:
            self.load_client.generate(
                model=self.model, prompt="", keep_alive=KEEP_ALIVE, **arguments
            )
        except OLLAMA_ERRORS as error:
            raise LLMError(f"ollama load of {self.model} failed: {error!r}") from error

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return the model's reply as JSON text that follows the schema."""
        return self._chat(system, user, format=schema, options={"temperature": TEMPERATURE})

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply as free text of at most max_tokens tokens."""
        options = {"temperature": TEMPERATURE, "num_predict": max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return self._chat(system, user, options=options)

    def _chat(self, system: str, user: str, **arguments) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=_messages(system, user),
                think=False,
                keep_alive=KEEP_ALIVE,
                **arguments,
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
        """Build the router's client from the settings."""
        return cls._build(settings, settings.llm_model, settings.llm_timeout_s)

    @classmethod
    def for_generation(cls, settings: Settings) -> OpenAICompatClient:
        """Build the generation client: its own model and a longer timeout."""
        return cls._build(settings, settings.gen_model, settings.gen_timeout_s)

    @classmethod
    def _build(cls, settings: Settings, model: str, timeout_s: float) -> OpenAICompatClient:
        import openai

        client = openai.OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=timeout_s,
            max_retries=MAX_RETRIES,
        )
        return cls(client, model)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        """Return the model's reply as JSON text that follows the schema."""
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "schema": schema, "strict": True},
        }
        return self._create(system, user, response_format=response_format)

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply as free text of at most max_tokens tokens."""
        return self._create(system, user, max_tokens=max_tokens)

    def _create(self, system: str, user: str, **arguments) -> str:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_messages(system, user),
                temperature=TEMPERATURE,
                **arguments,
            )
        except openai.APIError as error:
            raise LLMError(f"openai request to {self.model} failed: {error!r}") from error
        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"openai reply from {self.model} has no content")
        return content
