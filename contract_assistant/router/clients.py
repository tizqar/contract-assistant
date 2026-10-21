"""Clients for the language models behind the LLM router, the summaries and the explanations."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
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
# Ollama's MLX runner answers a schema-constrained request with 501 and this text.
NO_GRAMMAR_STATUS = 501
NO_GRAMMAR_TEXT = "structured output is unavailable"
SCHEMA_INSTRUCTION = (
    "\n\nReply with one JSON object that follows this JSON schema, and nothing else:\n{schema}"
)
FENCE = re.compile(r"^```[a-z]*\s*|\s*```$")


class LLMError(RuntimeError):
    """Raised for any transport, timeout or HTTP failure of an LLM call."""


class LLMClient(Protocol):
    """A chat model that answers with JSON constrained by a schema, or with free text."""

    model: str

    def complete_json(self, system: str, user: str, schema: dict) -> str: ...

    def complete_text(self, system: str, user: str, max_tokens: int) -> str: ...

    def stream_text(self, system: str, user: str, max_tokens: int) -> Iterator[str]: ...


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def json_object(text: str) -> str:
    """Return the first JSON object in a reply, without code fences; the reply if none."""
    stripped = FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else stripped


def _no_grammar(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return status == NO_GRAMMAR_STATUS or NO_GRAMMAR_TEXT in str(error)


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
        # None until the first schema-constrained call shows whether the runner has a grammar.
        self.grammar: bool | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        """Build the router's client; it shares the context size so one model can serve both."""
        return cls(
            ollama.Client(timeout=settings.llm_timeout_s),
            settings.llm_model,
            ollama.Client(timeout=LOAD_TIMEOUT_S),
            num_ctx=settings.gen_num_ctx,
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
        """Return the model's reply as JSON text that follows the schema.

        A runner without a grammar gets the schema in the prompt instead, and the first
        JSON object of its reply is returned; the caller validates it either way.
        """
        if self.grammar is not False:
            try:
                reply = self._chat(system, user, format=schema, options=self._options())
            except LLMError as error:
                if not _no_grammar(error.__cause__ or error):
                    raise
                logger.warning("%s has no grammar; the schema goes into the prompt", self.model)
                self.grammar = False
            else:
                self.grammar = True
                return reply
        prompted = system + SCHEMA_INSTRUCTION.format(schema=json.dumps(schema))
        return json_object(self._chat(prompted, user, options=self._options()))

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply as free text of at most max_tokens tokens."""
        return self._chat(system, user, options=self._options(num_predict=max_tokens))

    def stream_text(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """Yield the model's free-text reply piece by piece as it is written."""
        try:
            chunks = self.client.chat(
                model=self.model,
                messages=_messages(system, user),
                think=False,
                keep_alive=KEEP_ALIVE,
                stream=True,
                options=self._options(num_predict=max_tokens),
            )
            for chunk in chunks:
                piece = chunk.message.content
                if piece:
                    yield piece
        except OLLAMA_ERRORS as error:
            raise LLMError(f"ollama stream from {self.model} failed: {error!r}") from error

    def _options(self, **extra: int) -> dict[str, int | float]:
        options: dict[str, int | float] = {"temperature": TEMPERATURE, **extra}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options

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

    def stream_text(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """Yield the model's free-text reply piece by piece as it is written."""
        import openai

        try:
            chunks = self.client.chat.completions.create(
                model=self.model,
                messages=_messages(system, user),
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in chunks:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    yield piece
        except openai.APIError as error:
            raise LLMError(f"openai stream from {self.model} failed: {error!r}") from error

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
