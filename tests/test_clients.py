import json
import logging
from types import SimpleNamespace

import httpx
import ollama
import pytest

from contract_assistant.config import Settings
from contract_assistant.router.clients import (
    KEEP_ALIVE,
    LOAD_TIMEOUT_S,
    NO_GRAMMAR_STATUS,
    NO_GRAMMAR_TEXT,
    SCHEMA_INSTRUCTION,
    LLMError,
    OllamaClient,
    OpenAICompatClient,
    json_object,
)

MODEL = "test-model"
SYSTEM = "Map the question to clause types."
USER = "Which law governs this agreement?"
SCHEMA = {
    "type": "object",
    "properties": {"clause_types": {"type": "array", "items": {"type": "string"}}},
    "required": ["clause_types"],
    "additionalProperties": False,
}
REPLY = json.dumps({"clause_types": ["Governing Law"]})
TEXT_REPLY = "The agreement is governed by Delaware law."
MAX_TOKENS = 123
NUM_CTX = 4096


def raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def raise_connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def respond_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="internal error")


FAILURES = [
    pytest.param(raise_timeout, id="timeout"),
    pytest.param(raise_connect_error, id="unreachable"),
    pytest.param(respond_500, id="http-500"),
]


def ollama_client(handler, model: str = MODEL) -> OllamaClient:
    client = ollama.Client(host="http://test", transport=httpx.MockTransport(handler))
    return OllamaClient(client, model)


def chat_reply(content: str | None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    return httpx.Response(200, json={"model": MODEL, "message": message, "done": True})


def tags_reply(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"model": name} for name in names]})


def ollama_request_body() -> dict:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return chat_reply(REPLY)

    ollama_client(handler).complete_json(SYSTEM, USER, SCHEMA)
    return bodies[0]


def test_ollama_returns_the_message_content():
    client = ollama_client(lambda request: chat_reply(REPLY))
    assert client.complete_json(SYSTEM, USER, SCHEMA) == REPLY


def test_ollama_request_names_the_model():
    assert ollama_request_body()["model"] == MODEL


def test_ollama_request_sends_the_schema_as_format():
    assert ollama_request_body()["format"] == SCHEMA


def test_ollama_request_turns_thinking_off():
    assert ollama_request_body()["think"] is False


def test_ollama_request_uses_temperature_zero():
    assert ollama_request_body()["options"]["temperature"] == 0


def test_ollama_request_keeps_the_model_loaded():
    assert ollama_request_body()["keep_alive"] == KEEP_ALIVE


def generate_request_body() -> dict:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"model": MODEL, "response": "", "done": True})

    ollama_client(handler).load()
    [(path, body)] = bodies
    assert path == "/api/generate"
    return body


def test_ollama_load_posts_the_model_with_an_empty_prompt():
    body = generate_request_body()
    assert body["model"] == MODEL
    assert body["prompt"] == ""


def test_ollama_load_sets_keep_alive():
    assert generate_request_body()["keep_alive"] == KEEP_ALIVE


def test_ollama_load_without_a_context_size_sends_no_options():
    assert "options" not in generate_request_body()


def test_ollama_load_of_a_generation_client_sets_its_context_size():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"model": MODEL, "response": "", "done": True})

    client = ollama.Client(host="http://test", transport=httpx.MockTransport(handler))
    OllamaClient(client, MODEL, num_ctx=8192).load()
    assert bodies[0]["options"] == {"num_ctx": 8192}


@pytest.mark.parametrize("handler", FAILURES)
def test_ollama_load_failure_becomes_llm_error_naming_the_model(handler):
    with pytest.raises(LLMError, match=MODEL):
        ollama_client(handler).load()


def test_ollama_request_sends_system_then_user_message():
    assert ollama_request_body()["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]


@pytest.mark.parametrize("handler", FAILURES)
def test_ollama_failure_becomes_llm_error_naming_the_model(handler):
    with pytest.raises(LLMError, match=MODEL):
        ollama_client(handler).complete_json(SYSTEM, USER, SCHEMA)


def test_ollama_llm_error_keeps_the_original_exception_as_cause():
    with pytest.raises(LLMError) as raised:
        ollama_client(raise_timeout).complete_json(SYSTEM, USER, SCHEMA)
    assert isinstance(raised.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize("content", [None, ""], ids=["missing", "empty"])
def test_ollama_reply_without_content_becomes_llm_error(content):
    client = ollama_client(lambda request: chat_reply(content))
    with pytest.raises(LLMError, match=MODEL):
        client.complete_json(SYSTEM, USER, SCHEMA)


def test_ollama_model_in_the_server_list_is_available():
    assert ollama_client(lambda request: tags_reply("other", MODEL)).available() is True


def test_ollama_model_missing_from_the_server_list_is_not_available():
    assert ollama_client(lambda request: tags_reply("other")).available() is False


def test_ollama_model_name_without_tag_matches_the_latest_tag():
    client = ollama_client(lambda request: tags_reply("llama3.2:latest"), model="llama3.2")
    assert client.available() is True


@pytest.mark.parametrize("handler", FAILURES)
def test_ollama_failing_server_is_not_available(handler):
    assert ollama_client(handler).available() is False


def test_ollama_from_settings_uses_the_configured_model_and_timeout(monkeypatch):
    created = []

    def fake_client(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(ollama, "Client", fake_client)
    client = OllamaClient.from_settings(Settings(llm_model="other", llm_timeout_s=3.0))
    assert client.model == "other"
    assert created == [{"timeout": 3.0}, {"timeout": LOAD_TIMEOUT_S}]


def completion(content: str | None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    choice = {"index": 0, "finish_reason": "stop", "message": message}
    body = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": MODEL}
    return httpx.Response(200, json={**body, "choices": [choice]})


def openai_client(handler) -> OpenAICompatClient:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(
        base_url="http://test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return OpenAICompatClient(client, MODEL)


def openai_request_body() -> dict:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return completion(REPLY)

    openai_client(handler).complete_json(SYSTEM, USER, SCHEMA)
    return bodies[0]


def test_openai_returns_the_message_content():
    client = openai_client(lambda request: completion(REPLY))
    assert client.complete_json(SYSTEM, USER, SCHEMA) == REPLY


def test_openai_request_names_the_model():
    assert openai_request_body()["model"] == MODEL


def test_openai_request_uses_temperature_zero():
    assert openai_request_body()["temperature"] == 0


def test_openai_request_asks_for_strict_json_schema_output():
    response_format = openai_request_body()["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_openai_request_carries_the_schema():
    assert openai_request_body()["response_format"]["json_schema"]["schema"] == SCHEMA


def test_openai_request_sends_system_then_user_message():
    assert openai_request_body()["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]


@pytest.mark.parametrize("handler", FAILURES)
def test_openai_failure_becomes_llm_error_naming_the_model(handler):
    with pytest.raises(LLMError, match=MODEL):
        openai_client(handler).complete_json(SYSTEM, USER, SCHEMA)


def test_openai_llm_error_keeps_the_original_exception_as_cause():
    openai = pytest.importorskip("openai")
    with pytest.raises(LLMError) as raised:
        openai_client(respond_500).complete_json(SYSTEM, USER, SCHEMA)
    assert isinstance(raised.value.__cause__, openai.InternalServerError)


@pytest.mark.parametrize("content", [None, ""], ids=["missing", "empty"])
def test_openai_reply_without_content_becomes_llm_error(content):
    client = openai_client(lambda request: completion(content))
    with pytest.raises(LLMError, match=MODEL):
        client.complete_json(SYSTEM, USER, SCHEMA)


def ollama_text_request_body(num_ctx: int | None = None) -> dict:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return chat_reply(TEXT_REPLY)

    client = ollama_client(handler)
    client.num_ctx = num_ctx
    client.complete_text(SYSTEM, USER, MAX_TOKENS)
    return bodies[0]


def test_ollama_text_returns_the_message_content():
    client = ollama_client(lambda request: chat_reply(TEXT_REPLY))
    assert client.complete_text(SYSTEM, USER, MAX_TOKENS) == TEXT_REPLY


def test_ollama_text_request_has_no_format():
    assert "format" not in ollama_text_request_body()


def test_ollama_text_request_turns_thinking_off():
    assert ollama_text_request_body()["think"] is False


def test_ollama_text_request_uses_temperature_zero_and_the_token_limit():
    options = ollama_text_request_body()["options"]
    assert options["temperature"] == 0
    assert options["num_predict"] == MAX_TOKENS


def test_ollama_text_request_has_no_context_size_unless_the_client_has_one():
    assert "num_ctx" not in ollama_text_request_body()["options"]


def test_ollama_text_request_sets_the_context_size_of_the_client():
    assert ollama_text_request_body(num_ctx=NUM_CTX)["options"]["num_ctx"] == NUM_CTX


def test_ollama_text_request_keeps_the_model_loaded():
    assert ollama_text_request_body()["keep_alive"] == KEEP_ALIVE


@pytest.mark.parametrize("handler", FAILURES)
def test_ollama_text_failure_becomes_llm_error_naming_the_model(handler):
    with pytest.raises(LLMError, match=MODEL):
        ollama_client(handler).complete_text(SYSTEM, USER, MAX_TOKENS)


def test_ollama_text_reply_without_content_becomes_llm_error():
    client = ollama_client(lambda request: chat_reply(""))
    with pytest.raises(LLMError, match=MODEL):
        client.complete_text(SYSTEM, USER, MAX_TOKENS)


def test_ollama_from_settings_shares_the_generation_context_size(monkeypatch):
    monkeypatch.setattr(ollama, "Client", lambda **kwargs: object())
    settings = Settings(gen_num_ctx=6144)
    assert OllamaClient.from_settings(settings).num_ctx == 6144


def test_ollama_for_generation_uses_the_generation_settings(monkeypatch):
    created = []

    def fake_client(**kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(ollama, "Client", fake_client)
    settings = Settings(gen_model="gen", gen_timeout_s=90.0, gen_num_ctx=NUM_CTX)
    client = OllamaClient.for_generation(settings)
    assert client.model == "gen"
    assert client.num_ctx == NUM_CTX
    assert created == [{"timeout": 90.0}, {"timeout": LOAD_TIMEOUT_S}]


class FakeCompletions:
    def __init__(self, content: str | None = TEXT_REPLY):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_openai(completions: FakeCompletions) -> OpenAICompatClient:
    chat = SimpleNamespace(completions=completions)
    return OpenAICompatClient(SimpleNamespace(chat=chat), MODEL)


def test_openai_text_returns_the_content_with_the_token_limit_and_no_response_format():
    completions = FakeCompletions()
    reply = fake_openai(completions).complete_text(SYSTEM, USER, MAX_TOKENS)
    [call] = completions.calls
    assert reply == TEXT_REPLY
    assert call["max_tokens"] == MAX_TOKENS
    assert "response_format" not in call
    assert call["temperature"] == 0


def test_openai_text_request_sends_system_then_user_message():
    completions = FakeCompletions()
    fake_openai(completions).complete_text(SYSTEM, USER, MAX_TOKENS)
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]


@pytest.mark.parametrize("handler", FAILURES)
def test_openai_text_failure_becomes_llm_error_naming_the_model(handler):
    with pytest.raises(LLMError, match=MODEL):
        openai_client(handler).complete_text(SYSTEM, USER, MAX_TOKENS)


def test_openai_text_reply_without_content_becomes_llm_error():
    with pytest.raises(LLMError, match=MODEL):
        fake_openai(FakeCompletions(content="")).complete_text(SYSTEM, USER, MAX_TOKENS)


def test_openai_for_generation_uses_the_generation_settings():
    pytest.importorskip("openai")
    settings = Settings(
        llm_model="router",
        llm_timeout_s=3.0,
        gen_model="gen",
        gen_timeout_s=90.0,
        openai_base_url="http://test/v1",
        openai_api_key="test",
    )
    client = OpenAICompatClient.for_generation(settings)
    assert client.model == "gen"
    assert client.client.timeout == 90.0
    assert client.client.max_retries == 0


@pytest.fixture
def openai_from_settings() -> OpenAICompatClient:
    pytest.importorskip("openai")
    settings = Settings(
        llm_model="other",
        llm_timeout_s=3.0,
        openai_base_url="http://test/v1",
        openai_api_key="test",
    )
    return OpenAICompatClient.from_settings(settings)


def test_openai_from_settings_does_not_retry(openai_from_settings):
    assert openai_from_settings.client.max_retries == 0


def test_openai_from_settings_applies_the_settings(openai_from_settings):
    assert openai_from_settings.model == "other"
    assert str(openai_from_settings.client.base_url) == "http://test/v1/"
    assert openai_from_settings.client.timeout == 3.0


PIECES = ["The agreement ", "is governed ", "by Delaware law."]
JSON_OBJECT = {"clause_types": ["Governing Law"]}
NESTED = '{"a": {"b": [1, {"c": 2}]}}'
NO_GRAMMAR_ERROR = {"error": NO_GRAMMAR_TEXT}


def stream_body(pieces: list[str]) -> bytes:
    lines = []
    for piece in pieces:
        message = {"role": "assistant", "content": piece}
        lines.append(json.dumps({"model": MODEL, "message": message, "done": False}))
    final = {"role": "assistant", "content": ""}
    lines.append(json.dumps({"model": MODEL, "message": final, "done": True}))
    return ("\n".join(lines) + "\n").encode()


def stream_reply(pieces: list[str]) -> httpx.Response:
    headers = {"Content-Type": "application/x-ndjson"}
    return httpx.Response(200, content=stream_body(pieces), headers=headers)


def ollama_stream_request_body(num_ctx: int | None = None) -> dict:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return stream_reply(PIECES)

    client = ollama_client(handler)
    client.num_ctx = num_ctx
    list(client.stream_text(SYSTEM, USER, MAX_TOKENS))
    return bodies[0]


def test_ollama_stream_yields_the_pieces_in_order():
    client = ollama_client(lambda request: stream_reply(PIECES))
    assert list(client.stream_text(SYSTEM, USER, MAX_TOKENS)) == PIECES


def test_ollama_stream_skips_empty_pieces():
    client = ollama_client(lambda request: stream_reply(["a", "", "b"]))
    assert list(client.stream_text(SYSTEM, USER, MAX_TOKENS)) == ["a", "b"]


def test_ollama_stream_request_asks_for_a_stream_without_format():
    body = ollama_stream_request_body()
    assert body["stream"] is True
    assert "format" not in body


def test_ollama_stream_request_turns_thinking_off():
    assert ollama_stream_request_body()["think"] is False


def test_ollama_stream_request_uses_temperature_zero_and_the_token_limit():
    options = ollama_stream_request_body()["options"]
    assert options["temperature"] == 0
    assert options["num_predict"] == MAX_TOKENS


def test_ollama_stream_request_sets_the_context_size_of_the_client():
    assert ollama_stream_request_body(num_ctx=NUM_CTX)["options"]["num_ctx"] == NUM_CTX


def test_ollama_stream_request_has_no_context_size_unless_the_client_has_one():
    assert "num_ctx" not in ollama_stream_request_body()["options"]


@pytest.mark.parametrize("handler", FAILURES)
def test_ollama_stream_failure_becomes_llm_error_naming_the_model(handler):
    stream = ollama_client(handler).stream_text(SYSTEM, USER, MAX_TOKENS)
    with pytest.raises(LLMError, match=MODEL):
        list(stream)


def no_grammar_handler(bodies: list[dict], content: str, status: int = NO_GRAMMAR_STATUS):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "format" in body:
            return httpx.Response(status, json=NO_GRAMMAR_ERROR)
        return chat_reply(content)

    return handler


OBJECT_TEXT = json.dumps(JSON_OBJECT)
WRAPPED_REPLIES = [
    pytest.param(OBJECT_TEXT, id="plain"),
    pytest.param(f"```json\n{OBJECT_TEXT}\n```", id="fenced"),
    pytest.param(f"Here you go: {OBJECT_TEXT} Hope it helps.", id="prose"),
]


@pytest.mark.parametrize("content", WRAPPED_REPLIES)
def test_ollama_json_without_a_grammar_returns_the_json_object_only(content):
    client = ollama_client(no_grammar_handler([], content))
    assert client.complete_json(SYSTEM, USER, SCHEMA) == OBJECT_TEXT


def test_ollama_json_without_a_grammar_remembers_it():
    client = ollama_client(no_grammar_handler([], OBJECT_TEXT))
    client.complete_json(SYSTEM, USER, SCHEMA)
    assert client.grammar is False


def test_ollama_json_after_the_fallback_sends_one_request_without_format():
    bodies: list[dict] = []
    client = ollama_client(no_grammar_handler(bodies, OBJECT_TEXT))
    client.complete_json(SYSTEM, USER, SCHEMA)
    bodies.clear()
    assert client.complete_json(SYSTEM, USER, SCHEMA) == OBJECT_TEXT
    assert len(bodies) == 1
    assert "format" not in bodies[0]


def test_ollama_json_success_with_a_schema_sets_grammar():
    client = ollama_client(lambda request: chat_reply(REPLY))
    client.complete_json(SYSTEM, USER, SCHEMA)
    assert client.grammar is True


def test_ollama_json_fallback_puts_the_schema_at_the_end_of_the_system_message():
    bodies: list[dict] = []
    client = ollama_client(no_grammar_handler(bodies, OBJECT_TEXT))
    client.complete_json(SYSTEM, USER, SCHEMA)
    system = bodies[-1]["messages"][0]["content"]
    assert system == SYSTEM + SCHEMA_INSTRUCTION.format(schema=json.dumps(SCHEMA))


def test_ollama_json_fallback_by_message_text_alone():
    def handler(request: httpx.Request) -> httpx.Response:
        if "format" in json.loads(request.content):
            return httpx.Response(500, json=NO_GRAMMAR_ERROR)
        return chat_reply(OBJECT_TEXT)

    client = ollama_client(handler)
    assert client.complete_json(SYSTEM, USER, SCHEMA) == OBJECT_TEXT
    assert client.grammar is False


def test_ollama_json_server_error_still_raises_and_leaves_grammar_unknown():
    client = ollama_client(respond_500)
    with pytest.raises(LLMError, match=MODEL):
        client.complete_json(SYSTEM, USER, SCHEMA)
    assert client.grammar is None


def test_ollama_json_fallback_logs_one_warning_naming_the_model(caplog):
    client = ollama_client(no_grammar_handler([], OBJECT_TEXT))
    with caplog.at_level(logging.WARNING):
        client.complete_json(SYSTEM, USER, SCHEMA)
        client.complete_json(SYSTEM, USER, SCHEMA)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert MODEL in warnings[0].getMessage()


def test_json_object_returns_a_plain_object():
    assert json_object(OBJECT_TEXT) == OBJECT_TEXT


def test_json_object_removes_code_fences():
    assert json_object(f"```json\n{OBJECT_TEXT}\n```") == OBJECT_TEXT


def test_json_object_cuts_the_prose_around_the_object():
    assert json_object(f"Sure. {OBJECT_TEXT} Done.") == OBJECT_TEXT


def test_json_object_returns_the_stripped_text_when_there_is_no_object():
    assert json_object("  no object here \n") == "no object here"


def test_json_object_keeps_nested_braces():
    assert json_object(f"Result: {NESTED}!") == NESTED


def chunk(content: str | None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class StreamingCompletions:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.chunks)


def test_openai_stream_yields_the_pieces_in_order():
    completions = StreamingCompletions([chunk(piece) for piece in PIECES])
    stream = fake_openai(completions).stream_text(SYSTEM, USER, MAX_TOKENS)
    assert list(stream) == PIECES


def test_openai_stream_skips_empty_deltas_and_chunks_without_choices():
    chunks = [chunk("a"), chunk(None), SimpleNamespace(choices=[]), chunk(""), chunk("b")]
    stream = fake_openai(StreamingCompletions(chunks)).stream_text(SYSTEM, USER, MAX_TOKENS)
    assert list(stream) == ["a", "b"]


def test_openai_stream_request_asks_for_a_stream_with_the_token_limit():
    completions = StreamingCompletions([chunk("a")])
    list(fake_openai(completions).stream_text(SYSTEM, USER, MAX_TOKENS))
    [call] = completions.calls
    assert call["stream"] is True
    assert call["max_tokens"] == MAX_TOKENS
    assert call["temperature"] == 0


@pytest.mark.slow
def test_local_ollama_returns_json_that_follows_a_tiny_schema():
    client = OllamaClient.from_settings(Settings())
    if not client.available():
        pytest.skip(f"{client.model} is not served by a local Ollama")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    reply = client.complete_json("Reply with a JSON object.", "Say hello.", schema)
    assert "answer" in json.loads(reply)
