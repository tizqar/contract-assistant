import json

import httpx
import ollama
import pytest

from contract_assistant.config import Settings
from contract_assistant.router.clients import (
    KEEP_ALIVE,
    LOAD_TIMEOUT_S,
    LLMError,
    OllamaClient,
    OpenAICompatClient,
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
