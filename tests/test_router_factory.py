import logging
from types import SimpleNamespace

import pytest

from contract_assistant.clause_types import N_CLAUSE_TYPES
from contract_assistant.config import Settings
from contract_assistant.router import factory
from contract_assistant.router.embedding import EMBEDDING_MAX_TYPES, EmbeddingRouter
from contract_assistant.router.llm import LLMRouter
from tests.support.fakes import FakeEncoder, FakeLLMClient

OLLAMA_MODEL = "qwen-test"


class FakeOllama(FakeLLMClient):
    def __init__(self, available: bool):
        super().__init__([], model=OLLAMA_MODEL)
        self.is_available = available

    def available(self) -> bool:
        return self.is_available


def refuse_client(settings):
    raise AssertionError("no LLM client may be built")


@pytest.fixture(autouse=True)
def model_ids(monkeypatch) -> list[str]:
    ids = []

    def fake_encoder(model_id):
        ids.append(model_id)
        return FakeEncoder()

    monkeypatch.setattr(factory, "SentenceTransformerEncoder", fake_encoder)
    return ids


@pytest.fixture(autouse=True)
def no_clients(monkeypatch):
    monkeypatch.setattr(factory.OllamaClient, "from_settings", refuse_client)
    monkeypatch.setattr(factory.OpenAICompatClient, "from_settings", refuse_client)


@pytest.fixture
def clients(monkeypatch):
    ollama = FakeOllama(available=True)
    openai = FakeLLMClient([], model="openai-test")
    monkeypatch.setattr(factory.OllamaClient, "from_settings", lambda settings: ollama)
    monkeypatch.setattr(factory.OpenAICompatClient, "from_settings", lambda settings: openai)
    return SimpleNamespace(ollama=ollama, openai=openai)


def test_none_backend_returns_the_embedding_router_over_all_types(model_ids):
    settings = Settings(llm_backend="none")
    router = factory.build_router(settings)
    assert isinstance(router, EmbeddingRouter)
    assert router.threshold == settings.embedding_threshold
    assert router.max_types == EMBEDDING_MAX_TYPES
    assert len(router.names) == N_CLAUSE_TYPES
    assert model_ids == [settings.embedding_model_id]


def test_ollama_backend_returns_an_llm_router_over_the_ollama_client(clients):
    router = factory.build_router(Settings(llm_backend="ollama"))
    assert isinstance(router, LLMRouter)
    assert router.client is clients.ollama


def test_openai_backend_returns_an_llm_router_over_the_openai_client(clients):
    router = factory.build_router(Settings(llm_backend="openai"))
    assert isinstance(router, LLMRouter)
    assert router.client is clients.openai


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_llm_router_falls_back_to_the_embedding_router(backend, clients):
    router = factory.build_router(Settings(llm_backend=backend))
    assert isinstance(router.fallback, EmbeddingRouter)
    assert router.fallback.max_types == EMBEDDING_MAX_TYPES


def test_ollama_backend_without_the_model_returns_the_embedding_router(clients):
    clients.ollama.is_available = False
    router = factory.build_router(Settings(llm_backend="ollama"))
    assert isinstance(router, EmbeddingRouter)


def test_ollama_backend_without_the_model_warns_naming_the_model(clients, caplog):
    clients.ollama.is_available = False
    with caplog.at_level(logging.WARNING):
        factory.build_router(Settings(llm_backend="ollama"))
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert OLLAMA_MODEL in record.getMessage()
    assert "embedding router" in record.getMessage()


def test_unknown_backend_raises_before_any_model_is_loaded(model_ids):
    with pytest.raises(factory.UnknownBackendError, match="olama"):
        factory.build_router(Settings(llm_backend="olama"))
    assert model_ids == []


def test_unknown_backend_error_lists_the_valid_backends():
    with pytest.raises(factory.UnknownBackendError) as raised:
        factory.build_router(Settings(llm_backend="olama"))
    assert str(raised.value).endswith("use one of ollama, openai, none")


def test_embedding_router_is_built_without_any_llm_client(model_ids):
    settings = Settings(llm_backend="ollama")
    router = factory.build_embedding_router(settings)
    assert isinstance(router, EmbeddingRouter)
    assert model_ids == [settings.embedding_model_id]
