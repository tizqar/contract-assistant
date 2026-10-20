import json
import logging
from types import SimpleNamespace

import pytest

from contract_assistant.clause_types import N_CLAUSE_TYPES, ClauseType, load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router import factory
from contract_assistant.router.clients import LLMError
from contract_assistant.router.embedding import EMBEDDING_MAX_TYPES, EmbeddingRouter
from contract_assistant.router.llm import LLMRouter
from tests.support.fakes import FakeEncoder, FakeLLMClient

OLLAMA_MODEL = "qwen-test"
N_EXTRA_TYPES = 8


class FakeOllama(FakeLLMClient):
    def __init__(self, available: bool):
        super().__init__([], model=OLLAMA_MODEL)
        self.is_available = available
        self.loads = 0
        self.load_error: LLMError | None = None

    def available(self) -> bool:
        return self.is_available

    def load(self) -> None:
        self.loads += 1
        if self.load_error:
            raise self.load_error


def refuse_client(settings):
    raise AssertionError("no LLM client may be built")


@pytest.fixture(scope="module")
def cuad_types():
    return load_clause_types(Settings().clause_types_path)


@pytest.fixture(scope="module")
def extra_types():
    names = json.loads(Settings().extra_router_descriptions_path.read_text())
    return [
        ClauseType(f"extra-{i}", name, "", f"About {name}.") for i, name in enumerate(names)
    ]


@pytest.fixture(autouse=True)
def model_ids(monkeypatch):
    factory.load_encoder.cache_clear()
    ids = []

    def fake_encoder(model_id):
        ids.append(model_id)
        return FakeEncoder()

    monkeypatch.setattr(factory, "SentenceTransformerEncoder", fake_encoder)
    yield ids
    factory.load_encoder.cache_clear()


@pytest.fixture(autouse=True)
def no_clients(monkeypatch):
    monkeypatch.setattr(factory.OllamaClient, "from_settings", refuse_client)
    monkeypatch.setattr(factory.OpenAICompatClient, "from_settings", refuse_client)
    monkeypatch.setattr(factory.OllamaClient, "for_generation", refuse_client)
    monkeypatch.setattr(factory.OpenAICompatClient, "for_generation", refuse_client)


@pytest.fixture
def clients(monkeypatch):
    ollama = FakeOllama(available=True)
    openai = FakeLLMClient([], model="openai-test")
    monkeypatch.setattr(factory.OllamaClient, "from_settings", lambda settings: ollama)
    monkeypatch.setattr(factory.OpenAICompatClient, "from_settings", lambda settings: openai)
    return SimpleNamespace(ollama=ollama, openai=openai)


@pytest.fixture
def gen_clients(monkeypatch):
    ollama = FakeOllama(available=True)
    openai = FakeLLMClient([], model="openai-gen")
    monkeypatch.setattr(factory.OllamaClient, "for_generation", lambda settings: ollama)
    monkeypatch.setattr(factory.OpenAICompatClient, "for_generation", lambda settings: openai)
    return SimpleNamespace(ollama=ollama, openai=openai)


def test_none_backend_has_no_llm_client():
    assert factory.build_llm_client(Settings(llm_backend="none")) is None


def test_ollama_backend_returns_the_ollama_client(clients):
    assert factory.build_llm_client(Settings(llm_backend="ollama")) is clients.ollama


def test_openai_backend_returns_the_openai_client(clients):
    assert factory.build_llm_client(Settings(llm_backend="openai")) is clients.openai


def test_ollama_backend_without_the_model_has_no_llm_client(clients):
    clients.ollama.is_available = False
    assert factory.build_llm_client(Settings(llm_backend="ollama")) is None


def test_ollama_backend_without_the_model_warns_naming_the_model(clients, caplog):
    clients.ollama.is_available = False
    with caplog.at_level(logging.WARNING):
        factory.build_llm_client(Settings(llm_backend="ollama"))
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert OLLAMA_MODEL in record.getMessage()
    assert "embedding router" in record.getMessage()


def test_ollama_model_is_loaded_once_when_the_client_is_built(clients):
    factory.build_llm_client(Settings(llm_backend="ollama"))
    assert clients.ollama.loads == 1


def test_ollama_model_is_not_loaded_when_it_is_not_available(clients):
    clients.ollama.is_available = False
    factory.build_llm_client(Settings(llm_backend="ollama"))
    assert clients.ollama.loads == 0


def test_ollama_backend_whose_model_fails_to_load_has_no_llm_client(clients):
    clients.ollama.load_error = LLMError("load failed")
    assert factory.build_llm_client(Settings(llm_backend="ollama")) is None


def test_ollama_load_failure_warns_with_the_error(clients, caplog):
    clients.ollama.load_error = LLMError("load failed")
    with caplog.at_level(logging.WARNING):
        factory.build_llm_client(Settings(llm_backend="ollama"))
    [record] = caplog.records
    assert "load failed" in record.getMessage()
    assert "embedding router" in record.getMessage()


def test_unknown_backend_has_no_client_and_raises():
    with pytest.raises(factory.UnknownBackendError, match="olama"):
        factory.build_llm_client(Settings(llm_backend="olama"))


def test_unknown_backend_error_lists_the_valid_backends():
    with pytest.raises(factory.UnknownBackendError) as raised:
        factory.build_llm_client(Settings(llm_backend="olama"))
    assert str(raised.value).endswith("use one of ollama, openai, none")


def test_none_backend_returns_the_embedding_router_over_all_types(model_ids, cuad_types):
    settings = Settings(llm_backend="none")
    router = factory.build_router(settings, cuad_types)
    assert isinstance(router, EmbeddingRouter)
    assert router.threshold == settings.embedding_threshold
    assert router.max_types == EMBEDDING_MAX_TYPES
    assert len(router.names) == N_CLAUSE_TYPES
    assert model_ids == [settings.embedding_model_id]


def test_ollama_backend_returns_an_llm_router_over_the_ollama_client(clients, cuad_types):
    router = factory.build_router(Settings(llm_backend="ollama"), cuad_types)
    assert isinstance(router, LLMRouter)
    assert router.client is clients.ollama


def test_openai_backend_returns_an_llm_router_over_the_openai_client(clients, cuad_types):
    router = factory.build_router(Settings(llm_backend="openai"), cuad_types)
    assert isinstance(router, LLMRouter)
    assert router.client is clients.openai


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_llm_router_falls_back_to_the_embedding_router(backend, clients, cuad_types):
    router = factory.build_router(Settings(llm_backend=backend), cuad_types)
    assert isinstance(router.fallback, EmbeddingRouter)
    assert router.fallback.max_types == EMBEDDING_MAX_TYPES


def test_ollama_backend_without_the_model_returns_the_embedding_router(clients, cuad_types):
    clients.ollama.is_available = False
    router = factory.build_router(Settings(llm_backend="ollama"), cuad_types)
    assert isinstance(router, EmbeddingRouter)


def test_unknown_backend_raises_before_any_model_is_loaded(model_ids, cuad_types):
    with pytest.raises(factory.UnknownBackendError, match="olama"):
        factory.build_router(Settings(llm_backend="olama"), cuad_types)
    assert model_ids == []


def test_embedding_router_is_built_without_any_llm_client(model_ids, cuad_types):
    settings = Settings(llm_backend="ollama")
    router = factory.build_embedding_router(settings, cuad_types)
    assert isinstance(router, EmbeddingRouter)
    assert model_ids == [settings.embedding_model_id]


def test_embedding_router_over_cuad_types_has_the_cuad_descriptions_only(cuad_types):
    router = factory.build_embedding_router(Settings(), cuad_types)
    assert router.names == [clause_type.name for clause_type in cuad_types]


def test_embedding_router_over_extra_types_routes_over_all_of_them(cuad_types, extra_types):
    router = factory.build_embedding_router(Settings(), [*cuad_types, *extra_types])
    assert len(router.names) == N_CLAUSE_TYPES + N_EXTRA_TYPES
    assert router.names[N_CLAUSE_TYPES:] == [clause_type.name for clause_type in extra_types]


def test_extra_types_are_described_by_the_extra_descriptions_file(cuad_types, extra_types):
    settings = Settings()
    encoder = factory.load_encoder(settings.embedding_model_id)
    factory.build_embedding_router(settings, [*cuad_types, *extra_types])
    paraphrases = json.loads(settings.extra_router_descriptions_path.read_text())
    assert set(paraphrases["Force Majeure"]) <= set(encoder.document_texts)


def test_llm_router_over_extra_types_lists_them_in_prompt_and_schema(
    clients, cuad_types, extra_types
):
    router = factory.build_router(Settings(llm_backend="ollama"), [*cuad_types, *extra_types])
    enum = router.schema["properties"]["clause_types"]["items"]["enum"]
    assert len(enum) == N_CLAUSE_TYPES + N_EXTRA_TYPES
    assert all(f"- {clause_type.name}:" in router.system for clause_type in extra_types)


def test_encoder_is_loaded_once_for_two_builds_with_the_same_model(model_ids, cuad_types):
    settings = Settings()
    factory.build_embedding_router(settings, cuad_types)
    factory.build_embedding_router(settings, cuad_types)
    assert model_ids == [settings.embedding_model_id]


def test_encoder_is_loaded_again_for_another_model_id(model_ids):
    factory.load_encoder("model-a")
    factory.load_encoder("model-b")
    assert model_ids == ["model-a", "model-b"]


def test_router_unavailable_warning_still_says_embedding_router(clients, caplog):
    clients.ollama.is_available = False
    with caplog.at_level(logging.WARNING):
        factory.build_llm_client(Settings(llm_backend="ollama"))
    assert "using the embedding router" in caplog.text


def test_none_backend_has_no_gen_client():
    assert factory.build_gen_client(Settings(llm_backend="none")) is None


def test_ollama_backend_returns_the_ollama_gen_client(gen_clients):
    assert factory.build_gen_client(Settings(llm_backend="ollama")) is gen_clients.ollama


def test_ollama_gen_model_is_loaded_once(gen_clients):
    factory.build_gen_client(Settings(llm_backend="ollama"))
    assert gen_clients.ollama.loads == 1


def test_ollama_backend_without_the_gen_model_has_no_gen_client(gen_clients):
    gen_clients.ollama.is_available = False
    assert factory.build_gen_client(Settings(llm_backend="ollama")) is None
    assert gen_clients.ollama.loads == 0


def test_unavailable_gen_model_warns_naming_the_model_and_the_fallback(gen_clients, caplog):
    gen_clients.ollama.is_available = False
    with caplog.at_level(logging.WARNING):
        factory.build_gen_client(Settings(llm_backend="ollama"))
    [record] = caplog.records
    assert OLLAMA_MODEL in record.getMessage()
    assert factory.GENERATION_FALLBACK in record.getMessage()
    assert "summaries and explanations are off" in record.getMessage()


def test_gen_load_failure_gives_no_client_and_warns_with_the_error(gen_clients, caplog):
    gen_clients.ollama.load_error = LLMError("load failed")
    with caplog.at_level(logging.WARNING):
        assert factory.build_gen_client(Settings(llm_backend="ollama")) is None
    [record] = caplog.records
    assert "load failed" in record.getMessage()
    assert "summaries and explanations are off" in record.getMessage()


def test_openai_backend_returns_the_openai_gen_client(gen_clients):
    assert factory.build_gen_client(Settings(llm_backend="openai")) is gen_clients.openai


def test_unknown_backend_for_the_gen_client_raises_listing_the_backends():
    with pytest.raises(factory.UnknownBackendError) as raised:
        factory.build_gen_client(Settings(llm_backend="olama"))
    assert str(raised.value).endswith("use one of ollama, openai, none")


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_building_the_gen_client_never_builds_the_router_client(backend, gen_clients):
    assert factory.build_gen_client(Settings(llm_backend=backend)) is not None


@pytest.mark.parametrize("backend", ["ollama", "openai"])
def test_building_the_router_client_never_builds_the_gen_client(backend, clients):
    assert factory.build_llm_client(Settings(llm_backend=backend)) is not None
