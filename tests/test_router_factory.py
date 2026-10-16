from contract_assistant.clause_types import N_CLAUSE_TYPES
from contract_assistant.config import Settings
from contract_assistant.router import factory
from contract_assistant.router.embedding import EMBEDDING_MAX_TYPES, EmbeddingRouter
from tests.support.fakes import FakeEncoder


def test_build_router_returns_embedding_router_over_all_types(monkeypatch):
    model_ids = []

    def fake_encoder(model_id):
        model_ids.append(model_id)
        return FakeEncoder()

    monkeypatch.setattr(factory, "SentenceTransformerEncoder", fake_encoder)
    settings = Settings()
    router = factory.build_router(settings)
    assert isinstance(router, EmbeddingRouter)
    assert router.threshold == settings.embedding_threshold
    assert router.max_types == EMBEDDING_MAX_TYPES
    assert len(router.names) == N_CLAUSE_TYPES
    assert model_ids == [settings.embedding_model_id]
