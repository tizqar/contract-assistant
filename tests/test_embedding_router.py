import json

import huggingface_hub
import numpy as np
import pytest

from contract_assistant.clause_types import N_CLAUSE_TYPES, load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router.embedding import (
    QUERY_INSTRUCTION,
    EmbeddingRouter,
    RouterDataError,
    SentenceTransformerEncoder,
    load_descriptions,
    select_types,
)
from tests.support.fakes import FakeEncoder

MODEL_ID = "BAAI/bge-base-en-v1.5"
DESCRIPTIONS = {
    "Governing Law": ["which law applies", "choice of law"],
    "Insurance": ["required insurance coverage"],
    "Audit Rights": ["right to audit the books"],
}
SCORES = {"Insurance": 0.4, "Governing Law": 0.9, "Audit Rights": 0.6, "Term": 0.5}


def make_router(threshold=0.1, max_types=5, descriptions=DESCRIPTIONS):
    return EmbeddingRouter(FakeEncoder(), descriptions, threshold, max_types)


class StubModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 2))


def stub_encoder():
    encoder = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    encoder.model = StubModel()
    return encoder


def test_exact_match_scores_one():
    scores = make_router().scores("required insurance coverage")
    assert scores["Insurance"] == pytest.approx(1.0)


def test_score_is_max_over_a_types_texts():
    scores = make_router().scores("choice of law")
    assert scores["Governing Law"] == pytest.approx(1.0)


def test_type_just_below_threshold_is_excluded():
    router = make_router()
    score = router.scores("which law applies today")["Governing Law"]
    router.threshold = score + 1e-9
    assert router.route("which law applies today").clause_types == []
    router.threshold = score
    assert router.route("which law applies today").clause_types == ["Governing Law"]


def test_max_types_caps_the_route():
    router = make_router(threshold=0.0, max_types=2)
    assert len(router.route("law insurance audit").clause_types) == 2


def test_route_is_ordered_by_score():
    router = make_router(threshold=0.1)
    result = router.route("insurance coverage and which law")
    scores = router.scores("insurance coverage and which law")
    ranked = [scores[name] for name in result.clause_types]
    assert ranked == sorted(ranked, reverse=True)
    assert len(result.clause_types) >= 2


def test_question_matching_nothing_gives_empty_route():
    assert make_router(threshold=0.5).route("zzz qqq").clause_types == []


def test_select_types_keeps_scores_at_the_threshold_best_first():
    assert select_types(SCORES, 0.5, 5) == ["Governing Law", "Audit Rights", "Term"]


def test_select_types_stops_at_max_types():
    assert select_types(SCORES, 0.0, 2) == ["Governing Law", "Audit Rights"]


def test_select_types_is_empty_when_no_score_reaches_the_threshold():
    assert select_types(SCORES, 0.95, 5) == []


def test_route_follows_select_types():
    router = make_router(threshold=0.3, max_types=2)
    question = "insurance coverage and which law"
    expected = select_types(router.scores(question), router.threshold, router.max_types)
    assert router.route(question).clause_types == expected


def test_route_reports_kind_and_latency():
    result = make_router().route("which law applies")
    assert result.kind == "embedding"
    assert result.latency_ms >= 0


def test_descriptions_are_encoded_once_as_documents():
    encoder = FakeEncoder()
    router = EmbeddingRouter(encoder, DESCRIPTIONS, 0.1, 5)
    router.route("which law applies")
    router.route("which law applies")
    assert len(encoder.document_texts) == 4
    assert encoder.query_texts == ["which law applies"] * 2


def test_instruction_is_added_to_queries_only():
    encoder = stub_encoder()
    encoder.encode_queries(["question"])
    encoder.encode_documents(["passage"])
    (query_texts, _), (document_texts, _) = encoder.model.calls
    assert query_texts == [QUERY_INSTRUCTION + "question"]
    assert document_texts == ["passage"]


def test_encoder_requests_normalised_embeddings():
    encoder = stub_encoder()
    encoder.encode_queries(["question"])
    encoder.encode_documents(["passage"])
    assert all(kwargs["normalize_embeddings"] is True for _, kwargs in encoder.model.calls)


@pytest.fixture(scope="module")
def types():
    return load_clause_types(Settings().clause_types_path)


def test_every_description_name_is_a_clause_type(types):
    paraphrases = json.loads(Settings().router_descriptions_path.read_text())
    assert list(paraphrases) == [clause_type.name for clause_type in types]


def test_every_type_has_two_or_three_paraphrases():
    paraphrases = json.loads(Settings().router_descriptions_path.read_text())
    assert len(paraphrases) == N_CLAUSE_TYPES
    assert all(2 <= len(texts) <= 3 for texts in paraphrases.values())


def test_every_extra_description_name_is_an_extra_clause_type():
    settings = Settings()
    if not settings.extra_clause_types_path.exists():
        pytest.skip("data/extra_clause_types.json does not exist yet")
    from contract_assistant.clause_types import load_extra_clause_types

    extra = {t.name for t in load_extra_clause_types(settings.extra_clause_types_path)}
    paraphrases = json.loads(settings.extra_router_descriptions_path.read_text())
    assert set(paraphrases) == extra
    assert all(2 <= len(texts) <= 3 for texts in paraphrases.values())


def test_load_descriptions_puts_the_cuad_description_first(types):
    descriptions = load_descriptions(types, Settings().router_descriptions_path)
    assert descriptions[types[0].name][0] == types[0].description
    assert len(descriptions[types[0].name]) >= 3


def test_load_descriptions_rejects_unknown_name(types, tmp_path):
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps({**{t.name: ["x"] for t in types}, "Not A Type": ["y"]}))
    with pytest.raises(RouterDataError, match="Not A Type"):
        load_descriptions(types, path)


def test_load_descriptions_rejects_missing_name(types, tmp_path):
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps({t.name: ["x"] for t in types[1:]}))
    with pytest.raises(RouterDataError, match=types[0].name):
        load_descriptions(types, path)


@pytest.mark.slow
def test_governing_law_question_ranks_governing_law_first(types):
    if not isinstance(huggingface_hub.try_to_load_from_cache(MODEL_ID, "config.json"), str):
        pytest.skip(f"{MODEL_ID} is not in the Hugging Face cache")
    settings = Settings()
    descriptions = load_descriptions(types, settings.router_descriptions_path)
    router = EmbeddingRouter(SentenceTransformerEncoder(MODEL_ID), descriptions, 0.0, 5)
    scores = router.scores("Which law governs this agreement?")
    assert max(scores, key=scores.get) == "Governing Law"
