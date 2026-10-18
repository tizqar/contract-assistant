from dataclasses import replace

import pytest

from contract_assistant.clause_types import app_clause_types, load_clause_types
from contract_assistant.config import ROOT, Settings
from contract_assistant.extract import extractor as extractor_module
from contract_assistant.extract.extractor import Extractor, ModelNotFoundError
from contract_assistant.pdf import Document
from contract_assistant.store import read_cache, write_cache
from tests.support.fakes import FakeQA

MIN_DISPUTE_CONFIDENCE = 0.5

PAGES = [
    "Agreement between Acme and Beta.",
    "Governed by the laws of Delaware.",
    "Either party may terminate on notice.",
]


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def clause_types(settings):
    return load_clause_types(settings.clause_types_path)


def make_extractor(qa, clause_types, settings) -> Extractor:
    return Extractor(qa, clause_types, settings, device="cpu")


def pdf_document() -> Document:
    page_starts = []
    position = 0
    for page in PAGES:
        page_starts.append(position)
        position += len(page) + 1
    return Document("\n".join(PAGES), page_starts, "contract.pdf", "pdf", "unused")


def txt_document() -> Document:
    return Document("\n".join(PAGES), [], "contract.txt", "txt", "unused")


def question_of(clause_types, name: str) -> str:
    return next(c.cuad_question for c in clause_types if c.name == name)


def test_asks_every_question_once_in_file_order(settings, clause_types):
    qa = FakeQA()
    make_extractor(qa, clause_types, settings).extract(txt_document())
    assert len(qa.calls) == 41
    assert [question for question, _ in qa.calls] == [c.cuad_question for c in clause_types]


def test_decode_settings_come_from_settings(settings, clause_types):
    custom = replace(settings, top_k=5, doc_stride=128)
    qa = FakeQA()
    make_extractor(qa, clause_types, custom).extract(txt_document())
    expected = {
        "max_seq_len": custom.max_seq_len,
        "doc_stride": 128,
        "max_answer_len": custom.max_answer_len,
        "top_k": 5,
        "n_best_size": custom.n_best_size,
    }
    assert all(kwargs == expected for _, kwargs in qa.calls)


def test_progress_reports_each_clause_type(settings, clause_types):
    reports = []
    extractor = make_extractor(FakeQA(), clause_types, settings)
    extractor.extract(txt_document(), progress=lambda *args: reports.append(args))
    assert len(reports) == 41
    assert reports[0] == (1, 41, "Document Name")
    assert reports[-1] == (41, 41, "Third Party Beneficiary")


def test_pages_are_assigned_on_a_pdf_document(settings, clause_types):
    answers = {
        question_of(clause_types, "Governing Law"): [(PAGES[1], 0.9)],
        question_of(clause_types, "Termination For Convenience"): [(PAGES[2], 0.8)],
        question_of(clause_types, "Parties"): [(PAGES[0], 0.7)],
    }
    cache = make_extractor(FakeQA(answers), clause_types, settings).extract(pdf_document())
    assert cache.source.n_pages == 3
    assert cache.clauses["Governing Law"][0].page == 2
    assert cache.clauses["Termination For Convenience"][0].page == 3
    assert cache.clauses["Parties"][0].page == 1


def test_txt_document_has_no_pages(settings, clause_types):
    answers = {question_of(clause_types, "Governing Law"): [(PAGES[1], 0.9)]}
    cache = make_extractor(FakeQA(answers), clause_types, settings).extract(txt_document())
    assert cache.source.n_pages is None
    assert cache.clauses["Governing Law"][0].page is None


def test_candidates_keep_rank_order_and_offsets(settings, clause_types):
    question = question_of(clause_types, "Governing Law")
    answers = {question: [("Delaware", 0.9), ("laws", 0.5), ("Acme", 0.1)]}
    doc = pdf_document()
    cache = make_extractor(FakeQA(answers), clause_types, settings).extract(doc)
    candidates = cache.clauses["Governing Law"]
    assert [c.text for c in candidates] == ["Delaware", "laws", "Acme"]
    assert all(doc.text[c.start : c.end] == c.text for c in candidates)


def test_empty_n_best_gives_empty_list(settings, clause_types):
    cache = make_extractor(FakeQA(), clause_types, settings).extract(txt_document())
    assert cache.clauses["Document Name"] == []
    assert len(cache.clauses) == 41


def test_cache_survives_a_write_and_read(settings, clause_types, tmp_path):
    answers = {question_of(clause_types, "Governing Law"): [("Delaware", 0.9)]}
    cache = make_extractor(FakeQA(answers), clause_types, settings).extract(pdf_document())
    write_cache(cache, tmp_path / "cache.json")
    assert read_cache(tmp_path / "cache.json", settings) == cache


def test_model_path_is_relative_for_default_settings(settings, clause_types):
    cache = make_extractor(FakeQA(), clause_types, settings).extract(txt_document())
    assert not cache.model.path.startswith("/")


def test_model_path_is_absolute_outside_the_repository(settings, clause_types, tmp_path):
    outside = replace(settings, extractor_path=tmp_path / "model")
    cache = make_extractor(FakeQA(), clause_types, outside).extract(txt_document())
    assert cache.model.path == str(tmp_path / "model")


def test_from_settings_names_a_missing_model_folder(settings, tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ModelNotFoundError, match=str(missing)):
        Extractor.from_settings(replace(settings, extractor_path=missing))


def test_extractor_built_from_settings_covers_the_app_clause_types(
    settings, monkeypatch, tmp_path
):
    class StubModel(FakeQA):
        device = "cpu"

        def __init__(self, path, **kwargs):
            super().__init__()

    monkeypatch.setattr(extractor_module, "QAModel", StubModel)
    extractor = Extractor.from_settings(replace(settings, extractor_path=tmp_path))
    cache = extractor.extract(txt_document())
    assert list(cache.clauses) == [c.name for c in app_clause_types(settings)]
    assert len(cache.clauses) == 49


def test_extend_asks_only_the_missing_clause_types(settings, clause_types):
    first = make_extractor(FakeQA(), clause_types[:5], settings).extract(txt_document())
    qa = FakeQA()
    extended = make_extractor(qa, clause_types, settings).extend(first)
    missing = clause_types[5:]
    assert [question for question, _ in qa.calls] == [c.cuad_question for c in missing]
    assert list(extended.clauses) == [c.name for c in clause_types]


def test_extend_keeps_existing_candidates(settings, clause_types):
    question = question_of(clause_types, "Governing Law")
    qa = FakeQA({question: [("Delaware", 0.9)]})
    first = make_extractor(qa, clause_types[:20], settings).extract(txt_document())
    extended = make_extractor(FakeQA(), clause_types, settings).extend(first)
    assert extended.clauses["Governing Law"] == first.clauses["Governing Law"]


def test_extend_adds_candidates_of_the_new_types(settings, clause_types):
    question = clause_types[-1].cuad_question
    first = make_extractor(FakeQA(), clause_types[:-1], settings).extract(pdf_document())
    qa = FakeQA({question: [("Delaware", 0.7)]})
    extended = make_extractor(qa, clause_types, settings).extend(first)
    added = extended.clauses[clause_types[-1].name][0]
    assert (added.text, added.page) == ("Delaware", 2)


def test_extend_increases_the_extraction_time(settings, clause_types):
    first = make_extractor(FakeQA(), clause_types[:-1], settings).extract(txt_document())
    extended = make_extractor(FakeQA(), clause_types, settings).extend(first)
    assert extended.timing["extract_s"] >= first.timing["extract_s"]


def test_extend_returns_a_complete_cache_unchanged(settings, clause_types):
    extractor = make_extractor(FakeQA(), clause_types, settings)
    cache = extractor.extract(txt_document())
    qa = FakeQA()
    assert make_extractor(qa, clause_types, settings).extend(cache) is cache
    assert qa.calls == []


@pytest.mark.slow
def test_real_extractor_finds_the_arbitration_clause_of_the_short_demo(settings):
    text = (ROOT / "demo/short_supply_agreement.txt").read_text()
    doc = Document(text, [], "short_supply_agreement.txt", "txt", "unused")
    extractor = Extractor.from_settings(settings)
    name = "Dispute Resolution"
    dispute = next(c for c in extractor.clause_types if c.name == name)
    extractor.clause_types = [dispute]
    top = extractor.extract(doc).clauses[name][0]
    assert "arbitration" in top.text.lower()
    assert top.conf > MIN_DISPUTE_CONFIDENCE
