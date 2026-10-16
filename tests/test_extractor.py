from dataclasses import replace

import pytest

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.extract.extractor import Extractor, ModelNotFoundError
from contract_assistant.pdf import Document
from contract_assistant.store import read_cache, write_cache
from tests.support.fakes import FakeQA

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
