from dataclasses import replace

import pytest

from contract_assistant import cli
from contract_assistant.clause_types import app_clause_types, load_clause_types
from contract_assistant.cli import main
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.store import Store, read_cache, write_cache
from tests.support.caches import make_cache
from tests.support.fakes import FakeQA

SENTENCE = "This agreement is governed by the laws of Vanuatu."
DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


@pytest.fixture
def contract(tmp_path):
    path = tmp_path / "contract.txt"
    path.write_text(f"Preamble. {SENTENCE} Final words.", encoding="utf-8")
    return path


@pytest.fixture
def fake_qa(monkeypatch):
    settings = load_settings()
    clause_types = app_clause_types(settings)
    question = next(c for c in clause_types if c.name == "Governing Law").cuad_question
    qa = FakeQA({question: [(SENTENCE, 0.9)]})
    monkeypatch.setattr(
        Extractor,
        "from_settings",
        classmethod(lambda cls, s: Extractor(qa, clause_types, s, "cpu")),
    )
    return qa


def test_extract_writes_a_cache_and_returns_zero(contract, fake_qa, tmp_path, capsys):
    assert main(["extract", str(contract)]) == 0
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert "contract.txt" in capsys.readouterr().out


def test_second_extract_reports_cache_hit(contract, fake_qa, capsys):
    main(["extract", str(contract)])
    capsys.readouterr()
    assert main(["extract", str(contract)]) == 0
    assert "cache hit" in capsys.readouterr().out


def test_second_extract_does_not_call_the_qa_function(contract, fake_qa):
    main(["extract", str(contract)])
    calls = len(fake_qa.calls)
    main(["extract", str(contract)])
    assert len(fake_qa.calls) == calls


def test_show_prints_every_clause_type_once(capsys):
    assert main(["show", str(DEMO_TEXT)]) == 0
    lines = capsys.readouterr().out.splitlines()[1:]
    names = [c.name for c in app_clause_types(load_settings())]
    assert [next(n for n in names if line.startswith(n + " ")) for line in lines] == names


def test_show_accepts_a_cache_file(capsys):
    assert main(["show", str(DEMO_CACHE)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "20693 chars" in lines[0]
    assert len(lines) == 1 + len(app_clause_types(load_settings()))


def test_show_without_a_cache_returns_one(contract, capsys):
    assert main(["show", str(contract)]) == 1
    assert "no cache for" in capsys.readouterr().err


def test_extract_with_a_missing_model_prints_the_hint_and_returns_two(
    contract, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("EXTRACTOR_MODEL_ID", str(tmp_path / "missing"))
    assert main(["extract", str(contract)]) == 2
    assert "IMPLEMENTATION.md" in capsys.readouterr().err


def test_extract_re_extracts_a_cache_written_with_other_settings(
    contract, fake_qa, tmp_path, capsys
):
    settings = load_settings()
    text = contract.read_text(encoding="utf-8")
    stale = make_cache(text, replace(settings, extractor_name="other"))
    store = Store(tmp_path / "cache", settings.demo_cache_dir)
    store.save(stale)
    assert main(["extract", str(contract)]) == 0
    assert fake_qa.calls
    assert store.load(stale.doc_id, settings) is not None


def test_extract_demo_writes_into_the_demo_cache_dir(contract, fake_qa, tmp_path, monkeypatch):
    use_demo_dir(monkeypatch, tmp_path)
    assert main(["extract", "--demo", str(contract)]) == 0
    assert (tmp_path / "demo_cache" / "contract.json").is_file()
    assert not (tmp_path / "cache").exists()


def use_demo_dir(monkeypatch, tmp_path) -> Settings:
    settings = replace(Settings(), demo_cache_dir=tmp_path / "demo_cache")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    return settings


def older_cache(text, settings):
    cuad = load_clause_types(settings.clause_types_path)
    clauses = {c.name: [] for c in cuad}
    clauses["Governing Law"] = [(SENTENCE, 0.5)]
    return make_cache(text, settings, clauses)


def extra_questions(settings):
    cuad = {c.name for c in load_clause_types(settings.clause_types_path)}
    return {c.cuad_question for c in app_clause_types(settings) if c.name not in cuad}


def test_extract_adds_the_missing_types_to_a_cache_in_the_cache_folder(
    contract, fake_qa, tmp_path, capsys
):
    settings = load_settings()
    store = Store(tmp_path / "cache", settings.demo_cache_dir)
    older = store.save(older_cache(contract.read_text(encoding="utf-8"), settings))
    assert main(["extract", str(contract)]) == 0
    assert "extended with 8 clause types" in capsys.readouterr().out
    assert len(read_cache(older, settings).clauses) == 49


def test_extract_asks_only_the_missing_questions_of_an_older_cache(contract, fake_qa, tmp_path):
    settings = load_settings()
    store = Store(tmp_path / "cache", settings.demo_cache_dir)
    store.save(older_cache(contract.read_text(encoding="utf-8"), settings))
    main(["extract", str(contract)])
    assert {question for question, _ in fake_qa.calls} == extra_questions(settings)


def test_extract_demo_extends_the_demo_cache_in_place(contract, fake_qa, tmp_path, monkeypatch):
    settings = use_demo_dir(monkeypatch, tmp_path)
    path = tmp_path / "demo_cache" / "contract.json"
    older = older_cache(contract.read_text(encoding="utf-8"), settings)
    write_cache(older, path)
    assert main(["extract", "--demo", str(contract)]) == 0
    extended = read_cache(path, settings)
    assert len(extended.clauses) == 49
    assert extended.clauses["Governing Law"] == older.clauses["Governing Law"]
    assert not (tmp_path / "cache").exists()


def test_extract_demo_does_not_ask_the_questions_of_the_existing_types(
    contract, fake_qa, tmp_path, monkeypatch
):
    settings = use_demo_dir(monkeypatch, tmp_path)
    older = older_cache(contract.read_text(encoding="utf-8"), settings)
    write_cache(older, tmp_path / "demo_cache" / "contract.json")
    main(["extract", "--demo", str(contract)])
    assert {question for question, _ in fake_qa.calls} == extra_questions(settings)


def test_extract_demo_reports_a_cache_hit_when_no_type_is_missing(
    contract, fake_qa, tmp_path, monkeypatch, capsys
):
    use_demo_dir(monkeypatch, tmp_path)
    main(["extract", "--demo", str(contract)])
    capsys.readouterr()
    assert main(["extract", "--demo", str(contract)]) == 0
    assert "cache hit" in capsys.readouterr().out


def test_import_demo_without_thesis_data_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DIPLOMSKI_DIR", str(tmp_path))
    assert main(["import-demo"]) == 1
    assert "error" in capsys.readouterr().err
