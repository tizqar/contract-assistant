import pytest

from contract_assistant import cli
from contract_assistant.clause_types import load_clause_types
from contract_assistant.cli import main
from contract_assistant.config import ROOT, load_settings
from contract_assistant.extract.extractor import Extractor
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
    clause_types = load_clause_types(settings.clause_types_path)
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
    names = [c.name for c in load_clause_types(load_settings().clause_types_path)]
    assert [next(n for n in names if line.startswith(n + " ")) for line in lines] == names


def test_show_accepts_a_cache_file(capsys):
    assert main(["show", str(DEMO_CACHE)]) == 0
    assert "Supply Agreement.txt" in capsys.readouterr().out.splitlines()[0]


def test_show_without_a_cache_returns_one(contract, capsys):
    assert main(["show", str(contract)]) == 1
    assert "no cache for" in capsys.readouterr().err


def test_extract_with_a_missing_model_prints_the_hint_and_returns_two(
    contract, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("EXTRACTOR_MODEL_ID", str(tmp_path / "missing"))
    assert main(["extract", str(contract)]) == 2
    assert "IMPLEMENTATION.md" in capsys.readouterr().err
