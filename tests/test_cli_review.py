import json

import pytest

from contract_assistant import cli
from contract_assistant.cli import main
from contract_assistant.config import ROOT
from contract_assistant.review import Flag, RuleFileError
from tests.support.fakes import EchoTextClient

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
OTHER_TEXT = ROOT / "demo" / "long_outsourcing_agreement.txt"
NAME = "short_supply_agreement"
OTHER_NAME = "long_outsourcing_agreement"
TERM_COUNT = 8
CLAUSE_COUNT = 49


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def install_client(monkeypatch, client) -> None:
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: client)


def table_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("|")]


def test_review_prints_the_missing_cap_flag(capsys):
    assert main(["review", str(DEMO_TEXT)]) == 0
    assert "No cap on liability found" in capsys.readouterr().out


def test_review_json_is_a_list_of_flags(capsys):
    assert main(["review", str(DEMO_TEXT), "--json"]) == 0
    flags = [Flag.model_validate(entry) for entry in json.loads(capsys.readouterr().out)]
    assert flags


def test_review_with_a_bad_rule_file_exits_with_the_message(monkeypatch, capsys):
    def broken(known_types=None):
        raise RuleFileError("unknown clause types in review_rules.json: Nope")

    monkeypatch.setattr(cli, "load_rules", broken)
    assert main(["review", str(DEMO_TEXT)]) == 1
    assert "error: unknown clause types in review_rules.json: Nope" in capsys.readouterr().err


def test_compare_prints_both_tables_with_the_document_names(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["compare", str(DEMO_TEXT), str(OTHER_TEXT)]) == 0
    lines = table_lines(capsys.readouterr().out)
    headers = [line for line in lines if NAME in line and OTHER_NAME in line]
    assert [line.split("|")[1].strip() for line in headers] == ["Term", "Clause"]


def test_compare_marks_differing_rows(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    main(["compare", str(DEMO_TEXT), str(OTHER_TEXT)])
    assert "≠" in capsys.readouterr().out


def test_compare_json_has_the_terms_and_the_clauses(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["compare", str(DEMO_TEXT), str(OTHER_TEXT), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["documents"] == [NAME, OTHER_NAME]
    assert len(data["terms"]) == TERM_COUNT
    assert len(data["clauses"]) == CLAUSE_COUNT


def test_compare_without_a_generation_client_prints_the_clause_table_only(monkeypatch, capsys):
    install_client(monkeypatch, None)
    assert main(["compare", str(DEMO_TEXT), str(OTHER_TEXT)]) == 0
    captured = capsys.readouterr()
    assert [line.split("|")[1].strip() for line in table_lines(captured.out)[:1]] == ["Clause"]
    assert "key terms need a generation model" in captured.err


def test_report_writes_markdown_and_docx_named_after_the_document(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["report", str(DEMO_TEXT), "--out", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == [f"{NAME}.docx", f"{NAME}.md"]
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_report_notes_the_missing_summary_and_key_terms(tmp_path, capsys):
    main(["report", str(DEMO_TEXT), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert "no saved summary; run summarize first" in err
    assert "no saved key terms; run term-sheet first" in err


def test_report_includes_the_saved_summary(monkeypatch, tmp_path):
    install_client(monkeypatch, EchoTextClient())
    monkeypatch.setenv("GEN_MODEL", EchoTextClient.model)
    assert main(["summarize", str(DEMO_TEXT)]) == 0
    saved = next((tmp_path / "cache" / "summaries").glob("*.json"))
    text = json.loads(saved.read_text(encoding="utf-8"))["text"]
    out = tmp_path / "out"
    main(["report", str(DEMO_TEXT), "--out", str(out)])
    assert text in (out / f"{NAME}.md").read_text(encoding="utf-8")
