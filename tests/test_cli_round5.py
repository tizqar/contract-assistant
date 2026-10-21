import json

import pytest

from contract_assistant import cli
from contract_assistant.cli import main
from contract_assistant.config import ROOT, load_settings
from contract_assistant.fulltext import FullTextSummary
from contract_assistant.obligations import ObligationSheet
from contract_assistant.store import write_cache
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
GEN_MODEL = EchoTextClient.model
PARTY = "This agreement"
ACTION = "shall be governed by the Laws of England"
CLAUSE = "Governing Law"
VALID_REPLY = json.dumps(
    [{"clause_type": CLAUSE, "party": PARTY, "modality": "shall", "action": ACTION}]
)
NOT_JSON = "I cannot list the obligations."
SMALL_TEXT = 'The Customer shall pay the "fees" monthly.'


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GEN_MODEL", GEN_MODEL)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def install_client(monkeypatch, client) -> None:
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: client)


def obligation_client(replies: list[str]) -> FakeLLMClient:
    return FakeLLMClient(replies, model=GEN_MODEL)


def small_cache(tmp_path) -> str:
    path = tmp_path / "small.json"
    write_cache(make_cache(SMALL_TEXT, load_settings()), path)
    return str(path)


def test_obligations_prints_the_party_heading_and_the_entry(monkeypatch, capsys):
    install_client(monkeypatch, obligation_client([VALID_REPLY]))
    assert main(["obligations", str(DEMO_TEXT)]) == 0
    output = capsys.readouterr().out
    assert f"**{PARTY}**" in output
    assert f'- must: "{ACTION}" ({CLAUSE})' in output


def test_obligations_are_saved_under_the_cache_folder(monkeypatch, tmp_path):
    install_client(monkeypatch, obligation_client([VALID_REPLY]))
    main(["obligations", str(DEMO_TEXT)])
    assert len(list((tmp_path / "cache" / "obligations").glob("*.json"))) == 1


def test_obligations_second_run_makes_no_model_call(monkeypatch):
    client = obligation_client([VALID_REPLY])
    install_client(monkeypatch, client)
    main(["obligations", str(DEMO_TEXT)])
    assert main(["obligations", str(DEMO_TEXT)]) == 0
    assert len(client.text_calls) == 1


def test_obligations_fresh_asks_the_model_again(monkeypatch):
    client = obligation_client([VALID_REPLY, VALID_REPLY])
    install_client(monkeypatch, client)
    main(["obligations", str(DEMO_TEXT)])
    assert main(["obligations", str(DEMO_TEXT), "--fresh"]) == 0
    assert len(client.text_calls) == 2


def test_obligations_json_is_an_obligation_sheet(monkeypatch, capsys):
    install_client(monkeypatch, obligation_client([VALID_REPLY]))
    assert main(["obligations", str(DEMO_TEXT), "--json"]) == 0
    sheet = ObligationSheet.model_validate_json(capsys.readouterr().out)
    assert [o.party for o in sheet.obligations] == [PARTY]


def test_obligations_invalid_reply_twice_prints_the_withheld_line_and_saves_nothing(
    monkeypatch, tmp_path, capsys
):
    install_client(monkeypatch, obligation_client([NOT_JSON, NOT_JSON]))
    assert main(["obligations", str(DEMO_TEXT)]) == 0
    assert "Obligations withheld: " in capsys.readouterr().out
    assert not (tmp_path / "cache" / "obligations").exists()


def test_obligations_without_a_client_exits_one(monkeypatch, capsys):
    install_client(monkeypatch, None)
    assert main(["obligations", str(DEMO_TEXT)]) == 1
    assert capsys.readouterr().err.startswith("error: obligations needs a generation model (")


def test_obligations_without_clauses_exits_one(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, obligation_client([]))
    assert main(["obligations", small_cache(tmp_path)]) == 1
    assert capsys.readouterr().err == f"{cli.NO_OBLIGATIONS}\n"


def test_fulltext_prints_the_overview_and_the_section_line(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["fulltext", small_cache(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert 'The contract states "fees".' in output
    assert f"1. (chars 0-{len(SMALL_TEXT)})" in output


def test_fulltext_prints_the_progress_on_stderr(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, EchoTextClient())
    main(["fulltext", small_cache(tmp_path)])
    assert capsys.readouterr().err.splitlines() == ["section 1 of 1 · 0/2", "overview · 1/2"]


def test_fulltext_is_saved_under_the_cache_folder(monkeypatch, tmp_path):
    install_client(monkeypatch, EchoTextClient())
    main(["fulltext", small_cache(tmp_path)])
    assert len(list((tmp_path / "cache" / "fulltext").glob("*.json"))) == 1


def test_fulltext_json_is_a_full_text_summary(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["fulltext", small_cache(tmp_path), "--json"]) == 0
    summary = FullTextSummary.model_validate_json(capsys.readouterr().out)
    assert len(summary.sections) == 1


def test_fulltext_second_run_makes_no_model_call(monkeypatch, tmp_path):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    path = small_cache(tmp_path)
    main(["fulltext", path])
    calls = len(client.text_calls)
    assert main(["fulltext", path]) == 0
    assert len(client.text_calls) == calls


def test_report_notes_the_missing_obligations_and_overview(tmp_path, capsys):
    main(["report", str(DEMO_TEXT), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert cli.NO_SAVED_OBLIGATIONS in err
    assert cli.NO_SAVED_FULLTEXT in err


def test_report_includes_the_saved_obligations_and_overview(monkeypatch, tmp_path):
    install_client(monkeypatch, obligation_client([VALID_REPLY]))
    main(["obligations", str(DEMO_TEXT)])
    install_client(monkeypatch, EchoTextClient())
    main(["fulltext", str(DEMO_TEXT)])
    out = tmp_path / "out"
    main(["report", str(DEMO_TEXT), "--out", str(out)])
    report = (out / "short_supply_agreement.md").read_text(encoding="utf-8")
    assert ACTION in report
    assert "The contract states" in report
