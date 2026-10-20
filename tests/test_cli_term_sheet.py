import json

import pytest

from contract_assistant import cli
from contract_assistant.cli import main
from contract_assistant.config import ROOT, load_settings
from contract_assistant.router.clients import LLMError
from contract_assistant.store import write_cache
from contract_assistant.term_sheet import LABELS, TermSheet
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
OTHER_TEXT = ROOT / "demo" / "long_outsourcing_agreement.txt"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"
DEMO_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
NAME = "short_supply_agreement"
OTHER_NAME = "long_outsourcing_agreement"
NOT_JSON = "I cannot fill in this term sheet."


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def install_client(monkeypatch, client) -> None:
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: client)


def sheets_dir(tmp_path):
    return tmp_path / "cache" / "term_sheets"


def table_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("|")]


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_term_sheet_table_names_the_document_in_its_header(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["term-sheet", str(DEMO_TEXT)]) == 0
    assert cells(table_lines(capsys.readouterr().out)[0]) == ["Term", NAME]


def test_term_sheet_table_has_one_row_per_field_with_its_label(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    main(["term-sheet", str(DEMO_TEXT)])
    rows = table_lines(capsys.readouterr().out)[2:]
    assert [cells(row)[0] for row in rows] == list(LABELS.values())


def test_term_sheet_table_shows_a_value_copied_from_the_clause(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    main(["term-sheet", str(DEMO_TEXT)])
    rows = {cells(row)[0]: cells(row)[1] for row in table_lines(capsys.readouterr().out)[2:]}
    assert rows["Governing law"].startswith("This agreement shall be")


def test_two_targets_give_two_columns(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["term-sheet", str(DEMO_TEXT), str(OTHER_TEXT)]) == 0
    assert cells(table_lines(capsys.readouterr().out)[0]) == ["Term", NAME, OTHER_NAME]


def test_json_is_an_object_keyed_by_document_name_of_term_sheets(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["term-sheet", str(DEMO_TEXT), str(OTHER_TEXT), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data) == [NAME, OTHER_NAME]
    sheets = {name: TermSheet.model_validate(value) for name, value in data.items()}
    assert sheets[NAME].doc_id == DEMO_DOC_ID


def test_term_sheet_is_saved_under_the_cache_folder(monkeypatch, tmp_path):
    install_client(monkeypatch, EchoTextClient())
    main(["term-sheet", str(DEMO_TEXT)])
    assert len(list(sheets_dir(tmp_path).glob("*.json"))) == 1


def test_second_run_reads_the_saved_sheet_without_asking_the_model(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["term-sheet", str(DEMO_TEXT)])
    assert main(["term-sheet", str(DEMO_TEXT)]) == 0
    assert len(client.text_calls) == 1


def test_fresh_asks_the_model_again(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["term-sheet", str(DEMO_TEXT)])
    assert main(["term-sheet", str(DEMO_TEXT), "--fresh"]) == 0
    assert len(client.text_calls) == 2


def test_a_reply_that_is_not_json_twice_prints_the_note_and_saves_nothing(
    monkeypatch, tmp_path, capsys
):
    install_client(monkeypatch, FakeLLMClient([NOT_JSON, NOT_JSON]))
    assert main(["term-sheet", str(DEMO_TEXT)]) == 0
    assert f"{NAME}: term sheet withheld: " in capsys.readouterr().out
    assert not sheets_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "target", [str(DEMO_TEXT), str(DEMO_CACHE), DEMO_DOC_ID], ids=["document", "cache", "id"]
)
def test_term_sheet_takes_a_document_a_cache_file_or_a_document_id(
    target, monkeypatch, capsys
):
    install_client(monkeypatch, EchoTextClient())
    assert main(["term-sheet", target]) == 0
    assert cells(table_lines(capsys.readouterr().out)[0]) == ["Term", NAME]


def test_term_sheet_without_a_generation_client_exits_one_naming_the_settings(
    monkeypatch, capsys
):
    install_client(monkeypatch, None)
    assert main(["term-sheet", str(DEMO_TEXT)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: term-sheet needs a generation model (")
    assert "GEN_MODEL=" in error


def test_a_cache_without_key_term_clauses_exits_one(monkeypatch, tmp_path, capsys):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    path = tmp_path / "empty.json"
    write_cache(make_cache("A short contract text.", load_settings(), {}), path)
    assert main(["term-sheet", str(path)]) == 1
    assert capsys.readouterr().err == "error: no key-term clause was found in test\n"
    assert client.text_calls == []


def test_a_failing_model_exits_one_with_the_message(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, FakeLLMClient([LLMError("timed out")]))
    assert main(["term-sheet", str(DEMO_TEXT)]) == 1
    assert capsys.readouterr().err == "error: the language model did not answer: timed out\n"
    assert not sheets_dir(tmp_path).exists()
