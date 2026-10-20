import pytest

from contract_assistant import cli
from contract_assistant.clause_types import app_clause_types
from contract_assistant.cli import main
from contract_assistant.config import ROOT, load_settings
from contract_assistant.explain import Explanation
from contract_assistant.router.clients import LLMError
from contract_assistant.store import write_cache
from contract_assistant.summary import Summary
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"
DEMO_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
LAW = "Governing Law"
NOT_FOUND_TYPE = "Source Code Escrow"
LAW_QUOTE = "This agreement shall be governed by the Laws of England"
MISQUOTE = 'The contract says "this text is not in the contract at all".'


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def install_client(monkeypatch, client) -> None:
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: client)


def summaries_dir(tmp_path):
    return tmp_path / "cache" / "summaries"


def clause_names() -> list[str]:
    return [c.name for c in app_clause_types(load_settings())]


def empty_cache_file(tmp_path, clauses=None):
    settings = load_settings()
    path = tmp_path / "empty.json"
    write_cache(make_cache("A short contract text.", settings, clauses or {}), path)
    return path


def test_summarize_prints_the_reply_and_the_clauses_it_is_based_on(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["summarize", str(DEMO_TEXT)]) == 0
    output = capsys.readouterr().out
    assert "The contract states" in output
    assert "Based on: " in output
    assert f"{LAW} (found)" in output


def test_summarize_prints_the_model_and_the_time(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    main(["summarize", str(DEMO_TEXT)])
    assert "model: fake-gen · " in capsys.readouterr().out


def test_summarize_json_is_a_summary(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["summarize", str(DEMO_TEXT), "--json"]) == 0
    summary = Summary.model_validate_json(capsys.readouterr().out)
    assert (summary.doc_id, summary.model) == (DEMO_DOC_ID, "fake-gen")
    assert summary.text is not None


def test_summarize_saves_the_summary_under_the_cache_folder(monkeypatch, tmp_path):
    install_client(monkeypatch, EchoTextClient())
    main(["summarize", str(DEMO_TEXT)])
    assert len(list(summaries_dir(tmp_path).glob("*.json"))) == 1


def test_second_summarize_reads_the_saved_summary_without_asking_the_model(
    monkeypatch, capsys
):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["summarize", str(DEMO_TEXT), "--json"])
    first = Summary.model_validate_json(capsys.readouterr().out)
    assert main(["summarize", str(DEMO_TEXT), "--json"]) == 0
    second = Summary.model_validate_json(capsys.readouterr().out)
    assert len(client.text_calls) == 1
    assert second == first


def test_fresh_asks_the_model_again(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["summarize", str(DEMO_TEXT)])
    assert main(["summarize", str(DEMO_TEXT), "--fresh"]) == 0
    assert len(client.text_calls) == 2


def test_summarize_sends_only_the_clause_items_not_the_contract_text(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["summarize", str(DEMO_TEXT)])
    [(_, user, _)] = client.text_calls
    assert user.startswith("Items:\n")
    assert f'- {LAW} (found): "{LAW_QUOTE}' in user


@pytest.mark.parametrize(
    "target", [str(DEMO_TEXT), str(DEMO_CACHE), DEMO_DOC_ID], ids=["document", "cache", "id"]
)
def test_summarize_takes_a_document_a_cache_file_or_a_document_id(
    target, monkeypatch, capsys
):
    install_client(monkeypatch, EchoTextClient())
    assert main(["summarize", target]) == 0
    assert "Based on: " in capsys.readouterr().out


def test_summarize_without_a_generation_client_exits_one_naming_the_settings(
    monkeypatch, capsys
):
    install_client(monkeypatch, None)
    assert main(["summarize", str(DEMO_TEXT)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: summarize needs a generation model (")
    assert "LLM_BACKEND=" in error and "GEN_MODEL=" in error


def test_summarize_without_found_or_check_items_exits_one(monkeypatch, tmp_path, capsys):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    assert main(["summarize", str(empty_cache_file(tmp_path))]) == 1
    assert "nothing to summarize" in capsys.readouterr().err
    assert client.text_calls == []


def test_summarize_with_a_failing_model_exits_one_with_the_message(
    monkeypatch, tmp_path, capsys
):
    install_client(monkeypatch, FakeLLMClient([LLMError("timed out")]))
    assert main(["summarize", str(DEMO_TEXT)]) == 1
    assert capsys.readouterr().err == "error: the language model did not answer: timed out\n"
    assert not summaries_dir(tmp_path).exists()


def test_summarize_prints_that_a_twice_misquoting_summary_is_withheld(monkeypatch, capsys):
    install_client(monkeypatch, FakeLLMClient([MISQUOTE, MISQUOTE]))
    assert main(["summarize", str(DEMO_TEXT)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("Summary withheld: ")
    assert "this text is not in the contract at all" in output


def test_explain_prints_the_type_the_language_and_the_text(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["explain", str(DEMO_TEXT), "--type", LAW]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"**{LAW}** · English"
    assert lines[1].startswith("The contract states")


def test_explain_json_is_an_explanation(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["explain", str(DEMO_TEXT), "--type", LAW, "--json"]) == 0
    explanation = Explanation.model_validate_json(capsys.readouterr().out)
    assert (explanation.clause_type, explanation.language) == (LAW, "en")
    assert explanation.quote.startswith(LAW_QUOTE)


def test_explain_passes_the_language_to_the_model(monkeypatch, capsys):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    assert main(["explain", str(DEMO_TEXT), "--type", LAW, "--language", "sr"]) == 0
    [(system, _, _)] = client.text_calls
    assert "Serbian" in system
    assert capsys.readouterr().out.startswith(f"**{LAW}** · Serbian\n")


def test_explain_asks_for_the_quote_of_the_clause(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    main(["explain", str(DEMO_TEXT), "--type", LAW])
    [(_, user, _)] = client.text_calls
    assert user.startswith(f"Clause type: {LAW}\nClause: \"{LAW_QUOTE}")


def test_explain_with_an_unknown_type_exits_two_and_lists_the_names(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["explain", str(DEMO_TEXT), "--type", "Governing Laws"]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "error: unknown clause types: Governing Laws"
    assert lines[1:] == clause_names()


def test_explain_with_a_clause_that_was_not_found_exits_one(monkeypatch, capsys):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    assert main(["explain", str(DEMO_TEXT), "--type", NOT_FOUND_TYPE]) == 1
    assert capsys.readouterr().err == f"error: {NOT_FOUND_TYPE} was not found in this contract\n"
    assert client.text_calls == []


def test_explain_with_a_type_missing_from_the_cache_exits_one(monkeypatch, tmp_path, capsys):
    install_client(monkeypatch, EchoTextClient())
    path = empty_cache_file(tmp_path, {"Parties": [("A short", 0.9)]})
    assert main(["explain", str(path), "--type", LAW]) == 1
    assert capsys.readouterr().err == f"error: {LAW} is not in the analysis of this contract\n"


def test_explain_without_a_generation_client_exits_one_naming_the_settings(
    monkeypatch, capsys
):
    install_client(monkeypatch, None)
    assert main(["explain", str(DEMO_TEXT), "--type", LAW]) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: explain needs a generation model (")
    assert "LLM_BACKEND=" in error and "GEN_MODEL=" in error


def test_explain_with_a_failing_model_exits_one_with_the_message(monkeypatch, capsys):
    install_client(monkeypatch, FakeLLMClient([LLMError("timed out")]))
    assert main(["explain", str(DEMO_TEXT), "--type", LAW]) == 1
    assert capsys.readouterr().err == "error: the language model did not answer: timed out\n"


def test_explain_prints_that_a_misquoting_explanation_is_withheld(monkeypatch, capsys):
    install_client(monkeypatch, FakeLLMClient([MISQUOTE]))
    assert main(["explain", str(DEMO_TEXT), "--type", LAW]) == 0
    assert capsys.readouterr().out.startswith("Explanation withheld: ")


def test_explain_rejects_a_language_that_is_not_offered():
    with pytest.raises(SystemExit) as raised:
        main(["explain", str(DEMO_TEXT), "--type", LAW, "--language", "de"])
    assert raised.value.code == 2


def test_withheld_summary_is_not_saved_so_the_next_run_asks_again(
    monkeypatch, tmp_path, capsys
):
    client = FakeLLMClient([MISQUOTE, MISQUOTE, "A plain summary."])
    install_client(monkeypatch, client)
    assert main(["summarize", str(DEMO_TEXT)]) == 0
    assert not summaries_dir(tmp_path).exists()
    assert main(["summarize", str(DEMO_TEXT)]) == 0
    assert "A plain summary." in capsys.readouterr().out
    assert len(client.text_calls) == 3
