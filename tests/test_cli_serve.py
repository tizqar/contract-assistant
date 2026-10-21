import pytest

from contract_assistant import cli
from contract_assistant.background import precompute_all
from contract_assistant.summary import SummaryStore
from contract_assistant.term_sheet import TermSheetStore
from contract_assistant.ui import app
from tests.support.fakes import EchoTextClient, FakeRouter


@pytest.fixture
def served(monkeypatch, tmp_path):
    state = {"launched": {}, "background": [], "gen": EchoTextClient()}
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EXTRACTOR_MODEL_ID", "models/does-not-exist")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "build_router", lambda *args: FakeRouter())
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: state["gen"])
    monkeypatch.setattr(
        cli,
        "run_in_background",
        lambda target, *args: state["background"].append((target, args)),
    )
    monkeypatch.setattr(
        app,
        "launch_app",
        lambda demo, services, share, port: state["launched"].update(services=services),
    )
    return state


def test_serve_starts_the_precompute_in_the_background(served, capsys):
    assert cli.main(["serve"]) == 0
    assert [target for target, _ in served["background"]] == [precompute_all]
    assert cli.PRECOMPUTE_NOTICE in capsys.readouterr().err


def test_serve_shares_its_stores_with_the_precompute(served):
    cli.main(["serve"])
    services = served["launched"]["services"]
    args = served["background"][0][1]
    assert isinstance(services.summaries, SummaryStore)
    assert isinstance(services.term_sheets, TermSheetStore)
    assert args[3] is served["gen"]
    assert args[4] is services.summaries
    assert args[5] is services.term_sheets


def test_serve_skips_the_precompute_with_the_flag(served, capsys):
    assert cli.main(["serve", "--no-precompute"]) == 0
    assert served["background"] == []
    assert cli.PRECOMPUTE_NOTICE not in capsys.readouterr().err


def test_serve_skips_the_precompute_without_a_generation_client(served):
    served["gen"] = None
    assert cli.main(["serve"]) == 0
    assert served["background"] == []
