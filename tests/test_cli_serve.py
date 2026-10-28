import pytest

from contract_assistant import cli
from contract_assistant.background import precompute_all
from contract_assistant.fulltext import FullTextStore
from contract_assistant.obligations import ObligationStore
from contract_assistant.summary import SummaryStore
from contract_assistant.term_sheet import TermSheetStore
from contract_assistant.ui import app
from tests.support.fakes import EchoTextClient, FakeRouter

# Nothing listens on the discard port, so a connection there fails at once.
UNREACHABLE = "postgresql+psycopg://nobody:nothing@127.0.0.1:9/nowhere"


@pytest.fixture
def served(monkeypatch, tmp_path):
    state = {"launched": {}, "background": [], "gen": EchoTextClient()}
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
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
    monkeypatch.setattr(
        cli,
        "run_server",
        lambda web_app, host, port: state["launched"].update(
            services=web_app.state.workspace.services,
            database=web_app.state.workspace.database,
            precompute=web_app.state.precompute,
            host=host,
            port=port,
        ),
    )
    return state


def test_serve_runs_the_web_app_by_default(served):
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert served["launched"]["host"] == cli.DEFAULT_HOST
    assert served["launched"]["port"] == 8123


def test_serve_opens_the_database_of_the_settings(served, tmp_path):
    cli.main(["serve"])
    assert served["launched"]["database"].url == f"sqlite:///{tmp_path / 'app.db'}"


def test_serve_without_its_database_says_how_to_start_it(served, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE)
    assert cli.main(["serve"]) == cli.EXIT_ERROR
    assert "make db-up" in capsys.readouterr().err
    assert "host" not in served["launched"]


def test_serve_can_start_the_gradio_app(served):
    assert cli.main(["serve", "--ui", "gradio"]) == 0
    assert "host" not in served["launched"]
    assert served["launched"]["services"].rules


def test_the_web_app_prepares_the_summaries_itself(served, capsys):
    assert cli.main(["serve"]) == 0
    assert served["launched"]["precompute"] is True
    assert served["background"] == []
    assert cli.PRECOMPUTE_NOTICE in capsys.readouterr().err


def test_the_gradio_app_prepares_the_summaries_in_the_cache_folder(served, capsys):
    assert cli.main(["serve", "--ui", "gradio"]) == 0
    assert [target for target, _ in served["background"]] == [precompute_all]
    assert cli.PRECOMPUTE_NOTICE in capsys.readouterr().err


def test_the_gradio_precompute_shares_the_stores_of_the_app(served):
    cli.main(["serve", "--ui", "gradio"])
    services = served["launched"]["services"]
    args = served["background"][0][1]
    assert isinstance(services.summaries, SummaryStore)
    assert isinstance(services.term_sheets, TermSheetStore)
    assert args[3] is served["gen"]
    assert args[4] is services.summaries
    assert args[5] is services.term_sheets
    assert args[7] is services.obligations


def test_serve_skips_the_precompute_with_the_flag(served, capsys):
    assert cli.main(["serve", "--no-precompute"]) == 0
    assert served["launched"]["precompute"] is False
    assert cli.PRECOMPUTE_NOTICE not in capsys.readouterr().err


def test_serve_skips_the_precompute_without_a_generation_client(served):
    served["gen"] = None
    assert cli.main(["serve"]) == 0
    assert served["launched"]["precompute"] is False


def test_serve_passes_the_review_rules_to_the_app(served):
    cli.main(["serve"])
    assert served["launched"]["services"].rules


def test_serve_passes_the_obligation_and_full_text_stores_to_the_app(served):
    cli.main(["serve"])
    services = served["launched"]["services"]
    assert isinstance(services.obligations, ObligationStore)
    assert isinstance(services.fulltexts, FullTextStore)


def test_serve_fresh_starts_with_an_empty_cache_and_database(served, capsys):
    assert cli.main(["serve", "--fresh"]) == 0
    services = served["launched"]["services"]
    assert services.settings.cache_dir.is_dir()
    assert "contract-assistant-fresh-" in services.settings.cache_dir.name
    assert not services.settings.demo_cache_dir.exists()
    database = served["launched"]["database"]
    assert database.is_sqlite and str(services.settings.cache_dir) in database.url
    assert served["launched"]["precompute"] is False, "nothing to prepare on a fresh start"
    assert "fresh start" in capsys.readouterr().err
