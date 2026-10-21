import json

import pytest

from contract_assistant import cli
from contract_assistant.cli import main
from contract_assistant.config import ROOT
from contract_assistant.dates import KeyDate
from contract_assistant.feedback import FEEDBACK_FILE, Feedback, record_feedback
from contract_assistant.portfolio import headers
from tests.support.fakes import EchoTextClient

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
DEMO_NAMES = (
    "short_supply_agreement",
    "long_outsourcing_agreement",
    "medium_sponsorship_agreement",
)
NO_SHEET_MESSAGE = (
    "error: dates needs the key terms; run term-sheet first or set a generation model"
)
FIRST_ID = "sha256:" + "ab" * 32


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def install_client(monkeypatch, client) -> None:
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: client)


def summary_files(tmp_path) -> set[str]:
    return {path.name for path in (tmp_path / "cache" / "summaries").glob("*.json")}


def verdict(liked: bool, question: str, at: str) -> Feedback:
    return Feedback(
        doc_id=FIRST_ID,
        question=question,
        route_kind="clause",
        clause_types=["Governing Law"],
        liked=liked,
        at=at,
    )


def test_summarize_in_serbian_names_the_language_in_the_system_prompt(monkeypatch):
    client = EchoTextClient()
    install_client(monkeypatch, client)
    assert main(["summarize", str(DEMO_TEXT), "--language", "sr"]) == 0
    assert "Serbian" in client.text_calls[0][0]


def test_summarize_in_serbian_is_saved_apart_from_the_english_one(monkeypatch, tmp_path):
    install_client(monkeypatch, EchoTextClient())
    main(["summarize", str(DEMO_TEXT)])
    english = summary_files(tmp_path)
    main(["summarize", str(DEMO_TEXT), "--language", "sr"])
    assert len(summary_files(tmp_path)) == 2
    assert english < summary_files(tmp_path)


def test_dates_prints_a_table_with_the_agreement_date(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["dates", str(DEMO_TEXT)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("| Date | When | From | How |")
    assert "| Agreement date |" in output


def test_dates_json_is_a_list_of_key_dates(monkeypatch, capsys):
    install_client(monkeypatch, EchoTextClient())
    assert main(["dates", str(DEMO_TEXT), "--json"]) == 0
    dates = [KeyDate.model_validate(item) for item in json.loads(capsys.readouterr().out)]
    assert "Agreement date" in [item.label for item in dates]


def test_dates_without_a_model_or_a_saved_sheet_exits_with_a_message(monkeypatch, capsys):
    install_client(monkeypatch, None)
    assert main(["dates", str(DEMO_TEXT)]) == 1
    assert NO_SHEET_MESSAGE in capsys.readouterr().err


def test_portfolio_lists_the_three_demo_contracts_under_the_header(capsys):
    assert main(["portfolio"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "| " + " | ".join(headers()) + " |"
    assert len(lines) == 2 + len(DEMO_NAMES)


def test_portfolio_rows_name_each_demo_contract(capsys):
    main(["portfolio"])
    output = capsys.readouterr().out
    assert all(name in output for name in DEMO_NAMES)


def test_portfolio_csv_writes_the_file_and_prints_its_path(tmp_path, capsys):
    path = tmp_path / "portfolio.csv"
    assert main(["portfolio", "--csv", str(path)]) == 0
    assert capsys.readouterr().out.strip() == str(path)
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("Contract,Size")


def test_portfolio_json_is_a_list_with_a_row_per_contract(capsys):
    assert main(["portfolio", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == len(DEMO_NAMES)


def test_feedback_without_a_file_says_there_is_none(capsys):
    assert main(["feedback"]) == 0
    assert capsys.readouterr().out.strip() == "no feedback yet"


def test_feedback_prints_the_counts_and_one_line_per_verdict(tmp_path, capsys):
    path = tmp_path / "cache" / FEEDBACK_FILE
    record_feedback(path, verdict(True, "Who governs?", "2026-10-20T10:00:00+00:00"))
    record_feedback(path, verdict(False, "Is there a cap?", "2026-10-20T11:00:00+00:00"))
    assert main(["feedback"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "liked 1 · disliked 1",
        f"2026-10-20T10:00:00+00:00 liked {FIRST_ID[:16]} Who governs?",
        f"2026-10-20T11:00:00+00:00 disliked {FIRST_ID[:16]} Is there a cap?",
    ]


def test_feedback_json_is_a_list_of_verdicts(tmp_path, capsys):
    path = tmp_path / "cache" / FEEDBACK_FILE
    record_feedback(path, verdict(True, "Who governs?", "2026-10-20T10:00:00+00:00"))
    assert main(["feedback", "--json"]) == 0
    assert Feedback.model_validate(json.loads(capsys.readouterr().out)[0]).liked is True
