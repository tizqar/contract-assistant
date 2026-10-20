import dataclasses
import json
from pathlib import Path

import pytest

from contract_assistant import cli
from contract_assistant.clause_types import load_clause_types
from contract_assistant.cli import main
from contract_assistant.config import ROOT, Settings
from contract_assistant.evaluation.e2e import E2EReport, OracleRouter
from contract_assistant.evaluation.phrase_eval import PhrasingReport
from contract_assistant.evaluation.questions import EvalQuestion, load_questions
from contract_assistant.evaluation.router_eval import THRESHOLD_GRID, RouterRun
from contract_assistant.evaluation.summary_eval import SummaryReport
from contract_assistant.router.base import RouteResult
from contract_assistant.router.clients import LLMError
from contract_assistant.router.factory import UnknownBackendError
from contract_assistant.store import DocumentCache, HarnessImportError, write_cache
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient, FixedRouter

LAW = "Governing Law"
INSURANCE = "Insurance"
PARTIES = "Parties"
LAW_QUESTION = "Which law governs this agreement?"
INSURANCE_QUESTION = "Is insurance required?"
PARTIES_QUESTION = "Who are the parties?"
WEATHER_QUESTION = "What is the weather today?"
QUESTIONS = [
    EvalQuestion(id="q1", question=LAW_QUESTION, gold=[LAW], kind="single", split="dev"),
    EvalQuestion(
        id="q2", question=INSURANCE_QUESTION, gold=[INSURANCE], kind="single", split="dev"
    ),
    EvalQuestion(
        id="q3", question=PARTIES_QUESTION, gold=[PARTIES], kind="single", split="test"
    ),
    EvalQuestion(
        id="q4", question=WEATHER_QUESTION, gold=[], kind="out_of_scope", split="test"
    ),
]
DEV_QUESTIONS = [LAW_QUESTION, INSURANCE_QUESTION]
TEST_QUESTIONS = [PARTIES_QUESTION, WEATHER_QUESTION]

LLM_MODEL = "qwen3:1.7b"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_THRESHOLD = 0.5
TAU_LOW = 0.2
TAU_HIGH = 0.8
LATENCY_MS = 10.0

TEXT_ALPHA = "This Agreement is governed by the laws of Delaware. Acme Corp is the Supplier."
TEXT_BETA = "This Agreement is governed by the laws of Texas."
CLAUSES_ALPHA = {LAW: [("laws of Delaware", 0.9)], INSURANCE: [], PARTIES: [("Acme Corp", 0.5)]}
CLAUSES_BETA = {LAW: [("laws of Texas", 0.9)], INSURANCE: [], PARTIES: []}
GOLD = {
    "Alpha": {LAW: ["laws of Delaware"], INSURANCE: [], PARTIES: ["Acme Corp"]},
    "Beta": {LAW: ["laws of Texas"], INSURANCE: [], PARTIES: []},
}

DEV_SCORES = {
    LAW_QUESTION: {LAW: 0.8, INSURANCE: 0.4},
    INSURANCE_QUESTION: {LAW: 0.3, INSURANCE: 0.7},
}
BEST_THRESHOLD = 0.7

FAITHFUL = 'Delaware law governs: "laws of Delaware"'
UNQUOTED = "Texas law governs."
ALTERED = 'The insurer is "full cover"'
INVENTED = '"Ohio Ltd"'
LINE_PARTS = ("violations 2 of 5 replies (rate 0.400)", "unquoted 1", "failures 1", "invalid 1")


def reply(paragraph: str) -> str:
    return json.dumps({"paragraph": paragraph})


# One reply per call: the in-scope questions in order, each on alpha and then on beta.
MIXED_REPLIES = [
    reply(FAITHFUL),
    reply(UNQUOTED),
    reply(ALTERED),
    LLMError("timed out"),
    "not json",
    reply(INVENTED),
]


class FakeScorer:
    max_types = 5

    def __init__(self, scores: dict[str, dict[str, float]]):
        self.prepared = scores
        self.questions: list[str] = []

    def scores(self, question: str) -> dict[str, float]:
        self.questions.append(question)
        return self.prepared[question]


def refuse_build(*arguments):
    raise AssertionError("no router or client may be built")


def route_of(kind: str, *names: str) -> RouteResult:
    return RouteResult(clause_types=list(names), kind=kind, latency_ms=LATENCY_MS)


def fixed_router(kind: str) -> FixedRouter:
    routes = {
        LAW_QUESTION: route_of(kind, LAW),
        INSURANCE_QUESTION: route_of(kind, INSURANCE),
        PARTIES_QUESTION: route_of(kind, PARTIES),
        WEATHER_QUESTION: route_of(kind),
        cli.WARM_UP_QUESTION: route_of(kind),
    }
    router = FixedRouter(routes)
    router.kind = kind
    return router


def make_caches(settings: Settings) -> dict[str, DocumentCache]:
    return {
        "Alpha": make_cache(TEXT_ALPHA, settings, CLAUSES_ALPHA),
        "Beta": make_cache(TEXT_BETA, settings, CLAUSES_BETA),
    }


@pytest.fixture
def settings(tmp_path) -> Settings:
    questions = tmp_path / "router_eval.jsonl"
    questions.write_text(
        "".join(f"{q.model_dump_json()}\n" for q in QUESTIONS), encoding="utf-8"
    )
    return dataclasses.replace(
        Settings(),
        router_eval_path=questions,
        diplomski_dir=tmp_path / "thesis",
        demo_cache_dir=tmp_path / "demo_cache",
        cache_dir=tmp_path / "cache",
        extractor_path=tmp_path / "missing-model",
        llm_backend="ollama",
        llm_model=LLM_MODEL,
        embedding_model_id=EMBEDDING_MODEL,
        embedding_threshold=EMBEDDING_THRESHOLD,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
    )


@pytest.fixture
def results(tmp_path) -> Path:
    return tmp_path / "results"


@pytest.fixture(autouse=True)
def isolated_env(settings, results, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "RESULTS_DIR", results)
    monkeypatch.setattr(cli, "build_router", refuse_build)
    monkeypatch.setattr(cli, "build_embedding_router", refuse_build)
    monkeypatch.setattr(cli, "build_llm_client", refuse_build)


@pytest.fixture
def thesis(monkeypatch) -> list[tuple[Path, Path]]:
    opened = []

    def import_preds(preds_path, cuad_json, settings):
        opened.append((preds_path, cuad_json))
        return make_caches(settings)

    monkeypatch.setattr(cli, "import_harness_preds", import_preds)
    monkeypatch.setattr(cli, "load_cuad_gold", lambda cuad_json, clause_types: GOLD)
    return opened


@pytest.fixture
def demo_caches(settings):
    caches = make_caches(settings)
    write_cache(caches["Alpha"], settings.demo_cache_dir / "alpha.json")
    write_cache(caches["Beta"], settings.demo_cache_dir / "beta.json")


@pytest.fixture
def mixed_client() -> FakeLLMClient:
    return FakeLLMClient(MIXED_REPLIES, model=LLM_MODEL)


def install(monkeypatch, name: str, built) -> list[Settings]:
    received = []

    def build(settings, *clause_types):
        received.append(settings)
        return built

    monkeypatch.setattr(cli, name, build)
    return received


def install_router(monkeypatch, router) -> list[Settings]:
    return install(monkeypatch, "build_router", router)


def install_client(monkeypatch, client) -> list[Settings]:
    return install(monkeypatch, "build_llm_client", client)


def use_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: settings)


def reject_backend(monkeypatch, name: str) -> None:
    def reject(settings, *clause_types):
        raise UnknownBackendError("unknown LLM_BACKEND 'olama'")

    monkeypatch.setattr(cli, name, reject)


def file_names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def read_tuning(results: Path) -> dict:
    return json.loads((results / "embedding_threshold.json").read_text(encoding="utf-8"))


def read_run(results: Path, name: str) -> RouterRun:
    return RouterRun.model_validate_json((results / name).read_text(encoding="utf-8"))


def normalised(question: str) -> str:
    return question.casefold().strip().rstrip("?").strip()


def table_rows(text: str) -> list[list[str]]:
    lines = text.splitlines()[2:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def tune(monkeypatch) -> FakeScorer:
    scorer = FakeScorer(DEV_SCORES)
    monkeypatch.setattr(cli, "build_embedding_router", lambda settings, clause_types: scorer)
    return scorer


def test_router_tune_writes_the_threshold_with_the_best_dev_f1(monkeypatch, results):
    tune(monkeypatch)
    assert main(["router-tune"]) == 0
    tuning = read_tuning(results)
    assert tuning["threshold"] == BEST_THRESHOLD
    assert tuning["f1"] == pytest.approx(1.0)


def test_router_tune_chooses_a_threshold_of_the_grid(monkeypatch, results):
    tune(monkeypatch)
    main(["router-tune"])
    assert read_tuning(results)["threshold"] in THRESHOLD_GRID


def test_router_tune_writes_the_whole_sweep_under_two_decimal_keys(monkeypatch, results):
    tune(monkeypatch)
    main(["router-tune"])
    sweep = read_tuning(results)["sweep"]
    assert list(sweep) == [f"{threshold:.2f}" for threshold in THRESHOLD_GRID]
    assert sweep["0.70"] == pytest.approx(1.0)
    assert sweep["0.90"] == 0.0


def test_router_tune_records_the_model_the_split_and_the_question_count(monkeypatch, results):
    tune(monkeypatch)
    main(["router-tune"])
    tuning = read_tuning(results)
    assert (tuning["model"], tuning["split"], tuning["n_questions"]) == (
        EMBEDDING_MODEL,
        "dev",
        2,
    )


def test_router_tune_scores_only_the_dev_questions(monkeypatch):
    scorer = tune(monkeypatch)
    main(["router-tune"])
    assert set(scorer.questions) == set(DEV_QUESTIONS)


def test_router_tune_prints_the_result_and_how_to_use_it(monkeypatch, results, capsys):
    tune(monkeypatch)
    main(["router-tune"])
    first, second = capsys.readouterr().out.splitlines()
    assert first == f"threshold 0.70, dev F1 1.000 -> {results / 'embedding_threshold.json'}"
    assert second == "set EMBEDDING_THRESHOLD=0.70 in .env and .env.example to use it"


def test_router_tune_writes_no_other_file(monkeypatch, results):
    tune(monkeypatch)
    main(["router-tune"])
    assert file_names(results) == ["embedding_threshold.json"]


@pytest.mark.parametrize(
    ("choice", "name"),
    [
        ("llm", "router_llm_qwen3-1.7b_test.json"),
        ("embedding", "router_embedding_bge-base-en-v1.5_test.json"),
    ],
)
def test_router_eval_writes_the_run_file_named_after_router_model_and_split(
    choice, name, monkeypatch, results
):
    install_router(monkeypatch, fixed_router(choice))
    assert main(["router-eval", "--router", choice]) == 0
    assert file_names(results) == sorted([name, "router.md"])


@pytest.mark.parametrize(
    ("choice", "model", "threshold"),
    [("llm", LLM_MODEL, None), ("embedding", EMBEDDING_MODEL, EMBEDDING_THRESHOLD)],
)
def test_router_eval_run_records_router_model_split_and_threshold(
    choice, model, threshold, monkeypatch, results
):
    install_router(monkeypatch, fixed_router(choice))
    main(["router-eval", "--router", choice])
    [path] = results.glob("router_*.json")
    run = RouterRun.model_validate_json(path.read_text(encoding="utf-8"))
    assert (run.router, run.model, run.split, run.threshold) == (
        choice,
        model,
        "test",
        threshold,
    )


def test_router_eval_scores_the_test_questions_by_default(monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm"])
    run = read_run(results, "router_llm_qwen3-1.7b_test.json")
    assert [result.question for result in run.report.results] == TEST_QUESTIONS


def test_router_eval_split_option_selects_the_dev_questions(monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm", "--split", "dev"])
    run = read_run(results, "router_llm_qwen3-1.7b_dev.json")
    assert [result.question for result in run.report.results] == DEV_QUESTIONS


@pytest.mark.parametrize("choice", ["llm", "embedding"])
def test_router_eval_routes_the_warm_up_question_once_before_the_scored_ones(
    choice, monkeypatch
):
    router = fixed_router(choice)
    install_router(monkeypatch, router)
    main(["router-eval", "--router", choice])
    assert router.questions == [cli.WARM_UP_QUESTION, *TEST_QUESTIONS]


def test_warm_up_question_is_not_an_evaluation_question():
    evaluation = {normalised(q.question) for q in load_questions(Settings().router_eval_path)}
    assert normalised(cli.WARM_UP_QUESTION) not in evaluation


def test_router_table_has_one_row_per_run_file_sorted_by_file_name(monkeypatch, results):
    for choice in ("llm", "embedding"):
        install_router(monkeypatch, fixed_router(choice))
        main(["router-eval", "--router", choice])
    rows = table_rows((results / "router.md").read_text(encoding="utf-8"))
    assert [row[:3] for row in rows] == [
        ["embedding", EMBEDDING_MODEL, "test"],
        ["llm", LLM_MODEL, "test"],
    ]


def test_tables_ignore_the_result_files_of_the_other_commands(
    thesis, demo_caches, mixed_client, monkeypatch, results
):
    tune(monkeypatch)
    install_client(monkeypatch, mixed_client)
    install_router(monkeypatch, fixed_router("llm"))
    for arguments in (["router-tune"], ["phrase-eval"], ["e2e-eval", "--router", "oracle"]):
        assert main(arguments) == 0
    assert main(["router-eval", "--router", "llm"]) == 0
    router_rows = table_rows((results / "router.md").read_text(encoding="utf-8"))
    e2e_rows = table_rows((results / "e2e.md").read_text(encoding="utf-8"))
    assert [row[0] for row in router_rows] == ["llm"]
    assert [row[0] for row in e2e_rows] == ["oracle", "oracle"]


def test_router_eval_prints_the_table_it_writes(monkeypatch, results, capsys):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm"])
    assert capsys.readouterr().out == (results / "router.md").read_text(encoding="utf-8")


def test_result_files_end_with_a_single_newline(monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm"])
    for path in results.iterdir():
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")


def test_run_file_is_indented_json(monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm"])
    text = (results / "router_llm_qwen3-1.7b_test.json").read_text(encoding="utf-8")
    assert text.startswith('{\n  "router": "llm",\n')


@pytest.mark.parametrize(
    ("choice", "model", "name", "field"),
    [
        ("llm", "qwen3:4b", "router_llm_qwen3-4b_test.json", "llm_model"),
        (
            "embedding",
            "BAAI/bge-small-en-v1.5",
            "router_embedding_bge-small-en-v1.5_test.json",
            "embedding_model_id",
        ),
    ],
)
def test_router_eval_model_option_changes_the_file_name_and_the_settings(
    choice, model, name, field, monkeypatch, results
):
    received = install_router(monkeypatch, fixed_router(choice))
    assert main(["router-eval", "--router", choice, "--model", model]) == 0
    assert name in file_names(results)
    assert [getattr(settings, field) for settings in received] == [model]


def test_router_eval_model_option_is_recorded_in_the_run(monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm", "--model", "qwen3:4b"])
    path = results / "router_llm_qwen3-4b_test.json"
    assert RouterRun.model_validate_json(path.read_text(encoding="utf-8")).model == "qwen3:4b"


def test_router_eval_embedding_builds_the_router_with_the_llm_backend_off(monkeypatch):
    received = install_router(monkeypatch, fixed_router("embedding"))
    main(["router-eval", "--router", "embedding"])
    assert [settings.llm_backend for settings in received] == ["none"]


def test_router_eval_llm_builds_the_router_from_the_settings_as_they_are(monkeypatch):
    received = install_router(monkeypatch, fixed_router("llm"))
    main(["router-eval", "--router", "llm"])
    assert [(s.llm_backend, s.llm_model) for s in received] == [("ollama", LLM_MODEL)]


def test_router_eval_llm_with_another_kind_of_router_exits_one_and_writes_nothing(
    monkeypatch, results, capsys
):
    router = fixed_router("embedding")
    install_router(monkeypatch, router)
    assert main(["router-eval", "--router", "llm"]) == 1
    error = capsys.readouterr().err
    assert "LLM_BACKEND=ollama" in error and f"LLM_MODEL={LLM_MODEL}" in error
    assert not results.exists()
    assert router.questions == []


def test_router_eval_with_an_unknown_backend_exits_one_and_writes_nothing(
    monkeypatch, results, capsys
):
    reject_backend(monkeypatch, "build_router")
    assert main(["router-eval", "--router", "llm"]) == 1
    assert capsys.readouterr().err == "error: unknown LLM_BACKEND 'olama'\n"
    assert not results.exists()


def test_router_eval_requires_the_router_option():
    with pytest.raises(SystemExit) as raised:
        main(["router-eval"])
    assert raised.value.code == 2


def test_e2e_eval_oracle_writes_its_report(thesis, results):
    assert main(["e2e-eval", "--router", "oracle"]) == 0
    report = E2EReport.model_validate_json((results / "e2e_oracle.json").read_text("utf-8"))
    assert (report.router, report.model) == ("oracle", None)
    assert (report.n_questions, report.n_contracts) == (3, 2)
    assert (report.tau_low, report.tau_high) == (TAU_LOW, TAU_HIGH)


def test_e2e_eval_oracle_builds_no_router(thesis, monkeypatch):
    received = install_router(monkeypatch, fixed_router("llm"))
    assert main(["e2e-eval", "--router", "oracle"]) == 0
    assert received == []


def test_e2e_eval_oracle_writes_the_table_and_prints_it(thesis, results, capsys):
    main(["e2e-eval", "--router", "oracle"])
    table = (results / "e2e.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == table
    assert "| oracle | found |" in table and "| oracle | shown |" in table


def test_e2e_eval_oracle_ignores_the_model_option(thesis, results):
    assert main(["e2e-eval", "--router", "oracle", "--model", "qwen3:4b"]) == 0
    report = E2EReport.model_validate_json((results / "e2e_oracle.json").read_text("utf-8"))
    assert report.model is None


@pytest.mark.parametrize(
    ("choice", "model", "name"),
    [
        ("llm", LLM_MODEL, "e2e_llm_qwen3-1.7b.json"),
        ("embedding", EMBEDDING_MODEL, "e2e_embedding_bge-base-en-v1.5.json"),
    ],
)
def test_e2e_eval_report_of_a_router_names_its_model_in_the_file(
    choice, model, name, thesis, monkeypatch, results
):
    install_router(monkeypatch, fixed_router(choice))
    assert main(["e2e-eval", "--router", choice]) == 0
    report = E2EReport.model_validate_json((results / name).read_text(encoding="utf-8"))
    assert (report.router, report.model) == (choice, model)


@pytest.mark.parametrize(
    ("choice", "model", "name", "field"),
    [
        ("llm", "qwen3:4b", "e2e_llm_qwen3-4b.json", "llm_model"),
        (
            "embedding",
            "BAAI/bge-small-en-v1.5",
            "e2e_embedding_bge-small-en-v1.5.json",
            "embedding_model_id",
        ),
    ],
)
def test_e2e_eval_model_option_changes_the_file_name_and_the_settings(
    choice, model, name, field, thesis, monkeypatch, results
):
    received = install_router(monkeypatch, fixed_router(choice))
    assert main(["e2e-eval", "--router", choice, "--model", model]) == 0
    assert name in file_names(results)
    assert [getattr(settings, field) for settings in received] == [model]


def test_e2e_eval_embedding_builds_the_router_with_the_llm_backend_off(thesis, monkeypatch):
    received = install_router(monkeypatch, fixed_router("embedding"))
    main(["e2e-eval", "--router", "embedding"])
    assert [settings.llm_backend for settings in received] == ["none"]


@pytest.mark.parametrize("choice", ["llm", "embedding"])
def test_e2e_eval_routes_the_warm_up_question_once_before_the_scored_ones(
    choice, thesis, monkeypatch
):
    router = fixed_router(choice)
    install_router(monkeypatch, router)
    main(["e2e-eval", "--router", choice])
    assert router.questions == [
        cli.WARM_UP_QUESTION,
        LAW_QUESTION,
        INSURANCE_QUESTION,
        PARTIES_QUESTION,
    ]


def test_e2e_eval_does_not_score_the_warm_up_question(thesis, monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["e2e-eval", "--router", "llm"])
    text = (results / "e2e_llm_qwen3-1.7b.json").read_text(encoding="utf-8")
    scored = [entry.question for entry in E2EReport.model_validate_json(text).questions]
    assert scored == [LAW_QUESTION, INSURANCE_QUESTION, PARTIES_QUESTION]


def test_e2e_eval_oracle_is_not_warmed_up(thesis, monkeypatch):
    asked = []

    class RecordingOracle(OracleRouter):
        def route(self, question):
            asked.append(question)
            return super().route(question)

    monkeypatch.setattr(cli, "OracleRouter", RecordingOracle)
    assert main(["e2e-eval", "--router", "oracle"]) == 0
    assert asked == [LAW_QUESTION, INSURANCE_QUESTION, PARTIES_QUESTION]


def test_e2e_eval_second_router_adds_its_rows_and_the_routing_loss_table(
    thesis, monkeypatch, results
):
    main(["e2e-eval", "--router", "oracle"])
    install_router(monkeypatch, fixed_router("llm"))
    assert main(["e2e-eval", "--router", "llm"]) == 0
    table = (results / "e2e.md").read_text(encoding="utf-8")
    assert table.count("| llm (qwen3:1.7b) | found |") == 2
    assert table.count("| llm (qwen3:1.7b) | shown |") == 2
    assert "Routing loss (router minus oracle)" in table


def test_e2e_table_lists_the_oracle_before_the_other_routers(thesis, monkeypatch, results):
    install_router(monkeypatch, fixed_router("llm"))
    main(["e2e-eval", "--router", "llm"])
    main(["e2e-eval", "--router", "oracle"])
    rows = table_rows((results / "e2e.md").read_text(encoding="utf-8").split("\n\n")[0])
    assert [row[:2] for row in rows] == [
        ["oracle", "found"],
        ["oracle", "shown"],
        ["llm (qwen3:1.7b)", "found"],
        ["llm (qwen3:1.7b)", "shown"],
    ]


def test_e2e_eval_reads_the_thesis_files_from_the_diplomski_dir(thesis, settings):
    main(["e2e-eval", "--router", "oracle"])
    preds = settings.diplomski_dir / "results" / f"{settings.extractor_name}_preds.jsonl"
    assert thesis == [(preds, settings.diplomski_dir / "data" / "raw" / "test.json")]


def test_e2e_eval_llm_with_another_kind_of_router_exits_one_and_writes_nothing(
    thesis, monkeypatch, results, capsys
):
    router = fixed_router("embedding")
    install_router(monkeypatch, router)
    assert main(["e2e-eval", "--router", "llm"]) == 1
    error = capsys.readouterr().err
    assert "LLM_BACKEND=ollama" in error and f"LLM_MODEL={LLM_MODEL}" in error
    assert not results.exists()
    assert router.questions == []


def test_e2e_eval_with_missing_thesis_files_exits_one_and_writes_nothing(results, capsys):
    assert main(["e2e-eval", "--router", "oracle"]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert not results.exists()


def test_e2e_eval_import_error_exits_one_with_its_message(monkeypatch, results, capsys):
    def refuse_import(preds_path, cuad_json, settings):
        raise HarnessImportError("predictions were written with other settings: name")

    monkeypatch.setattr(cli, "import_harness_preds", refuse_import)
    assert main(["e2e-eval", "--router", "oracle"]) == 1
    error = "error: predictions were written with other settings: name\n"
    assert capsys.readouterr().err == error
    assert not results.exists()


def test_e2e_eval_stale_oracle_exits_one_after_writing_its_own_report(
    thesis, settings, monkeypatch, results, capsys
):
    main(["e2e-eval", "--router", "oracle"])
    stale_table = (results / "e2e.md").read_text(encoding="utf-8")
    capsys.readouterr()
    use_settings(monkeypatch, dataclasses.replace(settings, tau_low=0.1))
    install_router(monkeypatch, fixed_router("llm"))
    assert main(["e2e-eval", "--router", "llm"]) == 1
    error = capsys.readouterr().err
    assert "oracle" in error and "re-run e2e-eval" in error
    assert (results / "e2e_llm_qwen3-1.7b.json").is_file()
    assert (results / "e2e.md").read_text(encoding="utf-8") == stale_table


def test_e2e_eval_unreadable_result_file_exits_one_asking_for_a_rerun(thesis, results, capsys):
    results.mkdir()
    (results / "e2e_old.json").write_text("{}\n", encoding="utf-8")
    assert main(["e2e-eval", "--router", "oracle"]) == 1
    assert "re-run e2e-eval" in capsys.readouterr().err


def test_phrase_eval_writes_the_report_named_after_the_model(
    demo_caches, mixed_client, monkeypatch, results
):
    install_client(monkeypatch, mixed_client)
    assert main(["phrase-eval"]) == 0
    assert file_names(results) == ["phrasing_qwen3-1.7b.json"]


def test_phrase_eval_report_holds_the_counts_and_the_rate(
    demo_caches, mixed_client, monkeypatch, results
):
    install_client(monkeypatch, mixed_client)
    main(["phrase-eval"])
    text = (results / "phrasing_qwen3-1.7b.json").read_text(encoding="utf-8")
    report = PhrasingReport.model_validate_json(text)
    assert (report.model, report.calls, report.accepted, report.violations) == (
        LLM_MODEL,
        6,
        2,
        2,
    )
    assert (report.unquoted_replies, report.llm_failures, report.invalid_replies) == (1, 1, 1)
    assert report.violation_rate == pytest.approx(0.4)


def test_phrase_eval_prints_one_line_with_the_counts_and_the_path(
    demo_caches, mixed_client, monkeypatch, results, capsys
):
    install_client(monkeypatch, mixed_client)
    main(["phrase-eval"])
    [line] = capsys.readouterr().out.splitlines()
    assert all(part in line for part in LINE_PARTS)
    assert line.endswith(f"-> {results / 'phrasing_qwen3-1.7b.json'}")


def test_phrase_eval_prints_not_available_when_every_call_failed(
    demo_caches, monkeypatch, capsys
):
    install_client(monkeypatch, FakeLLMClient([LLMError("down")] * 6, model=LLM_MODEL))
    assert main(["phrase-eval"]) == 0
    assert "of 0 replies (rate n/a)" in capsys.readouterr().out


def test_phrase_eval_asks_every_in_scope_question_on_every_demo_document(
    demo_caches, mixed_client, monkeypatch
):
    install_client(monkeypatch, mixed_client)
    main(["phrase-eval"])
    assert len(mixed_client.calls) == 6


def test_phrase_eval_uses_the_configured_thresholds(demo_caches, mixed_client, monkeypatch):
    install_client(monkeypatch, mixed_client)
    main(["phrase-eval"])
    _, user, _ = mixed_client.calls[0]
    assert '- Governing Law (found): "laws of Delaware"' in user


def test_phrase_eval_model_option_overrides_the_llm_model(
    demo_caches, mixed_client, monkeypatch
):
    received = install_client(monkeypatch, mixed_client)
    main(["phrase-eval", "--model", "qwen3:4b"])
    assert [settings.llm_model for settings in received] == ["qwen3:4b"]


def test_phrase_eval_without_an_llm_exits_one_naming_the_settings(monkeypatch, results, capsys):
    install_client(monkeypatch, None)
    assert main(["phrase-eval"]) == 1
    error = "error: phrase-eval needs an LLM (LLM_BACKEND=ollama, LLM_MODEL=qwen3:1.7b)\n"
    assert capsys.readouterr().err == error
    assert not results.exists()


def test_phrase_eval_with_an_unknown_backend_exits_one(monkeypatch, capsys):
    reject_backend(monkeypatch, "build_llm_client")
    assert main(["phrase-eval"]) == 1
    assert capsys.readouterr().err == "error: unknown LLM_BACKEND 'olama'\n"


def test_serve_with_an_unknown_backend_exits_one(monkeypatch, capsys):
    reject_backend(monkeypatch, "build_router")
    assert main(["serve"]) == 1
    assert capsys.readouterr().err.endswith("error: unknown LLM_BACKEND 'olama'\n")


@pytest.mark.parametrize(
    ("builder", "arguments", "built"),
    [
        ("build_embedding_router", ["router-tune"], FakeScorer(DEV_SCORES)),
        ("build_router", ["router-eval", "--router", "llm"], fixed_router("llm")),
        ("build_router", ["e2e-eval", "--router", "llm"], fixed_router("llm")),
    ],
    ids=["router-tune", "router-eval", "e2e-eval"],
)
def test_evaluation_commands_build_the_router_over_the_41_cuad_types(
    builder, arguments, built, thesis, settings, monkeypatch
):
    received = []

    def build(settings, clause_types):
        received.append([clause_type.name for clause_type in clause_types])
        return built

    monkeypatch.setattr(cli, builder, build)
    main(arguments)
    expected = [c.name for c in load_clause_types(settings.clause_types_path)]
    assert received == [expected]
    assert len(expected) == 41


def use_real_demo_caches(monkeypatch, settings: Settings) -> None:
    demo = dataclasses.replace(settings, demo_cache_dir=ROOT / "demo" / "cache")
    use_settings(monkeypatch, demo)


def install_gen_client(monkeypatch, client) -> list[Settings]:
    return install(monkeypatch, "build_gen_client", client)


def test_summary_eval_writes_the_report_named_after_the_model(
    settings, monkeypatch, results
):
    use_real_demo_caches(monkeypatch, settings)
    install_gen_client(monkeypatch, EchoTextClient())
    assert main(["summary-eval"]) == 0
    assert file_names(results) == ["summary_fake-gen.json"]


def test_summary_eval_report_has_one_record_per_demo_document(
    settings, monkeypatch, results
):
    use_real_demo_caches(monkeypatch, settings)
    install_gen_client(monkeypatch, EchoTextClient())
    main(["summary-eval"])
    text = (results / "summary_fake-gen.json").read_text(encoding="utf-8")
    report = SummaryReport.model_validate_json(text)
    assert (report.model, report.documents, len(report.records)) == ("fake-gen", 3, 3)
    assert (report.withheld, report.llm_failures) == (0, 0)


def test_summary_eval_prints_one_line_with_the_counts_and_the_path(
    settings, monkeypatch, results, capsys
):
    use_real_demo_caches(monkeypatch, settings)
    install_gen_client(monkeypatch, EchoTextClient())
    main(["summary-eval"])
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith(
        "summary-eval: fake-gen · 3 documents · 0 withheld · 0 retried · 0 failures · mean "
    )
    assert line.endswith(f" s → {results / 'summary_fake-gen.json'}")


def test_summary_eval_prints_not_available_when_every_call_failed(
    settings, monkeypatch, capsys
):
    use_real_demo_caches(monkeypatch, settings)
    install_gen_client(monkeypatch, FakeLLMClient([LLMError("down")] * 3, model="fake-gen"))
    assert main(["summary-eval"]) == 0
    assert "3 failures · mean n/a → " in capsys.readouterr().out


def test_summary_eval_model_option_changes_the_file_name_and_the_settings(
    settings, monkeypatch, results
):
    use_real_demo_caches(monkeypatch, settings)
    client = EchoTextClient()
    client.model = "other-gen"
    received = install_gen_client(monkeypatch, client)
    assert main(["summary-eval", "--model", "qwen3.5:9b"]) == 0
    assert file_names(results) == ["summary_other-gen.json"]
    assert [s.gen_model for s in received] == ["qwen3.5:9b"]


def test_summary_eval_without_a_generation_client_exits_one_and_writes_nothing(
    monkeypatch, results, capsys
):
    install_gen_client(monkeypatch, None)
    assert main(["summary-eval"]) == 1
    error = capsys.readouterr().err
    assert error.startswith("error: summary-eval needs a generation model (")
    assert "LLM_BACKEND=ollama" in error and "GEN_MODEL=" in error
    assert not results.exists()
