import pytest

from contract_assistant.evaluation.questions import EvalQuestion, QuestionKind, Split
from contract_assistant.evaluation.router_eval import (
    THRESHOLD_GRID,
    RouterReport,
    RouterRun,
    evaluate_router,
    markdown_table,
    set_scores,
    threshold_sweep,
    tune_threshold,
)
from contract_assistant.router.base import RouteKind, RouteResult
from contract_assistant.router.embedding import EmbeddingRouter
from tests.support.fakes import FakeEncoder, FixedRouter

DESCRIPTIONS = {
    "Governing Law": ["which law applies", "choice of law"],
    "Insurance": ["required insurance coverage"],
    "Audit Rights": ["right to audit the books"],
}


def make_question(
    number: int, kind: QuestionKind, gold: list[str], split: Split = "test"
) -> EvalQuestion:
    return EvalQuestion(
        id=f"q{number}", question=f"question {number}", gold=gold, kind=kind, split=split
    )


def make_route(
    clause_types: list[str], latency_ms: float, kind: RouteKind = "llm"
) -> RouteResult:
    return RouteResult(clause_types=clause_types, kind=kind, latency_ms=latency_ms)


QUESTIONS = [
    make_question(1, "single", ["Governing Law"]),
    make_question(2, "single", ["Insurance"]),
    make_question(3, "compound", ["Exclusivity", "Non-Compete"]),
    make_question(4, "compound", ["Parties", "Insurance", "Audit Rights"]),
    make_question(5, "out_of_scope", []),
    make_question(6, "out_of_scope", []),
    make_question(7, "out_of_scope", []),
]
ROUTES = [
    make_route(["Governing Law"], 10.0),
    make_route(["Insurance", "Audit Rights"], 20.0),
    make_route(["Exclusivity", "Parties"], 30.0),
    make_route([], 40.0, "llm_fallback"),
    make_route([], 50.0),
    make_route(["Governing Law"], 60.0),
    make_route([], 70.0),
]


def fixed_router() -> FixedRouter:
    texts = [question.question for question in QUESTIONS]
    return FixedRouter(dict(zip(texts, ROUTES, strict=True)))


@pytest.fixture
def report() -> RouterReport:
    return evaluate_router(fixed_router(), QUESTIONS)


def test_two_empty_sets_score_one():
    assert set_scores(set(), set()) == (1.0, 1.0, 1.0)


def test_prediction_against_empty_gold_scores_zero():
    assert set_scores({"Governing Law"}, set()) == (0.0, 0.0, 0.0)


def test_empty_prediction_against_gold_scores_zero():
    assert set_scores(set(), {"Governing Law"}) == (0.0, 0.0, 0.0)


def test_disjoint_sets_score_zero():
    assert set_scores({"Insurance"}, {"Governing Law"}) == (0.0, 0.0, 0.0)


def test_extra_prediction_lowers_precision_only():
    scores = set_scores({"Insurance", "Parties"}, {"Insurance"})
    assert scores == pytest.approx((1 / 2, 1.0, 2 / 3))


def test_partial_overlap_is_scored_by_hand():
    pred = {"Parties", "Insurance", "Audit Rights"}
    gold = {"Insurance", "Audit Rights", "Exclusivity", "Non-Compete"}
    assert set_scores(pred, gold) == pytest.approx((2 / 3, 2 / 4, 2 * 2 / (3 + 4)))


def test_every_question_is_routed_once():
    router = fixed_router()
    evaluate_router(router, QUESTIONS)
    assert router.questions == [question.question for question in QUESTIONS]


def test_overall_means_are_hand_computed(report):
    assert report.overall.n == 7
    assert report.overall.precision == pytest.approx((1 + 1 / 2 + 1 / 2 + 0 + 1 + 0 + 1) / 7)
    assert report.overall.recall == pytest.approx((1 + 1 + 1 / 2 + 0 + 1 + 0 + 1) / 7)
    assert report.overall.f1 == pytest.approx((1 + 2 / 3 + 1 / 2 + 0 + 1 + 0 + 1) / 7)


@pytest.mark.parametrize(
    ("kind", "n", "precision", "recall", "f1"),
    [
        ("single", 2, (1 + 1 / 2) / 2, (1 + 1) / 2, (1 + 2 / 3) / 2),
        ("compound", 2, (1 / 2 + 0) / 2, (1 / 2 + 0) / 2, (1 / 2 + 0) / 2),
        ("out_of_scope", 3, (1 + 0 + 1) / 3, (1 + 0 + 1) / 3, (1 + 0 + 1) / 3),
    ],
)
def test_means_by_kind_are_hand_computed(report, kind, n, precision, recall, f1):
    scores = report.by_kind[kind]
    assert scores.n == n
    assert (scores.precision, scores.recall, scores.f1) == pytest.approx(
        (precision, recall, f1)
    )


def test_exact_set_accuracy_counts_identical_sets(report):
    assert report.exact_set_accuracy == pytest.approx(3 / 7)


def test_exact_set_accuracy_ignores_the_order_of_the_route():
    questions = [make_question(1, "compound", ["Exclusivity", "Non-Compete"])]
    router = FixedRouter({"question 1": make_route(["Non-Compete", "Exclusivity"], 1.0)})
    assert evaluate_router(router, questions).exact_set_accuracy == 1.0


def test_out_of_scope_accuracy_is_the_share_of_empty_routes(report):
    assert report.out_of_scope_accuracy == pytest.approx(2 / 3)


def test_out_of_scope_accuracy_is_none_without_out_of_scope_questions():
    report = evaluate_router(fixed_router(), QUESTIONS[:4])
    assert report.out_of_scope_accuracy is None


def test_kinds_without_questions_are_left_out_of_the_report():
    report = evaluate_router(fixed_router(), QUESTIONS[:4])
    assert list(report.by_kind) == ["single", "compound"]


def test_fallback_rate_is_the_share_of_fallback_routes(report):
    assert report.fallback_rate == pytest.approx(1 / 7)


def test_latency_median_and_p90_are_taken_from_the_routes(report):
    assert report.latency_median_ms == pytest.approx(40.0)
    assert report.latency_p90_ms == pytest.approx(60 + 0.4 * (70 - 60))


def test_results_follow_the_question_order(report):
    assert [result.id for result in report.results] == [q.id for q in QUESTIONS]


def test_result_keeps_the_route_and_its_scores(report):
    result = report.results[1]
    assert (result.id, result.question, result.kind) == ("q2", "question 2", "single")
    assert result.gold == ["Insurance"]
    assert result.route == ROUTES[1]
    assert (result.precision, result.recall, result.f1) == pytest.approx((1 / 2, 1.0, 2 / 3))


def test_evaluating_no_questions_raises():
    with pytest.raises(ValueError, match="no questions"):
        evaluate_router(fixed_router(), [])


def test_run_survives_a_json_round_trip(report):
    run = RouterRun(
        router="embedding", model="bge", split="test", threshold=0.55, report=report
    )
    assert RouterRun.model_validate_json(run.model_dump_json(indent=2)) == run


DEV = [
    make_question(1, "single", ["Governing Law"], "dev"),
    make_question(2, "single", ["Expiration Date"], "dev"),
    make_question(3, "out_of_scope", [], "dev"),
]
DEV_SCORES = {
    "question 1": {"Governing Law": 0.8, "Expiration Date": 0.55, "Insurance": 0.3},
    "question 2": {"Governing Law": 0.45, "Expiration Date": 0.65, "Insurance": 0.2},
    "question 3": {"Governing Law": 0.5, "Expiration Date": 0.35, "Insurance": 0.1},
}
COARSE_GRID = [0.4, 0.5, 0.6, 0.7]


class FakeScoringRouter:
    """Prepared scores per question; records the questions it was asked."""

    def __init__(self, scores: dict[str, dict[str, float]], max_types: int = 5):
        self.prepared = scores
        self.max_types = max_types
        self.questions: list[str] = []

    def scores(self, question: str) -> dict[str, float]:
        self.questions.append(question)
        return self.prepared[question]


def test_threshold_grid_has_61_values_from_030_to_090():
    assert len(THRESHOLD_GRID) == 61
    assert (THRESHOLD_GRID[0], THRESHOLD_GRID[-1]) == (0.30, 0.90)


def test_threshold_grid_is_increasing_in_hundredths():
    assert THRESHOLD_GRID == sorted(set(THRESHOLD_GRID))
    assert all(value == round(value, 2) for value in THRESHOLD_GRID)


def test_sweep_gives_the_mean_f1_per_threshold():
    sweep = threshold_sweep(FakeScoringRouter(DEV_SCORES), DEV, COARSE_GRID)
    assert sweep == pytest.approx(
        {
            0.4: (2 / 3 + 2 / 3 + 0) / 3,
            0.5: (2 / 3 + 1 + 0) / 3,
            0.6: (1 + 1 + 1) / 3,
            0.7: (1 + 0 + 1) / 3,
        }
    )


def test_sweep_applies_the_max_types_cap():
    scorer = FakeScoringRouter({"question 1": {"Governing Law": 0.9, "Insurance": 0.8}}, 1)
    dev = [make_question(1, "compound", ["Governing Law", "Insurance"], "dev")]
    assert threshold_sweep(scorer, dev, [0.5]) == pytest.approx({0.5: 2 / 3})


def test_tuning_picks_the_best_threshold_of_the_grid():
    assert tune_threshold(FakeScoringRouter(DEV_SCORES), DEV, COARSE_GRID) == 0.6


@pytest.mark.parametrize("grid", [[0.6, 0.62, 0.64], [0.64, 0.62, 0.6]])
def test_tuning_prefers_the_higher_threshold_on_a_tie(grid):
    assert tune_threshold(FakeScoringRouter(DEV_SCORES), DEV, grid) == 0.64


def test_scores_are_requested_once_per_question():
    scorer = FakeScoringRouter(DEV_SCORES)
    tune_threshold(scorer, DEV, THRESHOLD_GRID)
    assert scorer.questions == [question.question for question in DEV]


def test_tuning_without_questions_raises():
    with pytest.raises(ValueError, match="no questions"):
        tune_threshold(FakeScoringRouter({}), [], THRESHOLD_GRID)


def text_question(text: str, kind: QuestionKind, gold: list[str]) -> EvalQuestion:
    return EvalQuestion(id=text, question=text, gold=gold, kind=kind, split="dev")


TEXT_DEV = [
    text_question("which law applies here", "single", ["Governing Law"]),
    text_question("what insurance coverage is required", "single", ["Insurance"]),
    text_question(
        "insurance coverage and the right to audit", "compound", ["Insurance", "Audit Rights"]
    ),
    text_question(
        "which law applies and what insurance is required",
        "compound",
        ["Governing Law", "Insurance"],
    ),
    text_question("what is the weather today", "out_of_scope", []),
]


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7])
def test_sweep_agrees_with_evaluating_the_router_at_that_threshold(threshold):
    router = EmbeddingRouter(FakeEncoder(), DESCRIPTIONS, threshold, 3)
    swept = threshold_sweep(router, TEXT_DEV, [threshold])[threshold]
    assert evaluate_router(router, TEXT_DEV).overall.f1 == pytest.approx(swept)


def make_run(report: RouterReport, router: RouteKind = "llm", model: str = "qwen3:1.7b"):
    return RouterRun(router=router, model=model, split="test", report=report)


def table_rows(table: str) -> list[dict[str, str]]:
    lines = [line.strip("|").split("|") for line in table.splitlines()]
    header, _, *rows = [[cell.strip() for cell in cells] for cells in lines]
    return [dict(zip(header, row, strict=True)) for row in rows]


def test_table_has_a_header_a_separator_and_a_row_per_run(report):
    table = markdown_table([make_run(report, "embedding", "bge"), make_run(report)])
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("| router | model | split | n | P | R | F1 |")
    assert set(lines[1]) <= set("|-: ")
    assert [row["router"] for row in table_rows(table)] == ["embedding", "llm"]


def test_row_has_three_decimals_and_whole_millisecond_latencies(report):
    (row,) = table_rows(markdown_table([make_run(report)]))
    assert row == {
        "router": "llm",
        "model": "qwen3:1.7b",
        "split": "test",
        "n": "7",
        "P": "0.571",
        "R": "0.643",
        "F1": "0.595",
        "F1 single": "0.833",
        "F1 compound": "0.250",
        "out-of-scope acc.": "0.667",
        "exact-set acc.": "0.429",
        "fallback rate": "0.143",
        "median ms": "40",
        "p90 ms": "64",
    }


def test_row_marks_metrics_without_questions_as_not_available():
    report = evaluate_router(fixed_router(), QUESTIONS[:2])
    (row,) = table_rows(markdown_table([make_run(report)]))
    assert (row["F1 compound"], row["out-of-scope acc."]) == ("n/a", "n/a")
