"""Scores routers on the evaluation questions and tunes the embedding threshold."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Protocol, get_args

import numpy as np
from pydantic import BaseModel

from contract_assistant.evaluation.questions import EvalQuestion, QuestionKind, Split
from contract_assistant.router.base import RouteKind, Router, RouteResult
from contract_assistant.router.embedding import select_types

THRESHOLD_GRID = [percent / 100 for percent in range(30, 91)]
QUESTION_KINDS = get_args(QuestionKind)
LATENCY_PERCENTILES = (50, 90)
NOT_AVAILABLE = "n/a"
TEXT_COLUMNS = ["router", "model", "split"]
NUMBER_COLUMNS = [
    "n",
    "P",
    "R",
    "F1",
    "F1 single",
    "F1 compound",
    "out-of-scope acc.",
    "exact-set acc.",
    "fallback rate",
    "median ms",
    "p90 ms",
]


class ScoringRouter(Protocol):
    """A router that scores every clause type for a question."""

    max_types: int

    def scores(self, question: str) -> dict[str, float]: ...


class MeanScores(BaseModel):
    """Mean precision, recall and F1 over n questions."""

    n: int
    precision: float
    recall: float
    f1: float


class QuestionResult(BaseModel):
    """The route chosen for one question and its scores against the gold set."""

    id: str
    question: str
    kind: QuestionKind
    gold: list[str]
    route: RouteResult
    precision: float
    recall: float
    f1: float


class RouterReport(BaseModel):
    """Quality and latency of a router over a list of questions."""

    overall: MeanScores
    by_kind: dict[QuestionKind, MeanScores]
    exact_set_accuracy: float
    out_of_scope_accuracy: float | None
    fallback_rate: float
    latency_median_ms: float
    latency_p90_ms: float
    results: list[QuestionResult]


class RouterRun(BaseModel):
    """A router report with the router, model and split it was measured on."""

    router: RouteKind
    model: str
    split: Split
    threshold: float | None = None
    report: RouterReport


def set_scores(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Return precision, recall and F1; empty gold scores 1 only for an empty prediction."""
    if not gold:
        score = 0.0 if pred else 1.0
        return score, score, score
    hits = len(pred & gold)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def evaluate_router(router: Router, questions: list[EvalQuestion]) -> RouterReport:
    """Route every question once and score the routes against the gold sets."""
    _check_questions(questions)
    results = [_score(question, router.route(question.question)) for question in questions]
    latencies = [result.route.latency_ms for result in results]
    median, p90 = np.percentile(latencies, LATENCY_PERCENTILES)
    return RouterReport(
        overall=_mean_scores(results),
        by_kind=_mean_scores_by_kind(results),
        exact_set_accuracy=_share([_is_exact(result) for result in results]),
        out_of_scope_accuracy=_out_of_scope_accuracy(results),
        fallback_rate=_share([result.route.kind == "llm_fallback" for result in results]),
        latency_median_ms=float(median),
        latency_p90_ms=float(p90),
        results=results,
    )


def threshold_sweep(
    router: ScoringRouter, dev: list[EvalQuestion], grid: Sequence[float]
) -> dict[float, float]:
    """Mean F1 over the questions for each threshold in the grid."""
    _check_questions(dev)
    scored = [(router.scores(question.question), set(question.gold)) for question in dev]
    return {
        threshold: fmean(
            _f1(select_types(scores, threshold, router.max_types), gold)
            for scores, gold in scored
        )
        for threshold in grid
    }


def tune_threshold(
    router: ScoringRouter, dev: list[EvalQuestion], grid: Sequence[float]
) -> float:
    """Return the grid threshold with the highest mean F1; ties go to the higher one."""
    sweep = threshold_sweep(router, dev, grid)
    return max(sweep, key=lambda threshold: (sweep[threshold], threshold))


def markdown_table(runs: Sequence[RouterRun]) -> str:
    """Render the runs as a Markdown table, one row per run."""
    header = TEXT_COLUMNS + NUMBER_COLUMNS
    alignment = ["---"] * len(TEXT_COLUMNS) + ["---:"] * len(NUMBER_COLUMNS)
    rows = [_row(header), _row(alignment), *(_row(_cells(run)) for run in runs)]
    return "\n".join(rows)


def _check_questions(questions: list[EvalQuestion]) -> None:
    if not questions:
        raise ValueError("no questions to evaluate")


def _f1(pred: Sequence[str], gold: set[str]) -> float:
    _, _, f1 = set_scores(set(pred), gold)
    return f1


def _score(question: EvalQuestion, route: RouteResult) -> QuestionResult:
    precision, recall, f1 = set_scores(set(route.clause_types), set(question.gold))
    return QuestionResult(
        id=question.id,
        question=question.question,
        kind=question.kind,
        gold=question.gold,
        route=route,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _mean_scores(results: list[QuestionResult]) -> MeanScores:
    return MeanScores(
        n=len(results),
        precision=fmean(result.precision for result in results),
        recall=fmean(result.recall for result in results),
        f1=fmean(result.f1 for result in results),
    )


def _mean_scores_by_kind(results: list[QuestionResult]) -> dict[QuestionKind, MeanScores]:
    by_kind: dict[QuestionKind, MeanScores] = {}
    for kind in QUESTION_KINDS:
        group = [result for result in results if result.kind == kind]
        if group:
            by_kind[kind] = _mean_scores(group)
    return by_kind


def _is_exact(result: QuestionResult) -> bool:
    return set(result.route.clause_types) == set(result.gold)


def _out_of_scope_accuracy(results: list[QuestionResult]) -> float | None:
    empty = [
        not result.route.clause_types for result in results if result.kind == "out_of_scope"
    ]
    return _share(empty) if empty else None


def _share(flags: list[bool]) -> float:
    return sum(flags) / len(flags)


def _cells(run: RouterRun) -> list[str]:
    report = run.report
    return [
        run.router,
        run.model,
        run.split,
        str(report.overall.n),
        _number(report.overall.precision),
        _number(report.overall.recall),
        _number(report.overall.f1),
        _number(_kind_f1(report, "single")),
        _number(_kind_f1(report, "compound")),
        _number(report.out_of_scope_accuracy),
        _number(report.exact_set_accuracy),
        _number(report.fallback_rate),
        f"{report.latency_median_ms:.0f}",
        f"{report.latency_p90_ms:.0f}",
    ]


def _kind_f1(report: RouterReport, kind: QuestionKind) -> float | None:
    scores = report.by_kind.get(kind)
    return None if scores is None else scores.f1


def _number(value: float | None) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.3f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"
