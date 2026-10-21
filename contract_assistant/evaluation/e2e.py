"""End-to-end evaluation on CUAD's test split, scored as in the thesis operating point."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contract_assistant.answer import Answer, AnswerItem, Status, answer
from contract_assistant.clause_types import ClauseType
from contract_assistant.cuad_match import RELAXED_CLAUSE_TYPES, is_match
from contract_assistant.evaluation.questions import EvalQuestion, QuestionKind
from contract_assistant.router.base import RouteKind, Router, RouteResult, Turn
from contract_assistant.store import DocumentCache

Policy = Literal["found", "shown"]
Gold = dict[str, dict[str, list[str]]]

POLICY_STATUSES: dict[Policy, tuple[Status, ...]] = {
    "found": ("found",),
    "shown": ("found", "check"),
}
ORACLE: RouteKind = "oracle"
FALLBACK: RouteKind = "llm_fallback"
METRIC_TITLES = {
    "citation_precision": "citation precision",
    "citation_recall": "citation recall",
    "omission_rate": "omission rate",
    "false_citation_rate": "false-citation rate",
    "abstention_accuracy": "abstention accuracy",
    "hallucination_rate": "hallucination rate",
}
VALUE_FORMAT = ".3f"
LOSS_FORMAT = "+.3f"
NOT_AVAILABLE = "n/a"
LOSS_CAPTION = "Routing loss (router minus oracle)"


class GoldError(ValueError):
    """Raised when the CUAD gold does not fit the clause types or the caches."""


class Counts(BaseModel):
    """Outcome counts of one display policy, summed over questions and contracts."""

    correct: int = 0
    wrong: int = 0
    false: int = 0
    omission: int = 0
    abstention: int = 0
    missed: int = 0
    correct_asked: int = 0
    omission_asked: int = 0
    present_asked: int = 0
    quotes_attempted: int = 0
    quotes_dropped: int = 0


class Metrics(BaseModel):
    """Rates derived from the counts; None where the denominator is zero."""

    citation_precision: float | None
    citation_recall: float | None
    omission_rate: float | None
    false_citation_rate: float | None
    abstention_accuracy: float | None
    hallucination_rate: float | None


class PolicyResult(BaseModel):
    """The counts and metrics of one display policy."""

    counts: Counts
    metrics: Metrics


class RoutedQuestion(BaseModel):
    """A scored question and the route the router chose for it."""

    id: str
    question: str
    kind: QuestionKind
    gold: list[str]
    route: RouteResult


class E2EReport(BaseModel):
    """The end-to-end scores of one router."""

    router: RouteKind
    model: str | None
    tau_low: float
    tau_high: float
    n_questions: int
    n_contracts: int
    policies: dict[Policy, PolicyResult]
    fallback_rate: float | None
    latency_median_ms: float | None
    questions: list[RoutedQuestion]


class OracleRouter:
    """Router that returns the gold clause types of each question."""

    kind = ORACLE

    def __init__(self, questions: list[EvalQuestion]):
        self.gold = {question.question: question.gold for question in questions}

    def route(self, question: str, turns: Sequence[Turn] = ()) -> RouteResult:
        """Return the gold clause types of the question; turns are not used."""
        return RouteResult(clause_types=self.gold[question], kind=ORACLE, latency_ms=0.0)


def load_cuad_gold(cuad_json: Path, clause_types: list[ClauseType]) -> Gold:
    """Read CUAD's gold spans per contract title and clause type."""
    name_of = {clause_type.cuad_question: clause_type.name for clause_type in clause_types}
    gold: Gold = {}
    for entry in json.loads(cuad_json.read_text(encoding="utf-8"))["data"]:
        spans = {}
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["question"] not in name_of:
                    raise GoldError(f"unknown question in {entry['title']}: {qa['question']}")
                spans[name_of[qa["question"]]] = [span["text"] for span in qa["answers"]]
        gold[entry["title"]] = spans
    return gold


def evaluate_e2e(
    router: Router,
    questions: list[EvalQuestion],
    caches: dict[str, DocumentCache],
    gold: Gold,
    tau_low: float,
    tau_high: float,
    model: str | None = None,
) -> E2EReport:
    """Score a router's answers over every in-scope question and every contract."""
    _check_gold(caches, gold)
    in_scope = [question for question in questions if question.kind != "out_of_scope"]
    routes = [router.route(question.question) for question in in_scope]
    return E2EReport(
        router=router.kind,
        model=model,
        tau_low=tau_low,
        tau_high=tau_high,
        n_questions=len(in_scope),
        n_contracts=len(caches),
        policies=_score(in_scope, routes, caches, gold, tau_low, tau_high),
        fallback_rate=_ratio(sum(route.kind == FALLBACK for route in routes), len(routes)),
        latency_median_ms=_median([route.latency_ms for route in routes]),
        questions=[_routed(q, r) for q, r in zip(in_scope, routes, strict=True)],
    )


def _check_gold(caches: dict[str, DocumentCache], gold: Gold) -> None:
    missing = sorted(set(caches) - set(gold))
    if missing:
        raise GoldError(f"no CUAD gold for: {', '.join(missing)}")


def _score(
    questions: list[EvalQuestion],
    routes: list[RouteResult],
    caches: dict[str, DocumentCache],
    gold: Gold,
    tau_low: float,
    tau_high: float,
) -> dict[Policy, PolicyResult]:
    counts = {policy: Counts() for policy in POLICY_STATUSES}
    for question, route in zip(questions, routes, strict=True):
        for title, cache in caches.items():
            result = answer(question.question, route, cache, tau_low, tau_high)
            _count_answer(counts, result, set(question.gold), gold[title])
    return {
        policy: PolicyResult(counts=policy_counts, metrics=_metrics(policy_counts))
        for policy, policy_counts in counts.items()
    }


def _count_answer(
    counts: dict[Policy, Counts],
    result: Answer,
    asked: set[str],
    contract_gold: dict[str, list[str]],
) -> None:
    items = {item.clause_type: item for item in result.items}
    routed = set(items)
    present = {name for name in routed | asked if contract_gold[name]}
    for policy, statuses in POLICY_STATUSES.items():
        in_policy = {name for name in routed if items[name].status in statuses}
        shown = {name for name in in_policy if items[name].citation is not None}
        withheld = {name for name in in_policy if items[name].withheld}
        hidden = routed - shown
        shown_present = shown & present
        correct = {name for name in shown_present if _matches(items[name], contract_gold[name])}
        policy_counts = counts[policy]
        policy_counts.correct += len(correct)
        policy_counts.wrong += len(shown_present - correct)
        policy_counts.false += len(shown - present)
        policy_counts.omission += len(hidden & present)
        policy_counts.abstention += len(hidden - present)
        policy_counts.missed += len((asked - routed) & present)
        policy_counts.correct_asked += len(correct & asked)
        policy_counts.omission_asked += len(hidden & present & asked)
        policy_counts.present_asked += len(asked & present)
        policy_counts.quotes_attempted += len(shown) + len(withheld)
        policy_counts.quotes_dropped += len(withheld)


def _matches(item: AnswerItem, spans: list[str]) -> bool:
    relaxed = item.clause_type in RELAXED_CLAUSE_TYPES
    return any(is_match(span, item.citation.quote, relaxed) for span in spans)


def _metrics(counts: Counts) -> Metrics:
    routed_absent = counts.false + counts.abstention
    return Metrics(
        citation_precision=_ratio(counts.correct, counts.correct + counts.wrong + counts.false),
        citation_recall=_ratio(counts.correct_asked, counts.present_asked),
        omission_rate=_ratio(counts.omission_asked + counts.missed, counts.present_asked),
        false_citation_rate=_ratio(counts.false, routed_absent),
        abstention_accuracy=_ratio(counts.abstention, routed_absent),
        hallucination_rate=_ratio(counts.quotes_dropped, counts.quotes_attempted),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _routed(question: EvalQuestion, route: RouteResult) -> RoutedQuestion:
    return RoutedQuestion(
        id=question.id,
        question=question.question,
        kind=question.kind,
        gold=question.gold,
        route=route,
    )


def markdown_table(reports: list[E2EReport]) -> str:
    """Render the metrics per router and policy, and the loss against the oracle router."""
    columns = ["router", "policy", *METRIC_TITLES.values()]
    rows = [
        [_label(report), policy, *_format(_values(report, policy), VALUE_FORMAT)]
        for report in reports
        for policy in POLICY_STATUSES
    ]
    tables = [_table(columns, rows)]
    loss_rows = _loss_rows(reports)
    if loss_rows:
        tables.append(f"{LOSS_CAPTION}\n\n{_table(columns, loss_rows)}")
    return "\n\n".join(tables)


def _loss_rows(reports: list[E2EReport]) -> list[list[str]]:
    oracle = next((report for report in reports if report.router == ORACLE), None)
    if oracle is None:
        return []
    others = [report for report in reports if report.router != ORACLE]
    for report in others:
        _check_comparable(report, oracle)
    return [
        [_label(report), policy, *_format(_losses(report, oracle, policy), LOSS_FORMAT)]
        for report in others
        for policy in POLICY_STATUSES
    ]


def _check_comparable(report: E2EReport, oracle: E2EReport) -> None:
    if _setup(report) != _setup(oracle):
        raise ValueError(
            f"{_label(report)} and the oracle differ in thresholds, questions or contracts"
        )


def _setup(report: E2EReport) -> tuple:
    questions = [(entry.id, entry.question, entry.gold) for entry in report.questions]
    return report.tau_low, report.tau_high, report.n_contracts, questions


def _label(report: E2EReport) -> str:
    return report.router if report.model is None else f"{report.router} ({report.model})"


def _values(report: E2EReport, policy: Policy) -> list[float | None]:
    metrics = report.policies[policy].metrics.model_dump()
    return [metrics[name] for name in METRIC_TITLES]


def _losses(report: E2EReport, oracle: E2EReport, policy: Policy) -> list[float | None]:
    pairs = zip(_values(report, policy), _values(oracle, policy), strict=True)
    return [_difference(value, reference) for value, reference in pairs]


def _difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _format(values: list[float | None], spec: str) -> list[str]:
    return [NOT_AVAILABLE if value is None else format(value, spec) for value in values]


def _table(columns: list[str], rows: list[list[str]]) -> str:
    lines = [columns, ["---"] * len(columns), *rows]
    return "\n".join(f"| {' | '.join(line)} |" for line in lines)
