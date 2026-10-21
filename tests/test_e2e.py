import json
from pathlib import Path

import pytest

from contract_assistant.clause_types import ClauseType, load_clause_types
from contract_assistant.config import Settings
from contract_assistant.evaluation.e2e import (
    Counts,
    E2EReport,
    GoldError,
    OracleRouter,
    evaluate_e2e,
    load_cuad_gold,
    markdown_table,
)
from contract_assistant.evaluation.questions import EvalQuestion
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.store import import_harness_preds
from tests.support.caches import make_cache
from tests.support.fakes import FixedRouter

TAU_LOW = 0.2
TAU_HIGH = 0.8
THESIS_TAU_LOW = 0.06
THESIS_TAU_HIGH = 0.91
THESIS_COUNTS = {
    "shown": {"correct": 1078, "wrong": 138, "false": 595, "omission": 28, "abstention": 2343},
    "found": {"correct": 444, "wrong": 17, "false": 143, "omission": 783, "abstention": 2795},
}
THESIS_N_CONTRACTS = 102
THESIS_N_CLAUSE_TYPES = 41

LAW = "Governing Law"
END = "Expiration Date"
TERMINATION = "Termination For Convenience"
INSURANCE = "Insurance"
AUDIT = "Audit Rights"
PARTIES = "Parties"

LAW_QUESTION = "Which law governs the agreement?"
END_QUESTION = "When does it end, and can either side terminate?"
INSURANCE_QUESTION = "Does the licensee have to carry insurance?"
WEATHER_QUESTION = "What is the weather today?"
SOLO_QUESTION = "A question about one clause"

LAW_SPAN = "laws of Delaware"
LAW_SENTENCE = "This Agreement is governed by the laws of Delaware"
TEXT_ALPHA = (
    "This Agreement is governed by the laws of Delaware. The term is five years. "
    "Either party may terminate on thirty days notice. "
    "The Supplier shall maintain no special cover."
)
TEXT_BETA = (
    "This Agreement is governed by the laws of Texas. The term is ten years. "
    "The Licensee shall keep insurance in force and allow audits of its books."
)
CLAUSES_ALPHA = {
    LAW: [("laws of Delaware", 0.9)],
    END: [("five years", 0.5)],
    TERMINATION: [("terminate on thirty days notice", 0.1)],
    INSURANCE: [("special cover", 0.95)],
    AUDIT: [("shall maintain", 0.1)],
}
CLAUSES_BETA = {
    LAW: [("laws of Texas", 0.85)],
    END: [("ten years", 0.4)],
    TERMINATION: [],
    INSURANCE: [("keep insurance in force", 0.05)],
    AUDIT: [("allow audits of its books", 0.5)],
}
GOLD = {
    "Alpha": {
        LAW: ["laws of Delaware"],
        END: ["five years"],
        TERMINATION: ["Either party may terminate on thirty days notice"],
        INSURANCE: [],
        AUDIT: [],
    },
    "Beta": {
        LAW: ["This Agreement is governed by the laws of Texas"],
        END: [],
        TERMINATION: [],
        INSURANCE: ["The Licensee shall keep insurance in force"],
        AUDIT: ["allow audits of its books"],
    },
}
SHOWN_COUNTS = Counts(
    correct=3,
    wrong=1,
    false=2,
    omission=1,
    abstention=1,
    missed=1,
    correct_asked=2,
    omission_asked=1,
    present_asked=5,
    quotes_attempted=6,
    quotes_dropped=0,
)
FOUND_COUNTS = Counts(
    correct=1,
    wrong=1,
    false=1,
    omission=3,
    abstention=2,
    missed=1,
    correct_asked=1,
    omission_asked=2,
    present_asked=5,
    quotes_attempted=3,
    quotes_dropped=0,
)

OUTCOMES = ("correct", "wrong", "false", "omission", "abstention", "missed")
TABLE_HEADER = (
    "| router | policy | citation precision | citation recall | omission rate "
    "| false-citation rate | abstention accuracy | hallucination rate |"
)
CLAUSE_TYPES = [
    ClauseType("governing_law", LAW, "Question about Governing Law", "The governing law"),
    ClauseType("insurance", INSURANCE, "Question about Insurance", "The insurance required"),
]


def route_of(*names: str, kind: str = "embedding", latency_ms: float = 5.0) -> RouteResult:
    return RouteResult(clause_types=list(names), kind=kind, latency_ms=latency_ms)


def make_question(
    question_id: str, text: str, gold: list[str], kind: str = "single"
) -> EvalQuestion:
    return EvalQuestion(id=question_id, question=text, gold=gold, kind=kind, split="test")


QUESTIONS = [
    make_question("q1", LAW_QUESTION, [LAW]),
    make_question("q2", END_QUESTION, [END, TERMINATION], "compound"),
    make_question("q3", INSURANCE_QUESTION, [INSURANCE]),
    make_question("q4", WEATHER_QUESTION, [], "out_of_scope"),
]
ROUTES = {
    LAW_QUESTION: route_of(LAW),
    END_QUESTION: route_of(END),
    INSURANCE_QUESTION: route_of(INSURANCE, AUDIT),
}
NO_ROUTES = {question: route_of() for question in ROUTES}


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def caches(settings):
    return {
        "Alpha": make_cache(TEXT_ALPHA, settings, CLAUSES_ALPHA),
        "Beta": make_cache(TEXT_BETA, settings, CLAUSES_BETA),
    }


def run(caches, routes=ROUTES, questions=QUESTIONS) -> E2EReport:
    router = FixedRouter(routes)
    return evaluate_e2e(router, questions, caches, GOLD, TAU_LOW, TAU_HIGH)


@pytest.fixture
def report(caches) -> E2EReport:
    return run(caches)


@pytest.fixture
def oracle_report(caches) -> E2EReport:
    return evaluate_e2e(OracleRouter(QUESTIONS), QUESTIONS, caches, GOLD, TAU_LOW, TAU_HIGH)


@pytest.fixture
def narrow_report(caches) -> E2EReport:
    return run(caches, {**ROUTES, END_QUESTION: route_of()})


def solo_report(
    settings,
    spans,
    *,
    conf=0.9,
    clause_type=LAW,
    quote=LAW_SPAN,
    routed=True,
    asked=True,
) -> E2EReport:
    cache = make_cache(quote, settings, {clause_type: [(quote, conf)], INSURANCE: []})
    gold = {"Solo": {clause_type: spans, INSURANCE: []}}
    question = make_question("q", SOLO_QUESTION, [clause_type if asked else INSURANCE])
    route = route_of(clause_type) if routed else route_of()
    router = FixedRouter({SOLO_QUESTION: route})
    return evaluate_e2e(router, [question], {"Solo": cache}, gold, TAU_LOW, TAU_HIGH)


def counted(report: E2EReport, policy: str) -> dict[str, int]:
    counts = report.policies[policy].counts.model_dump()
    return {name: counts[name] for name in OUTCOMES if counts[name]}


def hallucination_rates(report: E2EReport) -> list[float | None]:
    return [report.policies[policy].metrics.hallucination_rate for policy in ("found", "shown")]


def qa_entry(question, texts, qa_id):
    answers = [{"text": text, "answer_start": 0} for text in texts]
    return {"id": qa_id, "question": question, "answers": answers, "is_impossible": not texts}


def write_cuad(tmp_path, qas):
    path = tmp_path / "test.json"
    data = [{"title": "Alpha", "paragraphs": [{"context": TEXT_ALPHA, "qas": qas}]}]
    path.write_text(json.dumps({"data": data}), encoding="utf-8")
    return path


def cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def split_tables(text: str) -> tuple[list[str], str, list[str]]:
    results, caption, losses = text.split("\n\n")
    return results.splitlines(), caption, losses.splitlines()


def test_found_policy_counts(report):
    assert report.policies["found"].counts == FOUND_COUNTS


def test_shown_policy_counts(report):
    assert report.policies["shown"].counts == SHOWN_COUNTS


@pytest.mark.parametrize(
    ("policy", "metric", "expected"),
    [
        ("found", "citation_precision", 1 / 3),
        ("found", "citation_recall", 1 / 5),
        ("found", "omission_rate", 3 / 5),
        ("found", "false_citation_rate", 1 / 3),
        ("found", "abstention_accuracy", 2 / 3),
        ("found", "hallucination_rate", 0 / 3),
        ("shown", "citation_precision", 3 / 6),
        ("shown", "citation_recall", 2 / 5),
        ("shown", "omission_rate", 2 / 5),
        ("shown", "false_citation_rate", 2 / 3),
        ("shown", "abstention_accuracy", 1 / 3),
        ("shown", "hallucination_rate", 0 / 6),
    ],
)
def test_metric_follows_its_definition(report, policy, metric, expected):
    assert getattr(report.policies[policy].metrics, metric) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("conf", "spans", "found", "shown"),
    [
        pytest.param(0.9, [LAW_SPAN], "correct", "correct", id="found-tier-matches-gold"),
        pytest.param(0.9, [LAW_SENTENCE], "wrong", "wrong", id="found-tier-misses-gold"),
        pytest.param(0.9, [], "false", "false", id="found-tier-gold-absent"),
        pytest.param(0.5, [LAW_SPAN], "omission", "correct", id="check-tier-matches-gold"),
        pytest.param(0.5, [LAW_SENTENCE], "omission", "wrong", id="check-tier-misses-gold"),
        pytest.param(0.5, [], "abstention", "false", id="check-tier-gold-absent"),
        pytest.param(0.1, [LAW_SPAN], "omission", "omission", id="below-tau-low-gold-present"),
        pytest.param(0.1, [], "abstention", "abstention", id="below-tau-low-gold-absent"),
    ],
)
def test_outcome_of_a_routed_type(settings, conf, spans, found, shown):
    report = solo_report(settings, spans, conf=conf)
    assert (counted(report, "found"), counted(report, "shown")) == ({found: 1}, {shown: 1})


def test_parties_citation_containing_the_gold_span_is_correct(settings):
    quote = "made between Acme Corp and Beta Ltd"
    report = solo_report(settings, ["Acme Corp"], clause_type=PARTIES, quote=quote)
    assert counted(report, "shown") == {"correct": 1}


def test_other_citation_containing_the_gold_span_is_wrong(settings):
    report = solo_report(settings, ["Delaware"], clause_type=LAW, quote=LAW_SPAN)
    assert counted(report, "shown") == {"wrong": 1}


def test_router_miss_on_a_present_type_is_counted_as_missed(settings):
    report = solo_report(settings, [LAW_SPAN], routed=False)
    assert report.policies["shown"].counts == Counts(missed=1, present_asked=1)


def test_router_miss_on_a_present_type_lowers_recall(settings):
    routed = solo_report(settings, [LAW_SPAN])
    missed = solo_report(settings, [LAW_SPAN], routed=False)
    recalls = [r.policies["shown"].metrics.citation_recall for r in (routed, missed)]
    assert recalls == [1.0, 0.0]


def test_router_miss_on_a_present_type_raises_the_omission_rate(settings):
    report = solo_report(settings, [LAW_SPAN], routed=False)
    assert report.policies["shown"].metrics.omission_rate == 1.0


def test_router_miss_on_an_absent_type_changes_nothing(settings):
    report = solo_report(settings, [], routed=False)
    assert report.policies["shown"].counts == Counts()


@pytest.mark.parametrize(
    ("conf", "outcome"),
    [
        pytest.param(0.9, "false", id="found-tier"),
        pytest.param(0.1, "abstention", id="below-tau-low"),
    ],
)
def test_extra_absent_type_is_false_or_abstention(settings, conf, outcome):
    report = solo_report(settings, [], conf=conf, asked=False)
    assert counted(report, "shown") == {outcome: 1}


def test_extra_absent_type_is_not_in_recall(settings):
    report = solo_report(settings, [], asked=False)
    assert report.policies["shown"].metrics.citation_recall is None


def test_extra_present_type_counts_for_precision_but_not_recall(settings):
    report = solo_report(settings, [LAW_SPAN], asked=False)
    metrics = report.policies["shown"].metrics
    assert (metrics.citation_precision, metrics.citation_recall) == (1.0, None)


def test_type_routed_twice_is_counted_once(caches):
    routes = {**ROUTES, LAW_QUESTION: route_of(LAW, LAW)}
    assert run(caches, routes).policies["shown"].counts == SHOWN_COUNTS


def test_tampered_quote_raises_the_hallucination_rate(caches):
    caches["Alpha"].clauses[LAW][0].text = "laws of Texas"
    assert hallucination_rates(run(caches)) == [pytest.approx(1 / 3), pytest.approx(1 / 6)]


def test_tampered_quote_is_not_shown(caches):
    caches["Alpha"].clauses[LAW][0].text = "laws of Texas"
    counts = run(caches).policies["shown"].counts
    assert (counts.correct, counts.omission) == (
        SHOWN_COUNTS.correct - 1,
        SHOWN_COUNTS.omission + 1,
    )


def test_tampered_check_tier_quote_is_not_counted_under_the_found_policy(caches):
    caches["Alpha"].clauses[END][0].text = "ten years"
    assert hallucination_rates(run(caches)) == [0.0, pytest.approx(1 / 6)]


def test_out_of_scope_question_is_skipped(caches):
    router = FixedRouter(ROUTES)
    report = evaluate_e2e(router, QUESTIONS, caches, GOLD, TAU_LOW, TAU_HIGH)
    assert (report.n_questions, WEATHER_QUESTION in router.questions) == (3, False)


def test_each_question_is_routed_once(caches):
    router = FixedRouter(ROUTES)
    evaluate_e2e(router, QUESTIONS, caches, GOLD, TAU_LOW, TAU_HIGH)
    assert router.questions == [LAW_QUESTION, END_QUESTION, INSURANCE_QUESTION]


def test_report_records_router_kind_and_thresholds(report):
    assert (report.router, report.tau_low, report.tau_high) == ("explicit", TAU_LOW, TAU_HIGH)


def test_report_records_scored_questions_and_contracts(report):
    assert (report.n_questions, report.n_contracts) == (3, 2)


def test_model_name_is_recorded(caches):
    router = FixedRouter(ROUTES)
    report = evaluate_e2e(
        router, QUESTIONS, caches, GOLD, TAU_LOW, TAU_HIGH, model="qwen3:1.7b"
    )
    assert report.model == "qwen3:1.7b"


def test_model_name_defaults_to_none(report):
    assert report.model is None


def test_report_keeps_the_route_of_each_question(caches):
    routes = {**ROUTES, END_QUESTION: route_of(END, kind="llm_fallback", latency_ms=12.5)}
    assert run(caches, routes).questions[1].model_dump() == {
        "id": "q2",
        "question": END_QUESTION,
        "kind": "compound",
        "gold": [END, TERMINATION],
        "route": {
            "clause_types": [END],
            "kind": "llm_fallback",
            "latency_ms": 12.5,
            "raw": None,
        },
    }


def test_fallback_rate_is_the_share_of_fallback_routes(caches):
    routes = {
        LAW_QUESTION: route_of(LAW, kind="llm"),
        END_QUESTION: route_of(END, kind="llm_fallback"),
        INSURANCE_QUESTION: route_of(INSURANCE, kind="llm"),
    }
    assert run(caches, routes).fallback_rate == pytest.approx(1 / 3)


def test_median_latency_is_taken_over_the_scored_questions(caches):
    routes = {
        LAW_QUESTION: route_of(LAW, latency_ms=10.0),
        END_QUESTION: route_of(END, latency_ms=60.0),
        INSURANCE_QUESTION: route_of(INSURANCE, latency_ms=20.0),
    }
    assert run(caches, routes).latency_median_ms == 20.0


def test_no_scored_question_leaves_every_metric_undefined(caches):
    report = run(caches, questions=[QUESTIONS[3]])
    metrics = [result.metrics.model_dump() for result in report.policies.values()]
    assert [value for values in metrics for value in values.values()] == [None] * 12


def test_no_scored_question_leaves_fallback_rate_and_latency_undefined(caches):
    report = run(caches, questions=[QUESTIONS[3]])
    assert (report.fallback_rate, report.latency_median_ms) == (None, None)


def test_undefined_rate_is_json_null(caches):
    report = run(caches, questions=[QUESTIONS[3]])
    metrics = json.loads(report.model_dump_json())["policies"]["shown"]["metrics"]
    assert metrics["citation_precision"] is None


def test_report_round_trips_through_json(report):
    assert E2EReport.model_validate_json(report.model_dump_json(indent=2)) == report


def test_contract_without_gold_raises(caches):
    gold = {"Alpha": GOLD["Alpha"]}
    with pytest.raises(GoldError, match="Beta"):
        evaluate_e2e(FixedRouter(ROUTES), QUESTIONS, caches, gold, TAU_LOW, TAU_HIGH)


def test_missing_gold_is_found_before_any_question_is_routed(caches):
    router = FixedRouter(ROUTES)
    with pytest.raises(GoldError):
        evaluate_e2e(router, QUESTIONS, caches, {}, TAU_LOW, TAU_HIGH)
    assert router.questions == []


def test_gold_spans_are_keyed_by_title_and_clause_type(tmp_path):
    qas = [
        qa_entry("Question about Governing Law", [LAW_SPAN, "Delaware"], "Alpha__Law"),
        qa_entry("Question about Insurance", [], "Alpha__Insurance"),
    ]
    gold = load_cuad_gold(write_cuad(tmp_path, qas), CLAUSE_TYPES)
    assert gold == {"Alpha": {LAW: [LAW_SPAN, "Delaware"], INSURANCE: []}}


def test_clause_type_comes_from_the_question_and_not_the_id(tmp_path):
    qas = [qa_entry("Question about Governing Law", [LAW_SPAN], "Alpha__Insurance")]
    gold = load_cuad_gold(write_cuad(tmp_path, qas), CLAUSE_TYPES)
    assert gold == {"Alpha": {LAW: [LAW_SPAN]}}


def test_unknown_question_raises(tmp_path):
    qas = [qa_entry("Question about Exclusivity", [], "Alpha__Exclusivity")]
    with pytest.raises(GoldError, match="Alpha"):
        load_cuad_gold(write_cuad(tmp_path, qas), CLAUSE_TYPES)


def test_oracle_router_returns_the_gold_types():
    route = OracleRouter(QUESTIONS).route(END_QUESTION)
    assert route.clause_types == [END, TERMINATION]


def test_oracle_router_ignores_the_earlier_turns():
    turns = [Turn(question=LAW_QUESTION, clause_types=[LAW])]
    route = OracleRouter(QUESTIONS).route(END_QUESTION, turns)
    assert route.clause_types == [END, TERMINATION]


def test_oracle_route_is_marked_oracle_with_no_latency():
    route = OracleRouter(QUESTIONS).route(LAW_QUESTION)
    assert (route.kind, route.latency_ms) == ("oracle", 0.0)


def test_oracle_report_names_the_oracle_router(oracle_report):
    assert oracle_report.router == "oracle"


def test_table_header_names_the_columns(oracle_report):
    assert markdown_table([oracle_report]).splitlines()[0] == TABLE_HEADER


def test_table_rows_show_metrics_with_three_decimals(oracle_report):
    rows = markdown_table([oracle_report]).splitlines()[2:]
    assert rows == [
        "| oracle | found | 0.333 | 0.200 | 0.600 | 0.333 | 0.667 | 0.000 |",
        "| oracle | shown | 0.400 | 0.400 | 0.400 | 0.667 | 0.333 | 0.000 |",
    ]


def test_table_has_a_row_for_each_router_and_policy(oracle_report, narrow_report):
    results, _, _ = split_tables(markdown_table([oracle_report, narrow_report]))
    assert [cells(row)[:2] for row in results[2:]] == [
        ["oracle", "found"],
        ["oracle", "shown"],
        ["explicit", "found"],
        ["explicit", "shown"],
    ]


def test_undefined_metric_is_shown_as_n_a(caches):
    nothing_routed = run(caches, NO_ROUTES)
    _, shown_row = markdown_table([nothing_routed]).splitlines()[2:]
    assert shown_row == "| explicit | shown | n/a | 0.000 | 1.000 | n/a | n/a | n/a |"


def test_model_name_is_part_of_the_row_label(caches):
    router = FixedRouter(ROUTES)
    llm = evaluate_e2e(router, QUESTIONS, caches, GOLD, TAU_LOW, TAU_HIGH, model="qwen3:1.7b")
    found_row = markdown_table([llm]).splitlines()[2]
    assert cells(found_row)[:2] == ["explicit (qwen3:1.7b)", "found"]


def test_routing_loss_table_subtracts_the_oracle(oracle_report, narrow_report):
    _, _, losses = split_tables(markdown_table([oracle_report, narrow_report]))
    assert losses[2:] == [
        "| explicit | found | +0.000 | +0.000 | +0.000 | +0.167 | -0.167 | +0.000 |",
        "| explicit | shown | +0.100 | -0.200 | +0.200 | -0.167 | +0.167 | +0.000 |",
    ]


def test_routing_loss_is_undefined_where_a_metric_is_undefined(caches, oracle_report):
    nothing_routed = run(caches, NO_ROUTES)
    _, _, losses = split_tables(markdown_table([oracle_report, nothing_routed]))
    assert losses[2:] == [
        "| explicit | found | n/a | -0.200 | +0.400 | n/a | n/a | n/a |",
        "| explicit | shown | n/a | -0.400 | +0.600 | n/a | n/a | n/a |",
    ]


def test_routing_loss_table_has_the_same_columns_as_the_results(oracle_report, narrow_report):
    results, _, losses = split_tables(markdown_table([oracle_report, narrow_report]))
    assert losses[0] == results[0] == TABLE_HEADER


def test_routing_loss_table_has_a_caption(oracle_report, narrow_report):
    _, caption, _ = split_tables(markdown_table([oracle_report, narrow_report]))
    assert caption == "Routing loss (router minus oracle)"


def test_oracle_alone_has_no_loss_table(oracle_report):
    assert "\n\n" not in markdown_table([oracle_report])


def test_routers_without_an_oracle_have_no_loss_table(narrow_report):
    assert "\n\n" not in markdown_table([narrow_report])


def test_loss_against_an_oracle_on_other_thresholds_raises(caches, narrow_report):
    other = evaluate_e2e(OracleRouter(QUESTIONS), QUESTIONS, caches, GOLD, 0.1, TAU_HIGH)
    with pytest.raises(ValueError, match="oracle"):
        markdown_table([other, narrow_report])


def test_loss_against_an_oracle_on_other_questions_raises(caches, narrow_report):
    questions = QUESTIONS[:2]
    other = evaluate_e2e(OracleRouter(questions), questions, caches, GOLD, TAU_LOW, TAU_HIGH)
    with pytest.raises(ValueError, match="oracle"):
        markdown_table([other, narrow_report])


def thesis_paths() -> tuple[Path, Path]:
    settings = Settings()
    preds = settings.diplomski_dir / "results" / f"{settings.extractor_name}_preds.jsonl"
    return preds, settings.diplomski_dir / "data" / "raw" / "test.json"


THESIS_PREDS, THESIS_CUAD = thesis_paths()
THESIS_FILES = (THESIS_PREDS, THESIS_PREDS.with_suffix(".config.json"), THESIS_CUAD)
needs_thesis_data = pytest.mark.skipif(
    not all(path.exists() for path in THESIS_FILES),
    reason="thesis predictions or CUAD test.json not found",
)


@pytest.fixture(scope="module")
def thesis_report() -> E2EReport:
    settings = Settings()
    clause_types = load_clause_types(settings.clause_types_path)
    questions = [
        make_question(f"t{number}", clause_type.name, [clause_type.name])
        for number, clause_type in enumerate(clause_types)
    ]
    return evaluate_e2e(
        OracleRouter(questions),
        questions,
        import_harness_preds(THESIS_PREDS, THESIS_CUAD, settings),
        load_cuad_gold(THESIS_CUAD, clause_types),
        THESIS_TAU_LOW,
        THESIS_TAU_HIGH,
    )


@needs_thesis_data
@pytest.mark.parametrize("policy", ["shown", "found"])
def test_oracle_counts_equal_the_thesis_operating_point(thesis_report, policy):
    expected = THESIS_COUNTS[policy]
    assert thesis_report.policies[policy].counts.model_dump(include=set(expected)) == expected


@needs_thesis_data
def test_thesis_run_scores_every_contract_and_clause_type(thesis_report):
    sizes = (thesis_report.n_contracts, thesis_report.n_questions)
    assert sizes == (THESIS_N_CONTRACTS, THESIS_N_CLAUSE_TYPES)
