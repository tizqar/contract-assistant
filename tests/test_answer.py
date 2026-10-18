from pathlib import Path

import pytest

from contract_assistant.answer import answer, compose, tier, verify_quotes
from contract_assistant.config import Settings
from contract_assistant.router.base import RouteResult
from contract_assistant.store import read_cache
from tests.support.caches import make_cache

TEXT = "This Agreement is governed by the laws of Delaware. Term is five years."
LAW = "laws of Delaware"
TERM = "five years"
TAU_LOW = 0.2
TAU_HIGH = 0.8
DEMO_CACHE = Path(__file__).resolve().parent.parent / "demo/cache/short_supply_agreement.json"


@pytest.fixture
def settings() -> Settings:
    return Settings()


def route_of(*names: str) -> RouteResult:
    return RouteResult(clause_types=list(names), kind="explicit", latency_ms=0.0)


def test_confidence_at_tau_high_is_found():
    assert tier(TAU_HIGH, TAU_LOW, TAU_HIGH) == "found"


def test_confidence_just_below_tau_high_is_check():
    assert tier(TAU_HIGH - 0.001, TAU_LOW, TAU_HIGH) == "check"


def test_confidence_at_tau_low_is_check():
    assert tier(TAU_LOW, TAU_LOW, TAU_HIGH) == "check"


def test_confidence_just_below_tau_low_is_not_found():
    assert tier(TAU_LOW - 0.001, TAU_LOW, TAU_HIGH) == "not_found"


def test_tau_low_above_tau_high_raises(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.5)]})
    with pytest.raises(ValueError):
        compose(cache, ["Governing Law"], 0.9, 0.1)


def test_found_and_check_carry_top_candidate(settings):
    cache = make_cache(
        TEXT,
        settings,
        {"Governing Law": [(LAW, 0.95), (TERM, 0.02)], "Term": [(TERM, 0.5)]},
    )
    found, check = compose(cache, ["Governing Law", "Term"], TAU_LOW, TAU_HIGH)
    assert (found.status, found.citation.quote) == ("found", LAW)
    assert (check.status, check.citation.quote) == ("check", TERM)


def test_not_found_carries_no_citation(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.03)]})
    (item,) = compose(cache, ["Governing Law"], TAU_LOW, TAU_HIGH)
    assert (item.status, item.citation, item.best_conf) == ("not_found", None, 0.03)


def test_empty_n_best_is_not_found_with_zero_confidence(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": []})
    (item,) = compose(cache, ["Governing Law"], TAU_LOW, TAU_HIGH)
    assert (item.status, item.citation, item.best_conf) == ("not_found", None, 0.0)


def test_empty_n_best_stays_not_found_at_zero_tau_low(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": []})
    (item,) = compose(cache, ["Governing Law"], 0.0, TAU_HIGH)
    assert item.status == "not_found"


def test_unknown_clause_type_raises_key_error(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)]})
    with pytest.raises(KeyError):
        compose(cache, ["Exclusivity"], TAU_LOW, TAU_HIGH)


def test_items_follow_route_order(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.9)], "Term": [(TERM, 0.9)]})
    result = answer("q", route_of("Term", "Governing Law"), cache, TAU_LOW, TAU_HIGH)
    assert [item.clause_type for item in result.items] == ["Term", "Governing Law"]


def test_answer_records_thresholds_and_question(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.9)]})
    result = answer("How long?", route_of("Term"), cache, TAU_LOW, TAU_HIGH)
    assert (result.question, result.tau_low, result.tau_high) == ("How long?", 0.2, 0.8)


def test_tampered_quote_is_withheld_and_counted(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.9)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH)
    items[0].citation.quote = "ten years"
    verified, dropped = verify_quotes(items, cache.text)
    assert (verified[0].citation, verified[0].withheld, dropped) == (None, True, 1)


def test_tampered_offset_is_withheld_and_counted(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.9)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH)
    items[0].citation.start += 1
    verified, dropped = verify_quotes(items, cache.text)
    assert (verified[0].citation, verified[0].withheld, dropped) == (None, True, 1)


def test_withheld_item_keeps_its_tier(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.9)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH)
    items[0].citation.quote = "ten years"
    verified, _ = verify_quotes(items, cache.text)
    assert (verified[0].status, verified[0].best_conf) == ("found", 0.9)


def test_untampered_demo_cache_drops_no_quotes(settings):
    cache = read_cache(DEMO_CACHE, settings)
    route = route_of(*cache.clauses)
    result = answer("q", route, cache, settings.tau_low, settings.tau_high)
    assert (len(result.items), result.dropped_quotes) == (41, 0)


def test_unvalidated_type_above_tau_high_is_check(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.95)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH, unvalidated={"Term"})
    assert items[0].status == "check"
    assert items[0].citation.conf == 0.95


def test_validated_type_above_tau_high_is_found_when_others_are_unvalidated(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.95)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH, unvalidated={"Other"})
    assert items[0].status == "found"


def test_unvalidated_type_below_tau_low_stays_not_found(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.05)]})
    items = compose(cache, ["Term"], TAU_LOW, TAU_HIGH, unvalidated={"Term"})
    assert items[0].status == "not_found"


def test_answer_caps_unvalidated_types(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.95)]})
    result = answer("q", route_of("Term"), cache, TAU_LOW, TAU_HIGH, unvalidated=["Term"])
    assert result.items[0].status == "check"


def test_answer_without_unvalidated_is_unchanged(settings):
    cache = make_cache(TEXT, settings, {"Term": [(TERM, 0.95)]})
    result = answer("q", route_of("Term"), cache, TAU_LOW, TAU_HIGH)
    assert result.items[0].status == "found"
