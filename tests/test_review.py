import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contract_assistant.answer import STATUS_WORDS
from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import Settings
from contract_assistant.review import (
    MAX_TOKENS,
    RULES_PATH,
    SEVERITY_ORDER,
    SOURCE,
    Flag,
    Rule,
    RuleFileError,
    flag_request,
    load_rules,
    render_markdown,
    review,
)
from contract_assistant.store import read_cache
from tests.conftest import ROOT
from tests.support.caches import make_cache

TAU_LOW = 0.2
TAU_HIGH = 0.8
LAW = "This Agreement is governed by the laws of Delaware."
CAP = "Liability is capped at   fees paid\nin the prior year."
RENEWAL = "It renews for one year."
NOTICE = "Notice to stop renewal is ninety days."
INSURANCE = "Each party shall carry insurance."
TEXT = f"{LAW} {CAP} {RENEWAL} {NOTICE} {INSURANCE}"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"
N_RULES = 18


@pytest.fixture
def settings() -> Settings:
    return Settings()


def cache_of(settings, clauses):
    return make_cache(TEXT, settings, clauses)


def rule(kind, clause_types, unless=(), severity="medium", rule_id="r") -> Rule:
    return Rule(
        id=rule_id,
        kind=kind,
        clause_types=list(clause_types),
        unless=list(unless),
        severity=severity,
        title=f"Title {rule_id}",
        note=f"Note {rule_id}.",
    )


def run(cache, rules, tau_low=TAU_LOW, tau_high=TAU_HIGH, unvalidated=()):
    return review(cache, rules, tau_low, tau_high, unvalidated)


def write_rules(path: Path, *entries: dict) -> Path:
    path.write_text(json.dumps({"rules": list(entries)}), encoding="utf-8")
    return path


def entry(**overrides) -> dict:
    base = {
        "id": "x",
        "kind": "missing",
        "clause_types": ["Governing Law"],
        "severity": "low",
        "title": "T",
        "note": "N",
    }
    return {**base, **overrides}


def test_shipped_rules_load_with_unique_ids_and_valid_severities():
    rules = load_rules(RULES_PATH)
    assert len(rules) == N_RULES
    assert len({r.id for r in rules}) == N_RULES
    assert {r.severity for r in rules} <= set(SEVERITY_ORDER)


def test_shipped_rules_name_only_the_app_clause_types(settings):
    known = [clause_type.name for clause_type in app_clause_types(settings)]
    assert len(load_rules(RULES_PATH, known)) == N_RULES


def test_unknown_clause_type_in_the_rule_file_is_named_in_the_error(tmp_path):
    path = write_rules(tmp_path / "rules.json", entry(clause_types=["Made Up Type"]))
    with pytest.raises(RuleFileError, match="Made Up Type"):
        load_rules(path, ["Governing Law"])


def test_unknown_clause_type_in_unless_is_named_in_the_error(tmp_path):
    path = write_rules(
        tmp_path / "rules.json",
        entry(kind="present_without", unless=["Other Made Up"]),
    )
    with pytest.raises(RuleFileError, match="Other Made Up"):
        load_rules(path, ["Governing Law"])


def test_rules_are_not_checked_without_known_types(tmp_path):
    path = write_rules(tmp_path / "rules.json", entry(clause_types=["Made Up Type"]))
    assert load_rules(path)[0].clause_types == ["Made Up Type"]


def test_extra_key_in_a_rule_fails_validation(tmp_path):
    path = write_rules(tmp_path / "rules.json", entry(colour="red"))
    with pytest.raises(ValidationError):
        load_rules(path)


def test_rules_keep_file_order(tmp_path):
    path = write_rules(tmp_path / "rules.json", entry(id="b"), entry(id="a"))
    assert [r.id for r in load_rules(path)] == ["b", "a"]


def test_missing_rule_fires_when_the_type_is_below_tau_low(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.1)]})
    (flag,) = run(cache, [rule("missing", ["Governing Law"])])
    assert flag.status == "not_found"
    assert [item.clause_type for item in flag.items] == ["Governing Law"]
    assert flag.items[0].citation is None


def test_missing_rule_fires_when_the_type_is_absent_from_the_cache(settings):
    cache = cache_of(settings, {})
    (flag,) = run(cache, [rule("missing", ["Governing Law"])])
    assert flag.status == "not_found"
    assert flag.items == []


def test_missing_rule_lists_every_checked_type_that_is_in_the_cache(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.1)], "Insurance": [(INSURANCE, 0.1)]})
    (flag,) = run(cache, [rule("missing", ["Governing Law", "Insurance"])])
    assert [item.clause_type for item in flag.items] == ["Governing Law", "Insurance"]


def test_missing_rule_does_not_fire_when_the_type_is_found(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.9)]})
    assert run(cache, [rule("missing", ["Governing Law"])]) == []


def test_missing_rule_does_not_fire_in_the_verify_band(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.5)]})
    assert run(cache, [rule("missing", ["Governing Law"])]) == []


def test_missing_rule_fires_when_the_only_quote_is_withheld(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.9)]})
    cache.clauses["Governing Law"][0].start += 1
    (flag,) = run(cache, [rule("missing", ["Governing Law"])])
    assert flag.status == "not_found"
    assert flag.items[0].citation is None


def test_present_rule_fires_as_found_for_a_found_type(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.9)]})
    (flag,) = run(cache, [rule("present", ["Governing Law"])])
    assert flag.status == "found"
    assert flag.items[0].citation.quote == LAW


def test_present_rule_fires_as_check_for_a_verify_type(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.5)]})
    (flag,) = run(cache, [rule("present", ["Governing Law"])])
    assert flag.status == "check"


def test_present_rule_does_not_fire_below_tau_low(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.1)]})
    assert run(cache, [rule("present", ["Governing Law"])]) == []


def test_present_rule_is_found_when_any_item_is_found(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.5)], "Insurance": [(INSURANCE, 0.9)]})
    (flag,) = run(cache, [rule("present", ["Governing Law", "Insurance"])])
    assert flag.status == "found"


def test_present_rule_lists_only_the_items_with_a_citation(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.9)], "Insurance": [(INSURANCE, 0.1)]})
    (flag,) = run(cache, [rule("present", ["Governing Law", "Insurance"])])
    assert [item.clause_type for item in flag.items] == ["Governing Law"]


def test_present_rule_does_not_fire_for_a_withheld_quote(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.9)]})
    cache.clauses["Governing Law"][0].start += 1
    assert run(cache, [rule("present", ["Governing Law"])]) == []


def test_all_present_rule_fires_when_every_type_has_a_citation(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.5)]})
    (flag,) = run(cache, [rule("all_present", ["Renewal Term", "Notice"])])
    assert flag.status == "found"
    assert [item.clause_type for item in flag.items] == ["Renewal Term", "Notice"]


def test_all_present_rule_does_not_fire_when_one_type_is_not_found(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.1)]})
    assert run(cache, [rule("all_present", ["Renewal Term", "Notice"])]) == []


def test_all_present_rule_does_not_fire_when_one_type_is_absent(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)]})
    assert run(cache, [rule("all_present", ["Renewal Term", "Notice"])]) == []


def test_all_present_rule_does_not_fire_when_one_quote_is_withheld(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.9)]})
    cache.clauses["Notice"][0].start += 1
    assert run(cache, [rule("all_present", ["Renewal Term", "Notice"])]) == []


def test_present_without_rule_fires_when_no_unless_type_has_a_citation(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.1)]})
    (flag,) = run(cache, [rule("present_without", ["Renewal Term"], ["Notice"])])
    assert flag.status == "found"
    assert [item.clause_type for item in flag.items] == ["Renewal Term"]


def test_present_without_rule_does_not_fire_when_an_unless_type_is_in_the_verify_band(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.5)]})
    assert run(cache, [rule("present_without", ["Renewal Term"], ["Notice"])]) == []


def test_present_without_rule_does_not_fire_when_the_type_itself_is_not_found(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.1)]})
    assert run(cache, [rule("present_without", ["Renewal Term"], ["Notice"])]) == []


def test_present_without_rule_fires_when_the_unless_quote_is_withheld(settings):
    cache = cache_of(settings, {"Renewal Term": [(RENEWAL, 0.9)], "Notice": [(NOTICE, 0.9)]})
    cache.clauses["Notice"][0].start += 1
    (flag,) = run(cache, [rule("present_without", ["Renewal Term"], ["Notice"])])
    assert [item.clause_type for item in flag.items] == ["Renewal Term"]


def test_flags_are_sorted_by_severity_and_keep_file_order_within_one(settings):
    cache = cache_of(settings, {})
    rules = [
        rule("missing", ["Governing Law"], severity="low", rule_id="l1"),
        rule("missing", ["Insurance"], severity="high", rule_id="h1"),
        rule("missing", ["Notice"], severity="medium", rule_id="m1"),
        rule("missing", ["Renewal Term"], severity="high", rule_id="h2"),
        rule("missing", ["Cap"], severity="low", rule_id="l2"),
    ]
    assert [flag.rule for flag in run(cache, rules)] == ["h1", "h2", "m1", "l1", "l2"]


def test_flag_is_unvalidated_when_a_rule_type_is_in_the_unvalidated_set(settings):
    cache = cache_of(settings, {"Insurance": [(INSURANCE, 0.9)]})
    rules = [
        rule("present", ["Insurance"], rule_id="a"),
        rule("missing", ["Notice"], rule_id="b"),
    ]
    flags = run(cache, rules, unvalidated=["Insurance"])
    assert {flag.rule: flag.unvalidated for flag in flags} == {"a": True, "b": False}


def test_unvalidated_type_is_capped_at_check_in_a_flag(settings):
    cache = cache_of(settings, {"Insurance": [(INSURANCE, 0.95)]})
    (flag,) = run(cache, [rule("present", ["Insurance"])], unvalidated=["Insurance"])
    assert flag.status == "check"


def test_a_conf_of_half_is_verify_at_a_low_tau_low_and_not_found_at_a_high_one(settings):
    cache = cache_of(settings, {"Governing Law": [(LAW, 0.5)]})
    present = [rule("present", ["Governing Law"])]
    missing = [rule("missing", ["Governing Law"])]
    assert run(cache, present, tau_low=0.2)[0].status == "check"
    assert run(cache, missing, tau_low=0.2) == []
    assert run(cache, present, tau_low=0.6, tau_high=0.9) == []
    assert run(cache, missing, tau_low=0.6, tau_high=0.9)[0].status == "not_found"


def test_shipped_rules_on_the_short_demo_cache_give_seven_flags(settings):
    cache = read_cache(DEMO_CACHE, settings)
    extra = [c.name for c in load_extra_clause_types(settings.extra_clause_types_path)]
    rules = load_rules(RULES_PATH)
    flags = run(cache, rules, settings.tau_low, settings.tau_high, extra)
    titles = [flag.title for flag in flags]
    assert len(flags) == 7
    assert titles[0] == "No cap on liability found"
    assert flags[0].severity == "medium"
    assert "Automatic renewal with a notice window" in titles


def make_flag(settings) -> Flag:
    cache = cache_of(
        settings,
        {"Governing Law": [(LAW, 0.9)], "Cap": [(CAP, 0.5)], "Insurance": [(INSURANCE, 0.1)]},
    )
    (flag,) = run(cache, [rule("present", ["Governing Law", "Cap", "Insurance"])])
    return flag


def test_flag_request_rejects_a_flag_without_quotes(settings):
    cache = cache_of(settings, {})
    (flag,) = run(cache, [rule("missing", ["Governing Law"])])
    with pytest.raises(ValueError):
        flag_request(flag)


def test_flag_request_user_message_names_the_title_and_lists_each_cited_item(settings):
    flag = make_flag(settings)
    request = flag_request(flag)
    collapsed = "Liability is capped at fees paid in the prior year."
    lines = [line for line in request.user.splitlines() if line.startswith("- ")]
    assert request.user.startswith(f"Review point: {flag.title}\n")
    assert lines == [
        f'- Governing Law ({STATUS_WORDS["found"]}): "{LAW}"',
        f'- Cap ({STATUS_WORDS["check"]}): "{collapsed}"',
    ]


def test_flag_request_carries_the_limit_source_and_label(settings):
    flag = make_flag(settings)
    request = flag_request(flag)
    assert request.max_tokens == MAX_TOKENS
    assert request.source == SOURCE
    assert flag.rule in request.label


def test_flag_request_check_rejects_a_misquote_and_accepts_a_faithful_sentence(settings):
    request = flag_request(make_flag(settings))
    assert request.check(f'The law is clear: "{LAW}"') is None
    assert request.check('The contract says "This Agreement is governed by French law."')


def test_render_markdown_says_no_flags_when_empty():
    assert render_markdown([]) == "No review flags."


def test_render_markdown_has_one_line_per_flag_and_an_indented_line_per_item(settings):
    flag = make_flag(settings)
    lines = render_markdown([flag]).splitlines()
    assert lines[0] == f"- [medium] {flag.title}: {flag.note}"
    assert lines[1] == f'    Governing Law (found): "{LAW}"'
    assert lines[2].startswith("    Cap (possible, verify): ")
    assert len(lines) == 3


def test_render_markdown_marks_an_unvalidated_flag(settings):
    cache = cache_of(settings, {"Insurance": [(INSURANCE, 0.9)]})
    flags = run(cache, [rule("present", ["Insurance"])], unvalidated=["Insurance"])
    assert render_markdown(flags).startswith(
        "- [medium] Title r (not validated on test data): Note r."
    )


def test_render_markdown_has_a_line_without_items_for_a_missing_flag(settings):
    cache = cache_of(settings, {})
    flags = run(cache, [rule("missing", ["Governing Law"], severity="high")])
    assert render_markdown(flags) == "- [high] Title r: Note r."
