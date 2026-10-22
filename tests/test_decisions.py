import json

import pytest

from contract_assistant.config import load_settings
from contract_assistant.decisions import (
    CONFIRMED_CONF,
    Decision,
    DecisionStore,
    apply,
    now,
)
from tests.support.caches import make_cache

LAW = "This agreement is governed by the laws of Vanuatu."
OTHER = "Disputes go to arbitration in Geneva."
TEXT = f"Preamble. {LAW} {OTHER} Final words."


@pytest.fixture
def settings():
    return load_settings()


def decision(verdict, text, clause_type="Governing Law") -> Decision:
    start = TEXT.find(text)
    return Decision(
        clause_type=clause_type, verdict=verdict, start=start, end=start + len(text), at=now()
    )


def test_a_confirmed_span_moves_to_the_top_with_full_confidence(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(OTHER, 0.5), (LAW, 0.3)]})
    decided = apply(cache, {"Governing Law": decision("confirmed", LAW)})
    top = decided.clauses["Governing Law"][0]
    assert (top.text, top.conf) == (LAW, CONFIRMED_CONF)
    assert [c.text for c in decided.clauses["Governing Law"]] == [LAW, OTHER]
    assert cache.clauses["Governing Law"][0].text == OTHER, "the cache itself is unchanged"


def test_a_rejected_span_is_removed_and_the_next_candidate_leads(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(OTHER, 0.5), (LAW, 0.3)]})
    decided = apply(cache, {"Governing Law": decision("rejected", OTHER)})
    assert [c.text for c in decided.clauses["Governing Law"]] == [LAW]
    alone = make_cache(TEXT, settings, {"Governing Law": [(OTHER, 0.5)]})
    assert apply(alone, {"Governing Law": decision("rejected", OTHER)}).clauses["Governing Law"] == []


def test_a_confirmed_span_the_model_no_longer_lists_is_rebuilt_from_the_text(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(OTHER, 0.5)]})
    decided = apply(cache, {"Governing Law": decision("confirmed", LAW)})
    top = decided.clauses["Governing Law"][0]
    assert (top.text, top.conf, top.page) == (LAW, CONFIRMED_CONF, None)


def test_a_span_outside_the_text_is_ignored(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(OTHER, 0.5)]})
    bad = Decision(clause_type="Governing Law", verdict="confirmed", start=5, end=9999, at=now())
    assert apply(cache, {"Governing Law": bad}).clauses == cache.clauses


def test_no_decisions_return_the_same_cache(settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW, 0.5)]})
    assert apply(cache, {}) is cache


def test_the_store_keeps_one_decision_per_clause_type(tmp_path):
    store = DecisionStore(tmp_path / "decisions")
    assert store.load("sha256:abc") == {}
    store.decide("sha256:abc", decision("confirmed", LAW))
    store.decide("sha256:abc", decision("rejected", OTHER, "Dispute Resolution"))
    store.decide("sha256:abc", decision("rejected", LAW))
    loaded = store.load("sha256:abc")
    assert loaded["Governing Law"].verdict == "rejected"
    assert loaded["Dispute Resolution"].verdict == "rejected"
    entries = json.loads(store.path_of("sha256:abc").read_text())
    assert len(entries) == 2
    store.forget("sha256:abc", "Governing Law")
    assert list(store.load("sha256:abc")) == ["Dispute Resolution"]


def test_a_confirmed_candidate_is_marked_and_never_capped(settings):
    from contract_assistant.answer import compose

    cache = make_cache(TEXT, settings, {"Confidentiality": [(LAW, 0.95)]})
    capped = compose(cache, ["Confidentiality"], 0.06, 0.91, {"Confidentiality"})
    assert capped[0].status == "check", "an extra clause type is capped at check"
    decided = apply(cache, {"Confidentiality": decision("confirmed", LAW, "Confidentiality")})
    assert decided.clauses["Confidentiality"][0].confirmed
    items = compose(decided, ["Confidentiality"], 0.06, 0.91, {"Confidentiality"})
    assert items[0].status == "found"
