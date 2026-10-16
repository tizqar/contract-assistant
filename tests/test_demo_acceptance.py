import json

import pytest

from contract_assistant.config import Settings
from contract_assistant.cuad_match import RELAXED_CLAUSE_TYPES, is_match
from contract_assistant.store import read_cache
from tests.conftest import ROOT

DEMO = ROOT / "demo"


def matched_clause_types(slug: str) -> int:
    """Count present clause types whose top candidate matches a gold span."""
    documents = json.loads((DEMO / "demo_documents.json").read_text(encoding="utf-8"))
    document = next(d for d in documents if d["slug"] == slug)
    cache = read_cache(DEMO / "cache" / f"{slug}.json", Settings())
    matched = 0
    for clause_type in document["present_clause_types"]:
        candidates = cache.clauses.get(clause_type, [])
        if not candidates:
            continue
        relaxed = clause_type in RELAXED_CLAUSE_TYPES
        gold_spans = document["gold_spans"][clause_type]
        if any(is_match(gold, candidates[0].text, relaxed) for gold in gold_spans):
            matched += 1
    return matched


@pytest.mark.parametrize(
    ("slug", "floor"),
    [
        pytest.param("short_supply_agreement", 14, id="short"),
        pytest.param("medium_sponsorship_agreement", 18, id="medium"),
        pytest.param("long_outsourcing_agreement", 15, id="long"),
    ],
)
def test_top_candidates_match_gold_for_enough_clause_types(slug, floor):
    assert matched_clause_types(slug) >= floor
