import pytest

from contract_assistant.cuad_match import (
    IOU_THRESH,
    RELAXED_CLAUSE_TYPES,
    get_jaccard,
    is_match,
)

TERM = "This agreement is for a term of ten (10) years."
PARTIES_PRED = (
    "This ODM Supply Agreement is made between AGAPE ATP INTERNATIONAL HOLDING LIMITED "
    "and ORGANIC PREPARATIONS INC. as the Customer"
)

CASES = [
    pytest.param(TERM, TERM, False, 1.0, True, id="exact"),
    pytest.param(
        TERM,
        "This agreement is for a term of ten (10) years and may be extended",
        False,
        0.7142857142857143,
        True,
        id="overlap_above_threshold",
    ),
    pytest.param(
        TERM,
        "This agreement is for a term of five (5) calendar months total",
        False,
        0.4666666666666667,
        False,
        id="overlap_below_threshold",
    ),
    pytest.param(
        TERM,
        "This agreement is for a term of five (5) calendar months",
        False,
        0.5,
        True,
        id="overlap_at_threshold",
    ),
    pytest.param(
        TERM,
        "The term of ten years begins on the date hereof and continues until terminated "
        "by either party",
        False,
        0.18181818181818182,
        False,
        id="mostly_different",
    ),
    pytest.param(TERM, "", False, 0.0, False, id="empty_prediction"),
    pytest.param("", "", False, 1.0, True, id="both_empty"),
    pytest.param(
        "Either party may terminate this agreement",
        "Either party  may terminate\nthis agreement",
        False,
        0.5,
        True,
        id="double_space_and_newline",
    ),
    pytest.param(
        "Either party may terminate this agreement",
        "Either party may terminate\nthis agreement",
        False,
        0.5714285714285714,
        True,
        id="newline_joins_words",
    ),
    pytest.param(
        "This agreement will be automatically renewed at the end of every ten (10) year "
        "term, with each subsequent term of renewal being for a ten (10) year term.",
        TERM,
        False,
        0.3333333333333333,
        False,
        id="renewal_versus_term",
    ),
    pytest.param(
        "15t h day of January 2018",
        "15t h day of January 2018, by and between",
        False,
        0.6666666666666666,
        True,
        id="agreement_date_with_tail",
    ),
    pytest.param(
        "ORGANIC PREPARATIONS INC.",
        PARTIES_PRED,
        False,
        0.15789473684210525,
        False,
        id="party_in_long_prediction",
    ),
    pytest.param(
        "ORGANIC PREPARATIONS INC.",
        PARTIES_PRED,
        True,
        0.15789473684210525,
        True,
        id="party_in_long_prediction_relaxed",
    ),
]


@pytest.mark.parametrize(("gold", "pred", "relaxed", "jaccard", "match"), CASES)
def test_jaccard_matches_reference_value(gold, pred, relaxed, jaccard, match):
    assert get_jaccard(gold, pred) == pytest.approx(jaccard)


@pytest.mark.parametrize(("gold", "pred", "relaxed", "jaccard", "match"), CASES)
def test_is_match_matches_reference_value(gold, pred, relaxed, jaccard, match):
    assert is_match(gold, pred, relaxed) is match


def test_constants():
    assert IOU_THRESH == 0.5
    assert RELAXED_CLAUSE_TYPES == ("Parties",)


def test_punctuation_is_ignored():
    assert get_jaccard("Term: ten, (10); years.", "Term ten (10) years") == 1.0


def test_slash_splits_words():
    assert get_jaccard("and/or its affiliates", "and or its affiliates") == 1.0


def test_case_is_ignored():
    assert get_jaccard("ODM SUPPLY AGREEMENT", "odm supply agreement.") == 1.0


def test_substring_gold_needs_relaxed_flag():
    gold = "ORGANIC PREPARATIONS INC."
    assert not is_match(gold, PARTIES_PRED)
    assert is_match(gold, PARTIES_PRED, relaxed=True)
