"""Unit tests for the n-best postprocessing, without a model."""

import math

from contract_assistant.extract.qa_core import Candidate, nbest_to_candidates

CTX = "Preamble. This Agreement shall be governed by the laws of New York.\n Other text."
GOV = CTX.index("This Agreement")
GOV_END = CTX.index("New York.") + len("New York.")


def test_dedupes_to_unique_texts_keeping_best_score_and_its_offsets():
    # the same span rediscovered by two windows, once with a trailing newline, once with
    # a lower score; plus a different span
    prelim = [
        (5.0, GOV, GOV_END),  # best occurrence of the governing-law span
        (4.5, GOV, GOV_END + 2),  # same text + "\n " -> same after strip
        (3.0, 0, len("Preamble.")),  # a second unique text
        (1.0, GOV, GOV_END),  # duplicate, lower score, must not win
    ]
    cands = nbest_to_candidates(prelim, null_score=-10.0, context=CTX, n_best_size=20, top_k=20)
    assert [c.text for c in cands] == [CTX[GOV:GOV_END], "Preamble."]
    gov = cands[0]
    assert (gov.start, gov.end) == (GOV, GOV_END)
    assert CTX[gov.start : gov.end] == gov.text  # the offset invariant
    assert cands[0].conf > cands[1].conf


def test_softmax_includes_null_and_sums_to_one_with_it():
    prelim = [(2.0, 0, 9), (1.0, GOV, GOV_END)]
    null = 2.0
    cands = nbest_to_candidates(prelim, null, CTX, n_best_size=20, top_k=20)
    z = sum(math.exp(s - 2.0) for s in (2.0, 1.0, null))
    assert math.isclose(cands[0].conf, math.exp(0.0) / z, rel_tol=1e-9)
    assert math.isclose(sum(c.conf for c in cands), 1 - math.exp(0.0) / z, rel_tol=1e-9)


def test_n_best_size_caps_unique_texts_before_softmax():
    prelim = [(float(10 - i), i, i + 1) for i in range(10)]  # ten distinct one-char texts
    cands = nbest_to_candidates(prelim, -50.0, "abcdefghij", n_best_size=3, top_k=20)
    assert [c.text for c in cands] == ["a", "b", "c"]


def test_strips_offsets_with_the_text():
    ctx = "  x  "
    (c,) = nbest_to_candidates([(1.0, 0, 5)], -5.0, ctx, 20, 20)
    assert c == Candidate("x", c.conf, 2, 3)


def test_empty_and_whitespace_only_candidates_are_dropped():
    assert nbest_to_candidates([], 0.0, CTX, 20, 20) == []
    assert nbest_to_candidates([(1.0, GOV_END, GOV_END + 2)], 0.0, CTX, 20, 20) == []  # "\n "
