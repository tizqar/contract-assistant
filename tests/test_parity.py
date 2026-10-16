"""QAModel must reproduce the thesis evaluation driver on two CUAD test contracts.

The fixture was produced by `diplomski/src/eval_cuad_qa.py` (CPU, fp32). A checkpoint with a
trained QA head is required; the base Ettin head is randomly initialised on every load.
"""

import json

import numpy as np
import pytest

from tests.conftest import FIXTURES

pytestmark = pytest.mark.slow

MODEL = "rankyx/ModernBERT-QnA-base-squad"


def _load_fixture():
    lines = (FIXTURES / "PARITY_modernbert_squad_cpu_preds.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {r["contract_id"]: r["preds"] for r in rows}


def test_vendored_core_matches_harness(parity_contracts, clause_types):
    from contract_assistant.extract.qa_core import QAModel

    fixture = _load_fixture()
    question_of = {ct["name"]: ct["cuad_question"] for ct in clause_types["clause_types"]}
    qa = QAModel(MODEL, device="cpu", batch_size=16)
    n_checked = 0
    for cid, per_type in fixture.items():
        text = parity_contracts[cid]
        for clause_type, expected in per_type.items():
            got = qa(
                question_of[clause_type],
                text,
                max_seq_len=1024,
                doc_stride=256,
                max_answer_len=512,
                top_k=20,
                n_best_size=20,
            )
            assert [c.text for c in got] == [t for t, _ in expected], (cid, clause_type)
            confs = [p for _, p in expected]
            assert np.allclose([c.conf for c in got], confs, atol=1e-5), (cid, clause_type)
            assert all(text[c.start : c.end] == c.text for c in got), (cid, clause_type)
            n_checked += len(got)
    assert n_checked > 0
