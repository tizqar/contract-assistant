import json
from pathlib import Path

import pytest

from contract_assistant.config import Settings
from contract_assistant.store import (
    HarnessImportError,
    import_harness_preds,
    read_cache,
    write_cache,
)

TITLE_A = "Alpha_Supply Agreement"
TITLE_B = "Beta_License Agreement"
TEXT_A = "Term is five years. The laws of Delaware apply. The laws of Delaware govern."
TEXT_B = "Licensor grants a license. Either party may terminate."


@pytest.fixture
def settings() -> Settings:
    return Settings()


def _stamp(settings: Settings, **overrides) -> dict:
    stamp = {
        "doc_stride": settings.doc_stride,
        "impl": 3,
        "max_answer_len": settings.max_answer_len,
        "max_seq_len": settings.max_seq_len,
        "model": f"runs/{settings.extractor_name}/final",
        "top_k": settings.top_k,
    }
    return stamp | overrides


def _write(tmp_path: Path, settings: Settings, preds: list[dict], **stamp_overrides) -> tuple:
    preds_path = tmp_path / "run_preds.jsonl"
    preds_path.write_text("".join(json.dumps(line) + "\n" for line in preds), encoding="utf-8")
    preds_path.with_suffix(".config.json").write_text(
        json.dumps(_stamp(settings, **stamp_overrides)), encoding="utf-8"
    )
    cuad_path = tmp_path / "test.json"
    contracts = [(TITLE_A, TEXT_A), (TITLE_B, TEXT_B)]
    data = [{"title": t, "paragraphs": [{"context": c}]} for t, c in contracts]
    cuad_path.write_text(json.dumps({"data": data}), encoding="utf-8")
    return preds_path, cuad_path


def _preds() -> list[dict]:
    return [
        {
            "contract_id": TITLE_A,
            "preds": {
                "Governing Law": [["laws of Delaware", 0.9], ["five years", 0.1]],
                "Expiration Date": [],
            },
        },
        {"contract_id": TITLE_B, "preds": {"Termination": [["terminate", 0.5]]}},
    ]


def test_repeated_text_points_at_first_occurrence(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    caches = import_harness_preds(preds_path, cuad_path, settings)
    candidate = caches[TITLE_A].clauses["Governing Law"][0]
    assert candidate.start == TEXT_A.find("laws of Delaware")
    assert candidate.end == candidate.start + len("laws of Delaware")


def test_caches_are_keyed_by_title(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    caches = import_harness_preds(preds_path, cuad_path, settings)
    assert list(caches) == [TITLE_A, TITLE_B]


def test_cache_records_harness_origin_and_txt_source(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    cache = import_harness_preds(preds_path, cuad_path, settings)[TITLE_A]
    assert cache.model.origin == "harness"
    assert cache.source.filename == f"{TITLE_A}.txt"
    assert cache.source.n_pages is None


def test_candidates_have_no_page(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    cache = import_harness_preds(preds_path, cuad_path, settings)[TITLE_A]
    assert all(c.page is None for cands in cache.clauses.values() for c in cands)


def test_cache_passes_read_checks_after_write(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    cache = import_harness_preds(preds_path, cuad_path, settings)[TITLE_A]
    write_cache(cache, tmp_path / "out.json")
    assert read_cache(tmp_path / "out.json", settings) == cache


def test_other_doc_stride_is_rejected(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds(), doc_stride=128)
    with pytest.raises(HarnessImportError, match="doc_stride"):
        import_harness_preds(preds_path, cuad_path, settings)


def test_other_run_tag_is_rejected(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds(), model="runs/another_run/final")
    with pytest.raises(HarnessImportError, match="name"):
        import_harness_preds(preds_path, cuad_path, settings)


def test_missing_predicted_text_is_rejected(tmp_path, settings):
    preds = [{"contract_id": TITLE_A, "preds": {"Governing Law": [["not in contract", 0.9]]}}]
    preds_path, cuad_path = _write(tmp_path, settings, preds)
    with pytest.raises(HarnessImportError, match="Governing Law"):
        import_harness_preds(preds_path, cuad_path, settings)


def test_titles_filter_returns_only_requested_contract(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    caches = import_harness_preds(preds_path, cuad_path, settings, titles=[TITLE_B])
    assert list(caches) == [TITLE_B]


def test_requested_title_without_predictions_is_rejected(tmp_path, settings):
    preds_path, cuad_path = _write(tmp_path, settings, _preds())
    with pytest.raises(HarnessImportError, match="Gamma"):
        import_harness_preds(preds_path, cuad_path, settings, titles=[TITLE_A, "Gamma"])
