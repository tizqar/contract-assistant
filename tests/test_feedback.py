import json
from datetime import datetime

from contract_assistant.feedback import (
    Feedback,
    dump_json,
    now,
    read_feedback,
    record_feedback,
    summary,
)


def verdict(liked: bool, question: str = "Who pays?", comment: str | None = None) -> Feedback:
    return Feedback(
        doc_id="sha256:abc",
        question=question,
        route_kind="clause",
        clause_types=["Cap On Liability"],
        liked=liked,
        comment=comment,
        at="2026-10-21T10:00:00+00:00",
    )


def test_record_feedback_appends_one_json_line_per_verdict(tmp_path):
    path = tmp_path / "feedback.jsonl"
    record_feedback(path, verdict(True))
    record_feedback(path, verdict(False))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["liked"] for line in lines] == [True, False]


def test_record_feedback_creates_the_directory(tmp_path):
    path = tmp_path / "deep" / "dir" / "feedback.jsonl"
    assert record_feedback(path, verdict(True)) == path
    assert path.is_file()


def test_read_feedback_returns_the_verdicts_in_order(tmp_path):
    path = tmp_path / "feedback.jsonl"
    verdicts = [verdict(True, "first"), verdict(False, "second", "wrong clause")]
    for item in verdicts:
        record_feedback(path, item)
    assert read_feedback(path) == verdicts


def test_read_feedback_of_a_missing_file_is_empty(tmp_path):
    assert read_feedback(tmp_path / "absent.jsonl") == []


def test_read_feedback_skips_blank_lines(tmp_path):
    path = tmp_path / "feedback.jsonl"
    line = verdict(True).model_dump_json()
    path.write_text(f"\n{line}\n\n   \n{line}\n", encoding="utf-8")
    assert len(read_feedback(path)) == 2


def test_now_is_iso_with_a_timezone():
    assert datetime.fromisoformat(now()).tzinfo is not None


def test_summary_counts_liked_and_disliked():
    verdicts = [verdict(True), verdict(True), verdict(False)]
    assert summary(verdicts) == {"liked": 2, "disliked": 1}


def test_summary_of_no_verdicts_is_zero():
    assert summary([]) == {"liked": 0, "disliked": 0}


def test_dump_json_is_a_json_list_of_the_verdicts():
    loaded = json.loads(dump_json([verdict(True), verdict(False)]))
    assert [item["liked"] for item in loaded] == [True, False]
