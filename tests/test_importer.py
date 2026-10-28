"""The import of the cache folder into the database: contracts, files, decisions and texts."""

import dataclasses
import json
import shutil
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from contract_assistant import audit, files
from contract_assistant.accounts import AccountError
from contract_assistant.answer import AnswerItem, Citation
from contract_assistant.config import load_settings
from contract_assistant.db.models import (
    PDF,
    TXT,
    Blob,
    Contract,
    Decision,
    Event,
    Feedback,
    Folder,
    Generated,
)
from contract_assistant.db.stores import DbRecords, FeedbackLog
from contract_assistant.feedback import FEEDBACK_FILE
from contract_assistant.feedback import Feedback as FeedbackVerdict
from contract_assistant.importer import (
    BAD_SPAN,
    DEFAULT_FOLDER,
    NO_ANALYSIS,
    NOT_FEEDBACK,
    OLD_RECORD,
    import_cache,
    render,
)
from contract_assistant.store import DOC_ID_PREFIX, read_cache, write_cache
from contract_assistant.summary import Summary, SummaryStore
from contract_assistant.term_sheet import TermSheet, TermSheetStore, TermValue
from tests.support.caches import make_cache
from tests.support.db import add_user

DEMOS = {
    "short": "short_supply_agreement",
    "medium": "medium_sponsorship_agreement",
    "long": "long_outsourcing_agreement",
}
UNKNOWN_DOC = DOC_ID_PREFIX + "f" * 64
MODEL = "fake-llm"
TAU_LOW = 0.06
TAU_HIGH = 0.91
DECIDED_AT = "2026-10-22T12:39:14+00:00"
FEEDBACK_AT = "2026-10-22T13:00:00+00:00"


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(load_settings(), cache_dir=tmp_path / "cache")


@pytest.fixture
def ana(database):
    return add_user(database, "ana@example.com", name="Ana")


@pytest.fixture
def demos(settings):
    """The three demo analyses by short name."""
    return {
        key: read_cache(settings.demo_cache_dir / f"{name}.json", settings)
        for key, name in DEMOS.items()
    }


def stem(doc_id: str) -> str:
    return doc_id.removeprefix(DOC_ID_PREFIX)


def make_summary(doc_id: str, language: str = "en", text: str = "A summary.") -> Summary:
    citation = Citation(quote="the laws of England", page=None, conf=1.0, start=1, end=2)
    item = AnswerItem(
        clause_type="Governing Law", status="found", best_conf=1.0, citation=citation
    )
    return Summary(
        doc_id=doc_id,
        model=MODEL,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        language=language,
        items=[item],
        text=text,
        latency_ms=10.0,
    )


def make_sheet(doc_id: str) -> TermSheet:
    value = TermValue(value="England", clause_type="Governing Law", page=None, start=1, end=8)
    return TermSheet(
        doc_id=doc_id,
        model=MODEL,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        latency_ms=1.0,
        items=[],
        terms={"governing_law": [value]},
        rejected={},
    )


def decision(clause_type: str, verdict: str, start: int, end: int) -> dict:
    return {
        "clause_type": clause_type,
        "verdict": verdict,
        "start": start,
        "end": end,
        "at": DECIDED_AT,
    }


def verdict(doc_id: str, question: str, liked: bool) -> str:
    line = FeedbackVerdict(
        doc_id=doc_id,
        question=question,
        route_kind="clauses",
        clause_types=["Parties"],
        liked=liked,
        at=FEEDBACK_AT,
    )
    return line.model_dump_json()


@pytest.fixture
def cache_dir(settings, demos):
    """A cache folder as the command line left it: analyses, a PDF, decisions, saved texts."""
    root = settings.cache_dir
    for cache in demos.values():
        write_cache(cache, root / f"{stem(cache.doc_id)}.json")
    short = demos["short"]
    documents = root / "documents"
    documents.mkdir()
    shutil.copyfile(
        settings.demo_cache_dir.parent / "short_supply_agreement.pdf",
        documents / f"{stem(short.doc_id)}.pdf",
    )
    parties = short.clauses["Parties"][0]
    exclusivity = short.clauses["Exclusivity"][0]
    decisions = [
        decision("Parties", "confirmed", parties.start, parties.end),
        decision("Exclusivity", "rejected", exclusivity.start, exclusivity.end),
        # Past the end of the text: left out.
        decision("Insurance", "confirmed", 10, len(short.text) + 5),
    ]
    (root / "decisions").mkdir()
    (root / "decisions" / f"{stem(short.doc_id)}.json").write_text(json.dumps(decisions))
    SummaryStore(root / "summaries").save(make_summary(short.doc_id))
    SummaryStore(root / "summaries", "sr").save(make_summary(short.doc_id, "sr", "Sažetak."))
    TermSheetStore(root / "term_sheets").save(make_sheet(short.doc_id))
    TermSheetStore(root / "term_sheets").save(make_sheet(UNKNOWN_DOC))
    lines = [
        verdict(short.doc_id, "Who are the parties?", liked=True),
        verdict(UNKNOWN_DOC, "Which law?", liked=False),
        "not json",
    ]
    (root / FEEDBACK_FILE).write_text("\n".join(lines) + "\n")
    return root


def snapshot(directory) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def count(database, model, *where) -> int:
    with database.session() as session:
        return session.scalar(select(func.count()).select_from(model).where(*where))


def contracts_by_doc(database) -> dict[str, Contract]:
    with database.session() as session:
        return {row.doc_id: row for row in session.scalars(select(Contract))}


# What an import brings in


def test_each_valid_analysis_becomes_a_contract_of_the_owner_in_one_folder(
    database, settings, ana, demos, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    names = sorted(f"{name}.txt" for name in DEMOS.values())
    assert sorted(report.imported) == names
    assert report.folder_created and report.reused == [] and report.skipped == []
    with database.session() as session:
        top = files.listing(session, ana)
        (folder,) = top.folders
        inside = files.listing(session, ana, folder.id)
    assert folder.name == DEFAULT_FOLDER
    assert sorted(contract.name for contract in inside.contracts) == names
    assert {contract.doc_id for contract in inside.contracts} == {
        cache.doc_id for cache in demos.values()
    }
    assert all(contract.created_by == ana.id for contract in inside.contracts)


def test_a_pdf_copy_makes_a_pdf_contract_and_the_others_keep_their_text(
    database, settings, ana, demos, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    contracts = contracts_by_doc(database)
    short, medium = contracts[demos["short"].doc_id], contracts[demos["medium"].doc_id]
    with database.session() as session:
        pdf = session.get(Blob, short.file_sha256)
        text = session.get(Blob, medium.file_sha256)
    pdf_copy = cache_dir / "documents" / f"{stem(demos['short'].doc_id)}.pdf"
    assert (short.kind, bytes(pdf.data), pdf.media_type) == (
        PDF,
        pdf_copy.read_bytes(),
        "application/pdf",
    )
    assert (medium.kind, bytes(text.data)) == (TXT, demos["medium"].text.encode("utf-8"))
    assert report.pdfs == 1


def test_decisions_come_in_with_their_time_and_no_one_as_who_decided(
    database, settings, ana, demos, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    contract = contracts_by_doc(database)[demos["short"].doc_id]
    with database.session() as session:
        decided = files.decisions_of(session, contract.id)
    assert sorted(decided) == ["Exclusivity", "Parties"]
    assert decided["Parties"].decision.verdict == "confirmed"
    assert decided["Exclusivity"].decision.verdict == "rejected"
    assert decided["Parties"].by_id is None and decided["Parties"].by_name is None
    with database.session() as session:
        row = session.get(Decision, (contract.id, "Parties"))
    assert row.decided_at == datetime(2026, 10, 22, 12, 39, 14, tzinfo=UTC)
    assert report.decisions == 2
    reason = BAD_SPAN.format(clause="Insurance")
    assert any(item.reason == reason for item in report.left_out)


def test_saved_texts_move_to_the_contract_id_in_both_summary_languages(
    database, settings, ana, demos, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    contract_id = contracts_by_doc(database)[demos["short"].doc_id].id
    records = DbRecords(database)
    keys = (contract_id, MODEL, TAU_LOW, TAU_HIGH)
    english = SummaryStore(records).load(*keys)
    serbian = SummaryStore(records, "sr").load(*keys)
    sheet = TermSheetStore(records).load(*keys)
    assert (english.doc_id, english.text) == (contract_id, "A summary.")
    assert (serbian.doc_id, serbian.text) == (contract_id, "Sažetak.")
    assert sheet.terms["governing_law"][0].value == "England"
    assert report.generated == {"summary": 1, "summary-sr": 1, "term-sheet": 1}
    with database.session() as session:
        doc_ids = set(session.scalars(select(Generated.doc_id)))
    assert doc_ids == {contract_id}


def test_feedback_lines_become_rows_of_their_contract(
    database, settings, ana, demos, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    contract_id = contracts_by_doc(database)[demos["short"].doc_id].id
    (verdict,) = FeedbackLog(database).read(contract_id)
    assert (verdict.question, verdict.liked) == ("Who are the parties?", True)
    assert report.feedback == 1
    left_out = {item.path: item.reason for item in report.left_out}
    assert left_out[f"{FEEDBACK_FILE}, line 2"] == NO_ANALYSIS
    assert left_out[f"{FEEDBACK_FILE}, line 3"] == NOT_FEEDBACK


def test_what_belongs_to_no_analysis_is_left_out_and_named(
    database, settings, ana, cache_dir
):
    report = import_cache(database, settings, "ana@example.com")
    sheet = TermSheetStore(cache_dir / "term_sheets").path_of(make_sheet(UNKNOWN_DOC))
    assert {"path": f"term_sheets/{sheet.name}", "reason": NO_ANALYSIS} in [
        dataclasses.asdict(item) for item in report.left_out
    ]


def test_each_new_contract_gets_one_import_event_without_an_actor(
    database, settings, ana, cache_dir
):
    import_cache(database, settings, "ana@example.com")
    with database.session() as session:
        events = audit.events(session, audit.EventFilter(actions=[audit.CONTRACT_IMPORTED]))
    assert len(events) == 3
    assert {event.actor_id for event in events} == {None}
    with database.session() as session:
        everything = audit.events(session)
    assert {event.actor_id for event in everything} == {None}, "nothing is put on the owner"
    assert audit.CONTRACT_ADDED not in {event.action for event in everything}
    assert {event.contract_id for event in events} == {
        row.id for row in contracts_by_doc(database).values()
    }
    sentences = {audit.describe(event).split(" was ")[1] for event in events}
    assert sentences == {"imported from the cache folder"}


# Running it again


def test_a_second_run_imports_nothing_new(database, settings, ana, cache_dir):
    import_cache(database, settings, "ana@example.com")
    tables = (Contract, Folder, Decision, Generated, Feedback, Event, Blob)
    before = {model: count(database, model) for model in tables}
    report = import_cache(database, settings, "ana@example.com")
    assert report.imported == [] and not report.folder_created
    assert len(report.reused) == 3
    assert (report.decisions, sum(report.generated.values()), report.feedback) == (0, 0, 0)
    assert not report.changed
    assert {model: count(database, model) for model in tables} == before
    assert "nothing new to import" in render(report)


def test_a_second_run_keeps_what_the_app_wrote_since(database, settings, ana, demos, cache_dir):
    import_cache(database, settings, "ana@example.com")
    contract_id = contracts_by_doc(database)[demos["short"].doc_id].id
    store = SummaryStore(DbRecords(database))
    store.save(make_summary(contract_id, text="Written again by the app."))
    with database.session() as session:
        files.undo_decision(session, ana, contract_id, "Parties")
    import_cache(database, settings, "ana@example.com")
    assert store.load(contract_id, MODEL, TAU_LOW, TAU_HIGH).text == "Written again by the app."
    with database.session() as session:
        assert "Parties" not in files.decisions_of(session, contract_id)


def test_a_folder_of_that_name_at_the_top_is_used(database, settings, ana, cache_dir):
    with database.session() as session:
        folder = files.create_folder(session, ana, None, "Old work")
    report = import_cache(database, settings, "ana@example.com", folder_name="old WORK")
    assert not report.folder_created
    assert {row.folder_id for row in contracts_by_doc(database).values()} == {folder.id}


# What is skipped


def test_an_analysis_of_other_settings_or_damaged_is_skipped_with_its_reason(
    database, settings, ana, cache_dir
):
    other = make_cache("A contract read with other settings.", settings)
    other.model.n_best_size = settings.n_best_size + 1
    write_cache(other, cache_dir / f"{stem(other.doc_id)}.json")
    (cache_dir / "broken.json").write_text("{")
    report = import_cache(database, settings, "ana@example.com")
    reasons = {item.path: item.reason for item in report.skipped}
    assert "other settings: n_best_size" in reasons[f"{stem(other.doc_id)}.json"]
    assert "not a valid cache" in reasons["broken.json"]
    assert len(report.imported) == 3
    assert other.doc_id not in contracts_by_doc(database)


def test_a_saved_text_of_an_older_version_is_left_out(
    database, settings, ana, demos, cache_dir
):
    summaries = cache_dir / "summaries"
    path = SummaryStore(summaries).path_of(make_summary(demos["short"].doc_id))
    old = summaries / f"{stem(demos['short'].doc_id)}-0123456789abcdef.json"
    shutil.copyfile(path, old)
    report = import_cache(database, settings, "ana@example.com")
    assert {"path": f"summaries/{old.name}", "reason": OLD_RECORD} in [
        dataclasses.asdict(item) for item in report.left_out
    ]


def test_an_unknown_owner_is_refused(database, settings, cache_dir):
    with pytest.raises(AccountError):
        import_cache(database, settings, "nobody@example.com")


def test_a_missing_cache_folder_is_refused(database, settings, ana, tmp_path):
    with pytest.raises(FileNotFoundError):
        import_cache(database, settings, "ana@example.com", cache_dir=tmp_path / "nowhere")


# The cache folder is only read


def test_the_cache_folder_is_unchanged(database, settings, ana, cache_dir):
    before = snapshot(cache_dir)
    import_cache(database, settings, "ana@example.com")
    import_cache(database, settings, "ana@example.com")
    assert snapshot(cache_dir) == before


@pytest.mark.pg
def test_an_import_on_postgres(pg_database, settings, demos, cache_dir):
    add_user(pg_database, "ana@example.com", name="Ana")
    first = import_cache(pg_database, settings, "ana@example.com")
    second = import_cache(pg_database, settings, "ana@example.com")
    assert len(first.imported) == 3 and first.decisions == 2 and first.feedback == 1
    assert second.imported == [] and len(second.reused) == 3 and not second.changed
    assert count(pg_database, Contract) == 3
