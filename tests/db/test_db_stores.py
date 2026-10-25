"""The database stores of analyses, files and generated records, on SQLite and on Postgres."""

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select, update

from contract_assistant.answer import AnswerItem, Citation
from contract_assistant.config import load_settings
from contract_assistant.db.models import Analysis, Blob, Generated
from contract_assistant.db.stores import AnalysisStore, BlobStore, DbRecords
from contract_assistant.generation import FileRecords
from contract_assistant.store import (
    CachedCandidate,
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    doc_id_of,
    read_cache,
)
from contract_assistant.summary import PROMPT_VERSION, Summary, SummaryStore
from contract_assistant.term_sheet import TermSheet, TermSheetStore, TermValue

SHORT_DEMO = "short_supply_agreement.json"
MEDIUM_DEMO = "medium_sponsorship_agreement.json"
MODEL = "fake-llm"
TAU_LOW = 0.1
TAU_HIGH = 0.5
DOC_A = "sha256:aaaa"
DOC_B = "contract-b"
PDF_BYTES = b"%PDF-1.4 a small file\n%%EOF\n"


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def demo(settings) -> DocumentCache:
    return read_cache(settings.demo_cache_dir / SHORT_DEMO, settings)


def count(database, model) -> int:
    with database.session() as session:
        return session.scalar(select(func.count()).select_from(model))


def make_summary(doc_id=DOC_A, **changes) -> Summary:
    citation = Citation(quote="the laws of Ohio", page=2, conf=0.912345678901, start=10, end=26)
    item = AnswerItem(
        clause_type="Governing Law", status="found", best_conf=0.912345678901, citation=citation
    )
    fields = {
        "doc_id": doc_id,
        "model": MODEL,
        "tau_low": TAU_LOW,
        "tau_high": TAU_HIGH,
        "items": [item],
        "text": 'The contract follows "the laws of Ohio".',
        "latency_ms": 1234.5,
    }
    return Summary(**{**fields, **changes})


def make_sheet(doc_id=DOC_A) -> TermSheet:
    value = TermValue(value="Ohio", clause_type="Governing Law", page=None, start=22, end=26)
    return TermSheet(
        doc_id=doc_id,
        model=MODEL,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        latency_ms=1.0,
        items=[],
        terms={"parties": [], "governing_law": [value]},
        rejected={"term": ["forever"]},
    )


# Analyses


def test_a_saved_analysis_loads_back_equal_with_its_clause_order(database, settings, demo):
    AnalysisStore(database, settings).save(demo)
    loaded = AnalysisStore(database, settings).load(demo.doc_id)
    assert loaded == demo
    assert list(loaded.clauses) == list(demo.clauses)


def test_save_replaces_the_analysis_of_the_same_text(database, settings, demo):
    store = AnalysisStore(database, settings)
    store.save(demo)
    store.save(demo.model_copy(update={"timing": {"extract_s": 2.5}}))
    assert count(database, Analysis) == 1
    assert AnalysisStore(database, settings).load(demo.doc_id).timing == {"extract_s": 2.5}


def test_the_row_names_the_extractor(database, settings, demo):
    AnalysisStore(database, settings).save(demo)
    with database.session() as session:
        assert session.scalar(select(Analysis.extractor)) == demo.model.name


def test_a_demo_analysis_is_copied_into_the_database_once(database, settings, demo, tmp_path):
    first = AnalysisStore(database, settings, settings.demo_cache_dir, memory=0)
    assert first.load(demo.doc_id) == demo
    assert first.load(demo.doc_id) == demo
    assert count(database, Analysis) == 1
    without_demos = AnalysisStore(database, settings, tmp_path / "no-demos")
    assert without_demos.load(demo.doc_id) == demo
    assert count(database, Analysis) == 1


def test_an_analysis_written_with_other_decode_settings_is_refused(database, settings, demo):
    AnalysisStore(database, settings).save(demo)
    other = dataclasses.replace(settings, top_k=settings.top_k + 1)
    with pytest.raises(CacheMismatchError, match="top_k"):
        AnalysisStore(database, other).load(demo.doc_id)


def test_a_demo_of_other_decode_settings_is_refused_and_not_copied(database, settings, demo):
    other = dataclasses.replace(settings, top_k=settings.top_k + 1)
    with pytest.raises(CacheMismatchError):
        AnalysisStore(database, other, settings.demo_cache_dir).load(demo.doc_id)
    assert count(database, Analysis) == 0


def test_a_tampered_text_is_refused(database, settings, demo):
    tampered = demo.model_copy(update={"text": demo.text.replace("a", "b", 1)})
    AnalysisStore(database, settings).save(tampered)
    with pytest.raises(CacheIntegrityError, match="doc_id"):
        AnalysisStore(database, settings).load(demo.doc_id)


def test_moved_offsets_are_refused(database, settings, demo):
    name, candidates = next((name, c) for name, c in demo.clauses.items() if c)
    moved = candidates[0].model_copy(update={"start": candidates[0].start + 1})
    clauses = {**demo.clauses, name: [moved, *candidates[1:]]}
    AnalysisStore(database, settings).save(demo.model_copy(update={"clauses": clauses}))
    with pytest.raises(CacheIntegrityError, match="offsets"):
        AnalysisStore(database, settings).load(demo.doc_id)


def test_a_row_that_is_not_an_analysis_is_refused(database, settings):
    with database.session() as session:
        session.add(Analysis(doc_id=DOC_A, data={"doc_id": DOC_A}, extractor="x"))
    with pytest.raises(CacheIntegrityError, match="not a valid analysis"):
        AnalysisStore(database, settings).load(DOC_A)


def test_an_analysis_saved_under_another_doc_id_is_refused(database, settings, demo):
    AnalysisStore(database, settings).save(demo)
    with database.session() as session:
        session.execute(update(Analysis).values(doc_id=DOC_A))
    with pytest.raises(CacheIntegrityError, match="belongs to"):
        AnalysisStore(database, settings).load(DOC_A)


def test_load_of_an_unknown_doc_id_is_none(database, settings):
    assert AnalysisStore(database, settings, load_settings().demo_cache_dir).load(DOC_A) is None


def test_load_keeps_the_analysis_in_memory_until_it_is_saved_again(database, settings, demo):
    store = AnalysisStore(database, settings)
    store.save(demo)
    first = store.load(demo.doc_id)
    assert store.load(demo.doc_id) is first
    store.save(demo.model_copy(update={"timing": {"extract_s": 9.0}}))
    assert store.load(demo.doc_id).timing == {"extract_s": 9.0}


def test_memory_forgets_the_analysis_used_least_recently(database, settings):
    store = AnalysisStore(database, settings, settings.demo_cache_dir, memory=1)
    short = read_cache(settings.demo_cache_dir / SHORT_DEMO, settings).doc_id
    medium = read_cache(settings.demo_cache_dir / MEDIUM_DEMO, settings).doc_id
    first = store.load(short)
    store.load(medium)
    assert store.load(short) is not first
    assert store.load(short) == first


def test_exists_finds_a_demo_without_copying_it(database, settings, demo):
    store = AnalysisStore(database, settings, settings.demo_cache_dir)
    assert store.exists(demo.doc_id)
    assert not store.exists(DOC_A)
    assert count(database, Analysis) == 0


def test_ensure_copies_a_demo_analysis_and_tells_whether_there_is_one(database, settings, demo):
    store = AnalysisStore(database, settings, settings.demo_cache_dir)
    assert store.ensure(demo.doc_id)
    assert store.ensure(demo.doc_id)
    assert not store.ensure(DOC_A)
    assert count(database, Analysis) == 1


def test_loads_from_many_threads_copy_a_demo_once(database, settings, demo):
    store = AnalysisStore(database, settings, settings.demo_cache_dir)
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(lambda _: store.load(demo.doc_id), range(16)))
    assert all(cache == demo for cache in loaded)
    assert count(database, Analysis) == 1


def test_an_invalid_demo_file_is_skipped_with_a_warning(database, settings, tmp_path, caplog):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="contract_assistant.db.stores"):
        assert AnalysisStore(database, settings, tmp_path).load(DOC_A) is None
    assert "broken.json" in caplog.text


# Files


def test_put_keeps_the_bytes_once(database, tmp_path):
    blobs = BlobStore(database, tmp_path)
    sha256 = blobs.put(PDF_BYTES, "application/pdf")
    assert blobs.put(PDF_BYTES, "application/pdf") == sha256
    assert count(database, Blob) == 1
    assert len(sha256) == 64
    assert blobs.get(sha256) == PDF_BYTES
    with database.session() as session:
        assert session.scalar(select(Blob.size)) == len(PDF_BYTES)


def test_get_of_an_unknown_or_malformed_sha256_is_none(database, tmp_path):
    blobs = BlobStore(database, tmp_path)
    assert blobs.get("0" * 64) is None
    assert blobs.get("../secret") is None


def test_local_path_writes_the_file_once(database, tmp_path, monkeypatch):
    blobs = BlobStore(database, tmp_path / "files")
    sha256 = blobs.put(PDF_BYTES, "application/pdf")
    path = blobs.local_path(sha256, ".pdf")
    assert path == tmp_path / "files" / f"{sha256}.pdf"
    assert path.read_bytes() == PDF_BYTES
    monkeypatch.setattr(blobs, "get", lambda _: pytest.fail("read from the database again"))
    assert blobs.local_path(sha256, ".pdf") == path
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_local_path_of_an_unknown_sha256_is_none_and_writes_nothing(database, tmp_path):
    blobs = BlobStore(database, tmp_path / "files")
    assert blobs.local_path("0" * 64, ".pdf") is None
    assert blobs.local_path("../../etc/passwd", "") is None
    assert not (tmp_path / "files").exists()


def test_local_path_refuses_a_suffix_that_is_not_one(database, tmp_path):
    blobs = BlobStore(database, tmp_path)
    with pytest.raises(ValueError, match="suffix"):
        blobs.local_path("0" * 64, "/../x")


# Generated records


def stores(database, tmp_path, make=SummaryStore):
    """The same kind of store on files and in the database."""
    return make(tmp_path / "files"), make(DbRecords(database))


def test_a_summary_in_the_database_loads_back_like_the_file_one(database, tmp_path):
    on_files, in_database = stores(database, tmp_path)
    summary = make_summary()
    on_files.save(summary)
    assert in_database.save(summary) is None
    loaded = in_database.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH)
    assert loaded == summary == on_files.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH)


def test_the_key_in_the_database_is_the_file_stem(database, tmp_path):
    on_files, in_database = stores(database, tmp_path)
    summary = make_summary()
    in_database.save(summary)
    with database.session() as session:
        row = session.scalars(select(Generated)).one()
    assert (row.kind, row.key, row.doc_id) == ("summary", on_files.path_of(summary).stem, DOC_A)
    assert row.key == in_database.key_of(DOC_A, MODEL, TAU_LOW, TAU_HIGH)


def test_a_term_sheet_in_the_database_loads_back_like_the_file_one(database, tmp_path):
    on_files, in_database = stores(database, tmp_path, TermSheetStore)
    sheet = make_sheet()
    on_files.save(sheet)
    in_database.save(sheet)
    loaded = in_database.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH)
    assert loaded == sheet == on_files.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH)


def test_save_replaces_the_record_of_the_same_keys(database):
    store = SummaryStore(DbRecords(database))
    store.save(make_summary())
    store.save(make_summary(text="Another text."))
    assert count(database, Generated) == 1
    assert store.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH).text == "Another text."


@pytest.mark.parametrize(
    ("model", "tau_low", "tau_high"),
    [("other", TAU_LOW, TAU_HIGH), (MODEL, 0.2, TAU_HIGH), (MODEL, TAU_LOW, 0.9)],
)
def test_records_of_other_keys_are_not_loaded(database, tmp_path, model, tau_low, tau_high):
    for store in stores(database, tmp_path):
        store.save(make_summary())
        assert store.load(DOC_A, model, tau_low, tau_high) is None
        assert store.load(DOC_B, MODEL, TAU_LOW, TAU_HIGH) is None


def test_a_record_of_another_prompt_version_is_not_loaded(database, tmp_path):
    for store in stores(database, tmp_path):
        store.save(make_summary(prompt_version=PROMPT_VERSION + 1))
        assert store.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None


def test_the_languages_of_summaries_are_kept_apart(database):
    records = DbRecords(database)
    english, serbian = SummaryStore(records), SummaryStore(records, "sr")
    english.save(make_summary())
    assert serbian.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None
    serbian.save(make_summary(language="sr", text="Ugovor prati zakone Ohaja."))
    assert english.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH).language == "en"
    assert serbian.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH).language == "sr"


def test_a_record_saved_under_the_key_of_others_is_not_loaded(database):
    store = SummaryStore(DbRecords(database))
    store.save(make_summary())
    with database.session() as session:
        session.execute(update(Generated).values(data=make_summary(model="other").model_dump()))
    assert store.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None


def test_an_invalid_record_is_none_with_a_warning_naming_kind_and_key(database, caplog):
    store = SummaryStore(DbRecords(database))
    store.save(make_summary())
    with database.session() as session:
        session.execute(update(Generated).values(data={"broken": True}))
    with caplog.at_level(logging.WARNING, logger="contract_assistant.generation"):
        assert store.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None
    assert "summary" in caplog.text
    assert store.key_of(DOC_A, MODEL, TAU_LOW, TAU_HIGH) in caplog.text


def test_delete_doc_removes_only_the_records_of_that_doc(database):
    records = DbRecords(database)
    summaries, sheets = SummaryStore(records), TermSheetStore(records)
    summaries.save(make_summary())
    sheets.save(make_sheet())
    summaries.save(make_summary(DOC_B))
    assert records.delete_doc(DOC_A) == 2
    assert summaries.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None
    assert sheets.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) is None
    assert summaries.load(DOC_B, MODEL, TAU_LOW, TAU_HIGH) == make_summary(DOC_B)
    assert records.delete_doc(DOC_A) == 0


def test_delete_doc_on_files_removes_only_the_files_of_that_doc(tmp_path):
    records = FileRecords(tmp_path / "summaries")
    store = SummaryStore(records)
    store.save(make_summary())
    store.save(make_summary(model="other"))
    kept = store.save(make_summary("sha256:aaaab"))
    assert records.delete_doc(DOC_A) == 2
    assert [path.name for path in records.directory.iterdir()] == [kept.name]
    assert FileRecords(tmp_path / "absent").delete_doc(DOC_A) == 0


def test_a_store_in_the_database_has_no_files(database):
    store = SummaryStore(DbRecords(database))
    with pytest.raises(TypeError, match="not kept in files"):
        store.path_of(make_summary())
    with pytest.raises(TypeError):
        _ = store.directory


# Postgres keeps analyses and records as JSONB: other key order, its own number format.


@pytest.mark.pg
def test_postgres_gives_back_analyses_that_pass_the_checks(pg_database, settings):
    copying = AnalysisStore(pg_database, settings, settings.demo_cache_dir, memory=0)
    for path in sorted(settings.demo_cache_dir.glob("*.json")):
        demo = read_cache(path, settings)
        assert copying.ensure(demo.doc_id)
        loaded = AnalysisStore(pg_database, settings).load(demo.doc_id)
        assert loaded == demo
        assert list(loaded.clauses) == list(demo.clauses)
    saved = demo.model_copy(update={"timing": {"extract_s": 0.1 + 0.2}})
    AnalysisStore(pg_database, settings).save(saved)
    assert AnalysisStore(pg_database, settings).load(demo.doc_id).timing == saved.timing


@pytest.mark.pg
def test_postgres_keeps_records_and_files_exactly(pg_database, tmp_path):
    records = DbRecords(pg_database)
    summaries, sheets = SummaryStore(records), TermSheetStore(records)
    summary = make_summary(tau_low=0.1 + 0.2, latency_ms=1e-7)
    summaries.save(summary)
    sheets.save(make_sheet())
    assert summaries.load(DOC_A, MODEL, 0.1 + 0.2, TAU_HIGH) == summary
    assert sheets.load(DOC_A, MODEL, TAU_LOW, TAU_HIGH) == make_sheet()
    assert records.delete_doc(DOC_A) == 2
    blobs = BlobStore(pg_database, tmp_path)
    data = bytes(range(256)) * 3
    sha256 = blobs.put(data, "application/pdf")
    assert blobs.put(data, "application/pdf") == sha256
    assert blobs.local_path(sha256, ".pdf").read_bytes() == data


@pytest.mark.pg
def test_postgres_keeps_an_analysis_whose_text_holds_a_nul(pg_database, settings, demo):
    text = "Section 1.\x00 This Agreement is governed by the laws of Ohio."
    start = text.index("the laws")
    end = len(text) - 1
    candidate = CachedCandidate(text=text[start:end], start=start, end=end, page=None, conf=0.9)
    cache = demo.model_copy(
        update={
            "doc_id": doc_id_of(text),
            "text": text,
            "page_starts": [],
            "clauses": {"Governing Law": [candidate], "Parties": []},
        }
    )
    AnalysisStore(pg_database, settings).save(cache)
    loaded = AnalysisStore(pg_database, settings, memory=0).load(cache.doc_id)
    assert loaded == cache
    assert list(loaded.clauses) == ["Governing Law", "Parties"]


@pytest.mark.pg
def test_postgres_keeps_an_event_whose_quote_holds_a_nul(pg_database):
    from contract_assistant import audit
    from tests.support.db import add_user

    actor = add_user(pg_database, "ana@example.com")
    with pg_database.session() as session:
        audit.record(
            session, audit.PASSWORD_CHANGED, actor_id=actor.id, details={"quote": "a\x00b"}
        )
    with pg_database.session() as session:
        assert audit.events(session)[0].details == {"quote": "ab"}
