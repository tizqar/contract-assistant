import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from contract_assistant.config import Settings
from contract_assistant.store import (
    CacheIntegrityError,
    CacheMismatchError,
    Store,
    doc_id_of,
    read_cache,
    write_cache,
)
from tests.support.caches import make_cache

TEXT = "This Agreement is governed by the laws of Delaware. Term is five years."
CLAUSES = {"Governing Law": [("laws of Delaware", 0.9)]}


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "cache", tmp_path / "demo")


def test_round_trip_returns_equal_cache(store, settings):
    cache = make_cache(TEXT, settings, CLAUSES)
    store.save(cache)
    assert store.load(cache.doc_id, settings) == cache


def test_save_names_file_by_hash(store, settings):
    cache = make_cache(TEXT, settings)
    path = store.save(cache)
    assert path == store.cache_dir / f"{doc_id_of(TEXT).removeprefix('sha256:')}.json"


def test_load_rejects_other_name(store, settings):
    store.save(make_cache(TEXT, replace(settings, extractor_name="other"), CLAUSES))
    with pytest.raises(CacheMismatchError, match="name"):
        store.load(doc_id_of(TEXT), settings)


def test_load_rejects_other_doc_stride(store, settings):
    store.save(make_cache(TEXT, replace(settings, doc_stride=128), CLAUSES))
    with pytest.raises(CacheMismatchError, match="doc_stride"):
        store.load(doc_id_of(TEXT), settings)


def test_load_rejects_changed_text(store, settings):
    path = store.save(make_cache(TEXT, settings))
    data = json.loads(path.read_text())
    data["text"] = data["text"].replace("Delaware", "Delaware!", 1)
    path.write_text(json.dumps(data))
    with pytest.raises(CacheIntegrityError, match="doc_id"):
        store.load(doc_id_of(TEXT), settings)


def test_load_rejects_offsets_that_do_not_give_the_text(store, settings):
    cache = make_cache(TEXT, settings, CLAUSES)
    cache.clauses["Governing Law"][0].start += 1
    path = store.save(cache)
    with pytest.raises(CacheIntegrityError, match="Governing Law"):
        store.load_path(path, settings)


def test_load_finds_cache_in_demo_dir(store, settings):
    cache = make_cache(TEXT, settings, CLAUSES)
    write_cache(cache, store.demo_dir / "sample.json")
    assert store.load(cache.doc_id, settings) == cache


def test_find_returns_demo_path_and_none_for_unknown_id(store, settings):
    cache = make_cache(TEXT, settings, CLAUSES)
    write_cache(cache, store.demo_dir / "sample.json")
    assert store.find(cache.doc_id) == store.demo_dir / "sample.json"
    assert store.find("sha256:" + "0" * 64) is None


def test_missing_id_returns_none(store, settings):
    assert store.load("sha256:" + "0" * 64, settings) is None


def test_save_never_writes_to_demo_dir(store, settings):
    demo_file = store.demo_dir / "sample.json"
    write_cache(make_cache("Demo contract text.", settings), demo_file)
    before = (demo_file.read_bytes(), os.stat(demo_file).st_mtime_ns)
    saved = store.save(make_cache(TEXT, settings))
    assert [p.name for p in store.demo_dir.iterdir()] == ["sample.json"]
    assert (demo_file.read_bytes(), os.stat(demo_file).st_mtime_ns) == before
    assert saved.parent == store.cache_dir


def test_list_covers_both_directories(store, settings):
    store.save(make_cache(TEXT, settings))
    write_cache(make_cache("Demo contract text.", settings), store.demo_dir / "sample.json")
    assert [e.doc_id for e in store.list()] == [
        doc_id_of(TEXT),
        doc_id_of("Demo contract text."),
    ]


def test_list_of_missing_directories_is_empty(store):
    assert store.list() == []


def test_write_cache_leaves_no_temp_file(tmp_path, settings):
    write_cache(make_cache(TEXT, settings), tmp_path / "out" / "cache.json")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["cache.json"]


def test_load_rejects_a_cache_stored_under_another_id(store, settings):
    cache = make_cache(TEXT, settings, CLAUSES)
    other = "sha256:" + "1" * 64
    write_cache(cache, store.cache_dir / f"{'1' * 64}.json")
    with pytest.raises(CacheIntegrityError):
        store.load(other, settings)


def test_scan_skips_invalid_files_and_warns(store, settings, caplog):
    cache = make_cache(TEXT, settings, CLAUSES)
    write_cache(cache, store.demo_dir / "good.json")
    (store.demo_dir / "bad.json").write_text("{not json", encoding="utf-8")
    assert [entry.path.name for entry in store.list()] == ["good.json"]
    assert store.find(cache.doc_id) == store.demo_dir / "good.json"
    assert "bad.json" in caplog.text


def test_read_cache_turns_invalid_json_into_integrity_error(tmp_path, settings):
    path = tmp_path / "broken.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CacheIntegrityError, match="broken.json"):
        read_cache(path, settings)
