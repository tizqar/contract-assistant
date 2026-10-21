import logging
import threading

import pytest

from contract_assistant.background import precompute, precompute_all, run_in_background, warm
from contract_assistant.config import Settings
from contract_assistant.router.clients import LLMError
from contract_assistant.store import Store
from contract_assistant.summary import SummaryStore
from contract_assistant.term_sheet import TermSheetStore
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeLLMClient

LAW = "This Agreement is governed by the laws of Delaware."
PARTIES = "The parties are Acme Corp and Beta LLC."
TEXT = f"{LAW} {PARTIES}"
OTHER_LAW = "This Agreement is governed by the laws of Ohio."
OTHER_TEXT = f"{OTHER_LAW} {PARTIES}"
TYPES = ["Governing Law", "Parties", "Term"]
CLAUSES = {"Governing Law": [(LAW, 0.99)], "Parties": [(PARTIES, 0.99)]}
OTHER_CLAUSES = {"Governing Law": [(OTHER_LAW, 0.99)], "Parties": [(PARTIES, 0.99)]}
TAU_LOW = 0.06
TAU_HIGH = 0.91
MISQUOTE = 'The contract says "words that are nowhere in it".'
SHEET_REPLY = '{"governing_law": "laws of Delaware"}'
ECHO_CALLS_PER_DOCUMENT = 2


@pytest.fixture
def settings() -> Settings:
    return Settings(tau_low=TAU_LOW, tau_high=TAU_HIGH)


@pytest.fixture
def cache(settings):
    return make_cache(TEXT, settings, CLAUSES)


@pytest.fixture
def summaries(tmp_path) -> SummaryStore:
    return SummaryStore(tmp_path / "s")


@pytest.fixture
def term_sheets(tmp_path) -> TermSheetStore:
    return TermSheetStore(tmp_path / "t")


@pytest.fixture
def store(tmp_path) -> Store:
    cache_dir, demo_dir = tmp_path / "cache", tmp_path / "demo"
    cache_dir.mkdir()
    demo_dir.mkdir()
    return Store(cache_dir, demo_dir)


class FailsOn(EchoTextClient):
    """An echo client that raises LLMError for a request that mentions a marker."""

    def __init__(self, marker: str):
        super().__init__()
        self.marker = marker

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        if self.marker in user:
            raise LLMError("model down")
        return super().complete_text(system, user, max_tokens)


def keys(client, cache):
    return (cache.doc_id, client.model, TAU_LOW, TAU_HIGH)


def run_precompute(cache, client, summaries, term_sheets, tau_high=TAU_HIGH):
    precompute(cache, TYPES, client, summaries, term_sheets, TAU_LOW, tau_high)


def test_precompute_saves_the_summary_and_the_term_sheet_when_both_are_missing(
    cache, summaries, term_sheets
):
    client = EchoTextClient()
    run_precompute(cache, client, summaries, term_sheets)
    assert len(list(summaries.directory.glob("*.json"))) == 1
    assert len(list(term_sheets.directory.glob("*.json"))) == 1
    assert summaries.load(*keys(client, cache)).text
    assert term_sheets.load(*keys(client, cache)).reason is None


def test_precompute_makes_no_model_call_when_both_exist(cache, summaries, term_sheets):
    run_precompute(cache, EchoTextClient(), summaries, term_sheets)
    again = EchoTextClient()
    run_precompute(cache, again, summaries, term_sheets)
    assert again.text_calls == []


def test_precompute_makes_only_the_missing_term_sheet(cache, summaries, term_sheets):
    client = EchoTextClient()
    run_precompute(cache, client, summaries, term_sheets)
    term_sheets.path_of(term_sheets.load(*keys(client, cache))).unlink()
    again = EchoTextClient()
    run_precompute(cache, again, summaries, term_sheets)
    assert len(again.text_calls) == 1
    assert "JSON" in again.text_calls[0][0]
    assert term_sheets.load(*keys(client, cache)) is not None


def test_precompute_makes_only_the_missing_summary(cache, summaries, term_sheets):
    client = EchoTextClient()
    run_precompute(cache, client, summaries, term_sheets)
    summaries.path_of(summaries.load(*keys(client, cache))).unlink()
    again = EchoTextClient()
    run_precompute(cache, again, summaries, term_sheets)
    assert len(again.text_calls) == 1
    assert "JSON" not in again.text_calls[0][0]
    assert summaries.load(*keys(client, cache)) is not None


def test_precompute_does_not_save_a_withheld_summary_but_saves_the_term_sheet(
    cache, summaries, term_sheets
):
    client = FakeLLMClient([MISQUOTE, MISQUOTE, SHEET_REPLY])
    run_precompute(cache, client, summaries, term_sheets)
    assert not list(summaries.directory.glob("*.json"))
    assert term_sheets.load(*keys(client, cache)) is not None


def test_precompute_of_a_cache_without_summary_items_saves_and_asks_nothing(
    settings, summaries, term_sheets
):
    client = EchoTextClient()
    run_precompute(make_cache(TEXT, settings), client, summaries, term_sheets)
    assert client.text_calls == []
    assert not list(summaries.directory.glob("*.json"))
    assert not list(term_sheets.directory.glob("*.json"))


def test_precompute_treats_another_tau_high_as_missing(cache, summaries, term_sheets):
    run_precompute(cache, EchoTextClient(), summaries, term_sheets)
    again = EchoTextClient()
    run_precompute(cache, again, summaries, term_sheets, tau_high=0.95)
    assert len(again.text_calls) == ECHO_CALLS_PER_DOCUMENT


def test_precompute_treats_another_model_as_missing(cache, summaries, term_sheets):
    run_precompute(cache, EchoTextClient(), summaries, term_sheets)
    other = EchoTextClient()
    other.model = "other-gen"
    run_precompute(cache, other, summaries, term_sheets)
    assert len(other.text_calls) == ECHO_CALLS_PER_DOCUMENT


def test_precompute_all_does_each_document_once_and_counts_them(
    store, settings, summaries, term_sheets
):
    store.save(make_cache(TEXT, settings, CLAUSES))
    store.save(make_cache(OTHER_TEXT, settings, OTHER_CLAUSES))
    client = EchoTextClient()
    done = precompute_all(store, settings, TYPES, client, summaries, term_sheets)
    assert done == 2
    assert len(client.text_calls) == 2 * ECHO_CALLS_PER_DOCUMENT


def test_precompute_all_counts_a_document_with_two_entries_once(
    store, settings, summaries, term_sheets
):
    cache = make_cache(TEXT, settings, CLAUSES)
    store.save(cache)
    (store.demo_dir / "copy.json").write_text(cache.model_dump_json(), encoding="utf-8")
    assert len(store.list()) == 2
    client = EchoTextClient()
    assert precompute_all(store, settings, TYPES, client, summaries, term_sheets) == 1
    assert len(client.text_calls) == ECHO_CALLS_PER_DOCUMENT


def test_precompute_all_skips_a_document_whose_model_fails_and_continues(
    store, settings, summaries, term_sheets, caplog
):
    first = make_cache(TEXT, settings, CLAUSES)
    second = make_cache(OTHER_TEXT, settings, OTHER_CLAUSES)
    store.save(first)
    store.save(second)
    client = FailsOn("Delaware")
    with caplog.at_level(logging.WARNING):
        done = precompute_all(store, settings, TYPES, client, summaries, term_sheets)
    assert done == 1
    assert summaries.load(*keys(client, second)) is not None
    assert summaries.load(*keys(client, first)) is None
    assert any(first.doc_id in r.getMessage() for r in caplog.records if r.levelno == 30)


def test_precompute_all_logs_the_count(store, settings, summaries, term_sheets, caplog):
    store.save(make_cache(TEXT, settings, CLAUSES))
    with caplog.at_level(logging.INFO):
        precompute_all(store, settings, TYPES, EchoTextClient(), summaries, term_sheets)
    assert "precomputed 1 documents" in caplog.text


def test_precompute_all_keys_the_results_by_the_settings_thresholds(
    store, tmp_path, summaries, term_sheets
):
    custom = Settings(tau_low=0.1, tau_high=0.5)
    cache = make_cache(TEXT, custom, CLAUSES)
    store.save(cache)
    client = EchoTextClient()
    precompute_all(store, custom, TYPES, client, summaries, term_sheets)
    assert summaries.load(cache.doc_id, client.model, 0.1, 0.5) is not None
    assert term_sheets.load(cache.doc_id, client.model, 0.1, 0.5) is not None
    assert summaries.load(*keys(client, cache)) is None


class Loadable:
    def __init__(self, model: str = "loadable", error: Exception | None = None):
        self.model = model
        self.error = error
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        if self.error:
            raise self.error


def test_warm_loads_each_client_that_can_be_loaded():
    first, second = Loadable(), Loadable()
    warm([first, second])
    assert (first.loads, second.loads) == (1, 1)


def test_warm_skips_none_and_clients_without_load():
    loadable = Loadable()
    warm([None, FakeLLMClient([]), loadable])
    assert loadable.loads == 1


def test_warm_logs_a_warning_and_continues_when_a_load_fails(caplog):
    failing, after = Loadable("broken", LLMError("gone")), Loadable()
    with caplog.at_level(logging.WARNING):
        warm([failing, after])
    assert after.loads == 1
    assert "broken" in caplog.text


def test_run_in_background_runs_the_target_with_its_arguments_in_a_daemon_thread():
    calls: list[tuple] = []

    def target(*args):
        calls.append((args, threading.current_thread()))

    thread = run_in_background(target, 1, "two")
    thread.join(timeout=5)
    assert thread.daemon is True
    assert calls == [((1, "two"), thread)]
