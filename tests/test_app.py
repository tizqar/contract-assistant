import dataclasses
import inspect
import socket
from types import SimpleNamespace
from typing import get_type_hints

import gradio as gr
import pytest
from gradio_client import Client

from contract_assistant import cli
from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.store import Store
from contract_assistant.ui import app
from contract_assistant.ui.app import Handlers, Services, build_app
from tests.support.caches import make_cache
from tests.support.fakes import FakeQA, FakeRouter, slow_fake_qa

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
SHORT_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
LAW_SENTENCE = "This agreement is governed by the laws of Vanuatu."
TEXT = f"Preamble. {LAW_SENTENCE} Final words."
TAU_LOW, TAU_HIGH = 0.06, 0.91


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(load_settings(), cache_dir=tmp_path / "cache")


@pytest.fixture
def clause_types(settings):
    return load_clause_types(settings.clause_types_path)


@pytest.fixture
def make_services(settings, clause_types):
    def make(extractor: Extractor | None = None, router=None) -> Services:
        store = Store(settings.cache_dir, settings.demo_cache_dir)
        return Services(store, router or FakeRouter(), extractor, clause_types, settings)

    return make


@pytest.fixture
def handlers(make_services) -> Handlers:
    return Handlers(make_services())


@pytest.fixture
def demo_cache(settings):
    return Store(settings.cache_dir, settings.demo_cache_dir).load(SHORT_DOC_ID, settings)


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(app.gr, "Warning", messages.append)
    return messages


def status_of(rows: list[list], clause_type: str) -> str:
    return next(row[1] for row in rows if row[0] == clause_type)


def highlighted(segments: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
    return [(text, label) for text, label in segments if label]


def is_skip(value) -> bool:
    return value == gr.skip()


def test_build_app_builds_without_an_extractor(make_services):
    assert isinstance(build_app(make_services()), gr.Blocks)


def test_default_document_is_the_short_supply_agreement(make_services):
    choices = app.document_choices(make_services().store)
    assert app.default_document(choices) == SHORT_DOC_ID


def test_document_choices_list_each_document_once(make_services):
    choices = app.document_choices(make_services().store)
    assert len({doc_id for _, doc_id in choices}) == len(choices) == 3


def test_open_document_resets_the_page(handlers, demo_cache):
    result = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH)
    cache, last, chat, title, rows, label, _, picker, sources, _ = result
    assert (cache.doc_id, last, chat, sources) == (SHORT_DOC_ID, None, [], {})
    assert title.startswith("short_supply_agreement.txt")
    assert len(rows) == len(demo_cache.clauses)
    assert label.startswith("characters 0")
    assert picker.visible is False


def test_ask_cites_the_governing_law_clause(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    result = Handlers(make_services()).ask("which law governs?", [], cache, TAU_LOW, TAU_HIGH)
    textbox, history, _, picker, _, segments, sources = result
    assert textbox == ""
    assert "found" in history[-1]["content"]
    assert highlighted(segments) == [(LAW_SENTENCE, "Governing Law")]
    assert picker.value == "Governing Law"
    assert sources["answer"]["question"] == "which law governs?"


def test_ask_with_two_types_offers_both_and_shows_the_first(handlers, demo_cache):
    question = "who are the parties and which law governs?"
    result = handlers.ask(question, [], demo_cache, TAU_LOW, TAU_HIGH)
    _, history, last, picker, _, segments, _ = result
    assert [name for name, _ in picker.choices] == ["Parties", "Governing Law"]
    assert picker.value == "Parties"
    assert highlighted(segments)[0][1] == "Parties"
    assert len(history) == 2 and len(last.items) == 2


def test_ask_keeps_earlier_messages(handlers, demo_cache):
    first = handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH)
    second = handlers.ask("who are the parties?", first[1], demo_cache, TAU_LOW, TAU_HIGH)
    assert len(second[1]) == 4


def test_ask_records_the_slider_thresholds(handlers, demo_cache):
    result = handlers.ask("which law governs?", [], demo_cache, 0.2, 0.7)
    assert (result[2].tau_low, result[2].tau_high) == (0.2, 0.7)


def test_ask_without_a_citation_hides_the_picker(handlers, demo_cache):
    result = handlers.ask("tell me a joke", [], demo_cache, TAU_LOW, TAU_HIGH)
    assert result[3].visible is False
    assert is_skip(result[5])


def test_empty_question_warns_and_changes_nothing(handlers, demo_cache, warnings):
    assert is_skip(handlers.ask("  ", [], demo_cache, TAU_LOW, TAU_HIGH))
    assert warnings == ["Type a question first."]


def test_ask_without_a_document_warns(handlers, warnings):
    assert is_skip(handlers.ask("which law governs?", [], None, TAU_LOW, TAU_HIGH))
    assert warnings == ["Open a document first."]


def test_ask_with_inverted_thresholds_warns(handlers, demo_cache, warnings):
    assert is_skip(handlers.ask("which law governs?", [], demo_cache, 0.9, 0.1))
    assert len(warnings) == 1


def test_pick_citation_shows_the_chosen_type(handlers, demo_cache):
    question = "who are the parties and which law governs?"
    last = handlers.ask(question, [], demo_cache, TAU_LOW, TAU_HIGH)[2]
    _, segments = handlers.pick_citation("Governing Law", last, demo_cache)
    assert highlighted(segments)[0][1] == "Governing Law"


def test_raising_tau_low_moves_rows_from_verify_to_not_found(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.3)]})
    handlers = Handlers(make_services())
    before = handlers.release_low(TAU_LOW, TAU_HIGH, cache)[1]
    after = handlers.release_low(0.5, TAU_HIGH, cache)[1]
    assert "verify" in status_of(before, "Governing Law")
    assert "not found" in status_of(after, "Governing Law")


def test_tau_low_released_above_tau_high_is_clamped(handlers, demo_cache):
    assert handlers.release_low(0.95, 0.91, demo_cache)[0] == 0.91


def test_tau_high_released_below_tau_low_is_clamped(handlers, demo_cache):
    assert handlers.release_high(0.5, 0.2, demo_cache)[0] == 0.5


def test_reset_restores_the_default_thresholds(handlers, demo_cache, settings):
    tau_low, tau_high, rows = handlers.reset_thresholds(demo_cache)
    assert (tau_low, tau_high) == (settings.tau_low, settings.tau_high)
    assert len(rows) == len(demo_cache.clauses)


def test_slider_release_without_a_document_changes_no_table(handlers):
    assert is_skip(handlers.release_low(0.1, 0.9, None)[1])


def test_show_row_shows_the_top_citation_of_that_type(handlers, demo_cache):
    event = SimpleNamespace(index=[0, 0], row_value=["Governing Law", "", 0.9, "n/a", ""])
    _, segments = handlers.show_row(demo_cache, event)
    top = demo_cache.clauses["Governing Law"][0]
    assert highlighted(segments) == [(top.text, "Governing Law")]


def test_upload_with_an_existing_cache_is_a_hit_and_skips_the_model(make_services):
    qa = FakeQA()
    extractor = Extractor(qa, [], load_settings(), "fake")
    handlers = Handlers(make_services(extractor))
    dropdown, status = handlers.extract_upload(str(DEMO_TEXT), progress=lambda *a, **k: None)
    assert (dropdown.value, status) == (SHORT_DOC_ID, "Cache hit")
    assert qa.calls == []


def test_upload_without_an_extractor_warns_with_the_model_path(
    handlers, tmp_path, warnings, settings
):
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    assert is_skip(handlers.extract_upload(str(path), progress=lambda *a, **k: None))
    assert warnings == [
        f"Extraction needs the model in {settings.extractor_path}; see IMPLEMENTATION.md, Step A."
    ]


def test_upload_of_a_new_text_writes_a_cache_and_selects_it(
    make_services, settings, clause_types, tmp_path
):
    extractor = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    calls = []
    dropdown, status = Handlers(make_services(extractor)).extract_upload(
        str(path), progress=lambda *args, **kwargs: calls.append(args)
    )
    assert len(list(settings.cache_dir.glob("*.json"))) == 1
    assert status.startswith("Extracted in")
    assert dropdown.value in [doc_id for _, doc_id in dropdown.choices]
    assert calls[-1][0] == (len(clause_types), len(clause_types))


def test_upload_of_an_unsupported_file_warns(handlers, tmp_path, warnings):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    assert is_skip(handlers.extract_upload(str(path), progress=lambda *a, **k: None))
    assert warnings == ["unsupported file type: .docx"]


def test_upload_with_no_file_warns(handlers, warnings):
    assert is_skip(handlers.extract_upload(None, progress=lambda *a, **k: None))
    assert len(warnings) == 1


def test_unexpected_exception_becomes_a_warning(make_services, demo_cache, warnings):
    class BrokenRouter:
        kind = "explicit"

        def route(self, question):
            raise RuntimeError("boom")

    handlers = Handlers(make_services(router=BrokenRouter()))
    result = handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH)
    assert is_skip(result)
    assert warnings == [app.UNEXPECTED_ERROR]


def test_guarded_handler_keeps_the_progress_default_gradio_looks_for(handlers):
    parameter = inspect.signature(handlers.extract_upload).parameters["progress"]
    assert isinstance(parameter.default, gr.Progress)


def test_guarded_handler_keeps_the_select_data_annotation_gradio_looks_for(handlers):
    assert get_type_hints(handlers.show_row)["evt"] is gr.SelectData


def test_serve_builds_the_services_without_a_model(monkeypatch, tmp_path, capsys):
    launched = {}
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EXTRACTOR_MODEL_ID", "models/does-not-exist")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "build_router", lambda settings: FakeRouter())
    monkeypatch.setattr(
        app,
        "launch_app",
        lambda demo, services, share, port: launched.update(
            demo=demo, services=services, share=share, port=port
        ),
    )
    assert cli.main(["serve", "--port", "7999", "--share"]) == 0
    assert launched["services"].extractor is None
    assert (launched["port"], launched["share"]) == (7999, True)
    assert isinstance(launched["demo"], gr.Blocks)
    assert "models/does-not-exist" in capsys.readouterr().err


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ask_over_the_client_uses_the_document_loaded_in_the_same_session(make_services):
    demo = build_app(make_services())
    demo.launch(prevent_thread_lock=True, quiet=True, server_port=free_port())
    try:
        client = Client(demo.local_url, verbose=False)
        client.predict(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, api_name="/load_document")
        question = "who are the parties and which law governs?"
        reply = client.predict(question, [], TAU_LOW, TAU_HIGH, api_name="/ask")
        picker, chat = reply[2], reply[1]
    finally:
        demo.close()
    assert [choice[0] for choice in picker["choices"]] == ["Parties", "Governing Law"]
    assert len(chat) == 2
