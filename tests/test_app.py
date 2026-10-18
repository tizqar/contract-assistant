import dataclasses
import inspect
import socket
from types import SimpleNamespace
from typing import get_type_hints

import gradio as gr
import pytest
from gradio_client import Client

from contract_assistant import cli
from contract_assistant.answer import Passage
from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.router.base import RouteResult
from contract_assistant.search import PassageSearch
from contract_assistant.store import Store
from contract_assistant.ui import app
from contract_assistant.ui.app import ALL_CLAUSES, DOCUMENT_TAB, Handlers, Services, build_app
from contract_assistant.ui.views import SUGGESTED_QUESTIONS, display_name
from tests.support.caches import make_cache
from tests.support.fakes import FakeEncoder, FakeQA, FakeRouter, FixedRouter, slow_fake_qa

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
    return app_clause_types(settings)


@pytest.fixture
def unvalidated(settings) -> frozenset[str]:
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    return frozenset(clause_type.name for clause_type in extra)


@pytest.fixture
def make_services(settings, clause_types, unvalidated):
    def make(extractor: Extractor | None = None, router=None) -> Services:
        store = Store(settings.cache_dir, settings.demo_cache_dir)
        search = PassageSearch(FakeEncoder())
        return Services(
            store,
            router or FakeRouter(),
            extractor,
            clause_types,
            settings,
            search,
            unvalidated,
        )

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
    return next(row[2] for row in rows if row[1] == display_name(clause_type))


def highlighted(segments: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
    return [(text, label) for text, label in segments if label]


def is_skip(value) -> bool:
    return value == gr.skip()


def row_event(name: str) -> SimpleNamespace:
    return SimpleNamespace(index=[0, 1], row_value=["Topic", name, "", "", "", ""])


def find_component(demo: gr.Blocks, elem_id: str):
    return next(block for block in demo.blocks.values() if block.elem_id == elem_id)


def test_build_app_builds_without_an_extractor(make_services):
    assert isinstance(build_app(make_services()), gr.Blocks)


def test_default_document_is_the_short_supply_agreement(make_services):
    choices = app.document_choices(make_services().store)
    assert app.default_document(choices) == SHORT_DOC_ID


def test_document_choices_show_readable_names(make_services):
    choices = app.document_choices(make_services().store)
    assert ("Short supply agreement", SHORT_DOC_ID) in choices


def test_header_names_no_model_or_router(make_services):
    header = find_component(build_app(make_services()), "header").value
    assert "Contract Assistant" in header
    assert not any(word in header.lower() for word in ("router", "qwen", "llm", "model"))


def test_technical_tab_names_the_router(make_services):
    demo = build_app(make_services())
    texts = [
        block.value
        for block in demo.blocks.values()
        if isinstance(block, gr.Markdown) and block.value
    ]
    assert any("explicit" in text for text in texts)


def test_document_choices_list_each_document_once(make_services):
    choices = app.document_choices(make_services().store)
    assert len({doc_id for _, doc_id in choices}) == len(choices) == 3


def test_open_document_resets_the_page(handlers, demo_cache):
    result = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)
    cache, last, chat, title, rows, _, _, _, picker, sources, _ = result
    assert (cache.doc_id, last, chat, sources) == (SHORT_DOC_ID, None, [], {})
    assert "Short supply agreement" in title
    assert len(rows) == len(demo_cache.clauses)
    assert picker.visible is False


def test_ask_cites_the_governing_law_clause(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    result = Handlers(make_services()).ask("which law governs?", [], cache, TAU_LOW, TAU_HIGH)
    textbox, history, _, picker, _, segments, sources, _ = result
    assert textbox == ""
    assert "tier-found" in history[-1]["content"]
    assert highlighted(segments) == [(LAW_SENTENCE, "Governing Law")]
    assert picker.value == "Governing Law"
    assert sources["answer"]["question"] == "which law governs?"


def test_ask_with_two_types_offers_both_and_shows_the_first(handlers, demo_cache):
    question = "who are the parties and which law governs?"
    result = handlers.ask(question, [], demo_cache, TAU_LOW, TAU_HIGH)
    _, history, last, picker, _, segments, _, _ = result
    assert picker.choices == [("Parties", "Parties"), ("Governing Law", "Governing Law")]
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


def test_ask_switches_to_the_document_tab(handlers, demo_cache):
    result = handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH)
    assert result[-1].selected == DOCUMENT_TAB


def test_suggested_questions_are_asked_through_the_ask_handler(make_services):
    demo = build_app(make_services())
    buttons = [
        b for b in demo.blocks.values() if getattr(b, "value", None) in SUGGESTED_QUESTIONS
    ]
    ask_handler = Handlers.ask
    targets = [
        fn
        for fn in demo.fns.values()
        if any(trigger[0] in {b._id for b in buttons} for trigger in fn.targets)
    ]
    assert len(buttons) == len(SUGGESTED_QUESTIONS)
    assert targets and all(fn.fn.__wrapped__ is ask_handler.__wrapped__ for fn in targets)


def test_ask_without_a_citation_hides_the_picker(handlers, demo_cache):
    result = handlers.ask("tell me a joke", [], demo_cache, TAU_LOW, TAU_HIGH)
    assert result[3].visible is False
    assert is_skip(result[5])


def test_empty_question_warns_and_changes_nothing(handlers, demo_cache, warnings):
    assert is_skip(handlers.ask("  ", [], demo_cache, TAU_LOW, TAU_HIGH))
    assert warnings == ["Type a question first."]


def test_ask_without_a_document_warns(handlers, warnings):
    assert is_skip(handlers.ask("which law governs?", [], None, TAU_LOW, TAU_HIGH))
    assert warnings == ["Select a contract first."]


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
    before = handlers.release_low(TAU_LOW, TAU_HIGH, cache, ALL_CLAUSES)[1]
    after = handlers.release_low(0.5, TAU_HIGH, cache, ALL_CLAUSES)[1]
    assert "tier-check" in status_of(before, "Governing Law")
    assert "tier-not_found" in status_of(after, "Governing Law")


def test_status_filter_keeps_only_rows_of_that_status(handlers, demo_cache):
    rows = handlers.filter_clauses("found", TAU_LOW, TAU_HIGH, demo_cache)
    assert rows
    assert all("tier-found" in row[2] for row in rows)


def test_status_filter_is_respected_after_a_threshold_change(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.3)]})
    handlers = Handlers(make_services())
    rows = handlers.release_low(0.5, TAU_HIGH, cache, "not_found")[1]
    assert all("tier-not_found" in row[2] for row in rows)
    assert status_of(rows, "Governing Law")


def test_tau_low_released_above_tau_high_is_clamped(handlers, demo_cache):
    assert handlers.release_low(0.95, 0.91, demo_cache, ALL_CLAUSES)[0] == 0.91


def test_tau_high_released_below_tau_low_is_clamped(handlers, demo_cache):
    assert handlers.release_high(0.5, 0.2, demo_cache, ALL_CLAUSES)[0] == 0.5


def test_reset_restores_the_default_thresholds(handlers, demo_cache, settings):
    tau_low, tau_high, rows = handlers.reset_thresholds(demo_cache, ALL_CLAUSES)
    assert (tau_low, tau_high) == (settings.tau_low, settings.tau_high)
    assert len(rows) == len(demo_cache.clauses)


def test_slider_release_without_a_document_changes_no_table(handlers):
    assert is_skip(handlers.release_low(0.1, 0.9, None, ALL_CLAUSES)[1])


def test_show_row_shows_the_top_citation_of_that_type(handlers, demo_cache):
    _, segments, _ = handlers.show_row(demo_cache, row_event("Governing Law"))
    top = demo_cache.clauses["Governing Law"][0]
    assert highlighted(segments) == [(top.text, "Governing Law")]


def test_show_row_maps_a_display_name_back_to_its_clause_type(handlers, demo_cache):
    _, segments, _ = handlers.show_row(demo_cache, row_event("Cap on Liability"))
    top = demo_cache.clauses["Cap On Liability"][0]
    assert highlighted(segments) == [(top.text, "Cap on Liability")]


def test_show_row_switches_to_the_document_tab(handlers, demo_cache):
    assert (
        handlers.show_row(demo_cache, row_event("Governing Law"))[-1].selected == DOCUMENT_TAB
    )


def test_upload_with_an_existing_cache_is_a_hit_and_skips_the_model(make_services):
    qa = FakeQA()
    extractor = Extractor(qa, [], load_settings(), "fake")
    handlers = Handlers(make_services(extractor))
    dropdown, status, _ = handlers.extract_upload(str(DEMO_TEXT), progress=lambda *a, **k: None)
    assert (dropdown.value, status) == (SHORT_DOC_ID, "Already analysed")
    assert qa.calls == []


def test_upload_without_an_extractor_says_the_model_is_missing(handlers, tmp_path, warnings):
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    assert is_skip(handlers.extract_upload(str(path), progress=lambda *a, **k: None))
    assert warnings == [app.NO_MODEL]


def test_upload_of_a_new_text_writes_a_cache_and_selects_it(
    make_services, settings, clause_types, tmp_path
):
    extractor = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    calls = []
    dropdown, status, _ = Handlers(make_services(extractor)).extract_upload(
        str(path), progress=lambda *args, **kwargs: calls.append(args)
    )
    assert len(list(settings.cache_dir.glob("*.json"))) == 1
    assert status == "Analysis complete"
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


def test_upload_of_a_cache_missing_clause_types_extends_and_saves_it(
    make_services, settings, clause_types, tmp_path
):
    path = tmp_path / "older.txt"
    path.write_text(TEXT, encoding="utf-8")
    older = Extractor(slow_fake_qa(0), clause_types[:41], settings, "fake")
    full = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    Handlers(make_services(older)).extract_upload(str(path), progress=lambda *a, **k: None)
    calls = []
    dropdown, status, refresh = Handlers(make_services(full)).extract_upload(
        str(path), progress=lambda *args, **kwargs: calls.append(args)
    )
    saved = Store(settings.cache_dir, settings.demo_cache_dir).load(dropdown.value, settings)
    assert (status, refresh) == ("Analysis updated", dropdown.value)
    assert "Payment Terms" in saved.clauses
    assert calls[-1][0] == (8, 8)


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
    monkeypatch.setattr(cli, "build_router", lambda *args: FakeRouter())
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


def payment_cache(settings, conf=0.97):
    return make_cache(TEXT, settings, {"Payment Terms": [(LAW_SENTENCE, conf)]})


def route_to(*names: str) -> FixedRouter:
    return FixedRouter(
        {"q": RouteResult(clause_types=list(names), kind="explicit", latency_ms=0)}
    )


def test_ask_fills_passages_for_an_empty_route(make_services, settings):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage()]
    result = handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)
    assert len(result[2].passages) == 1
    assert "tier-related" in result[1][-1]["content"]


def passage(start: int = 10) -> Passage:
    return Passage(quote=LAW_SENTENCE, page=None, score=0.7, start=start, end=start + 49)


def test_ask_leaves_passages_empty_when_the_route_has_clause_types(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    handlers = Handlers(make_services(router=route_to("Governing Law")))
    handlers.services.search.search = lambda cache, question: [passage()]
    assert handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)[2].passages == []


def test_ask_about_a_type_missing_from_an_older_cache_does_not_raise(make_services):
    cache = make_cache(TEXT, load_settings(), {"Governing Law": []})
    handlers = Handlers(make_services(router=route_to("Payment Terms")))
    result = handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)
    assert result[2].items == []


def test_ask_caps_an_unvalidated_type_at_verify(make_services, settings):
    cache = payment_cache(settings)
    handlers = Handlers(make_services(router=route_to("Payment Terms")))
    reply = handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)[1][-1]["content"]
    assert "tier-check" in reply and "tier-found" not in reply
    assert "Not validated on test data" in reply


def test_ask_with_passages_lists_them_in_the_picker_and_shows_the_first(
    make_services, settings
):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage(10), passage(20)]
    result = handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)
    picker, segments = result[3], result[5]
    assert [label for label, _ in picker.choices] == ["Related passage 1", "Related passage 2"]
    assert highlighted(segments)[0][1] == "Related passage"
    assert picker.value == picker.choices[0][1]


def test_pick_citation_shows_the_chosen_related_passage(make_services, settings):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage(0), passage(10)]
    result = handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH)
    value = result[3].choices[1][1]
    _, segments = handlers.pick_citation(value, result[2], cache)
    assert highlighted(segments)[0][0] == TEXT[10:59]


def test_checklist_lists_extra_types_as_verify_at_most(make_services, settings):
    cache = payment_cache(settings)
    handlers = Handlers(make_services())
    rows = handlers.filter_clauses(ALL_CLAUSES, TAU_LOW, TAU_HIGH, cache)
    assert "tier-check" in status_of(rows, "Payment Terms")
    assert next(row[0] for row in rows if row[1] == "Payment Terms") == "Other common clauses"


def test_technical_tab_names_the_unvalidated_types(make_services):
    demo = build_app(make_services())
    texts = [b.value for b in demo.blocks.values() if isinstance(b, gr.Markdown) and b.value]
    assert any(
        "Not validated on test data:** " in text and "Payment Terms" in text for text in texts
    )


def test_dev_app_builds():
    from tests.support import dev_app

    assert isinstance(dev_app.demo, gr.Blocks)


def test_reload_after_an_extension_reopens_the_contract_and_clears_the_flag(handlers):
    result = handlers.reload_extended(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)
    assert result[0].doc_id == SHORT_DOC_ID
    assert result[-1] is None


def test_reload_without_an_extension_leaves_the_page_as_it_is(handlers):
    assert is_skip(handlers.reload_extended(None, TAU_LOW, TAU_HIGH, ALL_CLAUSES))


def test_skipped_upload_does_not_flag_a_reload(handlers, warnings):
    assert is_skip(handlers.extract_upload(None, progress=lambda *a, **k: None))


def test_new_upload_does_not_flag_a_reload(make_services, settings, clause_types, tmp_path):
    extractor = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    result = Handlers(make_services(extractor)).extract_upload(
        str(path), progress=lambda *a, **k: None
    )
    assert result[2] is None
