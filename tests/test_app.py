import dataclasses
import inspect
import json
import socket
from types import SimpleNamespace
from typing import get_type_hints

import gradio as gr
import pytest
from gradio_client import Client

from contract_assistant import cli
from contract_assistant.answer import Answer, GeneratedAnswer, Passage
from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.feedback import FEEDBACK_FILE
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.router.clients import LLMError
from contract_assistant.search import PassageSearch
from contract_assistant.store import Store
from contract_assistant.term_sheet import FIELDS, INVALID_REASON, LABELS, TermSheet
from contract_assistant.ui import app
from contract_assistant.ui.app import (
    ALL_CLAUSES,
    DOCUMENT_TAB,
    Handlers,
    Services,
    build_app,
    document_choices,
)
from contract_assistant.ui.views import SUGGESTED_QUESTIONS, display_name
from tests.support.caches import make_cache
from tests.support.fakes import (
    EchoTextClient,
    FakeEncoder,
    FakeLLMClient,
    FakeQA,
    FakeRouter,
    FixedRouter,
    slow_fake_qa,
)

LONG_DOC_ID = "sha256:5ace7a03650252f3f2a08d44b3bf4c689d6b9e1bc62a90a455a23bcb4e484b02"
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
    def make(extractor: Extractor | None = None, router=None, gen=None) -> Services:
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
            gen=gen,
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


def last(stream):
    return list(stream)[-1]


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
    cache, last, chat, title, rows, _, _, _, picker, sources, _, turns, *_ = result
    assert (cache.doc_id, last, chat, sources, turns) == (SHORT_DOC_ID, None, [], {}, [])
    assert "Short supply agreement" in title
    assert len(rows) == len(demo_cache.clauses)
    assert picker.visible is False


def test_ask_cites_the_governing_law_clause(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    result = last(
        Handlers(make_services()).ask("which law governs?", [], cache, TAU_LOW, TAU_HIGH)
    )
    textbox, history, _, picker, _, segments, sources, _, _ = result
    assert textbox == ""
    assert "tier-found" in history[-1]["content"]
    assert highlighted(segments) == [(LAW_SENTENCE, "Governing Law")]
    assert picker.value == "Governing Law"
    assert sources["answer"]["question"] == "which law governs?"


def test_ask_with_two_types_offers_both_and_shows_the_first(handlers, demo_cache):
    question = "who are the parties and which law governs?"
    result = last(handlers.ask(question, [], demo_cache, TAU_LOW, TAU_HIGH))
    _, history, answered, picker, _, segments, _, _, _ = result
    assert picker.choices == [("Parties", "Parties"), ("Governing Law", "Governing Law")]
    assert picker.value == "Parties"
    assert highlighted(segments)[0][1] == "Parties"
    assert len(history) == 2 and len(answered.items) == 2


def test_ask_keeps_earlier_messages(handlers, demo_cache):
    first = last(handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH))
    second = last(handlers.ask("who are the parties?", first[1], demo_cache, TAU_LOW, TAU_HIGH))
    assert len(second[1]) == 4


def test_ask_records_the_slider_thresholds(handlers, demo_cache):
    result = last(handlers.ask("which law governs?", [], demo_cache, 0.2, 0.7))
    assert (result[2].tau_low, result[2].tau_high) == (0.2, 0.7)


def test_ask_switches_to_the_document_tab(handlers, demo_cache):
    result = last(handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH))
    assert result[7].selected == DOCUMENT_TAB


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
    result = last(handlers.ask("tell me a joke", [], demo_cache, TAU_LOW, TAU_HIGH))
    assert result[3].visible is False
    assert is_skip(result[5])


def test_empty_question_warns_and_changes_nothing(handlers, demo_cache, warnings):
    assert is_skip(last(handlers.ask("  ", [], demo_cache, TAU_LOW, TAU_HIGH)))
    assert warnings == ["Type a question first."]


def test_ask_without_a_document_warns(handlers, warnings):
    assert is_skip(last(handlers.ask("which law governs?", [], None, TAU_LOW, TAU_HIGH)))
    assert warnings == ["Select a contract first."]


def test_ask_with_inverted_thresholds_warns(handlers, demo_cache, warnings):
    assert is_skip(last(handlers.ask("which law governs?", [], demo_cache, 0.9, 0.1)))
    assert len(warnings) == 1


def test_pick_citation_shows_the_chosen_type(handlers, demo_cache):
    question = "who are the parties and which law governs?"
    answered = last(handlers.ask(question, [], demo_cache, TAU_LOW, TAU_HIGH))[2]
    _, segments = handlers.pick_citation("Governing Law", answered, demo_cache)
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
    tau_low, tau_high, rows, *_ = handlers.reset_thresholds(demo_cache, ALL_CLAUSES)
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
    result = last(handlers.ask("which law governs?", [], demo_cache, TAU_LOW, TAU_HIGH))
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
    gen = EchoTextClient()
    monkeypatch.setattr(cli, "build_gen_client", lambda settings: gen)
    monkeypatch.setattr(cli, "run_in_background", lambda *args: None)
    monkeypatch.setattr(
        app,
        "launch_app",
        lambda demo, services, share, port: launched.update(
            demo=demo, services=services, share=share, port=port
        ),
    )
    assert cli.main(["serve", "--port", "7999", "--share"]) == 0
    assert launched["services"].extractor is None
    assert launched["services"].gen is gen
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
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert len(result[2].passages) == 1
    assert "tier-related" in result[1][-1]["content"]


def passage(start: int = 10) -> Passage:
    return Passage(quote=LAW_SENTENCE, page=None, score=0.7, start=start, end=start + 49)


def test_ask_leaves_passages_empty_when_the_route_has_clause_types(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    handlers = Handlers(make_services(router=route_to("Governing Law")))
    handlers.services.search.search = lambda cache, question: [passage()]
    assert last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))[2].passages == []


def test_ask_about_a_type_missing_from_an_older_cache_does_not_raise(make_services):
    cache = make_cache(TEXT, load_settings(), {"Governing Law": []})
    handlers = Handlers(make_services(router=route_to("Payment Terms")))
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert result[2].items == []


def test_ask_caps_an_unvalidated_type_at_verify(make_services, settings):
    cache = payment_cache(settings)
    handlers = Handlers(make_services(router=route_to("Payment Terms")))
    reply = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))[1][-1]["content"]
    assert "tier-check" in reply and "tier-found" not in reply
    assert "Not validated on test data" in reply


def test_ask_with_passages_lists_them_in_the_picker_and_shows_the_first(
    make_services, settings
):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage(10), passage(20)]
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    picker, segments = result[3], result[5]
    assert [label for label, _ in picker.choices] == ["Related passage 1", "Related passage 2"]
    assert highlighted(segments)[0][1] == "Related passage"
    assert picker.value == picker.choices[0][1]


def test_pick_citation_shows_the_chosen_related_passage(make_services, settings):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage(0), passage(10)]
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
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


FAITHFUL_REPLY = f'The law is set by "{LAW_SENTENCE}" (Governing Law).'
MISQUOTING_REPLY = 'The contract says "This agreement ends on Friday." (Governing Law).'
ALTERNATIVE_TAU_HIGH = 0.9


@pytest.fixture
def infos(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(app.gr, "Info", messages.append)
    return messages


@pytest.fixture
def law_cache(settings):
    return make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})


def generating(make_services, replies: list[str | Exception]):
    client = FakeLLMClient(replies)
    return Handlers(make_services(gen=client)), client


def view_segments(label: str | None, text: str = LAW_SENTENCE) -> list[dict]:
    return [
        {"token": "Preamble. ", "class_or_confidence": None},
        {"token": text, "class_or_confidence": label},
    ]


def component_values(demo: gr.Blocks) -> list[str]:
    return [b.value for b in demo.blocks.values() if isinstance(b, gr.Markdown) and b.value]


def test_summarize_returns_the_reply_and_saves_it(make_services, law_cache, settings):
    handlers, client = generating(make_services, [FAITHFUL_REPLY])
    result = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert FAITHFUL_REPLY in result
    assert len(client.text_calls) == 1
    assert len(list((settings.cache_dir / "summaries").glob("*.json"))) == 1


def test_summarize_again_with_the_same_thresholds_makes_no_model_call(make_services, law_cache):
    handlers, client = generating(make_services, [FAITHFUL_REPLY])
    first = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    second = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert len(client.text_calls) == 1
    assert second == first


def test_summarize_with_other_thresholds_asks_the_model_again(make_services, law_cache):
    handlers, client = generating(make_services, [FAITHFUL_REPLY, FAITHFUL_REPLY])
    last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    last(handlers.summarize(law_cache, TAU_LOW, ALTERNATIVE_TAU_HIGH))
    assert len(client.text_calls) == 2


def test_summarize_withholds_a_reply_that_misquotes_twice(make_services, law_cache):
    handlers, client = generating(make_services, [MISQUOTING_REPLY, MISQUOTING_REPLY])
    result = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert "not shown" in result
    assert "ends on Friday" not in result
    assert len(client.text_calls) == 2


def test_summarize_shows_the_second_reply_when_the_first_misquotes(make_services, law_cache):
    handlers, client = generating(make_services, [MISQUOTING_REPLY, FAITHFUL_REPLY])
    result = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert FAITHFUL_REPLY in result
    assert len(client.text_calls) == 2


def test_summarize_without_a_model_warns(make_services, law_cache, warnings):
    result = last(Handlers(make_services()).summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert is_skip(result)
    assert warnings == [app.SUMMARIES_OFF]


def test_summarize_without_a_document_warns(make_services, warnings):
    handlers, client = generating(make_services, [])
    assert is_skip(last(handlers.summarize(None, TAU_LOW, TAU_HIGH)))
    assert warnings == [app.SELECT_CONTRACT]
    assert client.text_calls == []


def test_summarize_with_inverted_thresholds_warns(make_services, law_cache, warnings):
    handlers, client = generating(make_services, [])
    assert is_skip(last(handlers.summarize(law_cache, 0.9, 0.1)))
    assert len(warnings) == 1
    assert client.text_calls == []


def test_summarize_without_found_clauses_says_there_is_nothing_to_summarize(
    make_services, settings
):
    handlers, client = generating(make_services, [])
    cache = make_cache(TEXT, settings)
    assert last(handlers.summarize(cache, TAU_LOW, TAU_HIGH)) == app.NOTHING_TO_SUMMARIZE
    assert client.text_calls == []


def test_summarize_warns_when_the_model_does_not_answer(make_services, law_cache, warnings):
    handlers, _ = generating(make_services, [LLMError("down")])
    assert is_skip(last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH)))
    assert warnings == [app.GENERATION_FAILED]


def test_explain_restates_the_highlighted_quote(make_services, law_cache):
    handlers, client = generating(make_services, ["The law of Vanuatu applies."])
    result = last(handlers.explain(view_segments("Governing Law"), "en", law_cache))
    assert LAW_SENTENCE in client.text_calls[0][1]
    assert "**Governing Law**" in result and "The law of Vanuatu applies." in result


def test_explain_keeps_the_related_passage_label(make_services, law_cache):
    handlers, client = generating(make_services, ["The law of Vanuatu applies."])
    result = last(handlers.explain(view_segments(app.RELATED_LABEL), "sr", law_cache))
    assert f"Clause type: {app.RELATED_LABEL}" in client.text_calls[0][1]
    assert f"**{app.RELATED_LABEL}** · in plain Serbian" in result


def test_explain_without_a_highlight_says_so(make_services, law_cache, infos):
    handlers, client = generating(make_services, [])
    assert is_skip(last(handlers.explain(view_segments(None), "en", law_cache)))
    assert infos == [app.NOTHING_HIGHLIGHTED]
    assert client.text_calls == []


def test_explain_without_a_model_warns(make_services, law_cache, warnings):
    result = last(
        Handlers(make_services()).explain(view_segments("Governing Law"), "en", law_cache)
    )
    assert is_skip(result)
    assert warnings == [app.SUMMARIES_OFF]


def test_explain_without_a_document_warns(make_services, warnings):
    handlers, _ = generating(make_services, [])
    assert is_skip(last(handlers.explain(view_segments("Governing Law"), "en", None)))
    assert warnings == [app.SELECT_CONTRACT]


def test_explain_warns_when_the_model_does_not_answer(make_services, law_cache, warnings):
    handlers, _ = generating(make_services, [LLMError("down")])
    assert is_skip(last(handlers.explain(view_segments("Governing Law"), "en", law_cache)))
    assert warnings == [app.GENERATION_FAILED]


def test_buttons_are_off_and_the_intro_says_so_without_a_model(make_services):
    demo = build_app(make_services())
    assert find_component(demo, "summarize").interactive is False
    assert find_component(demo, "explain").interactive is False
    assert find_component(demo, "summary-intro").value == app.SUMMARY_INTRO_OFF


def test_buttons_are_on_with_a_model(make_services):
    demo = build_app(make_services(gen=FakeLLMClient([])))
    assert find_component(demo, "summarize").interactive is True
    assert find_component(demo, "explain").interactive is True
    assert find_component(demo, "summary-intro").value == app.SUMMARY_INTRO_ON


def test_technical_tab_names_the_generation_model(make_services):
    demo = build_app(make_services(gen=FakeLLMClient([], model="gen-model-x")))
    assert any("`gen-model-x`" in text for text in component_values(demo))


def test_summarize_over_the_client_returns_the_summary(make_services):
    demo = build_app(make_services(gen=EchoTextClient()))
    demo.launch(prevent_thread_lock=True, quiet=True, server_port=free_port())
    try:
        client = Client(demo.local_url, verbose=False)
        client.predict(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, api_name="/load_document")
        reply = client.predict(TAU_LOW, TAU_HIGH, api_name="/summarize")
    finally:
        demo.close()
    assert "Based on" in reply


def test_summarize_does_not_save_a_withheld_summary_and_asks_again(
    make_services, law_cache, settings
):
    replies = [MISQUOTING_REPLY, MISQUOTING_REPLY, FAITHFUL_REPLY]
    handlers, client = generating(make_services, replies)
    first = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert "not shown" in first
    assert not (settings.cache_dir / "summaries").exists()
    second = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert FAITHFUL_REPLY in second
    assert len(client.text_calls) == 3


def test_term_sheet_returns_rows_in_field_order_and_a_download(
    make_services, demo_cache, settings
):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    rows, button, *_ = last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    assert [row[0] for row in rows] == [LABELS[field] for field in FIELDS]
    assert len(rows) == 8
    assert button.visible is True
    [path] = (settings.cache_dir / "term_sheets").glob("*.json")
    assert button.value["orig_name"] == path.name
    assert TermSheet.model_validate_json(path.read_text()).doc_id == demo_cache.doc_id


def test_term_sheet_again_makes_no_model_call(make_services, demo_cache):
    client = EchoTextClient()
    handlers = Handlers(make_services(gen=client))
    last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    calls = len(client.text_calls)
    last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    assert calls > 0 and len(client.text_calls) == calls


def test_term_sheet_without_a_model_warns(make_services, demo_cache, warnings):
    assert is_skip(last(Handlers(make_services()).term_sheet(demo_cache, TAU_LOW, TAU_HIGH)))
    assert warnings == [app.SUMMARIES_OFF]


def test_term_sheet_without_a_document_warns(make_services, warnings):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    assert is_skip(last(handlers.term_sheet(None, TAU_LOW, TAU_HIGH)))
    assert warnings == [app.SELECT_CONTRACT]


def test_term_sheet_without_key_term_clauses_says_so(make_services, settings, infos):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    cache = make_cache(TEXT, settings)
    assert is_skip(last(handlers.term_sheet(cache, TAU_LOW, TAU_HIGH)))
    assert infos == [app.NO_TERM_CLAUSES]


def test_term_sheet_with_inverted_thresholds_warns(make_services, demo_cache, warnings):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    assert is_skip(last(handlers.term_sheet(demo_cache, 0.9, 0.1)))
    assert len(warnings) == 1


def test_term_sheet_warns_when_the_model_does_not_answer(make_services, demo_cache, warnings):
    handlers, _ = generating(make_services, [LLMError("down")])
    assert all(
        is_skip(value) for value in last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))[:2]
    )
    assert warnings == [app.GENERATION_FAILED]


def test_term_sheet_withheld_shows_not_stated_hides_the_download_and_is_not_saved(
    make_services, demo_cache, warnings, settings
):
    handlers, _ = generating(make_services, ["no json", "still no json"])
    rows, button, *_ = last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    assert {row[1] for row in rows} == {"not stated"}
    assert button.visible is False
    assert warnings == [app.TERM_SHEET_WITHHELD.format(reason=INVALID_REASON)]
    assert not (settings.cache_dir / "term_sheets").exists()


def test_term_sheet_button_follows_the_model(make_services):
    assert find_component(build_app(make_services()), "term-sheet").interactive is False
    demo = build_app(make_services(gen=FakeLLMClient([])))
    assert find_component(demo, "term-sheet").interactive is True
    assert find_component(demo, "term-download").visible is False


def test_ask_writes_an_answer_from_the_passages(make_services, settings):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to(), gen=EchoTextClient()))
    handlers.services.search.search = lambda cache, question: [passage()]
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert isinstance(result[2].generated, GeneratedAnswer)
    assert "tier-generated" in result[1][-1]["content"]


def test_ask_keeps_the_passages_when_the_model_does_not_answer(make_services, settings):
    cache = make_cache(TEXT, settings)
    client = FakeLLMClient([LLMError("down"), LLMError("down")])
    handlers = Handlers(make_services(router=route_to(), gen=client))
    handlers.services.search.search = lambda cache, question: [passage()]
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert len(result[2].passages) == 1 and result[2].generated is None
    assert "tier-generated" not in result[1][-1]["content"]


def test_ask_without_a_model_has_no_generated_answer(make_services, settings):
    cache = make_cache(TEXT, settings)
    handlers = Handlers(make_services(router=route_to()))
    handlers.services.search.search = lambda cache, question: [passage()]
    assert last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))[2].generated is None


def test_ask_with_clause_types_requests_no_generated_answer(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    client = EchoTextClient()
    handlers = Handlers(make_services(router=route_to("Governing Law"), gen=client))
    result = last(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert result[2].generated is None and client.text_calls == []


def test_summarize_streams_the_status_and_the_growing_reply(make_services, law_cache):
    handlers, _ = generating(make_services, [FAITHFUL_REPLY])
    yields = list(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert yields[0].startswith(app.WRITING_SUMMARY)
    assert any(FAITHFUL_REPLY.split()[0] in text and "▌" in text for text in yields[:-1])
    assert "▌" not in yields[-1]
    assert FAITHFUL_REPLY in yields[-1]


def test_summarize_shows_that_it_asks_again_when_the_first_reply_is_withheld(
    make_services, law_cache
):
    handlers, _ = generating(make_services, [MISQUOTING_REPLY, FAITHFUL_REPLY])
    yields = list(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    retry = [i for i, text in enumerate(yields) if "asking again" in text]
    assert retry and "not shown because it quotes text" in yields[retry[0]]
    assert all("ends on Friday" not in text for text in yields[retry[0] :])
    assert FAITHFUL_REPLY in yields[-1]


def test_summarize_of_a_saved_summary_yields_once(make_services, law_cache):
    handlers, _ = generating(make_services, [FAITHFUL_REPLY])
    last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert len(list(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))) == 1


def test_explain_streams_the_status_and_ends_with_the_checked_text(make_services, law_cache):
    handlers, _ = generating(make_services, ["The law of Vanuatu applies."])
    yields = list(handlers.explain(view_segments("Governing Law"), "en", law_cache))
    assert yields[0].startswith(app.WRITING_EXPLANATION)
    assert any("The law" in text and "▌" in text for text in yields[:-1])
    assert "**Governing Law**" in yields[-1] and "▌" not in yields[-1]


def passage_handlers(make_services, settings, gen):
    handlers = Handlers(make_services(router=route_to(), gen=gen))
    handlers.services.search.search = lambda cache, question: [passage()]
    return handlers, make_cache(TEXT, settings)


def test_ask_streams_the_answer_from_the_passages(make_services, settings):
    handlers, cache = passage_handlers(
        make_services, settings, FakeLLMClient([f'It says "{LAW_SENTENCE}".'])
    )
    yields = list(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    written = [y[1][-1]["content"] for y in yields[:-1]]
    assert written[0].startswith(app.WRITING_ANSWER)
    assert any("▌" in text and "It says" in text for text in written)
    assert all(y[1][-2] == {"role": "user", "content": "q"} for y in yields)
    assert "tier-generated" in yields[-1][1][-1]["content"]


def test_ask_keeps_the_other_outputs_while_the_answer_is_written(make_services, settings):
    handlers, cache = passage_handlers(
        make_services, settings, FakeLLMClient([f'It says "{LAW_SENTENCE}".'])
    )
    yields = list(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))
    assert all(is_skip(value) for value in yields[0][2:])
    assert yields[0][0] == ""


def test_ask_with_a_route_yields_once(make_services, settings):
    cache = make_cache(TEXT, settings, {"Governing Law": [(LAW_SENTENCE, 0.95)]})
    handlers = Handlers(make_services(router=route_to("Governing Law"), gen=EchoTextClient()))
    assert len(list(handlers.ask("q", [], cache, TAU_LOW, TAU_HIGH))) == 1


def test_term_sheet_yields_a_status_and_then_the_rows(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    yields = list(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    assert yields[0][2] == app.TERM_SHEET_STATUS
    assert len(yields[-1][0]) == 8 and yields[-1][2] == ""


def test_error_inside_a_generator_handler_becomes_a_warning_and_one_skip(
    make_services, law_cache, warnings, monkeypatch
):
    handlers, _ = generating(make_services, [FAITHFUL_REPLY])
    monkeypatch.setattr(app, "summary_items", lambda *a, **k: 1 / 0)
    yields = list(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert len(yields) == 1 and is_skip(yields[0])
    assert warnings == [app.UNEXPECTED_ERROR]


def test_guarded_generator_handler_stays_a_generator_function(handlers):
    assert inspect.isgeneratorfunction(handlers.summarize)


def stop_functions(demo: gr.Blocks) -> list:
    stop = find_component(demo, "stop")
    return [fn for fn in demo.fns.values() if any(t[0] == stop._id for t in fn.targets)]


def test_stop_button_cancels_the_running_events(make_services):
    demo = build_app(make_services(gen=FakeLLMClient([])))
    [stop_fn] = [fn for fn in stop_functions(demo) if fn.cancels]
    assert find_component(demo, "stop").variant == "stop"
    assert len(stop_fn.cancels) == 1 + len(SUGGESTED_QUESTIONS) + 5


def test_term_status_is_an_output_of_the_term_sheet_button(make_services):
    demo = build_app(make_services(gen=FakeLLMClient([])))
    status = find_component(demo, "term-status")
    button = find_component(demo, "term-sheet")
    [fn] = [f for f in demo.fns.values() if any(t[0] == button._id for t in f.targets)]
    assert status in fn.outputs


def test_keep_warm_loads_every_client_that_can_be_loaded(make_services):
    loaded = []

    class Loadable(FakeLLMClient):
        def load(self):
            loaded.append(self.model)

    router = SimpleNamespace(kind="llm", client=Loadable([], model="router-model"))
    handlers = Handlers(make_services(router=router, gen=Loadable([], model="gen-model")))
    handlers.keep_warm()
    assert loaded == ["gen-model", "router-model"]


def test_keep_warm_skips_clients_that_cannot_be_loaded(make_services):
    Handlers(make_services(gen=FakeLLMClient([]))).keep_warm()


def test_keep_warm_runs_on_a_timer(make_services):
    demo = build_app(make_services())
    timers = [b for b in demo.blocks.values() if isinstance(b, gr.Timer)]
    assert [timer.value for timer in timers] == [app.KEEP_WARM_S]


@pytest.fixture
def background_calls(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(app, "run_in_background", lambda *args: calls.append(args))
    return calls


def upload_new_text(handlers, tmp_path):
    path = tmp_path / "new.txt"
    path.write_text(TEXT, encoding="utf-8")
    return handlers.extract_upload(str(path), progress=lambda *a, **k: None)


def test_upload_of_a_new_text_starts_precomputing_with_a_model(
    make_services, settings, clause_types, tmp_path, background_calls
):
    extractor = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    gen = EchoTextClient()
    upload_new_text(Handlers(make_services(extractor, gen=gen)), tmp_path)
    [(target, cache, names, client, *_)] = background_calls
    assert target is app.precompute and client is gen
    assert cache.text == TEXT and names == [c.name for c in clause_types]


def test_upload_of_a_new_text_does_not_precompute_without_a_model(
    make_services, settings, clause_types, tmp_path, background_calls
):
    extractor = Extractor(slow_fake_qa(0), clause_types, settings, "fake")
    upload_new_text(Handlers(make_services(extractor)), tmp_path)
    assert background_calls == []


def test_ask_passes_the_earlier_turns_to_the_router(make_services, law_cache):
    router = route_to("Governing Law")
    handlers = Handlers(make_services(router=router))
    earlier = [{"question": "who?", "clause_types": ["Parties"]}]
    last(handlers.ask("q", [], law_cache, TAU_LOW, TAU_HIGH, earlier))
    assert router.turns == [[Turn(question="who?", clause_types=["Parties"])]]


def test_ask_appends_a_turn_after_a_routed_question(make_services, law_cache):
    handlers = Handlers(make_services(router=route_to("Governing Law")))
    result = last(handlers.ask("q", [], law_cache, TAU_LOW, TAU_HIGH, []))
    assert result[8] == [{"question": "q", "clause_types": ["Governing Law"]}]


def test_ask_keeps_at_most_eight_turns(make_services, law_cache):
    handlers = Handlers(make_services(router=route_to("Governing Law")))
    earlier = [{"question": f"q{n}", "clause_types": []} for n in range(8)]
    result = last(handlers.ask("q", [], law_cache, TAU_LOW, TAU_HIGH, earlier))
    assert [turn["question"] for turn in result[8]] == [*(f"q{n}" for n in range(1, 8)), "q"]


def test_open_document_resets_the_turns(handlers):
    result = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)
    assert result[11] == []


def test_a_definition_question_calls_no_router_and_adds_no_turn(make_services, demo_cache):
    router = FixedRouter({})
    handlers = Handlers(make_services(router=router))
    result = last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))
    assert router.questions == []
    assert is_skip(result[8])


def test_a_definition_question_shows_the_definition(handlers, demo_cache):
    result = last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))
    _, history, answered, picker, _, segments, _, _, _ = result
    assert answered.definition.term == "RMA"
    assert "tier-definition" in history[-1]["content"]
    assert picker.choices == [("Definition: RMA", "definition")]
    assert highlighted(segments) == [(answered.definition.quote, "Definition")]


def test_pick_citation_shows_the_definition(handlers, demo_cache):
    answered = last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))[2]
    _, segments = handlers.pick_citation("definition", answered, demo_cache)
    assert highlighted(segments) == [(answered.definition.quote, "Definition")]


def test_the_definitions_are_found_once_per_document(handlers, demo_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "find_definitions", lambda *args: calls.append(args) or [])
    last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))
    last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))
    assert len(calls) == 1


def routed_answer(make_services, law_cache):
    router = route_to("Governing Law")
    handlers = Handlers(make_services(router=router, gen=EchoTextClient()))
    answered = last(handlers.ask("q", [], law_cache, TAU_LOW, TAU_HIGH, []))
    return handlers, router, answered


def test_explain_that_streams_an_explanation_into_the_chat(make_services, law_cache):
    handlers, _, first = routed_answer(make_services, law_cache)
    yields = list(
        handlers.ask("explain that", first[1], law_cache, TAU_LOW, TAU_HIGH, [], first[2])
    )
    content = yields[-1][1][-1]["content"]
    assert "The contract states" in content and "a restatement, not a quote" in content
    assert len(yields) > 1
    assert len(yields[-1]) == 9


def test_explain_that_leaves_the_answer_and_turns_alone(make_services, law_cache):
    handlers, router, first = routed_answer(make_services, law_cache)
    result = last(
        handlers.ask("explain that", first[1], law_cache, TAU_LOW, TAU_HIGH, [], first[2])
    )
    assert is_skip(result[2]) and is_skip(result[5]) and is_skip(result[8])
    assert router.questions == ["q"]


def test_explain_with_nothing_shown_says_so(make_services, law_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    result = last(handlers.ask("explain that", [], law_cache, TAU_LOW, TAU_HIGH, [], None))
    assert result[1][-1]["content"] == app.NOTHING_TO_EXPLAIN


def test_explain_that_explains_the_last_definition(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    first = last(handlers.ask("What is an RMA?", [], demo_cache, TAU_LOW, TAU_HIGH, []))
    result = last(
        handlers.ask("explain that", first[1], demo_cache, TAU_LOW, TAU_HIGH, [], first[2])
    )
    assert "a restatement, not a quote" in result[1][-1]["content"]


def test_explain_that_without_a_model_replies_in_the_chat(make_services, law_cache, warnings):
    _, _, first = routed_answer(make_services, law_cache)
    handlers = Handlers(make_services())
    result = last(
        handlers.ask("explain that", first[1], law_cache, TAU_LOW, TAU_HIGH, [], first[2])
    )
    assert result[1][-1]["content"] == app.SUMMARIES_OFF
    assert warnings == []


def test_summarize_in_the_chat_streams_the_summary(make_services, law_cache):
    router = FixedRouter({})
    handlers = Handlers(make_services(router=router, gen=EchoTextClient()))
    yields = list(handlers.ask("summarize", [], law_cache, TAU_LOW, TAU_HIGH, [], None))
    assert "Based on: " in yields[-1][1][-1]["content"]
    assert len(yields) > 1
    assert router.questions == []


def test_summarize_in_the_chat_without_a_model_replies_in_the_chat(make_services, law_cache):
    result = last(Handlers(make_services()).ask("summarize", [], law_cache, TAU_LOW, TAU_HIGH))
    assert result[1][-1]["content"] == app.SUMMARIES_OFF


def test_open_document_fills_the_defined_terms(handlers):
    rows = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)[12]
    assert [row[0] for row in rows] == ["Manufacturer", "Customer", "RMA"]
    assert all(len(row[1]) <= 120 and row[2] == "—" for row in rows)


def test_selecting_a_defined_term_highlights_it_and_shows_the_document(handlers, demo_cache):
    event = SimpleNamespace(index=[2, 0], row_value=["RMA", "", "—"])
    _, segments, tabs = handlers.show_definition(demo_cache, event)
    assert highlighted(segments)[0][1] == "Definition"
    assert "RMA" in highlighted(segments)[0][0]
    assert tabs.selected == DOCUMENT_TAB


def test_the_defined_terms_are_in_a_closed_accordion_with_a_table(make_services):
    demo = build_app(make_services())
    assert find_component(demo, "terms").open is False
    assert find_component(demo, "definitions-table").headers == ["Term", "Defined as", "Page"]


def opened_flags(handlers) -> list[dict]:
    return handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)[14]


def flag_event(title: str) -> SimpleNamespace:
    return SimpleNamespace(index=[0, 1], row_value=["", title, "", ""])


def test_open_document_fills_the_review_rows(handlers):
    rows = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)[13]
    assert rows[0][1] == "No cap on liability found"
    assert "tier-medium" in rows[0][0]
    assert rows[0][2] == "Cap on Liability (not found)"


def test_open_document_keeps_the_flags_and_no_selected_flag(handlers):
    result = handlers.open_document(SHORT_DOC_ID, TAU_LOW, TAU_HIGH, ALL_CLAUSES)
    assert result[14][0]["rule"] == "no_liability_cap" and result[15] is None


def test_the_threshold_handlers_refresh_the_review_rows(handlers, demo_cache):
    low = handlers.release_low(0.5, TAU_HIGH, demo_cache, ALL_CLAUSES)
    high = handlers.release_high(TAU_LOW, 0.95, demo_cache, ALL_CLAUSES)
    _, _, _, rows, flags, selected = handlers.reset_thresholds(demo_cache, ALL_CLAUSES)
    assert low[2][0][1] == high[2][0][1] == rows[0][1] == "No cap on liability found"
    assert low[3][0]["rule"] == flags[0]["rule"] and selected is None


def test_the_threshold_handlers_change_no_review_without_a_document(handlers):
    assert all(
        is_skip(value) for value in handlers.release_low(0.1, 0.9, None, ALL_CLAUSES)[1:]
    )


def test_selecting_a_flag_with_quotes_highlights_it_and_shows_the_document(
    handlers, demo_cache
):
    flags = opened_flags(handlers)
    _, segments, tabs, index = handlers.show_flag(
        flags, demo_cache, flag_event("Automatic renewal with a notice window")
    )
    assert highlighted(segments)[0][1] == "Renewal Term"
    assert tabs.selected == DOCUMENT_TAB and index == 1


def test_selecting_a_flag_without_quotes_gives_an_info(handlers, demo_cache, infos):
    result = handlers.show_flag(
        opened_flags(handlers), demo_cache, flag_event("No cap on liability found")
    )
    assert infos == [app.NO_PASSAGE_ATTACHED] and result[3] == 0


def test_explain_flag_streams_and_ends_with_the_note_line(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    yields = list(handlers.explain_flag(opened_flags(handlers), 1))
    assert len(yields) > 1
    assert "The contract states" in yields[-1]
    assert yields[-1].endswith("</small>") and "not legal advice" in yields[-1]


def test_explain_flag_without_quotes_returns_the_note(make_services):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    [text] = list(handlers.explain_flag(opened_flags(handlers), 0))
    assert text.startswith("A missing cap means") and "No passage is attached" in text


def test_explain_flag_without_a_selection_gives_an_info(handlers, infos):
    assert is_skip(last(handlers.explain_flag(opened_flags(handlers), None)))
    assert infos == [app.NO_FLAG_SELECTED]


def test_explain_flag_without_a_model_warns(handlers, warnings):
    assert is_skip(last(handlers.explain_flag(opened_flags(handlers), 1)))
    assert warnings == [app.SUMMARIES_OFF]


def test_the_review_tab_sits_between_the_summary_and_the_checklist(make_services):
    demo = build_app(make_services())
    ids = [tab.id for tab in demo.blocks.values() if isinstance(tab, gr.Tab)]
    assert ids.index("tab-summary") + 1 == ids.index("tab-review")
    assert ids.index("tab-review") + 1 == ids.index("tab-compare")
    assert ids.index("tab-compare") + 1 == ids.index("tab-portfolio")
    assert ids.index("tab-portfolio") + 1 == ids.index("tab-checklist")


def compare_rows(handlers, cache=None, other=LONG_DOC_ID):
    cache = cache or handlers.services.store.load(SHORT_DOC_ID, handlers.services.settings)
    return last(handlers.compare(cache, other, TAU_LOW, TAU_HIGH))


def test_compare_gives_the_term_and_clause_rows_with_marks(make_services):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    names, terms, clauses, status = compare_rows(handlers)
    assert "Short supply agreement" in names and "Long outsourcing agreement" in names
    assert len(terms) == 8 and len(clauses) == 49 and status == ""
    assert any(row[3] == "≠" for row in terms) and any(row[3] == "≠" for row in clauses)


def test_compare_extracts_a_missing_term_sheet_and_saves_it(make_services, settings):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    compare_rows(handlers)
    assert len(list((settings.cache_dir / "term_sheets").glob("*.json"))) == 2


def test_compare_says_which_contract_is_being_read_while_it_waits(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    first = next(iter(handlers.compare(demo_cache, LONG_DOC_ID, TAU_LOW, TAU_HIGH)))
    assert first[3] == "Extracting the key terms of Short supply agreement"


def test_compare_without_a_model_compares_only_the_clauses(make_services):
    _, terms, clauses, status = compare_rows(Handlers(make_services()))
    assert terms == [] and len(clauses) == 49 and "language model" in status


def test_compare_with_the_same_document_warns(handlers, demo_cache, warnings):
    assert is_skip(last(handlers.compare(demo_cache, SHORT_DOC_ID, TAU_LOW, TAU_HIGH)))
    assert warnings == [app.CHOOSE_DIFFERENT]


def test_compare_without_another_document_or_a_document_warns(handlers, demo_cache, warnings):
    last(handlers.compare(demo_cache, None, TAU_LOW, TAU_HIGH))
    last(handlers.compare(None, LONG_DOC_ID, TAU_LOW, TAU_HIGH))
    assert warnings == [app.CHOOSE_OTHER, app.SELECT_CONTRACT]


def test_compare_with_an_unknown_document_warns(handlers, demo_cache, warnings):
    assert is_skip(last(handlers.compare(demo_cache, "sha256:missing", TAU_LOW, TAU_HIGH)))
    assert warnings == [app.OTHER_UNAVAILABLE]


def test_build_report_files_writes_both_files_from_the_saved_parts(
    make_services, demo_cache, settings
):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    last(handlers.summarize(demo_cache, TAU_LOW, TAU_HIGH))
    last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))
    docx, markdown, status = handlers.build_report_files(demo_cache, TAU_LOW, TAU_HIGH)
    assert docx.visible and markdown.visible and status == app.REPORT_BUILT
    folder = settings.cache_dir / "reports"
    assert sorted(path.suffix for path in folder.iterdir()) == [".docx", ".md"]
    assert "No cap on liability found" in next(folder.glob("*.md")).read_text()


def test_build_report_files_says_what_is_missing(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    docx, _, status = handlers.build_report_files(demo_cache, TAU_LOW, TAU_HIGH)
    assert docx.visible and "the summary and the key terms" in status
    assert "Summary tab" in status


def test_build_report_files_without_a_document_warns(handlers, warnings):
    assert is_skip(handlers.build_report_files(None, TAU_LOW, TAU_HIGH))
    assert warnings == [app.SELECT_CONTRACT]


def test_the_report_buttons_start_hidden(make_services):
    demo = build_app(make_services())
    assert find_component(demo, "report-docx").visible is False
    assert find_component(demo, "report-md").visible is False


def test_term_sheet_returns_the_key_dates_in_a_fourth_output(make_services, demo_cache):
    handlers = Handlers(make_services(gen=EchoTextClient()))
    dates = last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))[3]
    assert "Agreement date" in [row[0] for row in dates]
    assert all(len(row) == 4 for row in dates)


def test_term_sheet_withheld_returns_no_date_rows(make_services, demo_cache, warnings):
    handlers, _ = generating(make_services, ["no json", "still no json"])
    assert last(handlers.term_sheet(demo_cache, TAU_LOW, TAU_HIGH))[3] == []


def test_the_summary_tab_has_the_dates_table_and_its_intro(make_services):
    demo = build_app(make_services())
    assert find_component(demo, "dates-table").headers == ["Date", "When", "From", "How"]
    assert "formula" in find_component(demo, "dates-intro").value


def answered(cache, question="who governs?") -> Answer:
    route = RouteResult(clause_types=["Governing Law"], kind="explicit", latency_ms=1.0)
    return Answer(
        question=question,
        route=route,
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        items=[],
        dropped_quotes=0,
    )


def like(liked: bool = True) -> SimpleNamespace:
    return SimpleNamespace(liked=liked, index=[0, 1], value="x")


def test_feedback_keeps_the_like_data_annotation_gradio_looks_for(handlers):
    assert get_type_hints(handlers.feedback)["evt"] is gr.LikeData


def test_feedback_writes_the_question_and_the_verdict(handlers, demo_cache, settings, infos):
    handlers.feedback(demo_cache, answered(demo_cache), like(False))
    [line] = (settings.cache_dir / FEEDBACK_FILE).read_text().splitlines()
    saved = json.loads(line)
    assert saved["question"] == "who governs?"
    assert saved["liked"] is False
    assert saved["doc_id"] == demo_cache.doc_id
    assert infos == [app.FEEDBACK_THANKS]


def test_feedback_without_an_answer_does_nothing(handlers, demo_cache, settings, infos):
    handlers.feedback(demo_cache, None, like())
    assert not (settings.cache_dir / FEEDBACK_FILE).exists()
    assert infos == []


def test_feedback_without_a_contract_does_nothing(handlers, demo_cache, settings):
    handlers.feedback(None, answered(demo_cache), like())
    assert not (settings.cache_dir / FEEDBACK_FILE).exists()


def test_technical_tab_names_the_feedback_file(make_services):
    texts = component_values(build_app(make_services()))
    assert any("feedback.jsonl" in text for text in texts)


def test_portfolio_gives_one_row_per_cached_document_and_a_csv(handlers, settings):
    rows, button = handlers.portfolio()
    documents = {doc_id for _, doc_id in document_choices(handlers.services.store)}
    assert len(rows) == len(documents)
    assert button.visible is True
    assert (settings.cache_dir / app.PORTFOLIO_CSV).exists()
    assert button.value["orig_name"] == app.PORTFOLIO_CSV


def test_the_portfolio_tab_and_its_components_exist(make_services):
    demo = build_app(make_services())
    assert "tab-portfolio" in [b.id for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert find_component(demo, "portfolio-table").interactive is False
    assert find_component(demo, "portfolio-refresh").value == "Refresh"
    assert find_component(demo, "portfolio-csv").visible is False
    assert find_component(demo, "portfolio-intro").value


def test_summarize_in_serbian_asks_for_serbian_and_saves_under_its_own_store(
    make_services, law_cache, settings
):
    client = EchoTextClient()
    handlers = Handlers(make_services(gen=client))
    result = last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH, "sr"))
    assert "Serbian" in client.text_calls[0][0]
    assert "written in Serbian" in result
    last(handlers.summarize(law_cache, TAU_LOW, TAU_HIGH))
    assert len(client.text_calls) == 2
    assert len(list((settings.cache_dir / "summaries").glob("*.json"))) == 2


def test_summarize_in_serbian_in_the_chat_does_the_same(make_services, law_cache):
    client = EchoTextClient()
    handlers = Handlers(make_services(router=FixedRouter({}), gen=client))
    yields = list(
        handlers.ask("summarize in Serbian", [], law_cache, TAU_LOW, TAU_HIGH, [], None)
    )
    assert "Serbian" in client.text_calls[0][0]
    assert "written in Serbian" in yields[-1][1][-1]["content"]


def test_the_summary_tab_has_a_language_dropdown(make_services):
    dropdown = find_component(build_app(make_services()), "summary-language")
    assert dropdown.value == "en"
    assert ("Serbian", "sr") in dropdown.choices
