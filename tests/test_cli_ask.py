import dataclasses

import pytest

from contract_assistant import cli
from contract_assistant.answer import EMPTY_ROUTE_MESSAGE, STATUS_WORDS, Answer
from contract_assistant.clause_types import app_clause_types
from contract_assistant.cli import main
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.router.base import RouteResult
from contract_assistant.router.factory import UnknownBackendError
from contract_assistant.store import Store
from tests.support.caches import make_cache
from tests.support.fakes import FakeEncoder, FixedRouter

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"
DEMO_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
QUESTION = "who are the parties and which law governs?"
LATENCY_MS = 12.6


def refuse_router(*arguments):
    raise AssertionError("no router may be built")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "build_router", refuse_router)


@pytest.fixture
def contract(tmp_path):
    path = tmp_path / "contract.txt"
    path.write_text("Preamble. Final words.", encoding="utf-8")
    return path


def fixed_router(kind: str) -> FixedRouter:
    route = RouteResult(
        clause_types=["Parties", "Governing Law"], kind=kind, latency_ms=LATENCY_MS
    )
    router = FixedRouter({QUESTION: route})
    router.kind = kind
    return router


def install_router(monkeypatch, router) -> list[Settings]:
    received = []

    def build(settings, clause_types):
        received.append(settings)
        return router

    monkeypatch.setattr(cli, "build_router", build)
    return received


def clause_names() -> list[str]:
    return [c.name for c in app_clause_types(load_settings())]


def headings(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("**")]


def answered_types(output: str) -> list[str]:
    return [line.split("**")[1] for line in headings(output)]


def test_types_print_each_type_with_its_status_word(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Governing Law,Parties"]) == 0
    statuses = [line.split(" — ")[1] for line in headings(capsys.readouterr().out)]
    assert len(statuses) == 2
    assert all(status.startswith(tuple(STATUS_WORDS.values())) for status in statuses)


def test_types_give_a_found_item_and_a_not_found_item(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Governing Law,Source Code Escrow"]) == 0
    output = capsys.readouterr().out
    assert "**Governing Law** — found" in output
    assert "**Source Code Escrow** — not found" in output


def test_type_names_are_stripped_of_spaces(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", " Governing Law ,  Parties "]) == 0
    assert answered_types(capsys.readouterr().out) == ["Governing Law", "Parties"]


def test_items_follow_the_order_of_the_given_types(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Parties,Governing Law"]) == 0
    assert answered_types(capsys.readouterr().out) == ["Parties", "Governing Law"]


def test_unknown_type_exits_two_and_lists_the_valid_names(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Governing Laws"]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "error: unknown clause types: Governing Laws"
    assert lines[1:] == clause_names()


def test_every_unknown_type_is_reported(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Governing Law,Nope,Never"]) == 2
    assert capsys.readouterr().err.startswith("error: unknown clause types: Nope, Never\n")


def test_types_build_no_router(monkeypatch):
    received = install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), "--types", "Parties"]) == 0
    assert received == []


def test_types_print_no_router_line(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Parties"]) == 0
    assert "router:" not in capsys.readouterr().out


def test_json_output_is_an_answer_with_an_explicit_route(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Governing Law", "--json"]) == 0
    result = Answer.model_validate_json(capsys.readouterr().out)
    assert result.route.kind == "explicit"
    assert [item.clause_type for item in result.items] == ["Governing Law"]


def test_question_is_empty_when_only_types_are_given(capsys):
    assert main(["ask", str(DEMO_TEXT), "--types", "Parties", "--json"]) == 0
    assert Answer.model_validate_json(capsys.readouterr().out).question == ""


def test_question_is_kept_when_given_with_types(capsys):
    arguments = ["ask", str(DEMO_TEXT), "which law?", "--types", "Governing Law", "--json"]
    assert main(arguments) == 0
    assert Answer.model_validate_json(capsys.readouterr().out).question == "which law?"


def test_answer_uses_the_configured_thresholds(monkeypatch, capsys):
    monkeypatch.setenv("TAU_LOW", "0.2")
    monkeypatch.setenv("TAU_HIGH", "0.5")
    assert main(["ask", str(DEMO_TEXT), "--types", "Parties", "--json"]) == 0
    result = Answer.model_validate_json(capsys.readouterr().out)
    assert (result.tau_low, result.tau_high) == (0.2, 0.5)


@pytest.mark.parametrize(
    "target",
    [str(DEMO_TEXT), str(DEMO_CACHE), DEMO_DOC_ID],
    ids=["document", "cache_file", "document_id"],
)
def test_target_can_be_a_document_a_cache_file_or_a_document_id(target, capsys):
    assert main(["ask", target, "--types", "Parties"]) == 0
    assert answered_types(capsys.readouterr().out) == ["Parties"]


def test_question_is_routed_by_the_built_router(monkeypatch, capsys):
    install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 0
    assert answered_types(capsys.readouterr().out) == ["Parties", "Governing Law"]


def test_router_line_follows_the_answer_after_a_blank_line(monkeypatch, capsys):
    install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == ["", "router: embedding · 13 ms"]


def test_router_line_rounds_the_latency_to_whole_milliseconds(monkeypatch, capsys):
    install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 0
    assert "· 13 ms" in capsys.readouterr().out


def test_json_output_of_a_routed_question_has_no_router_line(monkeypatch, capsys):
    install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION, "--json"]) == 0
    assert Answer.model_validate_json(capsys.readouterr().out).route.kind == "embedding"


@pytest.mark.parametrize("kind", ["embedding", "llm"])
def test_without_the_router_option_the_built_router_is_used(kind, monkeypatch, capsys):
    install_router(monkeypatch, fixed_router(kind))
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 0
    assert f"router: {kind} · " in capsys.readouterr().out


def test_router_embedding_builds_the_router_with_the_llm_backend_off(monkeypatch):
    received = install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION, "--router", "embedding"]) == 0
    assert [settings.llm_backend for settings in received] == ["none"]


def test_router_llm_builds_the_router_from_the_settings_as_they_are(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    received = install_router(monkeypatch, fixed_router("llm"))
    assert main(["ask", str(DEMO_TEXT), QUESTION, "--router", "llm"]) == 0
    assert [settings.llm_backend for settings in received] == ["openai"]


def test_router_llm_with_another_kind_of_router_exits_one_naming_the_settings(
    monkeypatch, capsys
):
    install_router(monkeypatch, fixed_router("embedding"))
    assert main(["ask", str(DEMO_TEXT), QUESTION, "--router", "llm"]) == 1
    captured = capsys.readouterr()
    assert "LLM_BACKEND" in captured.err
    assert "LLM_MODEL" in captured.err
    assert captured.out == ""


def test_unknown_llm_backend_exits_one_with_the_error(monkeypatch, capsys):
    def reject(settings, clause_types):
        raise UnknownBackendError("unknown LLM_BACKEND 'olama'")

    monkeypatch.setattr(cli, "build_router", reject)
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 1
    assert capsys.readouterr().err == "error: unknown LLM_BACKEND 'olama'\n"


def test_types_together_with_router_exits_two():
    with pytest.raises(SystemExit) as raised:
        main(["ask", str(DEMO_TEXT), "--types", "Parties", "--router", "embedding"])
    assert raised.value.code == 2


@pytest.mark.parametrize("arguments", [[], [""], ["  "]], ids=["missing", "empty", "blank"])
def test_missing_question_without_types_exits_two(arguments, capsys):
    assert main(["ask", str(DEMO_TEXT), *arguments]) == 2
    assert "question" in capsys.readouterr().err


def test_document_without_a_cache_exits_one(contract, capsys):
    assert main(["ask", str(contract), "--types", "Parties"]) == 1
    assert "no cache for" in capsys.readouterr().err


def test_unsupported_document_exits_one(tmp_path, capsys):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    assert main(["ask", str(path), "--types", "Parties"]) == 1
    assert "unsupported file type" in capsys.readouterr().err


def test_cache_written_with_other_settings_exits_one(contract, tmp_path, capsys):
    settings = load_settings()
    text = contract.read_text(encoding="utf-8")
    stale = make_cache(text, dataclasses.replace(settings, extractor_name="other"))
    Store(tmp_path / "cache", settings.demo_cache_dir).save(stale)
    assert main(["ask", str(contract), "--types", "Parties"]) == 1
    assert "other settings" in capsys.readouterr().err


PAYMENT_CLAUSE = "Invoices are payable within sixty days of receipt."
PAYMENT_TEXT = (
    f"{PAYMENT_CLAUSE} The buyer pays every invoice by bank transfer to the supplier."
)
PAYMENT_QUESTION = "invoices are payable within sixty days of receipt"
PAYMENT_PASSAGE_LINE = "> Invoices are payable within sixty days of receipt. The buyer pays"


def save_cache(tmp_path, text, clauses, page_starts=None) -> str:
    settings = load_settings()
    cache = make_cache(text, settings, clauses)
    cache.page_starts = page_starts or []
    Store(tmp_path / "cache", settings.demo_cache_dir).save(cache)
    return cache.doc_id


def install_empty_router(monkeypatch, question) -> None:
    route = RouteResult(clause_types=[], kind="embedding", latency_ms=LATENCY_MS)
    router = FixedRouter({question: route})
    router.kind = "embedding"
    install_router(monkeypatch, router)
    monkeypatch.setattr(cli, "load_encoder", lambda model_id: FakeEncoder())


def test_app_commands_build_the_router_over_the_49_app_types(monkeypatch):
    seen = []
    router = fixed_router("embedding")
    monkeypatch.setattr(
        cli, "build_router", lambda settings, types: seen.append(types) or router
    )
    assert main(["ask", str(DEMO_TEXT), QUESTION]) == 0
    assert [[t.name for t in types] for types in seen] == [clause_names()]
    assert len(seen[0]) == 49


def test_explicit_extra_type_is_capped_at_check(tmp_path, capsys):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Payment Terms": [(PAYMENT_CLAUSE, 0.99)]})
    assert main(["ask", doc_id, "--types", "Payment Terms", "--json"]) == 0
    result = Answer.model_validate_json(capsys.readouterr().out)
    assert result.items[0].status == "check"


def test_cuad_type_is_not_capped_at_check(tmp_path, capsys):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Parties": [(PAYMENT_CLAUSE, 0.99)]})
    assert main(["ask", doc_id, "--types", "Parties", "--json"]) == 0
    assert Answer.model_validate_json(capsys.readouterr().out).items[0].status == "found"


def test_types_missing_from_an_older_cache_are_dropped_from_the_answer(tmp_path, capsys):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Parties": []})
    assert main(["ask", doc_id, "--types", "Parties,Payment Terms", "--json"]) == 0
    result = Answer.model_validate_json(capsys.readouterr().out)
    assert [item.clause_type for item in result.items] == ["Parties"]


def test_empty_route_shows_the_closest_passages_after_the_message(
    tmp_path, monkeypatch, capsys
):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Parties": []})
    install_empty_router(monkeypatch, PAYMENT_QUESTION)
    assert main(["ask", doc_id, PAYMENT_QUESTION]) == 0
    output = capsys.readouterr().out
    assert output.index(EMPTY_ROUTE_MESSAGE) < output.index(cli.PASSAGES_MESSAGE)
    assert PAYMENT_PASSAGE_LINE in output


def test_passage_is_preceded_by_its_page(tmp_path, monkeypatch, capsys):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Parties": []}, page_starts=[0])
    install_empty_router(monkeypatch, PAYMENT_QUESTION)
    assert main(["ask", doc_id, PAYMENT_QUESTION]) == 0
    assert f"p. 1\n{PAYMENT_PASSAGE_LINE}" in capsys.readouterr().out


def test_empty_route_puts_the_passages_in_the_json_answer(tmp_path, monkeypatch, capsys):
    doc_id = save_cache(tmp_path, PAYMENT_TEXT, {"Parties": []})
    install_empty_router(monkeypatch, PAYMENT_QUESTION)
    assert main(["ask", doc_id, PAYMENT_QUESTION, "--json"]) == 0
    result = Answer.model_validate_json(capsys.readouterr().out)
    assert [passage.quote for passage in result.passages] == [PAYMENT_TEXT]


def test_routed_question_searches_no_passages(monkeypatch, capsys):
    install_router(monkeypatch, fixed_router("embedding"))
    monkeypatch.setattr(cli, "load_encoder", refuse_router)
    assert main(["ask", str(DEMO_TEXT), QUESTION, "--json"]) == 0
    assert Answer.model_validate_json(capsys.readouterr().out).passages == []
