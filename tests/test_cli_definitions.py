import json

import pytest

from contract_assistant import cli
from contract_assistant.answer import Definition
from contract_assistant.cli import main
from contract_assistant.config import ROOT, load_settings
from contract_assistant.store import Store
from tests.support.caches import make_cache

DEMO_TEXT = ROOT / "demo" / "short_supply_agreement.txt"
DEMO_CACHE = ROOT / "demo" / "cache" / "short_supply_agreement.json"
DEMO_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
NO_TERMS_TEXT = "The supplier delivers goods on time. The buyer pays within sixty days."


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def save_cache(tmp_path, text) -> str:
    settings = load_settings()
    cache = make_cache(text, settings)
    Store(tmp_path / "cache", settings.demo_cache_dir).save(cache)
    return cache.doc_id


def test_the_demo_text_gives_a_table_row_for_each_defined_term(capsys):
    assert main(["definitions", str(DEMO_TEXT)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "| Term | Defined as | Page |"
    assert [line.split(" | ")[0] for line in lines[2:]] == [
        "| Manufacturer",
        "| Customer",
        "| RMA",
    ]


def test_a_long_definition_is_cut_to_one_line_with_an_ellipsis(capsys):
    assert main(["definitions", str(DEMO_TEXT)]) == 0
    quote = capsys.readouterr().out.splitlines()[2].split(" | ")[1]
    assert len(quote) == cli.DEFINITION_WIDTH
    assert quote.endswith(cli.ELLIPSIS)


def test_the_page_column_says_n_a_for_a_plain_text_document(capsys):
    assert main(["definitions", str(DEMO_TEXT)]) == 0
    assert capsys.readouterr().out.splitlines()[2].endswith("| n/a |")


def test_json_is_a_list_of_definitions(capsys):
    assert main(["definitions", str(DEMO_TEXT), "--json"]) == 0
    items = [Definition.model_validate(item) for item in json.loads(capsys.readouterr().out)]
    assert [item.term for item in items] == ["Manufacturer", "Customer", "RMA"]


def test_a_contract_without_defined_terms_says_so_and_exits_zero(tmp_path, capsys):
    doc_id = save_cache(tmp_path, NO_TERMS_TEXT)
    assert main(["definitions", doc_id]) == 0
    assert capsys.readouterr().out == "no defined terms found\n"


def test_json_of_a_contract_without_defined_terms_prints_the_message(tmp_path, capsys):
    doc_id = save_cache(tmp_path, NO_TERMS_TEXT)
    assert main(["definitions", doc_id, "--json"]) == 0
    assert capsys.readouterr().out == "no defined terms found\n"


@pytest.mark.parametrize(
    "target",
    [str(DEMO_TEXT), str(DEMO_CACHE), DEMO_DOC_ID],
    ids=["document", "cache_file", "document_id"],
)
def test_target_can_be_a_document_a_cache_file_or_a_document_id(target, capsys):
    assert main(["definitions", target]) == 0
    assert "| RMA |" in capsys.readouterr().out


def test_a_document_without_a_cache_exits_one(tmp_path, capsys):
    path = tmp_path / "contract.txt"
    path.write_text("Preamble. Final words.", encoding="utf-8")
    assert main(["definitions", str(path)]) == 1
    assert "no cache for" in capsys.readouterr().err
