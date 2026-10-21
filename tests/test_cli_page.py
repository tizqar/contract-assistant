import pytest
from fpdf import FPDF

from contract_assistant import cli
from contract_assistant.clause_types import app_clause_types
from contract_assistant.cli import main
from contract_assistant.config import load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.pdf import load_document
from contract_assistant.store import doc_id_of
from tests.support.fakes import FakeQA

SENTENCE = "This agreement is governed by the laws of Vanuatu."
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


@pytest.fixture(autouse=True)
def fake_qa(monkeypatch):
    settings = load_settings()
    clause_types = app_clause_types(settings)
    question = next(c for c in clause_types if c.name == "Governing Law").cuad_question
    qa = FakeQA({question: [(SENTENCE, 0.9)]})
    monkeypatch.setattr(
        Extractor,
        "from_settings",
        classmethod(lambda cls, s: Extractor(qa, clause_types, s, "cpu")),
    )


@pytest.fixture
def pdf(tmp_path):
    document = FPDF()
    document.set_font("Helvetica", size=12)
    for line in ("Preamble of the contract.", SENTENCE):
        document.add_page()
        document.cell(text=line)
    path = tmp_path / "contract.pdf"
    document.output(str(path))
    return path


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "contract.txt"
    path.write_text(f"Preamble. {SENTENCE} Final words.", encoding="utf-8")
    return path


def _copy(tmp_path, source):
    doc_id = doc_id_of(load_document(source).text)
    return tmp_path / "cache" / "documents" / f"{doc_id.removeprefix('sha256:')}.pdf"


def test_extract_of_a_pdf_keeps_a_copy(pdf, tmp_path):
    assert main(["extract", str(pdf)]) == 0
    assert _copy(tmp_path, pdf).read_bytes() == pdf.read_bytes()


def test_extract_of_a_text_file_keeps_no_copy(text_file, tmp_path):
    assert main(["extract", str(text_file)]) == 0
    assert not (tmp_path / "cache" / "documents").exists()


def test_page_writes_a_png_under_the_cache_and_prints_its_path(pdf, tmp_path, capsys):
    main(["extract", str(pdf)])
    capsys.readouterr()
    assert main(["page", str(pdf), "2", "--quote", SENTENCE]) == 0
    written = list((tmp_path / "cache" / "pages").glob("*-2.png"))
    assert len(written) == 1
    assert written[0].read_bytes().startswith(PNG_SIGNATURE)
    assert str(written[0]) in capsys.readouterr().out


def test_page_writes_to_the_out_file(pdf, tmp_path):
    main(["extract", str(pdf)])
    out = tmp_path / "shot" / "page.png"
    assert main(["page", str(pdf), "1", "--out", str(out)]) == 0
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_page_beyond_the_last_page_returns_one(pdf, capsys):
    main(["extract", str(pdf)])
    capsys.readouterr()
    assert main(["page", str(pdf), "5"]) == 1
    assert "error: page 5 is not in" in capsys.readouterr().err


def test_page_of_a_text_file_returns_one(text_file, capsys):
    main(["extract", str(text_file)])
    capsys.readouterr()
    assert main(["page", str(text_file), "1"]) == 1
    err = capsys.readouterr().err
    assert "error: contract.txt is a text file; it has no pages to show" in err


def test_page_without_the_pdf_copy_returns_one(pdf, tmp_path, capsys):
    main(["extract", str(pdf)])
    _copy(tmp_path, pdf).unlink()
    capsys.readouterr()
    assert main(["page", str(pdf), "1"]) == 1
    assert "error: no PDF copy of this document; run extract on the PDF again" in (
        capsys.readouterr().err
    )
