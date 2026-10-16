import hashlib
from pathlib import Path

import pytest
from fpdf import FPDF

from contract_assistant.pdf import (
    EmptyDocumentError,
    UnsupportedFileError,
    load_document,
    page_of,
)
from tests.conftest import ROOT

PAGE_LINES = [
    "Alpha supplies the goods.",
    "Beta pays within thirty days.",
    "Gamma governs this agreement.",
]
DEMO = ROOT / "demo" / "short_supply_agreement.txt"


def write_pdf(path: Path, lines: list[str]) -> Path:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        pdf.add_page()
        if line:
            pdf.cell(text=line)
    pdf.output(str(path))
    return path


@pytest.fixture
def pdf_doc(tmp_path):
    return load_document(write_pdf(tmp_path / "contract.pdf", PAGE_LINES))


def test_page_starts_has_one_entry_per_page_beginning_at_zero(pdf_doc):
    assert len(pdf_doc.page_starts) == 3
    assert pdf_doc.page_starts[0] == 0


def test_offset_at_second_page_start_is_page_two(pdf_doc):
    assert page_of(pdf_doc, pdf_doc.page_starts[1]) == 2


def test_separator_before_second_page_belongs_to_page_one(pdf_doc):
    assert page_of(pdf_doc, pdf_doc.page_starts[1] - 1) == 1


def test_last_character_is_on_the_last_page(pdf_doc):
    assert page_of(pdf_doc, len(pdf_doc.text) - 1) == 3


@pytest.mark.parametrize("page, line", list(enumerate(PAGE_LINES, start=1)))
def test_known_line_is_found_on_its_page(pdf_doc, page, line):
    assert page_of(pdf_doc, pdf_doc.text.index(line)) == page


def test_txt_document_has_no_pages():
    doc = load_document(DEMO)
    assert doc.kind == "txt"
    assert page_of(doc, 0) is None


def test_txt_text_is_the_file_decoded_without_changes():
    doc = load_document(DEMO)
    assert doc.text == DEMO.read_bytes().decode("utf-8")


def test_txt_hash_is_sha256_of_the_text_bytes():
    doc = load_document(DEMO)
    assert doc.sha256 == hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


def test_txt_keeps_carriage_returns(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"one\r\ntwo\r\n")
    assert load_document(path).text == "one\r\ntwo\r\n"


def test_pdf_without_text_is_rejected(tmp_path):
    path = write_pdf(tmp_path / "blank.pdf", [""])
    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_docx_is_rejected(tmp_path):
    with pytest.raises(UnsupportedFileError):
        load_document(tmp_path / "contract.docx")
