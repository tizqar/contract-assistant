import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from contract_assistant import ocr
from contract_assistant.pdf import EmptyDocumentError, load_document

LINE = "GOVERNING LAW This agreement is governed by the laws of England."


def scanned_pdf(path, pages=1):
    """Write a PDF whose pages are images of text, without a text layer."""
    images = []
    for number in range(pages):
        image = Image.new("RGB", (1400, 400), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=36)
        draw.text((40, 120), f"Page {number + 1}. {LINE}", fill="black", font=font)
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


def test_a_scanned_pdf_is_read_page_by_page_with_the_reader_given(tmp_path):
    pdf = scanned_pdf(tmp_path / "scan.pdf", pages=2)
    pages_seen = []

    def fake_reader(page):
        pages_seen.append(page.page_number)
        return f"text of page {page.page_number}"

    document = load_document(pdf, ocr=fake_reader)
    assert document.text == "text of page 1\ntext of page 2"
    assert document.page_starts == [0, 15]
    assert document.ocr_pages == (1, 2)
    assert pages_seen == [1, 2]


def test_a_scanned_pdf_without_a_reader_is_an_empty_document(tmp_path):
    pdf = scanned_pdf(tmp_path / "scan.pdf")
    with pytest.raises(EmptyDocumentError, match="Tesseract"):
        load_document(pdf, ocr=False)


def test_is_scanned_looks_at_the_length_of_the_text_layer():
    assert ocr.is_scanned("")
    assert ocr.is_scanned("  3 \n")
    assert not ocr.is_scanned("x" * ocr.MIN_CHARS)


def test_usable_languages_keeps_only_the_installed_ones(monkeypatch):
    monkeypatch.setattr(ocr, "installed_languages", lambda: ("eng", "osd"))
    assert ocr.usable_languages("eng+srp_latn+srp") == "eng"
    with pytest.raises(ocr.OcrUnavailableError, match="tesseract-lang"):
        ocr.usable_languages("srp")


def test_without_tesseract_there_are_no_languages(monkeypatch):
    ocr.installed_languages.cache_clear()
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    assert ocr.installed_languages() == ()
    ocr.installed_languages.cache_clear()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")
def test_tesseract_reads_a_scanned_page(tmp_path):
    pdf = scanned_pdf(tmp_path / "scan.pdf")
    document = load_document(pdf, ocr=ocr.reader("eng"))
    assert "governed by the laws of England" in document.text
    assert document.ocr_pages == (1,)
