import pdfplumber
import pytest
from fpdf import FPDF

from contract_assistant.pdf import load_document
from contract_assistant.pdf_view import (
    PageError,
    copy_path,
    find_quote,
    keep_copy,
    page_count,
    page_image,
)

SENTENCE_ONE = "This agreement shall be governed by the laws of England and Wales."
SENTENCE_TWO = "The supplier delivers the goods within thirty days of each order."
FILLER = "The parties agree to the terms set out below and sign in good faith."
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DOC_ID = "sha256:" + "ab12" * 16
SHORTER_TAIL = "This agreement shall be governed by the laws of France and Germany plus more words"
SHORTEST_TAIL = "This agreement shall be ruled by the courts of France and Germany today"
NOT_ON_PAGE = "Payment falls due on the last day of every calendar month"


def write_pdf(path, pages):
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    pdf.output(str(path))
    return path


@pytest.fixture
def pdf_path(tmp_path):
    pages = [f"{FILLER}\n{SENTENCE_ONE}\n{FILLER}", f"{FILLER}\n{SENTENCE_TWO}"]
    return write_pdf(tmp_path / "contract.pdf", pages)


def boxes_on(pdf_path, page_number, quote):
    with pdfplumber.open(pdf_path) as pdf:
        return find_quote(pdf.pages[page_number - 1], quote)


def test_page_count_is_two(pdf_path):
    assert page_count(pdf_path) == 2


def test_find_quote_gives_one_box_for_the_known_sentence(pdf_path):
    assert len(boxes_on(pdf_path, 1, SENTENCE_ONE)) == 1


def test_found_box_has_positive_width_and_height(pdf_path):
    box = boxes_on(pdf_path, 1, SENTENCE_ONE)[0]
    assert box["x0"] < box["x1"]
    assert box["top"] < box["bottom"]


def test_found_box_lies_within_the_page(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        box = find_quote(page, SENTENCE_ONE)[0]
        assert 0 <= box["x0"] and box["x1"] <= page.width
        assert 0 <= box["top"] and box["bottom"] <= page.height


def test_quote_with_changed_tail_is_found_by_the_eight_word_fallback(pdf_path):
    assert len(boxes_on(pdf_path, 1, SHORTER_TAIL)) == 1


def test_quote_matching_only_four_words_is_found_by_the_shortest_fallback(pdf_path):
    assert len(boxes_on(pdf_path, 1, SHORTEST_TAIL)) == 1


def test_quote_without_words_on_the_page_gives_no_boxes(pdf_path):
    assert boxes_on(pdf_path, 1, NOT_ON_PAGE) == []


def test_repeated_words_give_only_the_first_match(tmp_path):
    path = write_pdf(tmp_path / "twice.pdf", [f"{SENTENCE_ONE}\n{FILLER}\n{SENTENCE_ONE}"])
    assert len(boxes_on(path, 1, SENTENCE_ONE)) == 1


def test_first_match_is_the_upper_one(tmp_path):
    path = write_pdf(tmp_path / "twice.pdf", [f"{SENTENCE_ONE}\n{FILLER}\n{SENTENCE_ONE}"])
    with pdfplumber.open(path) as pdf:
        every = pdf.pages[0].search(SENTENCE_ONE)
    assert boxes_on(path, 1, SENTENCE_ONE)[0]["top"] == min(box["top"] for box in every)


def test_page_image_without_quote_is_a_png(pdf_path):
    assert page_image(pdf_path, 1).startswith(PNG_SIGNATURE)


def test_page_image_with_quote_is_a_png(pdf_path):
    assert page_image(pdf_path, 1, SENTENCE_ONE).startswith(PNG_SIGNATURE)


def test_boxed_quote_changes_the_image(pdf_path):
    assert page_image(pdf_path, 1, SENTENCE_ONE) != page_image(pdf_path, 1)


def test_quote_absent_from_the_page_gives_the_plain_image(pdf_path):
    assert page_image(pdf_path, 1, NOT_ON_PAGE) == page_image(pdf_path, 1)


def test_page_zero_is_an_error(pdf_path):
    with pytest.raises(PageError):
        page_image(pdf_path, 0)


def test_page_after_the_last_is_an_error(pdf_path):
    with pytest.raises(PageError):
        page_image(pdf_path, 3)


def test_quote_is_boxed_on_its_own_page(pdf_path):
    assert page_image(pdf_path, 2, SENTENCE_TWO) != page_image(pdf_path, 2)


def test_quote_is_not_searched_on_another_page(pdf_path):
    assert page_image(pdf_path, 1, SENTENCE_TWO) == page_image(pdf_path, 1)


def test_copy_path_is_the_doc_id_without_prefix_and_a_pdf_suffix(tmp_path):
    assert copy_path(tmp_path, DOC_ID) == tmp_path / f"{'ab12' * 16}.pdf"


def test_keep_copy_copies_the_file_to_the_target(pdf_path, tmp_path):
    target = keep_copy(pdf_path, tmp_path / "kept", DOC_ID)
    assert target.read_bytes() == pdf_path.read_bytes()


def test_keep_copy_returns_the_copy_path(pdf_path, tmp_path):
    directory = tmp_path / "kept"
    assert keep_copy(pdf_path, directory, DOC_ID) == copy_path(directory, DOC_ID)


def test_keep_copy_creates_the_directory(pdf_path, tmp_path):
    directory = tmp_path / "deep" / "kept"
    keep_copy(pdf_path, directory, DOC_ID)
    assert directory.is_dir()


def test_keep_copy_leaves_the_source_in_place(pdf_path, tmp_path):
    before = pdf_path.read_bytes()
    keep_copy(pdf_path, tmp_path / "kept", DOC_ID)
    assert pdf_path.read_bytes() == before


def test_keep_copy_onto_itself_leaves_the_file_intact(pdf_path, tmp_path):
    directory = tmp_path / "kept"
    target = keep_copy(pdf_path, directory, DOC_ID)
    before = target.read_bytes()
    assert keep_copy(target, directory, DOC_ID) == target
    assert target.read_bytes() == before


def test_page_views_agree_with_the_page_starts_of_the_cache(pdf_path):
    doc = load_document(pdf_path)
    assert len(doc.page_starts) == 2
    assert doc.text.index(SENTENCE_TWO) >= doc.page_starts[1]
