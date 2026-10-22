"""A page of the source PDF as an image, with the cited passage marked on it."""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

import pdfplumber

from contract_assistant.store import DOC_ID_PREFIX
from contract_assistant.textsearch import pattern_of

DOCUMENTS_DIR = "documents"
RESOLUTION = 110
# The first words of a quote are enough to find it on the page; fewer when the line wraps oddly.
SEARCH_WORDS = (14, 8, 4)
FILL = (255, 221, 0, 90)
STROKE = (217, 119, 6)
STROKE_WIDTH = 2


class PageError(ValueError):
    """Raised when the page asked for does not exist in the PDF."""


def copy_path(directory: Path, doc_id: str) -> Path:
    """Return where the copy of a document's PDF is kept."""
    return directory / f"{doc_id.removeprefix(DOC_ID_PREFIX)}.pdf"


def keep_copy(source: Path, directory: Path, doc_id: str) -> Path:
    """Copy an uploaded PDF next to the caches so its pages can be shown later."""
    directory.mkdir(parents=True, exist_ok=True)
    target = copy_path(directory, doc_id)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def page_count(pdf_path: Path) -> int:
    """Return the number of pages of a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def find_quote(page: pdfplumber.page.Page, quote: str) -> list[dict]:
    """Return the boxes of the first match of the quote's opening words on the page."""
    words = quote.split()
    for count in SEARCH_WORDS:
        pattern = r"\s*".join(re.escape(word) for word in words[:count])
        matches = page.search(pattern, regex=True, case=True)
        if matches:
            return [matches[0]]
    return []


def find_all(page: pdfplumber.page.Page, term: str) -> list[dict]:
    """Return the boxes of every occurrence of a term on the page, whatever its case."""
    pattern = pattern_of(term)
    if pattern is None:
        return []
    return page.search(pattern.pattern, regex=True, case=False)


def page_image(
    pdf_path: Path, page_number: int, quote: str | None = None, term: str | None = None
) -> bytes:
    """Render one page as PNG bytes, marking the quote and every occurrence of the term."""
    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= page_number <= len(pdf.pages):
            raise PageError(f"page {page_number} is not in {pdf_path.name}")
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=RESOLUTION)
        boxes = find_quote(page, quote) if quote else []
        if term:
            boxes = [*boxes, *find_all(page, term)]
        if boxes:
            image.draw_rects(boxes, fill=FILL, stroke=STROKE, stroke_width=STROKE_WIDTH)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
