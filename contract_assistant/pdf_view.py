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
RESOLUTION = 170
# The whole quote is looked for first; when the text layer differs from the extracted text,
# its opening words are enough to find it, fewer when the line wraps oddly.
SEARCH_WORDS = (14, 8, 4)
# A translucent marker over each line, with no frame: the page reads as highlighted.
FILL = (255, 225, 0, 115)
STROKE = (255, 225, 0, 0)
STROKE_WIDTH = 0
LINE_TOLERANCE = 2.5
PADDING = 1.5


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
    """Return the boxes of the quote on the page, one per line of text, else nothing."""
    words = quote.split()
    counts = [len(words), *(count for count in SEARCH_WORDS if count < len(words))]
    for count in counts:
        pattern = r"\s*".join(re.escape(word) for word in words[:count])
        matches = page.search(pattern, regex=True, case=True)
        if matches:
            return line_boxes(matches[0])
    return []


def line_boxes(match: dict) -> list[dict]:
    """Split one match into a box per line, so a passage reads as marked, not framed."""
    chars = [char for char in match.get("chars", []) if char.get("text", "").strip()]
    if not chars:
        return [{key: match[key] for key in ("x0", "top", "x1", "bottom")}]
    lines: list[dict] = []
    for char in sorted(chars, key=lambda c: (round(c["top"]), c["x0"])):
        line = lines[-1] if lines else None
        if line is None or abs(char["top"] - line["top"]) > LINE_TOLERANCE:
            lines.append(
                {
                    "x0": char["x0"] - PADDING,
                    "top": char["top"] - PADDING,
                    "x1": char["x1"] + PADDING,
                    "bottom": char["bottom"] + PADDING,
                }
            )
            continue
        line["x0"] = min(line["x0"], char["x0"] - PADDING)
        line["x1"] = max(line["x1"], char["x1"] + PADDING)
        line["bottom"] = max(line["bottom"], char["bottom"] + PADDING)
    return lines


def find_all(page: pdfplumber.page.Page, term: str) -> list[dict]:
    """Return the boxes of every occurrence of a term on the page, whatever its case."""
    pattern = pattern_of(term)
    if pattern is None:
        return []
    matches = page.search(pattern.pattern, regex=True, case=False)
    return [box for match in matches for box in line_boxes(match)]


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
