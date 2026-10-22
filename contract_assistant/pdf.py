"""Load contracts from PDF or plain text and map offsets to pages."""

import hashlib
import logging
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber

from contract_assistant import ocr as ocr_module
from contract_assistant.config import load_settings

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n"
NO_TEXT = "no text layer"
NO_TEXT_SCANNED = f"{NO_TEXT}; {ocr_module.INSTALL_HINT}"


@dataclass(frozen=True)
class Document:
    """Extracted text of one contract, with the offset where each page starts."""

    text: str
    page_starts: list[int]
    filename: str
    kind: Literal["pdf", "txt"]
    sha256: str
    # The pages read with OCR because they had no text layer, 1-based.
    ocr_pages: tuple[int, ...] = ()


class UnsupportedFileError(ValueError):
    """Raised for files other than .pdf and .txt."""


class EmptyDocumentError(ValueError):
    """Raised when a PDF has no extractable text."""


def load_document(
    path: Path, ocr: ocr_module.PageReader | bool | None = None
) -> Document:
    """Read a .pdf or .txt file into a Document.

    A PDF page without a text layer is read with `ocr`: a page reader, or None for Tesseract
    with the configured languages when it is installed, or False to leave scans unread.
    """
    suffix = path.suffix.lower()
    ocr_pages: tuple[int, ...] = ()
    if suffix == ".pdf":
        text, page_starts, ocr_pages = _read_pdf(path, ocr)
        kind = "pdf"
    elif suffix == ".txt":
        text, page_starts = _read_txt(path), []
        kind = "txt"
    else:
        raise UnsupportedFileError(f"unsupported file type: {path.suffix}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(text, page_starts, path.name, kind, digest, ocr_pages)


def _read_txt(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnsupportedFileError(f"not a UTF-8 text file: {path.name}") from error


def page_of(doc: Document, offset: int) -> int | None:
    """Return the 1-based page containing offset, or None without page information."""
    if not doc.page_starts:
        return None
    return bisect_right(doc.page_starts, offset)


def _read_pdf(
    path: Path, ocr: ocr_module.PageReader | bool | None
) -> tuple[str, list[int], tuple[int, ...]]:
    pages: list[str] = []
    scanned: list[int] = []
    unread = False
    reader: ocr_module.PageReader | None | bool = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if ocr_module.is_scanned(text) and ocr is not False:
                if reader is None:
                    reader = ocr if callable(ocr) else _default_reader()
                if reader:
                    text = reader(page)
                    scanned.append(page.page_number)
                else:
                    unread = True
            elif ocr_module.is_scanned(text):
                unread = True
            pages.append(text)
    page_starts = []
    position = 0
    for page in pages:
        page_starts.append(position)
        position += len(page) + len(PAGE_SEPARATOR)
    text = PAGE_SEPARATOR.join(pages)
    if not text.strip():
        raise EmptyDocumentError(NO_TEXT_SCANNED if unread else NO_TEXT)
    return text, page_starts, tuple(scanned)


def _default_reader() -> ocr_module.PageReader | bool:
    """Return Tesseract for the configured languages, or False when it cannot run."""
    try:
        return ocr_module.reader(load_settings().ocr_languages)
    except ocr_module.OcrUnavailableError as error:
        logger.warning("scanned pages left unread: %s", error)
        return False
