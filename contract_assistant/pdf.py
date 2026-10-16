"""Load contracts from PDF or plain text and map offsets to pages."""

import hashlib
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pdfplumber

PAGE_SEPARATOR = "\n"


@dataclass(frozen=True)
class Document:
    """Extracted text of one contract, with the offset where each page starts."""

    text: str
    page_starts: list[int]
    filename: str
    kind: Literal["pdf", "txt"]
    sha256: str


class UnsupportedFileError(ValueError):
    """Raised for files other than .pdf and .txt."""


class EmptyDocumentError(ValueError):
    """Raised when a PDF has no extractable text."""


def load_document(path: Path) -> Document:
    """Read a .pdf or .txt file into a Document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, page_starts = _read_pdf(path)
        kind = "pdf"
    elif suffix == ".txt":
        text, page_starts = _read_txt(path), []
        kind = "txt"
    else:
        raise UnsupportedFileError(f"unsupported file type: {path.suffix}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Document(text, page_starts, path.name, kind, digest)


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


def _read_pdf(path: Path) -> tuple[str, list[int]]:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    page_starts = []
    position = 0
    for page in pages:
        page_starts.append(position)
        position += len(page) + len(PAGE_SEPARATOR)
    text = PAGE_SEPARATOR.join(pages)
    if not text.strip():
        raise EmptyDocumentError("no text layer; scanned PDFs are not supported")
    return text, page_starts
