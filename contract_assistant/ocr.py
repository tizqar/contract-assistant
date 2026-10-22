"""Text of scanned PDF pages, read with Tesseract when a page has no text layer.

A page whose text layer is shorter than MIN_CHARS is treated as a scan and rendered to an
image for Tesseract. The languages come from OCR_LANGUAGES (default English and Serbian in
both scripts); the ones not installed are dropped, and when none is left the page cannot be
read and the caller says so.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from functools import cache

import pdfplumber

logger = logging.getLogger(__name__)

RESOLUTION = 300
MIN_CHARS = 25
DEFAULT_LANGUAGES = "eng+srp_latn+srp"
TESSERACT = "tesseract"
INSTALL_HINT = (
    "scanned pages need Tesseract with its language packs "
    "(brew install tesseract tesseract-lang)"
)

PageReader = Callable[[pdfplumber.page.Page], str]


class OcrUnavailableError(RuntimeError):
    """Raised when a scanned page needs OCR and Tesseract or its languages are missing."""


def is_scanned(text: str) -> bool:
    """Return whether a page's text layer is too short to be real text."""
    return len(text.strip()) < MIN_CHARS


@cache
def installed_languages() -> tuple[str, ...]:
    """Return the Tesseract languages installed on this computer, none without Tesseract."""
    if shutil.which(TESSERACT) is None:
        return ()
    try:
        import pytesseract

        return tuple(pytesseract.get_languages(config=""))
    except (OSError, RuntimeError) as error:  # the binding raises subclasses of these
        logger.warning("tesseract languages could not be listed: %s", error)
        return ()


def usable_languages(requested: str = DEFAULT_LANGUAGES) -> str:
    """Return the requested languages that are installed, joined for Tesseract."""
    installed = set(installed_languages())
    kept = [code for code in requested.split("+") if code in installed]
    if not kept:
        raise OcrUnavailableError(f"{INSTALL_HINT}; requested {requested}")
    return "+".join(kept)


def read_page(page: pdfplumber.page.Page, languages: str) -> str:
    """Return the text Tesseract reads on a page rendered at RESOLUTION."""
    import pytesseract

    image = page.to_image(resolution=RESOLUTION).original
    return pytesseract.image_to_string(image, lang=languages)


def reader(requested: str = DEFAULT_LANGUAGES) -> PageReader:
    """Return a page reader for the requested languages, or raise when none is installed."""
    languages = usable_languages(requested)

    def read(page: pdfplumber.page.Page) -> str:
        return read_page(page, languages)

    return read
