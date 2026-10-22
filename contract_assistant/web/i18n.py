"""Two interface languages: English, the source of every string, and Serbian from a catalogue.

Every string the pages show is written in English in the templates, the modules and
`copy.py`. The Serbian versions live in `web/translations/sr_*.py`, one catalogue per area,
keyed by the English text. `translate` looks a string up for a language; `copy_for` returns
the vocabulary of `copy.py` in a language; the language of a request is a cookie.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import cache
from types import SimpleNamespace
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from contract_assistant.web import copy, translations

logger = logging.getLogger(__name__)

ENGLISH = "en"
SERBIAN = "sr"
LANGUAGES = (ENGLISH, SERBIAN)
LANGUAGE_NAMES = {ENGLISH: "English", SERBIAN: "Srpski"}
COOKIE = "lang"
COOKIE_MAX_AGE = 365 * 24 * 3600
SEE_OTHER = 303

router = APIRouter()


@cache
def catalogue(lang: str) -> dict[str, str]:
    """Return the translations of a language, merged from its catalogue modules."""
    merged: dict[str, str] = {}
    prefix = f"{lang}_"
    for module in pkgutil.iter_modules(translations.__path__):
        if module.name.startswith(prefix):
            loaded = importlib.import_module(f"{translations.__name__}.{module.name}")
            merged.update(getattr(loaded, "CATALOGUE", {}))
    return merged


def translate(text: str, lang: str = ENGLISH) -> str:
    """Return the text in a language; English, or a missing entry, returns the text itself."""
    if lang == ENGLISH or not text:
        return text
    return catalogue(lang).get(text, text)


# Strings of copy.py that are identifiers, not words shown: the section ids.
IDENTIFIERS = frozenset(section[0] for section in copy.SECTIONS)


def missing(lang: str, texts: list[str]) -> list[str]:
    """Return the texts a language has no translation for."""
    known = catalogue(lang)
    return [text for text in texts if text and text not in known and text not in IDENTIFIERS]


def _translated(value, lang: str):
    if isinstance(value, str):
        return translate(value, lang)
    if isinstance(value, tuple):
        return tuple(_translated(item, lang) for item in value)
    if isinstance(value, list):
        return [_translated(item, lang) for item in value]
    if isinstance(value, dict):
        return {key: _translated(item, lang) for key, item in value.items()}
    return value


@cache
def copy_for(lang: str) -> SimpleNamespace:
    """Return the vocabulary of copy.py with every string in the language."""
    names = {name: getattr(copy, name) for name in dir(copy) if name.isupper()}
    return SimpleNamespace(**{name: _translated(value, lang) for name, value in names.items()})


def copy_strings() -> list[str]:
    """Return every string of copy.py, for the check that a language covers them all."""
    found: list[str] = []

    def walk(value) -> None:
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for name in dir(copy):
        if name.isupper():
            walk(getattr(copy, name))
    return found


def language_of(request: Request) -> str:
    """Return the language of a request from its cookie, English by default."""
    lang = request.cookies.get(COOKIE, ENGLISH)
    return lang if lang in LANGUAGES else ENGLISH


@router.post("/language")
def set_language(lang: Annotated[str, Form()], next: Annotated[str, Form()] = "/"):
    """Remember the interface language in a cookie and go back to the page."""
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    response = RedirectResponse(target, status_code=SEE_OTHER)
    if lang in LANGUAGES:
        response.set_cookie(COOKIE, lang, max_age=COOKIE_MAX_AGE, samesite="lax")
    return response
