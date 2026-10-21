"""Defined terms of a contract: the term and the sentence that defines it, found by pattern."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence

from contract_assistant.answer import Definition

TERM = r"[A-Z][A-Za-z0-9&/\-]*(?: [A-Za-z0-9&/\-]+){0,5}"
QUOTE_OPEN = "[\"'“‘]"
QUOTE_CLOSE = "[\"'”’]"
# ("Term"), (the "Term"), (hereinafter the "Term"), ('the Term')
PARENTHETICAL = re.compile(
    rf"\((?:hereinafter (?:referred to as )?)?(?:the |this |each an? )?{QUOTE_OPEN}"
    rf"(?:the |this )?(?P<term>{TERM}){QUOTE_CLOSE}\)"
)
# "Term" means ..., "Term" shall mean ..., Term means ...
MEANS = re.compile(
    rf"(?:{QUOTE_OPEN}(?P<quoted>{TERM}){QUOTE_CLOSE}|\b(?P<bare>{TERM}))"
    r" (?:shall mean|means|shall have the meaning|has the meaning|refers to)\b"
)
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+|\n\s*\n")
MAX_DEFINITION_CHARS = 600
MIN_TERM_CHARS = 2
MAX_TERM_CHARS = 60
STOP_TERMS = frozenset({"Agreement"})
STOP_FIRST_WORDS = frozenset({"The", "This", "Each", "Any", "Such", "Following", "All"})
# A defining sentence shorter than this is extended backwards, as addresses split on periods.
MIN_CONTEXT_CHARS = 80
MAX_EXTENSIONS = 3
# The form naming "the term" comes first, or "What does the term X mean" would keep "term".
QUESTION_FORMS = (
    re.compile(
        r"^(?:define|definition of|meaning of|what do(?:es)? (?:the )?(?:term|word)\s+)"
        r"(?:the\s+)?[\"'“‘]?(?P<term>.+?)[\"'”’]?(?:\s+mean)?\s*\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:what|who)\s+(?:is|are|does|do)\s+(?:the\s+|a\s+|an\s+)?(?P<term>.+?)"
        r"(?:\s+mean|\s+refer to|\s+defined as|\s+stand for)?\s*\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:šta|sta|ko)\s+(?:je|su|znači|znaci)\s+(?P<term>.+?)\s*\??$", re.IGNORECASE
    ),
)


def find_definitions(text: str, page_starts: Sequence[int] = ()) -> list[Definition]:
    """Return the terms the text defines, each once, with the defining sentence as its quote."""
    found: dict[str, Definition] = {}
    for match in sorted(_matches(text), key=lambda m: m.start()):
        groups = match.groupdict()
        term = (groups.get("term") or groups.get("quoted") or groups.get("bare") or "").strip()
        if not _plausible(term) or term.lower() in found:
            continue
        start, end = _sentence_around(text, match.start(), match.end())
        found[term.lower()] = Definition(
            term=term,
            quote=text[start:end],
            page=_page_of(page_starts, start),
            start=start,
            end=end,
        )
    return list(found.values())


def _matches(text: str) -> list[re.Match[str]]:
    return [*PARENTHETICAL.finditer(text), *MEANS.finditer(text)]


def _plausible(term: str) -> bool:
    if not MIN_TERM_CHARS <= len(term) <= MAX_TERM_CHARS:
        return False
    if term in STOP_TERMS or term.split()[0] in STOP_FIRST_WORDS:
        return False
    return any(char.isalpha() for char in term)


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    begin = _sentence_start(text, start)
    extensions = 0
    while start - begin < MIN_CONTEXT_CHARS and begin > 0 and extensions < MAX_EXTENSIONS:
        begin = _sentence_start(text, begin - 1)
        extensions += 1
    stop = len(text)
    tail = SENTENCE_END.search(text, end)
    if tail is not None:
        stop = tail.start()
    if stop - begin > MAX_DEFINITION_CHARS:
        stop = _cut(text, begin, begin + MAX_DEFINITION_CHARS, end)
    return _trim(text, begin, stop)


def _sentence_start(text: str, position: int) -> int:
    begin = 0
    for match in SENTENCE_END.finditer(text, max(0, position - MAX_DEFINITION_CHARS), position):
        begin = match.end()
    return begin


def _cut(text: str, begin: int, limit: int, at_least: int) -> int:
    cut = text.rfind(" ", at_least, limit)
    return cut if cut > begin else max(limit, at_least)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _page_of(page_starts: Sequence[int], offset: int) -> int | None:
    if not page_starts:
        return None
    return max(bisect_right(list(page_starts), offset), 1)


def asked_term(question: str) -> str | None:
    """Return the term a question asks the meaning of, or None if it is not such a question."""
    stripped = question.strip()
    for form in QUESTION_FORMS:
        match = form.match(stripped)
        if match:
            term = match.group("term").strip(" \"'“”‘’")
            return term or None
    return None


def lookup(definitions: Sequence[Definition], question: str) -> Definition | None:
    """Return the definition a question asks for, matched on the term, or None."""
    term = asked_term(question)
    if term is None:
        return None
    wanted = _normalise(term)
    for definition in definitions:
        if _normalise(definition.term) == wanted:
            return definition
    for definition in definitions:
        known = _normalise(definition.term)
        if re.search(rf"\b{re.escape(known)}\b", wanted) or re.search(
            rf"\b{re.escape(wanted)}\b", known
        ):
            return definition
    return None


def _normalise(term: str) -> str:
    words = term.lower().split()
    if words and words[0] in ("the", "a", "an", "this"):
        words = words[1:]
    return " ".join(words)
