"""A summary of the whole text, section by section, each part checked against its own section."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from contract_assistant.generation import (
    GeneratedRecord,
    GeneratedStore,
    Generation,
    Request,
    faithful_check,
    run,
)
from contract_assistant.router.clients import LLMClient
from contract_assistant.search import split_passages
from contract_assistant.store import DocumentCache

logger = logging.getLogger(__name__)

PROMPT_VERSION = 1
# About 1,500 tokens of contract per section; a long contract has 15 to 20 sections.
SECTION_CHARS = 6000
SECTION_MAX_TOKENS = 220
MERGE_MAX_TOKENS = 700
MAX_SECTION_SENTENCES = 3
MAX_OVERVIEW_SENTENCES = 10
APP_ATTEMPTS = 2
FULLTEXT_DIR = "fulltext"
KIND = "fulltext"
SECTION_SOURCE = "section"
MERGE_SOURCE = "section summaries"
SECTION_PROMPT = (
    "Summarize this part of a contract for a lawyer in at most "
    f"{MAX_SECTION_SENTENCES} sentences. Quote it only word for word, in double quotes, and "
    "use double quotes for nothing else. Copy every number, date and amount exactly as the "
    "text states it and do not add facts it does not state. If the part is only headings, "
    "signatures or boilerplate, say so in one sentence. Reply with the sentences only."
)
MERGE_PROMPT = (
    "Combine these summaries of the parts of one contract, in their order, into one overview "
    f"for a lawyer of at most {MAX_OVERVIEW_SENTENCES} sentences in one or two paragraphs. "
    "Quote only text that appears in the summaries, word for word, in double quotes, and use "
    "double quotes for nothing else. Copy every number, date and amount exactly as the "
    "summaries state it and do not add facts they do not state. Reply with the overview only."
)


class SectionSummary(BaseModel):
    """One section of the text and what the model wrote about it, or why that is withheld."""

    start: int
    end: int
    page: int | None
    text: str | None
    reason: str | None = None
    attempts: int


class FullTextSummary(GeneratedRecord):
    """The section summaries of a document and the overview merged from them."""

    prompt_version: int = PROMPT_VERSION
    sections: list[SectionSummary] = Field(default_factory=list)
    text: str | None
    reason: str | None = None


class Progress(BaseModel):
    """Where a full-text run is: the section being written out of all of them."""

    done: int
    total: int
    stage: str


def sections(text: str) -> list[tuple[int, int]]:
    """Cut the text into sections of about SECTION_CHARS that never split a passage."""
    spans = split_passages(text)
    if not spans:
        return [(0, len(text))] if text.strip() else []
    result: list[tuple[int, int]] = []
    start, end = spans[0]
    for span_start, span_end in spans[1:]:
        if span_end - start > SECTION_CHARS and end > start:
            result.append((start, end))
            start = span_start
        end = span_end
    result.append((start, end))
    return result


def section_request(doc_id: str, index: int, text: str) -> Request:
    """Return the request that summarizes one section, checked against that section."""
    return Request(
        SECTION_PROMPT,
        f"Part {index + 1}:\n{text}",
        SECTION_MAX_TOKENS,
        faithful_check([text], SECTION_SOURCE),
        SECTION_SOURCE,
        f"section {index + 1} of {doc_id}",
    )


def merge_request(doc_id: str, parts: Sequence[str]) -> Request:
    """Return the request that merges the section summaries, checked against them."""
    user = "\n\n".join(f"Part {index + 1}: {part}" for index, part in enumerate(parts))
    return Request(
        MERGE_PROMPT,
        user,
        MERGE_MAX_TOKENS,
        faithful_check(list(parts), MERGE_SOURCE),
        MERGE_SOURCE,
        f"overview of {doc_id}",
    )


def write_fulltext(
    cache: DocumentCache,
    client: LLMClient,
    tau_low: float,
    tau_high: float,
    attempts: int = 1,
) -> Iterator[Progress | FullTextSummary]:
    """Yield the progress through the sections, then the full-text summary."""
    spans = sections(cache.text)
    total = len(spans) + 1
    done: list[SectionSummary] = []
    latency = 0.0
    attempts_made = 0
    for index, (start, end) in enumerate(spans):
        yield Progress(done=index, total=total, stage=f"section {index + 1} of {len(spans)}")
        result = run(
            client, section_request(cache.doc_id, index, cache.text[start:end]), attempts
        )
        latency += result.latency_ms
        attempts_made = max(attempts_made, result.attempts)
        done.append(
            SectionSummary(
                start=start,
                end=end,
                page=_page_of(cache.page_starts, start),
                text=result.text,
                reason=result.reason,
                attempts=result.attempts,
            )
        )
    parts = [section.text for section in done if section.text]
    text: str | None = None
    reason: str | None = None
    if parts:
        yield Progress(done=len(spans), total=total, stage="overview")
        merged: Generation = run(client, merge_request(cache.doc_id, parts), attempts)
        latency += merged.latency_ms
        attempts_made = max(attempts_made, merged.attempts)
        text, reason = merged.text, merged.reason
    else:
        reason = "no section could be summarized"
    yield FullTextSummary(
        doc_id=cache.doc_id,
        model=client.model,
        tau_low=tau_low,
        tau_high=tau_high,
        sections=done,
        text=text,
        reason=reason,
        attempts=attempts_made,
        latency_ms=latency,
    )


def request_fulltext(
    cache: DocumentCache, client: LLMClient, tau_low: float, tau_high: float, attempts: int = 1
) -> FullTextSummary:
    """Run the full-text summary to its end and return it."""
    outcome: Progress | FullTextSummary | None = None
    for item in write_fulltext(cache, client, tau_low, tau_high, attempts):
        outcome = item
    if not isinstance(outcome, FullTextSummary):
        raise TypeError("the full-text run ended without a summary")
    return outcome


def _page_of(page_starts: Sequence[int], offset: int) -> int | None:
    if not page_starts:
        return None
    from bisect import bisect_right

    return max(bisect_right(list(page_starts), offset), 1)


def render_markdown(summary: FullTextSummary) -> str:
    """Render the overview and the section summaries for the command line."""
    lines = []
    if summary.text:
        lines += [summary.text, ""]
    else:
        lines += [f"Overview withheld: {summary.reason}.", ""]
    for index, section in enumerate(summary.sections, start=1):
        page = f", p. {section.page}" if section.page is not None else ""
        if section.text:
            lines.append(f"{index}. (chars {section.start}-{section.end}{page}) {section.text}")
        else:
            lines.append(
                f"{index}. (chars {section.start}-{section.end}{page}) withheld: {section.reason}"
            )
    seconds = summary.latency_ms / 1000
    lines += [
        "",
        f"model: {summary.model} · {len(summary.sections)} sections · {seconds:.0f} s",
    ]
    return "\n".join(lines)


class FullTextStore(GeneratedStore[FullTextSummary]):
    """Full-text summaries saved per document, model, thresholds and prompt version."""

    def __init__(self, directory: Path):
        super().__init__(directory, KIND, PROMPT_VERSION, FullTextSummary)
