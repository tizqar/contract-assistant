"""Pure functions from caches and answers to Gradio component values."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Collection
from pathlib import Path

from contract_assistant.answer import (
    ELLIPSIS,
    Answer,
    AnswerItem,
    Citation,
    GeneratedAnswer,
    Passage,
    Status,
    blockquote,
    tier,
)
from contract_assistant.config import Settings
from contract_assistant.explain import LANGUAGES, Explanation
from contract_assistant.store import DocumentCache
from contract_assistant.summary import MS_PER_S, Summary
from contract_assistant.term_sheet import FIELDS, LABELS, TermSheet, cell

CLAUSE_GROUPS: dict[str, tuple[str, ...]] = {
    "General terms": (
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
        "Governing Law",
    ),
    "Competition, termination and assignment": (
        "Most Favored Nation",
        "Non-Compete",
        "Exclusivity",
        "No-Solicit Of Customers",
        "Competitive Restriction Exception",
        "No-Solicit Of Employees",
        "Non-Disparagement",
        "Termination For Convenience",
        "Rofr/Rofo/Rofn",
        "Change Of Control",
        "Anti-Assignment",
    ),
    "Revenue and pricing": (
        "Revenue/Profit Sharing",
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
    ),
    "Intellectual property and licences": (
        "IP Ownership Assignment",
        "Joint IP Ownership",
        "License Grant",
        "Non-Transferable License",
        "Affiliate License-Licensor",
        "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License",
        "Source Code Escrow",
    ),
    "Liability and other terms": (
        "Post-Termination Services",
        "Audit Rights",
        "Uncapped Liability",
        "Cap On Liability",
        "Liquidated Damages",
        "Warranty Duration",
        "Insurance",
        "Covenant Not To Sue",
        "Third Party Beneficiary",
    ),
}
OTHER_TOPIC = "Other common clauses"
STATUS_LABELS: dict[Status, str] = {
    "found": "Found",
    "check": "Verify",
    "not_found": "Not found",
}
CLAUSE_HEADERS = ["Topic", "Clause", "Status", "Confidence", "Page", "Passage"]
SUGGESTED_QUESTIONS = [
    "Who are the parties?",
    "Which law governs the contract?",
    "When does the agreement expire, and does it renew?",
    "Can either party terminate without cause?",
    "Is there a non-compete or exclusivity obligation?",
    "What are the payment terms?",
]
SMALL_WORDS = frozenset({"Of", "To", "On", "Or", "For"})
DISPLAY_OVERRIDES = {
    "Rofr/Rofo/Rofn": "Right of First Refusal, Offer or Negotiation",
    "Unlimited/All-You-Can-Eat-License": "Unlimited Usage License",
    "Affiliate License-Licensor": "License from the Licensor's Affiliates",
    "Affiliate License-Licensee": "License to the Licensee's Affiliates",
}
NOT_FOUND_TEXT = "The contract was checked for this clause; no passage matches it."
WITHHELD_TEXT = (
    "A matching passage was found but it could not be verified word for word "
    "against the contract, so it is not shown."
)
CHECKLIST_INTRO = (
    "All {count} clauses the assistant checks, by topic. Select a row to see its passage."
)
CHECKLIST_EXTRA_INTRO = (
    " The other common clauses are checked without validation data "
    "and are shown as Verify at most."
)
EMPTY_ROUTE_TEXT = "This question does not match any of the clauses I check."
PASSAGES_TEXT = (
    "This question does not match a clause I check. These are the passages of the "
    "contract closest to it; read them to find the answer."
)
UNVALIDATED_TEXT = "Not validated on test data"
RELATED_LABEL = "Related passage"
PASSAGE_PREFIX = "passage-"
HIGHLIGHT_TEXT = {
    1: "The passage is highlighted in the document on the right.",
    2: (
        "The first passage is highlighted in the document on the right; "
        "choose another under \u201cPassage shown\u201d."
    ),
}
NO_PAGE = "—"
EXCERPT_LABEL = "Excerpt around the highlighted passage"
TEXT_START_LABEL = "Beginning of the contract"
TABLE_PASSAGE_CHARS = 120
WORD_ROUNDING = 100
CONTEXT_CHARS = 1500
# Longer than any line of the demo texts; a text without line breaks is cut mid-line.
MAX_WIDEN_CHARS = 3000
EXAMPLE_QUESTIONS = 2

SUMMARY_WITHHELD_TEXT = "The summary is not shown because {reason}."
SUMMARY_MISMATCH_TEXT = "{count} quoted passage(s) did not match the contract."
SUMMARY_META_TEXT = (
    "Written by the language model {model} from {count} clause(s) in {seconds} s. "
    "Text in double quotes is verified word for word against the contract; "
    "the rest is the model's own wording."
)
SUMMARY_BASED_ON = "Based on: "
UNVALIDATED_MARK = "*"
EXPLANATION_HEADING = "**{name}** · in plain {language}"
EXPLANATION_WITHHELD_TEXT = "The explanation is not shown because {reason}."
EXPLANATION_META_TEXT = (
    "Written by the language model {model}. This is a restatement, not a quote; "
    "check it against the highlighted passage."
)
GENERATED_ANSWER_HEADING = (
    '<span class="tier tier-generated">Generated</span> **Answer from the passages**'
)
GENERATED_ANSWER_META = (
    "Written by the language model {model} from the passages below. "
    "Text in double quotes is verified against them; the rest is the model's wording. "
    "This is not a clause match."
)
GENERATED_ANSWER_WITHHELD = "The language model's answer is not shown because {reason}."
MATCHING_BACKENDS = {"ollama": "Ollama"}
OPENAI_BACKEND = "an OpenAI-compatible API"
GENERATION_OFF = "off"

Segments = list[tuple[str, str | None]]


def display_name(clause_type: str) -> str:
    """Return the name of a clause type as it is shown to the user."""
    if clause_type in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[clause_type]
    first, *rest = clause_type.split(" ")
    return " ".join([first, *(word.lower() if word in SMALL_WORDS else word for word in rest)])


def document_name(filename: str) -> str:
    """Return a file name as a readable title."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    name = " ".join(stem.split())
    return name[:1].upper() + name[1:]


def document_title(cache: DocumentCache) -> str:
    """Return the document name with its page count or approximate word count."""
    source = cache.source
    name = f"**{document_name(source.filename)}**"
    if source.n_pages is not None:
        return f"{name} · {source.n_pages} pages"
    words = len(cache.text.split())
    if words >= WORD_ROUNDING:
        words = round(words / WORD_ROUNDING) * WORD_ROUNDING
        return f"{name} · about {words:,} words"
    return f"{name} · {words} words"


def _badge(status: Status) -> str:
    return f'<span class="tier tier-{status}">{STATUS_LABELS[status]}</span>'


def _percent(conf: float) -> str:
    return f"{conf:.0%}"


def chat_reply(answer: Answer, unvalidated: Collection[str] = ()) -> str:
    """Return the Markdown shown in the chat for an answer."""
    if not answer.items:
        return (
            _passages_reply(answer.passages, answer.generated)
            if answer.passages
            else _empty_route_reply(unvalidated)
        )
    blocks = [_item_block(item, item.clause_type in unvalidated) for item in answer.items]
    shown = sum(item.citation is not None for item in answer.items)
    if shown:
        blocks.append(f"<small>{HIGHLIGHT_TEXT[min(shown, 2)]}</small>")
    return "\n\n".join(blocks)


def _passages_reply(passages: list[Passage], generated: GeneratedAnswer | None = None) -> str:
    blocks = [
        *_generated_blocks(generated),
        PASSAGES_TEXT,
        *(_passage_block(i, p) for i, p in enumerate(passages, start=1)),
    ]
    blocks.append(f"<small>{HIGHLIGHT_TEXT[min(len(passages), 2)]}</small>")
    return "\n\n".join(blocks)


def _generated_blocks(generated: GeneratedAnswer | None) -> list[str]:
    if generated is None:
        return []
    if generated.text is None:
        return [f"<small>{GENERATED_ANSWER_WITHHELD.format(reason=generated.reason)}</small>"]
    meta = GENERATED_ANSWER_META.format(model=generated.model)
    return [GENERATED_ANSWER_HEADING, generated.text, f"<small>{meta}</small>"]


def _passage_block(number: int, passage: Passage) -> str:
    heading = f'<span class="tier tier-related">Related</span> **Passage {number}**'
    block = f"{heading}\n{blockquote(passage.quote)}"
    if passage.page is None:
        return block
    return f"{block}\n\n<small>Page {passage.page}</small>"


def passage_choices(passages: list[Passage]) -> list[tuple[str, str]]:
    """Return the (label, value) choices that select each related passage."""
    return [
        (f"{RELATED_LABEL} {number}", f"{PASSAGE_PREFIX}{number - 1}")
        for number in range(1, len(passages) + 1)
    ]


def passage_citation(passage: Passage) -> Citation:
    """Return a related passage as a citation, with its score as the confidence."""
    return Citation(
        quote=passage.quote,
        page=passage.page,
        conf=passage.score,
        start=passage.start,
        end=passage.end,
    )


def _item_block(item: AnswerItem, unvalidated: bool) -> str:
    heading = f"{_badge(item.status)} **{display_name(item.clause_type)}**"
    if item.withheld:
        return f"{heading}\n\n{WITHHELD_TEXT}"
    if item.citation is None:
        return f"{heading}\n\n{NOT_FOUND_TEXT}"
    citation = item.citation
    meta = f"{_percent(citation.conf)} confidence"
    if unvalidated:
        meta = f"{UNVALIDATED_TEXT} · {meta}"
    if citation.page is not None:
        meta = f"Page {citation.page} · {meta}"
    return f"{heading}\n{blockquote(citation.quote)}\n\n<small>{meta}</small>"


def _empty_route_reply(unvalidated: Collection[str]) -> str:
    topics = dict(CLAUSE_GROUPS)
    if unvalidated:
        topics[OTHER_TOPIC] = tuple(sorted(unvalidated))
    lines = "\n".join(
        f"- **{topic}:** {', '.join(display_name(name) for name in names)}"
        for topic, names in topics.items()
    )
    examples = " or ".join(f"\u201c{q}\u201d" for q in SUGGESTED_QUESTIONS[:EXAMPLE_QUESTIONS])
    return f"{EMPTY_ROUTE_TEXT}\n\nI can answer questions about:\n\n{lines}\n\nFor example: {examples}"


def clause_rows(
    cache: DocumentCache,
    tau_low: float,
    tau_high: float,
    status: Status | None = None,
    unvalidated: Collection[str] = (),
) -> list[list[str]]:
    """Return one row per clause type in topic order, optionally only those with a status."""
    rows = []
    for topic, names in _topics(cache, unvalidated).items():
        for position, name in enumerate(names):
            candidates = cache.clauses.get(name, [])
            row_status = _row_status(candidates, tau_low, tau_high, name in unvalidated)
            if status is not None and row_status != status:
                continue
            label = topic if position == 0 or status is not None else ""
            rows.append(_row(label, name, row_status, candidates))
    return rows


def checklist_intro(cache: DocumentCache, unvalidated: Collection[str] = ()) -> str:
    """Return the line under the checklist tab, counting the clauses listed for cache."""
    topics = _topics(cache, unvalidated)
    intro = CHECKLIST_INTRO.format(count=sum(len(names) for names in topics.values()))
    return intro + CHECKLIST_EXTRA_INTRO if OTHER_TOPIC in topics else intro


def _topics(cache: DocumentCache, unvalidated: Collection[str]) -> dict[str, tuple[str, ...]]:
    extra = tuple(name for name in cache.clauses if name in unvalidated)
    return {**CLAUSE_GROUPS, OTHER_TOPIC: extra} if extra else CLAUSE_GROUPS


def _row_status(candidates: list, tau_low: float, tau_high: float, capped: bool) -> Status:
    if not candidates:
        return "not_found"
    status = tier(candidates[0].conf, tau_low, tau_high)
    return "check" if capped and status == "found" else status


def _row(topic: str, name: str, status: Status, candidates: list) -> list[str]:
    if not candidates:
        return [topic, display_name(name), _badge(status), _percent(0.0), NO_PAGE, ""]
    top = candidates[0]
    page = NO_PAGE if top.page is None else str(top.page)
    return [
        topic,
        display_name(name),
        _badge(status),
        _percent(top.conf),
        page,
        _short_passage(top.text),
    ]


def _short_passage(text: str) -> str:
    passage = " ".join(text.split())
    if len(passage) <= TABLE_PASSAGE_CHARS:
        return passage
    return passage[: TABLE_PASSAGE_CHARS - len(ELLIPSIS)] + ELLIPSIS


def top_citation(cache: DocumentCache, clause_type: str) -> Citation | None:
    """Return the top candidate of a clause type as a citation, whatever its tier."""
    candidates = cache.clauses.get(clause_type, [])
    if not candidates:
        return None
    top = candidates[0]
    return Citation(quote=top.text, page=top.page, conf=top.conf, start=top.start, end=top.end)


def document_view(
    cache: DocumentCache, clause_type: str | None, citation: Citation | None
) -> tuple[str, Segments]:
    """Return the label and highlighted segments for the part of the text around a citation."""
    if cache.page_starts:
        first, last, lo, hi = _page_window(cache, citation)
        label = f"Page {first}" if first == last else f"Pages {first}–{last}"
    else:
        lo, hi = _text_window(cache.text, citation)
        label = EXCERPT_LABEL if citation else TEXT_START_LABEL
    return label, _segments(cache.text, lo, hi, clause_type, citation)


def _text_window(text: str, citation: Citation | None) -> tuple[int, int]:
    first = 0 if citation is None else citation.start - CONTEXT_CHARS
    last = 2 * CONTEXT_CHARS if citation is None else citation.end + CONTEXT_CHARS
    return _widen_left(text, max(first, 0)), _widen_right(text, min(last, len(text)))


def _widen_left(text: str, position: int) -> int:
    newline = text.rfind("\n", max(position - MAX_WIDEN_CHARS, 0), position)
    if newline >= 0:
        return newline + 1
    return 0 if position <= MAX_WIDEN_CHARS else position


def _widen_right(text: str, position: int) -> int:
    newline = text.find("\n", position, position + MAX_WIDEN_CHARS)
    if newline >= 0:
        return newline
    return len(text) if position + MAX_WIDEN_CHARS >= len(text) else position


def _page_window(cache: DocumentCache, citation: Citation | None) -> tuple[int, int, int, int]:
    starts = cache.page_starts
    first = 1 if citation is None else bisect_right(starts, citation.start)
    last = (
        first
        if citation is None
        else bisect_right(starts, max(citation.end - 1, citation.start))
    )
    first = max(first, 1)
    last = max(last, first)
    hi = starts[last] if last < len(starts) else len(cache.text)
    return first, last, starts[first - 1], hi


def _segments(
    text: str, lo: int, hi: int, clause_type: str | None, citation: Citation | None
) -> Segments:
    if citation is None:
        return [(text[lo:hi], None)]
    parts = [
        (text[lo : citation.start], None),
        (text[citation.start : citation.end], clause_type and display_name(clause_type)),
        (text[citation.end : hi], None),
    ]
    return [part for part in parts if part[0]]


def technical_markdown(
    settings: Settings,
    route_kind: str,
    unvalidated: Collection[str] = (),
    gen_model: str | None = None,
) -> str:
    """Return the technical facts behind the answers as a short Markdown list."""
    lines = [
        f"- **Extraction model:** `{settings.extractor_name}`",
        f"- **Question matching:** {_matching(settings, route_kind)}",
        f"- **Found from:** {_percent(settings.tau_high)} confidence ({settings.tau_source})",
        f"- **Verify from:** {_percent(settings.tau_low)} confidence ({settings.tau_source})",
        f"- **Summaries and explanations:** {_generation(settings, gen_model)}",
    ]
    if unvalidated:
        names = ", ".join(display_name(name) for name in sorted(unvalidated))
        lines.append(f"- **Not validated on test data:** {names}")
    return "\n".join(lines)


def _backend(settings: Settings) -> str:
    return MATCHING_BACKENDS.get(settings.llm_backend, OPENAI_BACKEND)


def _generation(settings: Settings, gen_model: str | None) -> str:
    if gen_model is None:
        return GENERATION_OFF
    return f"`{gen_model}` through {_backend(settings)}"


def _matching(settings: Settings, route_kind: str) -> str:
    if route_kind == "llm":
        return f"`{settings.llm_model}` through {_backend(settings)}"
    if route_kind == "embedding":
        return f"embedding model `{settings.embedding_model_id}`"
    return route_kind


def sources_json(answer: Answer, cache: DocumentCache) -> dict:
    """Return the answer and the other n-best candidates of each of its clause types."""
    others = {
        item.clause_type: [
            {"text": c.text, "conf": c.conf, "start": c.start, "end": c.end, "page": c.page}
            for c in cache.clauses[item.clause_type][1:]
        ]
        for item in answer.items
    }
    return {"answer": answer.model_dump(mode="json"), "other_candidates": others}


def highlighted_quote(segments: object) -> tuple[str, str] | None:
    """Return the text and label of the first labelled segment, or None if there is none."""
    for segment in segments or []:
        if isinstance(segment, dict):
            text, label = segment.get("token"), segment.get("class_or_confidence")
        else:
            text, label = segment
        if label and text:
            return text, label
    return None


def summary_markdown(summary: Summary, unvalidated: Collection[str] = ()) -> str:
    """Return the Markdown shown for a summary: its text, or why it is withheld."""
    if summary.text is None:
        withheld = SUMMARY_WITHHELD_TEXT.format(reason=summary.reason)
        if not summary.unfaithful_spans:
            return withheld
        count = SUMMARY_MISMATCH_TEXT.format(count=len(summary.unfaithful_spans))
        return f"{withheld}\n\n<small>{count}</small>"
    meta = SUMMARY_META_TEXT.format(
        model=summary.model,
        count=len(summary.items),
        seconds=round(summary.latency_ms / MS_PER_S),
    )
    names = ", ".join(
        _based_on(item, item.clause_type in unvalidated) for item in summary.items
    )
    if any(item.clause_type in unvalidated for item in summary.items):
        names = f"{names} · {UNVALIDATED_MARK} {UNVALIDATED_TEXT}"
    return (
        f"{summary.text}\n\n<small>{meta}</small>\n\n<small>{SUMMARY_BASED_ON}{names}</small>"
    )


def _based_on(item: AnswerItem, unvalidated: bool) -> str:
    name = f"{display_name(item.clause_type)} {_badge(item.status)}"
    return f"{name}{UNVALIDATED_MARK}" if unvalidated else name


def explanation_markdown(explanation: Explanation) -> str:
    """Return the Markdown shown for an explanation: its text, or why it is withheld."""
    if explanation.text is None:
        return EXPLANATION_WITHHELD_TEXT.format(reason=explanation.reason)
    name = (
        explanation.clause_type
        if explanation.clause_type == RELATED_LABEL
        else display_name(explanation.clause_type)
    )
    language = LANGUAGES[explanation.language].split(",")[0]
    heading = EXPLANATION_HEADING.format(name=name, language=language)
    meta = EXPLANATION_META_TEXT.format(model=explanation.model)
    return f"{heading}\n\n{explanation.text}\n\n<small>{meta}</small>"


def term_rows(sheet: TermSheet) -> list[list[str]]:
    """Return one row per key term: label, value, clause types it comes from and page."""
    rows = []
    for field in FIELDS:
        values = sheet.terms.get(field, [])
        names = dict.fromkeys(display_name(value.clause_type) for value in values)
        page = values[0].page if values else None
        rows.append(
            [
                LABELS[field],
                cell(sheet, field),
                ", ".join(names),
                NO_PAGE if page is None else str(page),
            ]
        )
    return rows
