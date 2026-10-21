from itertools import pairwise

import pytest

from contract_assistant import fulltext
from contract_assistant.config import Settings
from contract_assistant.fulltext import (
    MERGE_MAX_TOKENS,
    MERGE_PROMPT,
    SECTION_CHARS,
    SECTION_MAX_TOKENS,
    SECTION_PROMPT,
    FullTextStore,
    FullTextSummary,
    Progress,
    SectionSummary,
    merge_request,
    render_markdown,
    request_fulltext,
    section_request,
    sections,
    write_fulltext,
)
from contract_assistant.generation import EMPTY_REASON, MISQUOTE_REASON, Generation
from contract_assistant.search import split_passages
from tests.conftest import ROOT
from tests.support.caches import make_cache
from tests.support.fakes import FakeLLMClient

TAU_LOW = 0.2
TAU_HIGH = 0.8
DOC_ID = "sha256:abc"
PARAGRAPH = (
    "The Supplier shall deliver the goods to the Customer at the agreed place and the "
    "Customer shall inspect them on arrival. Any defect found must be reported in writing "
    "without delay so that the Supplier can repair or replace the goods at its own cost."
)
SHORT_TEXT = f"{PARAGRAPH}\n\nThe Customer shall pay the fee."
DEMO_SECTIONS = {
    "short_supply_agreement.txt": 4,
    "medium_sponsorship_agreement.txt": 9,
    "long_outsourcing_agreement.txt": 18,
}
FAITHFUL = "The part says that the Supplier delivers the goods."
MISQUOTE = 'The part says "words that are nowhere in it".'
OVERVIEW = "The contract covers delivery and payment."


def long_text(paragraphs: int = 60) -> str:
    return "\n\n".join(f"{n}. {PARAGRAPH}" for n in range(paragraphs))


@pytest.fixture
def settings() -> Settings:
    return Settings()


def cache_of(text, settings):
    return make_cache(text, settings)


def test_a_short_text_is_one_section_covering_its_passages():
    spans = split_passages(SHORT_TEXT)
    assert sections(SHORT_TEXT) == [(spans[0][0], spans[-1][1])]


def test_an_empty_text_has_no_section():
    assert sections("") == []
    assert sections("  \n\n ") == []


def test_a_long_text_is_cut_into_sections_of_about_the_section_size():
    text = long_text()
    result = sections(text)
    assert len(result) > 1
    assert all(end - start <= SECTION_CHARS for start, end in result)


def test_sections_never_split_a_passage():
    text = long_text()
    spans = split_passages(text)
    starts = {start for start, _ in spans}
    ends = {end for _, end in spans}
    result = sections(text)
    assert all(start in starts and end in ends for start, end in result)


def test_sections_cover_the_passages_in_order_without_a_gap():
    text = long_text()
    spans = split_passages(text)
    result = sections(text)
    assert result[0][0] == spans[0][0]
    assert result[-1][1] == spans[-1][1]
    for (_, end), (start, _) in pairwise(result):
        assert end <= start
        assert any(span_start == start for span_start, _ in spans)
        previous = [span for span in spans if span[1] == end]
        assert previous
        following = spans[spans.index(previous[0]) + 1]
        assert following[0] == start


@pytest.mark.parametrize(("name", "count"), list(DEMO_SECTIONS.items()))
def test_the_demo_texts_have_their_number_of_sections(name, count):
    text = (ROOT / "demo" / name).read_text(encoding="utf-8")
    assert len(sections(text)) == count


def test_section_request_has_the_prompt_the_message_and_the_token_limit():
    request = section_request(DOC_ID, 2, "Some text.")
    assert request.system == SECTION_PROMPT
    assert request.user == "Part 3:\nSome text."
    assert request.max_tokens == SECTION_MAX_TOKENS
    assert "3" in request.label
    assert DOC_ID in request.label


def test_section_check_accepts_a_faithful_sentence():
    request = section_request(DOC_ID, 0, PARAGRAPH)
    assert request.check(FAITHFUL) is None
    assert request.check('It says "The Supplier shall deliver the goods".') is None


def test_section_check_rejects_a_misquote_of_the_section():
    request = section_request(DOC_ID, 0, PARAGRAPH)
    assert request.check(MISQUOTE) == MISQUOTE_REASON.format(source=request.source)


def test_section_check_rejects_an_empty_reply():
    assert section_request(DOC_ID, 0, PARAGRAPH).check(" ") == EMPTY_REASON


def test_merge_request_has_the_prompt_the_numbered_parts_and_the_token_limit():
    request = merge_request(DOC_ID, ["First.", "Second."])
    assert request.system == MERGE_PROMPT
    assert request.user == "Part 1: First.\n\nPart 2: Second."
    assert request.max_tokens == MERGE_MAX_TOKENS
    assert "overview" in request.label
    assert DOC_ID in request.label


def test_merge_check_accepts_a_faithful_sentence_and_rejects_a_misquote_of_the_parts():
    request = merge_request(DOC_ID, ['The Supplier "shall deliver" goods.', "Payment is due."])
    assert request.check('Overall, the Supplier "shall deliver" goods.') is None
    assert request.check(MISQUOTE) == MISQUOTE_REASON.format(source=request.source)


def test_write_fulltext_yields_a_progress_per_section_then_the_overview_then_the_summary(
    settings,
):
    cache = cache_of(long_text(), settings)
    count = len(sections(cache.text))
    client = FakeLLMClient([FAITHFUL] * count + [OVERVIEW])
    items = list(write_fulltext(cache, client, TAU_LOW, TAU_HIGH))
    *progress, summary = items
    assert all(isinstance(p, Progress) for p in progress)
    assert [(p.done, p.total) for p in progress] == [(n, count + 1) for n in range(count + 1)]
    assert [p.stage for p in progress[:-1]] == [
        f"section {n + 1} of {count}" for n in range(count)
    ]
    assert progress[-1].stage == "overview"
    assert isinstance(summary, FullTextSummary)


def test_the_summary_has_the_sections_with_offsets_text_cache_pages_and_the_overview(settings):
    cache = cache_of(long_text(), settings)
    spans = sections(cache.text)
    client = FakeLLMClient([f"{FAITHFUL}"] * len(spans) + [OVERVIEW])
    summary = request_fulltext(cache, client, TAU_LOW, TAU_HIGH)
    assert [(s.start, s.end) for s in summary.sections] == spans
    assert all(s.page is None for s in summary.sections)
    assert all(s.text == FAITHFUL and s.reason is None for s in summary.sections)
    assert summary.text == OVERVIEW
    assert summary.reason is None
    assert (summary.doc_id, summary.model) == (cache.doc_id, "fake-llm")
    assert (summary.tau_low, summary.tau_high) == (TAU_LOW, TAU_HIGH)


def test_the_section_requests_carry_the_text_of_each_section(settings):
    cache = cache_of(long_text(), settings)
    spans = sections(cache.text)
    client = FakeLLMClient([FAITHFUL] * len(spans) + [OVERVIEW])
    request_fulltext(cache, client, TAU_LOW, TAU_HIGH)
    for (start, end), (_, user, _) in zip(spans, client.text_calls, strict=False):
        assert cache.text[start:end] in user


def test_the_overview_is_asked_from_the_section_texts(settings):
    cache = cache_of(SHORT_TEXT, settings)
    client = FakeLLMClient([FAITHFUL, OVERVIEW])
    summary = request_fulltext(cache, client, TAU_LOW, TAU_HIGH)
    assert summary.text == OVERVIEW
    assert client.text_calls[-1][1] == f"Part 1: {FAITHFUL}"


def test_a_section_with_a_page_start_has_a_page(settings):
    cache = cache_of(SHORT_TEXT, settings).model_copy(update={"page_starts": [0, 10_000]})
    client = FakeLLMClient([FAITHFUL, OVERVIEW])
    assert request_fulltext(cache, client, TAU_LOW, TAU_HIGH).sections[0].page == 1


def test_a_section_withheld_twice_has_no_text_and_the_overview_comes_from_the_rest(settings):
    cache = cache_of(long_text(), settings)
    count = len(sections(cache.text))
    replies = [MISQUOTE, MISQUOTE] + [FAITHFUL] * (count - 1) + [OVERVIEW]
    client = FakeLLMClient(replies)
    summary = request_fulltext(cache, client, TAU_LOW, TAU_HIGH, attempts=2)
    first = summary.sections[0]
    assert first.text is None
    assert first.reason == MISQUOTE_REASON.format(source="section")
    assert first.attempts == 2
    assert summary.text == OVERVIEW
    assert client.text_calls[-1][1].count("Part ") == count - 1


def test_when_every_section_is_withheld_the_overview_is_skipped(settings):
    cache = cache_of(SHORT_TEXT, settings)
    client = FakeLLMClient([MISQUOTE])
    items = list(write_fulltext(cache, client, TAU_LOW, TAU_HIGH))
    summary = items[-1]
    assert summary.text is None
    assert summary.reason == "no section could be summarized"
    assert [p.stage for p in items[:-1]] == ["section 1 of 1"]
    assert len(client.text_calls) == 1


def test_a_text_without_sections_gives_a_summary_without_a_call(settings):
    cache = cache_of("", settings)
    client = FakeLLMClient([])
    summary = request_fulltext(cache, client, TAU_LOW, TAU_HIGH)
    assert summary.sections == []
    assert summary.reason == "no section could be summarized"


def test_attempts_is_the_maximum_and_latency_the_sum_over_the_runs(settings, monkeypatch):
    cache = cache_of(long_text(), settings)
    count = len(sections(cache.text))
    outcomes = [
        Generation(text="A.", reply="A.", attempts=n % 2 + 1, latency_ms=10.0)
        for n in range(count)
    ] + [Generation(text="O.", reply="O.", attempts=1, latency_ms=5.0)]
    calls = iter(outcomes)
    monkeypatch.setattr(fulltext, "run", lambda client, request, attempts=1: next(calls))
    summary = request_fulltext(cache, FakeLLMClient([]), TAU_LOW, TAU_HIGH, attempts=2)
    assert summary.attempts == 2
    assert summary.latency_ms == pytest.approx(10.0 * count + 5.0)


def test_request_fulltext_returns_the_summary_that_write_fulltext_ends_with(settings):
    cache = cache_of(SHORT_TEXT, settings)
    written = list(
        write_fulltext(cache, FakeLLMClient([FAITHFUL, OVERVIEW]), TAU_LOW, TAU_HIGH)
    )[-1]
    requested = request_fulltext(cache, FakeLLMClient([FAITHFUL, OVERVIEW]), TAU_LOW, TAU_HIGH)
    assert requested.model_dump(exclude={"latency_ms"}) == written.model_dump(
        exclude={"latency_ms"}
    )


def summary_of(text, *section_summaries) -> FullTextSummary:
    return FullTextSummary(
        doc_id=DOC_ID,
        model="fake-llm",
        tau_low=TAU_LOW,
        tau_high=TAU_HIGH,
        latency_ms=2000.0,
        sections=list(section_summaries),
        text=text,
        reason=None if text else "it quotes text that is not in the section summaries",
    )


def test_render_markdown_lists_the_overview_and_each_section_with_its_span():
    summary = summary_of(
        OVERVIEW,
        SectionSummary(start=0, end=120, page=None, text="One.", attempts=1),
        SectionSummary(start=130, end=300, page=2, text=None, reason="empty", attempts=2),
    )
    lines = render_markdown(summary).splitlines()
    assert lines[0] == OVERVIEW
    assert "1. (chars 0-120) One." in lines
    assert "2. (chars 130-300, p. 2) withheld: empty" in lines
    assert lines[-1] == "model: fake-llm · 2 sections · 2 s"


def test_render_markdown_says_when_the_overview_is_withheld():
    text = render_markdown(summary_of(None))
    assert text.startswith("Overview withheld: it quotes text")


def test_fulltext_store_saves_and_loads_a_summary(tmp_path):
    store = FullTextStore(tmp_path / "f")
    summary = summary_of(
        OVERVIEW, SectionSummary(start=0, end=5, page=None, text="One.", attempts=1)
    )
    assert store.save(summary).is_file()
    assert store.load(DOC_ID, "fake-llm", TAU_LOW, TAU_HIGH) == summary
    assert store.load(DOC_ID, "other", TAU_LOW, TAU_HIGH) is None
