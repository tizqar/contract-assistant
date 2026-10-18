from itertools import pairwise
from pathlib import Path

import huggingface_hub
import pytest

from contract_assistant.config import Settings
from contract_assistant.search import (
    MAX_PASSAGE_CHARS,
    MAX_PASSAGES,
    MIN_PASSAGE_CHARS,
    PassageSearch,
    split_passages,
)
from contract_assistant.store import read_cache
from tests.support.caches import make_cache
from tests.support.fakes import FakeEncoder

MODEL_ID = "BAAI/bge-base-en-v1.5"
DEMO_DIR = Path(__file__).parent.parent / "demo"
SENTENCE = "The supplier shall deliver the ordered goods to the buyer within thirty days. "
PAYMENT = (
    "Invoices are payable within sixty days of receipt and late payment bears interest. "
    "The buyer pays every invoice by bank transfer to the account named by the supplier."
)
DELIVERY = (
    "Goods are shipped by sea freight from the port of Rotterdam and delivered to the warehouse. "
    "Delivery dates are confirmed by the carrier before each shipment leaves the port."
)
ARBITRATION = (
    "Disputes are settled by arbitration under the rules of the chamber of commerce. "
    "The arbitrators sit in Geneva and their award is final and binding on both parties."
)
ARBITRATION_QUESTION = "disputes are settled by arbitration under the rules of the chamber"
PAYMENT_QUESTION = "invoices are payable within sixty days of receipt"


def contract_text() -> str:
    return f"{PAYMENT}\n\n{DELIVERY}\n\n{ARBITRATION}"


def make_search() -> tuple[PassageSearch, FakeEncoder]:
    encoder = FakeEncoder()
    return PassageSearch(encoder), encoder


def search(text: str, question: str, page_starts: list[int] | None = None):
    cache = make_cache(text, Settings())
    cache.page_starts = page_starts or []
    return make_search()[0].search(cache, question)


def test_passages_are_exact_slices_of_trimmed_text():
    text = "  " + SENTENCE * 12 + "\n\n" + SENTENCE * 3 + "  \n"
    for start, end in split_passages(text):
        passage = text[start:end]
        assert passage == passage.strip()
        assert passage


def test_passages_do_not_overlap_and_keep_order():
    spans = split_passages(SENTENCE * 40)
    assert all(first[1] <= second[0] for first, second in pairwise(spans))


def test_passages_cover_all_words_of_the_text():
    text = (SENTENCE * 15 + "\n\n" + SENTENCE * 9 + "\nA line.\n").strip()
    covered = " ".join(text[start:end] for start, end in split_passages(text))
    assert covered.split() == text.split()


def test_passages_never_cross_a_blank_line():
    text = SENTENCE * 5 + "\n\n" + SENTENCE * 5 + "\n \t\n" + SENTENCE * 5
    for start, end in split_passages(text):
        assert "\n\n" not in text[start:end]
        assert "\n \t\n" not in text[start:end]


def test_passages_respect_the_length_bounds():
    spans = split_passages(SENTENCE * 60)
    lengths = [end - start for start, end in spans]
    assert max(lengths) <= MAX_PASSAGE_CHARS
    assert min(lengths) >= MIN_PASSAGE_CHARS


def test_sentence_longer_than_the_maximum_is_cut_at_spaces():
    text = " ".join(["clause"] * 300)
    spans = split_passages(text)
    assert len(spans) > 1
    assert all(end - start <= MAX_PASSAGE_CHARS for start, end in spans)
    assert all(
        text[start:end].split() == ["clause"] * len(text[start:end].split())
        for start, end in spans
    )


def test_long_line_is_split_at_sentence_ends():
    spans = split_passages(SENTENCE * 20)
    text = SENTENCE * 20
    assert all(text[start:end].endswith(".") for start, end in spans)


def test_empty_text_has_no_passages():
    assert split_passages("  \n\n \n") == []


def test_search_returns_the_matching_passage_first():
    passages = search(contract_text(), ARBITRATION_QUESTION)
    assert passages[0].quote == ARBITRATION


def test_search_quotes_are_slices_of_the_document():
    text = contract_text()
    for passage in search(text, f"{ARBITRATION_QUESTION} {PAYMENT_QUESTION}"):
        assert text[passage.start : passage.end] == passage.quote


def test_search_returns_at_most_three_passages():
    text = "\n\n".join(f"{SENTENCE}Shipment {n} is delivered to the buyer." for n in range(8))
    passages = search(text, "supplier shall deliver the ordered goods to the buyer")
    assert len(passages) == MAX_PASSAGES


def test_search_returns_best_passage_first():
    scores = [p.score for p in search(contract_text(), PAYMENT_QUESTION)]
    assert scores == sorted(scores, reverse=True)


def test_search_scores_reach_the_threshold():
    passages = search(contract_text(), ARBITRATION_QUESTION)
    assert passages
    assert all(p.score >= 0.55 for p in passages)


def test_search_passages_do_not_overlap():
    text = SENTENCE * 40
    passages = search(text, "supplier shall deliver the ordered goods")
    spans = sorted((p.start, p.end) for p in passages)
    assert all(first[1] <= second[0] for first, second in pairwise(spans))


def test_question_matching_nothing_returns_no_passages():
    assert search(contract_text(), "what is the weather today") == []


def test_document_is_encoded_once_across_searches():
    search_, encoder = make_search()
    cache = make_cache(contract_text(), Settings())
    search_.search(cache, ARBITRATION_QUESTION)
    first = list(encoder.document_texts)
    search_.search(cache, PAYMENT_QUESTION)
    assert encoder.document_texts == first
    assert len(encoder.query_texts) == 2


def test_least_recently_used_document_is_encoded_again():
    search_, encoder = make_search()
    settings = Settings()
    caches = [make_cache(f"{PAYMENT} Document {n}.", settings) for n in range(5)]
    for cache in caches:
        search_.search(cache, PAYMENT_QUESTION)
    encoded = len(encoder.document_texts)
    search_.search(caches[0], PAYMENT_QUESTION)
    assert len(encoder.document_texts) > encoded


def test_page_is_none_without_page_starts():
    assert search(contract_text(), ARBITRATION_QUESTION)[0].page is None


def test_page_follows_the_page_starts():
    text = contract_text()
    page_starts = [0, text.index(DELIVERY), text.index(ARBITRATION)]
    passages = search(text, ARBITRATION_QUESTION, page_starts)
    assert passages[0].page == 3
    assert (
        search(
            text, "goods are shipped by sea freight from the port of Rotterdam", page_starts
        )[0].page
        == 2
    )


@pytest.mark.slow
def test_real_model_finds_the_arbitration_clause_and_ignores_the_weather():
    if not isinstance(huggingface_hub.try_to_load_from_cache(MODEL_ID, "config.json"), str):
        pytest.skip(f"{MODEL_ID} is not in the Hugging Face cache")
    from contract_assistant.router.embedding import SentenceTransformerEncoder

    cache = read_cache(DEMO_DIR / "cache" / "short_supply_agreement.json", Settings())
    finder = PassageSearch(SentenceTransformerEncoder(MODEL_ID))
    assert "ARBITRATION" in finder.search(cache, "How are disputes resolved?")[0].quote
    assert finder.search(cache, "What is the weather today?") == []
