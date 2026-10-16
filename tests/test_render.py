import pytest

from contract_assistant.answer import (
    ELLIPSIS,
    MAX_CHAT_QUOTE_CHARS,
    Answer,
    AnswerItem,
    Citation,
    render_markdown,
)
from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.router.base import RouteResult


def citation(quote: str, page: int | None = 12, conf: float = 0.94) -> Citation:
    return Citation(quote=quote, page=page, conf=conf, start=0, end=len(quote))


def found_item(quote: str, page: int | None = 12, conf: float = 0.94) -> AnswerItem:
    return AnswerItem(
        clause_type="Governing Law",
        status="found",
        best_conf=conf,
        citation=citation(quote, page, conf),
    )


def answer_of(*items: AnswerItem) -> Answer:
    route = RouteResult(
        clause_types=[i.clause_type for i in items], kind="explicit", latency_ms=0
    )
    return Answer(
        question="q",
        route=route,
        tau_low=0.06,
        tau_high=0.91,
        items=list(items),
        dropped_quotes=0,
    )


def render(*items: AnswerItem) -> str:
    return render_markdown(answer_of(*items), [])


def test_found_item_shows_page_and_confidence():
    text = render(found_item("Governed by New York law."))
    assert text == (
        "**Governing Law** — found · p. 12 · confidence 0.94\n> Governed by New York law."
    )


def test_confidence_has_two_decimals():
    assert "confidence 0.50" in render(found_item("x", conf=0.5))


def test_missing_page_is_shown_as_not_available():
    assert "p. n/a" in render(found_item("x", page=None))


def test_check_item_uses_verify_wording():
    item = found_item("x").model_copy(update={"status": "check"})
    assert "possible, verify" in render(item)


def test_not_found_item_shows_best_candidate():
    item = AnswerItem(
        clause_type="Exclusivity", status="not_found", best_conf=0.03, citation=None
    )
    assert render(item) == "**Exclusivity** — not found (checked; best candidate 0.03)"


def test_withheld_item_states_failed_verification():
    item = AnswerItem(
        clause_type="Exclusivity",
        status="check",
        best_conf=0.42,
        citation=None,
        withheld=True,
    )
    assert (
        render(item)
        == "**Exclusivity** — possible, verify · quote withheld (failed verification)"
    )


def test_items_are_separated_by_a_blank_line():
    other = found_item("y").model_copy(update={"clause_type": "Term"})
    assert "\n\n**Term**" in render(found_item("x"), other)


@pytest.mark.parametrize("char", ["*", "_", "#", ">"])
def test_markdown_characters_in_quote_are_escaped(char):
    assert f"a\\{char}b" in render(found_item(f"a{char}b"))


def test_multiline_quote_stays_one_blockquote():
    text = render(found_item("first\n\nsecond"))
    assert text.endswith("> first\n>\n> second")


def test_long_quote_is_cut_to_a_verbatim_prefix_with_ellipsis():
    quote = "abcdefghij" * 100
    body = render(found_item(quote)).split("\n> ", 1)[1]
    assert body.endswith(ELLIPSIS)
    assert len(body) == MAX_CHAT_QUOTE_CHARS
    assert quote.startswith(body.removesuffix(ELLIPSIS))


def test_quote_at_the_limit_is_not_cut():
    quote = "a" * MAX_CHAT_QUOTE_CHARS
    assert ELLIPSIS not in render(found_item(quote))


def test_empty_route_lists_all_clause_types():
    names = [t.name for t in load_clause_types(Settings().clause_types_path)]
    text = render_markdown(answer_of(), names)
    assert text.startswith("I can only answer about these clause types:")
    assert all(f"- {name}" in text for name in names)
    assert text.count("\n- ") == 41
