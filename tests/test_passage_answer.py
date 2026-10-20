import pytest

from contract_assistant.answer import Answer, GeneratedAnswer, Passage
from contract_assistant.passage_answer import MAX_TOKENS, SYSTEM_PROMPT, request_passage_answer
from contract_assistant.router.base import RouteResult
from contract_assistant.router.clients import LLMError
from tests.support.fakes import FakeLLMClient

QUESTION = "What happens on a change of weather?"
FIRST = "The Term is five years\nand renews annually."
SECOND = "Fees are due within 30 days."
FAITHFUL = 'The contract says "five years" and nothing else.'
MISQUOTE = 'The contract says "ten years".'
NUMBER = "Fees are due within 45 days."


def passages() -> list[Passage]:
    return [
        Passage(quote=FIRST, page=None, score=0.7, start=0, end=len(FIRST)),
        Passage(quote=SECOND, page=None, score=0.6, start=50, end=50 + len(SECOND)),
    ]


def ask(replies, attempts=1):
    client = FakeLLMClient(replies)
    return request_passage_answer(QUESTION, passages(), client, attempts), client


def test_request_without_passages_raises_and_does_not_call():
    client = FakeLLMClient([])
    with pytest.raises(ValueError):
        request_passage_answer(QUESTION, [], client)
    assert client.text_calls == []


def test_request_message_has_the_question_and_each_quoted_passage():
    _, client = ask([FAITHFUL])
    user = client.text_calls[0][1]
    assert f"Question: {QUESTION}" in user
    assert '- "The Term is five years and renews annually."' in user
    assert f'- "{SECOND}"' in user


def test_request_passes_the_system_prompt_and_max_tokens():
    _, client = ask([FAITHFUL])
    system, _, max_tokens = client.text_calls[0]
    assert (system, max_tokens) == (SYSTEM_PROMPT, MAX_TOKENS)


def test_a_reply_quoting_part_of_a_passage_is_kept():
    generated, _ = ask([FAITHFUL])
    assert generated.text == FAITHFUL
    assert generated.reason is None
    assert generated.unfaithful_spans == []


def test_a_reply_without_quotes_is_kept():
    generated, _ = ask(["The passages say the term runs for years."])
    assert generated.text == "The passages say the term runs for years."


def test_a_misquote_is_withheld_naming_the_passages_and_the_span():
    generated, _ = ask([MISQUOTE])
    assert generated.text is None
    assert "passages" in generated.reason
    assert generated.unfaithful_spans == ["ten years"]


def test_a_number_not_in_the_passages_is_withheld():
    generated, _ = ask([NUMBER])
    assert generated.text is None
    assert "passages" in generated.reason
    assert "45" in generated.reason


def test_a_rejected_first_reply_and_an_accepted_second_give_the_second():
    generated, client = ask([MISQUOTE, FAITHFUL], attempts=2)
    assert generated.text == FAITHFUL
    assert generated.attempts == 2
    assert len(client.text_calls) == 2


def test_the_answer_records_the_model_attempts_and_latency():
    generated, _ = ask([FAITHFUL])
    assert generated.model == "fake-llm"
    assert generated.attempts == 1
    assert generated.latency_ms >= 0


def test_request_propagates_llm_errors():
    with pytest.raises(LLMError):
        ask([LLMError("down")])


def make_answer(generated: GeneratedAnswer | None = None) -> Answer:
    return Answer(
        question=QUESTION,
        route=RouteResult(clause_types=[], kind="explicit", latency_ms=0.0),
        tau_low=0.2,
        tau_high=0.8,
        items=[],
        dropped_quotes=0,
        passages=passages(),
        generated=generated,
    )


def test_answer_generated_defaults_to_none():
    assert make_answer().generated is None


def test_answer_with_a_generated_answer_round_trips_through_json():
    generated = GeneratedAnswer(
        model="fake-llm",
        text=None,
        reason="why",
        unfaithful_spans=["x"],
        attempts=2,
        latency_ms=3.0,
    )
    answer = make_answer(generated)
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
