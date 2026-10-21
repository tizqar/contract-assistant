import pytest

from contract_assistant.intents import Intent, detect_intent


@pytest.mark.parametrize(
    "question",
    ["explain that", "objasni to", "can you explain the passage above"],
)
def test_explain_command_in_english_asks_for_english(question):
    assert detect_intent(question) == Intent(kind="explain", language="en")


def test_explain_in_serbian_asks_for_serbian():
    assert detect_intent("Explain in Serbian") == Intent(kind="explain", language="sr")


def test_long_explain_question_without_a_reference_word_is_routed():
    assert detect_intent("Explain the governing law clause in detail please") is None


@pytest.mark.parametrize(
    "question",
    [
        "summarize",
        "Summary please",
        "Can you summarise this contract in a few sentences?",
        "give me an overview of the agreement",
    ],
)
def test_summary_command_is_detected(question):
    assert detect_intent(question) == Intent(kind="summary", language="en")


def test_summary_of_one_clause_is_routed():
    assert detect_intent("Summarize the termination clause") is None


@pytest.mark.parametrize(
    "question", ["Which law governs?", "and what about the notice period?"]
)
def test_plain_questions_are_not_commands(question):
    assert detect_intent(question) is None


def test_intent_language_defaults_to_english():
    assert Intent(kind="summary").language == "en"
