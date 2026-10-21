import json
import logging
import time

import pytest

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.evaluation.questions import load_questions
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.router.clients import LLMError
from contract_assistant.router.llm import (
    CONTEXT_FOOTER,
    CONTEXT_HEADER,
    CURRENT_QUESTION,
    EXAMPLES,
    MAX_TURNS,
    MS_PER_S,
    NO_TYPES,
    LLMRouter,
    route_schema,
    user_message,
)
from tests.support.fakes import FakeLLMClient, FixedRouter

QUESTION = "Which law governs this agreement?"
EMPTY_REPLY = '{"clause_types": []}'
FALLBACK_ROUTE = RouteResult(clause_types=["Parties"], kind="embedding", latency_ms=1.0)
FAILED_ATTEMPT_S = 0.05
ARROW = "\u2192"
TURNS = [
    Turn(question="Which law governs?", clause_types=["Governing Law"]),
    Turn(question="Who are the parties?", clause_types=["Parties", "Agreement Date"]),
]


class SlowFailingClient:
    model = "slow-llm"

    def complete_json(self, system, user, schema):
        time.sleep(FAILED_ATTEMPT_S)
        raise LLMError("timed out")


@pytest.fixture(scope="module")
def types():
    return load_clause_types(Settings().clause_types_path)


def route_with(reply, types):
    fallback = FixedRouter({QUESTION: FALLBACK_ROUTE})
    return LLMRouter(FakeLLMClient([reply]), fallback, types).route(QUESTION)


def sent_request(types):
    client = FakeLLMClient([EMPTY_REPLY])
    LLMRouter(client, FixedRouter({}), types).route(QUESTION)
    return client.calls[0]


def normalised(question):
    return question.casefold().strip().rstrip("?").strip()


def test_valid_reply_gives_the_names_in_the_replys_order(types):
    result = route_with('{"clause_types": ["Insurance", "Governing Law"]}', types)
    assert result.clause_types == ["Insurance", "Governing Law"]
    assert result.kind == "llm"


def test_valid_reply_is_kept_as_raw(types):
    reply = '{"clause_types": ["Insurance"]}'
    assert route_with(reply, types).raw == reply


def test_empty_list_is_an_out_of_scope_route_not_a_fallback(types):
    fallback = FixedRouter({})
    result = LLMRouter(FakeLLMClient([EMPTY_REPLY]), fallback, types).route(QUESTION)
    assert result.clause_types == []
    assert result.kind == "llm"
    assert fallback.questions == []


@pytest.mark.parametrize(
    "reply",
    [
        '{"clause_types": ["Governing Law"',
        '{"clause_types": ["Force Majeure"]}',
        '{"clause_types": [], "answer": "New York"}',
    ],
    ids=["malformed_json", "unknown_clause_type", "extra_key"],
)
def test_unusable_reply_gives_the_fallback_route(reply, types):
    result = route_with(reply, types)
    assert result.clause_types == FALLBACK_ROUTE.clause_types
    assert result.kind == "llm_fallback"
    assert result.raw == "ValidationError"


def test_llm_error_gives_the_fallback_route_with_the_error_class_as_raw(types):
    result = route_with(LLMError("timed out"), types)
    assert result.clause_types == FALLBACK_ROUTE.clause_types
    assert result.kind == "llm_fallback"
    assert result.raw == "LLMError"


def test_fallback_is_asked_once_with_the_question(types):
    fallback = FixedRouter({QUESTION: FALLBACK_ROUTE})
    LLMRouter(FakeLLMClient(["not json"]), fallback, types).route(QUESTION)
    assert fallback.questions == [QUESTION]


def test_other_errors_propagate_without_asking_the_fallback(types):
    fallback = FixedRouter({})
    router = LLMRouter(FakeLLMClient([KeyError("bug")]), fallback, types)
    with pytest.raises(KeyError):
        router.route(QUESTION)
    assert fallback.questions == []


def test_duplicate_names_are_removed_keeping_the_first_occurrences(types):
    reply = '{"clause_types": ["Insurance", "Governing Law", "Insurance", "Governing Law"]}'
    assert route_with(reply, types).clause_types == ["Insurance", "Governing Law"]


def test_fallback_is_logged_once_with_model_and_error_class(types, caplog):
    client = FakeLLMClient(["not json"], model="qwen-test")
    router = LLMRouter(client, FixedRouter({QUESTION: FALLBACK_ROUTE}), types)
    with caplog.at_level(logging.WARNING):
        router.route(QUESTION)
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "qwen-test" in record.getMessage()
    assert "ValidationError" in record.getMessage()


def test_fallback_latency_includes_the_failed_attempt(types):
    router = LLMRouter(SlowFailingClient(), FixedRouter({QUESTION: FALLBACK_ROUTE}), types)
    assert router.route(QUESTION).latency_ms >= FAILED_ATTEMPT_S * MS_PER_S


def test_system_prompt_lists_every_clause_type_with_its_description_in_file_order(types):
    system, _, _ = sent_request(types)
    listed = [line for line in system.splitlines() if line.startswith("- ")]
    assert listed == [f"- {t.name}: {' '.join(t.description.split())}" for t in types]


def test_system_prompt_has_no_irregular_whitespace(types):
    system, _, _ = sent_request(types)
    assert "\xa0" not in system
    assert "  " not in system


def test_system_prompt_shows_each_worked_example_with_its_json_output(types):
    system, _, _ = sent_request(types)
    for question, names in EXAMPLES:
        assert f"Question: {question}\nOutput: {json.dumps({'clause_types': names})}" in system


def test_user_message_is_the_question_as_given(types):
    _, user, _ = sent_request(types)
    assert user == QUESTION


def test_user_message_without_turns_is_the_question_alone():
    assert user_message(QUESTION, ()) == QUESTION


def test_user_message_with_turns_lists_them_then_the_current_question():
    assert user_message("and the notice period?", TURNS) == "\n".join(
        [
            CONTEXT_HEADER,
            f"- Which law governs? {ARROW} Governing Law",
            f"- Who are the parties? {ARROW} Parties, Agreement Date",
            "",
            CONTEXT_FOOTER,
            "",
            f"{CURRENT_QUESTION}and the notice period?",
        ]
    )


def test_user_message_shows_none_for_a_turn_without_types():
    turns = [Turn(question="How many pages?", clause_types=[])]
    assert f"- How many pages? {ARROW} {NO_TYPES}" in user_message(QUESTION, turns)


def test_user_message_keeps_only_the_last_turns_in_order():
    turns = [Turn(question=f"Question {n}", clause_types=["Parties"]) for n in range(6)]
    listed = [line for line in user_message(QUESTION, turns).splitlines() if line[:1] == "-"]
    assert len(listed) == MAX_TURNS
    assert listed == [f"- Question {n} {ARROW} Parties" for n in range(6 - MAX_TURNS, 6)]
    assert "Question 0" not in user_message(QUESTION, turns)
    assert "Question 1" not in user_message(QUESTION, turns)


def test_route_sends_the_message_with_turns_as_the_user_content(types):
    client = FakeLLMClient([EMPTY_REPLY])
    LLMRouter(client, FixedRouter({}), types).route(QUESTION, TURNS)
    assert client.calls[0][1] == user_message(QUESTION, TURNS)


def test_route_keeps_the_system_prompt_when_turns_are_given(types):
    client = FakeLLMClient([EMPTY_REPLY, EMPTY_REPLY])
    router = LLMRouter(client, FixedRouter({}), types)
    router.route(QUESTION)
    router.route(QUESTION, TURNS)
    assert client.calls[0][0] == client.calls[1][0] == router.system


def test_fallback_gets_the_question_without_turns(types):
    fallback = FixedRouter({QUESTION: FALLBACK_ROUTE})
    LLMRouter(FakeLLMClient(["not json"]), fallback, types).route(QUESTION, TURNS)
    assert fallback.questions == [QUESTION]
    assert fallback.turns == [[]]


def test_schema_is_route_schema_of_the_names_in_file_order(types):
    _, _, schema = sent_request(types)
    assert schema == route_schema([t.name for t in types])


def test_route_schema_requires_a_list_of_names_and_forbids_other_keys():
    assert route_schema(["A", "B"]) == {
        "type": "object",
        "additionalProperties": False,
        "required": ["clause_types"],
        "properties": {
            "clause_types": {"type": "array", "items": {"type": "string", "enum": ["A", "B"]}},
        },
    }


def test_worked_examples_only_use_clause_type_names(types):
    names = {t.name for t in types}
    assert all(set(example) <= names for _, example in EXAMPLES)


def test_out_of_scope_worked_example_is_not_a_clause_type_question(types):
    assert ("How many pages does this contract have?", []) in EXAMPLES


def test_worked_examples_are_not_evaluation_questions():
    evaluation = {normalised(q.question) for q in load_questions(Settings().router_eval_path)}
    assert not evaluation & {normalised(question) for question, _ in EXAMPLES}


@pytest.mark.slow
def test_local_model_routes_a_governing_law_question(types):
    from contract_assistant.router.clients import OllamaClient

    settings = Settings()
    client = OllamaClient.from_settings(settings)
    if not client.available():
        pytest.skip(f"{settings.llm_model} is not available from the local Ollama server")
    fallback = FixedRouter({QUESTION: FALLBACK_ROUTE})
    result = LLMRouter(client, fallback, types).route(QUESTION)
    assert result.kind == "llm", result.raw
    assert "Governing Law" in result.clause_types
