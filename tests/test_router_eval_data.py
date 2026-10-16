import json
from collections import Counter

import pytest

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import Settings
from contract_assistant.evaluation.questions import load_questions

N_QUESTIONS = 60
KIND_COUNTS = {"single": 30, "compound": 20, "out_of_scope": 10}
SPLIT_COUNTS = {"dev": 30, "test": 30}
KIND_COUNTS_PER_SPLIT = {"single": 15, "compound": 10, "out_of_scope": 5}
MIN_COMPOUND_TYPES = 2
MAX_COMPOUND_TYPES = 5
PLAN_QUESTIONS = {
    "Which law governs this agreement?": {"Governing Law"},
    "Who are the parties?": {"Parties"},
    "When was it signed, and when does it take effect?": {"Agreement Date", "Effective Date"},
    "When does it expire, and does it renew on its own?": {
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
    },
    "Can either side walk away without cause?": {"Termination For Convenience"},
    "Is there a non-compete?": {"Non-Compete"},
    "Are there exclusivity obligations?": {"Exclusivity"},
    "Is liability capped anywhere? Is anything uncapped?": {
        "Cap On Liability",
        "Uncapped Liability",
    },
    "Can this be assigned if the company is sold?": {"Anti-Assignment", "Change Of Control"},
    "What licences are granted, and are any perpetual?": {
        "License Grant",
        "Irrevocable Or Perpetual License",
    },
    "Give me all the important dates.": {
        "Agreement Date",
        "Effective Date",
        "Expiration Date",
        "Renewal Term",
        "Notice Period To Terminate Renewal",
    },
    "What does it say about data privacy?": set(),
}


def normalise(text):
    return " ".join(text.lower().split())


@pytest.fixture(scope="module")
def questions():
    return load_questions(Settings().router_eval_path)


@pytest.fixture(scope="module")
def types():
    return load_clause_types(Settings().clause_types_path)


def test_file_has_sixty_questions(questions):
    assert len(questions) == N_QUESTIONS


def test_ids_are_unique(questions):
    ids = [q.id for q in questions]
    assert len(set(ids)) == len(ids)


def test_ids_run_from_q01_to_q60(questions):
    assert {q.id for q in questions} == {f"q{i:02d}" for i in range(1, N_QUESTIONS + 1)}


def test_questions_are_unique(questions):
    texts = [normalise(q.question) for q in questions]
    assert len(set(texts)) == len(texts)


def test_every_gold_name_is_a_clause_type(questions, types):
    names = {clause_type.name for clause_type in types}
    assert {name for q in questions for name in q.gold} - names == set()


def test_kind_counts(questions):
    assert dict(Counter(q.kind for q in questions)) == KIND_COUNTS


def test_split_counts(questions):
    assert dict(Counter(q.split for q in questions)) == SPLIT_COUNTS


@pytest.mark.parametrize("split", SPLIT_COUNTS)
def test_each_split_has_the_same_kind_mix(questions, split):
    kinds = Counter(q.kind for q in questions if q.split == split)
    assert dict(kinds) == KIND_COUNTS_PER_SPLIT


def test_out_of_scope_questions_have_empty_gold(questions):
    assert all(q.gold == [] for q in questions if q.kind == "out_of_scope")


def test_single_questions_have_exactly_one_gold_type(questions):
    assert all(len(q.gold) == 1 for q in questions if q.kind == "single")


def test_compound_questions_have_two_to_five_gold_types(questions):
    sizes = [len(q.gold) for q in questions if q.kind == "compound"]
    assert all(MIN_COMPOUND_TYPES <= size <= MAX_COMPOUND_TYPES for size in sizes)


def test_gold_has_no_duplicates(questions):
    assert all(len(set(q.gold)) == len(q.gold) for q in questions)


def test_every_clause_type_is_in_some_gold_set(questions, types):
    covered = {name for q in questions for name in q.gold}
    assert {clause_type.name for clause_type in types} - covered == set()


def test_plan_questions_are_present(questions):
    assert set(PLAN_QUESTIONS) <= {q.question for q in questions}


def test_plan_questions_have_the_planned_gold_sets(questions):
    wrong = [
        q.id
        for q in questions
        if q.question in PLAN_QUESTIONS and set(q.gold) != PLAN_QUESTIONS[q.question]
    ]
    assert wrong == []


def test_no_question_copies_a_router_paraphrase(questions):
    paraphrases = json.loads(Settings().router_descriptions_path.read_text(encoding="utf-8"))
    known = {normalise(text) for texts in paraphrases.values() for text in texts}
    assert [q.id for q in questions if normalise(q.question) in known] == []


def test_no_question_copies_a_cuad_description(questions, types):
    known = {normalise(clause_type.description) for clause_type in types}
    assert [q.id for q in questions if normalise(q.question) in known] == []
