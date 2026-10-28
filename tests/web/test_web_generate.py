import pytest

from contract_assistant.explain import Explanation
from contract_assistant.summary import Summary
from contract_assistant.term_sheet import TermSheet
from contract_assistant.web.core import SUMMARIES_OFF, Thresholds, Workspace, WorkspaceError
from contract_assistant.web.generate import (
    NOTHING_TO_SUMMARIZE,
    Step,
    explanation_steps,
    saved_summary,
    saved_term_sheet,
    summary_steps,
    term_sheet_steps,
)
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient
from tests.web.conftest import SHORT_DOC_ID, seed_demo

TAU = Thresholds(0.06, 0.91)


@pytest.fixture
def ws(make_services, database, settings, reader):
    seed_demo(database, settings, reader)
    return Workspace(make_services(gen=EchoTextClient()), database).for_user(reader)


def test_the_summary_is_written_once_then_loaded(ws):
    cache = ws.load(SHORT_DOC_ID)
    assert saved_summary(ws, cache, TAU, "en") is None
    steps = list(summary_steps(ws, cache, TAU, "en"))
    assert isinstance(steps[0], Step) and steps[0].text == ""
    assert isinstance(steps[-1], Summary) and steps[-1].text
    assert saved_summary(ws, cache, TAU, "en") is not None
    assert list(summary_steps(ws, cache, TAU, "en")) == [steps[-1]]


def test_a_serbian_summary_is_kept_apart(ws):
    cache = ws.load(SHORT_DOC_ID)
    list(summary_steps(ws, cache, TAU, "en"))
    assert saved_summary(ws, cache, TAU, "sr") is None


def test_step_status_names_the_retry_reason():
    assert Step("Writing", "", 3).status == "Writing · 3 s"
    assert Step("Writing", "x", 4, "a number was wrong").status.startswith(
        "The first reply was not shown because a number was wrong"
    )


def test_a_contract_without_clauses_has_nothing_to_summarize(ws, settings):
    cache = make_cache("Nothing here.", settings)
    with pytest.raises(WorkspaceError, match=NOTHING_TO_SUMMARIZE):
        list(summary_steps(ws, cache, TAU, "en"))


def test_generation_off_is_a_notice(make_services, database, settings, reader):
    seed_demo(database, settings, reader)
    ws = Workspace(make_services(), database).for_user(reader)
    cache = ws.load(SHORT_DOC_ID)
    with pytest.raises(WorkspaceError, match=SUMMARIES_OFF):
        list(summary_steps(ws, cache, TAU, "en"))
    assert saved_summary(ws, cache, TAU, "en") is None


def test_the_term_sheet_is_extracted_once_then_loaded(ws):
    cache = ws.load(SHORT_DOC_ID)
    assert saved_term_sheet(ws, cache, TAU) is None
    steps = list(term_sheet_steps(ws, cache, TAU))
    assert isinstance(steps[0], Step)
    assert isinstance(steps[-1], TermSheet)
    assert list(term_sheet_steps(ws, cache, TAU)) == [steps[-1]]


def test_an_explanation_streams_then_ends_with_the_explanation(ws):
    cache = ws.load(SHORT_DOC_ID)
    quote = cache.clauses["Governing Law"][0].text
    steps = list(explanation_steps(ws, "Governing Law", quote, "en"))
    assert all(isinstance(step, Step) for step in steps[:-1])
    assert isinstance(steps[-1], Explanation)
    assert steps[-1].text
