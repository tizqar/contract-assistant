import html
import json
import re

import pytest

from contract_assistant.answer import tier
from contract_assistant.term_sheet import NOT_STATED
from contract_assistant.ui.views import definition_citation, top_citation
from contract_assistant.web.core import DEFINITION_LABEL, SUMMARIES_OFF, Thresholds, view_url
from contract_assistant.web.generate import (
    EXTRACTING_TERMS,
    NO_TERM_CLAUSES,
    NOTHING_TO_SUMMARIZE,
    WRITING_SUMMARY,
)
from contract_assistant.web.insights import (
    FLAG_CHANGED,
    NO_OBLIGATION_CLAUSES,
    NO_PASSAGE_ATTACHED,
    UNKNOWN_FILTER,
    UNKNOWN_LANGUAGE,
    WRITING_FLAG,
    WRITING_OBLIGATIONS,
)
from tests.support.caches import make_cache
from tests.support.fakes import ITEM, EchoTextClient, FakeLLMClient
from tests.web.conftest import SHORT_DOC_ID, frames

DOC = f"/documents/{SHORT_DOC_ID}"
TAU = Thresholds(0.06, 0.91)
ACTION_WORDS = slice(2, 6)
INVENTED = {"clause_type": "Governing Law", "party": "Mars", "modality": "must", "action": "x"}


class ObligationClient(EchoTextClient):
    """Echo client that also answers an obligations request: one entry per clause item."""

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        if "JSON array" not in system:
            return super().complete_text(system, user, max_tokens)
        entries = [
            {
                "clause_type": name,
                "party": " ".join(words[:2]),
                "modality": "must",
                "action": " ".join(words[ACTION_WORDS]),
            }
            for name, quote in ITEM.findall(user)
            if len(words := quote.split()) >= ACTION_WORDS.stop
        ]
        return json.dumps([*entries, INVENTED])


@pytest.fixture
def gen_client(make_client):
    return make_client(gen=EchoTextClient())


def written_summary(client, language: str = "en") -> dict:
    return frames(client, "POST", f"{DOC}/summary", data={"language": language})[-1]


def test_the_summary_tab_starts_writing_when_nothing_is_saved(gen_client):
    page = gen_client.get(f"{DOC}/summary").text
    assert "data-autosubmit" in page and f'action="{DOC}/summary"' in page
    assert 'name="language" value="en"' in page and "data-stop" in page
    assert WRITING_SUMMARY in page


def test_the_summary_stream_shows_the_status_then_the_summary(gen_client):
    lines = frames(gen_client, "POST", f"{DOC}/summary", data={"language": "en"})
    assert "status-line" in lines[0]["html"] and WRITING_SUMMARY in lines[0]["html"]
    assert any("The contract states" in line.get("html", "") for line in lines[:-1])
    assert lines[-1]["done"] is True
    panel = lines[-1]["oob"]["summary-panel-en"]
    assert "badge-generated" in panel and "Based on" in panel
    assert "source-dot-found" in panel and "Written by the language model fake-gen" in panel


def test_a_saved_summary_is_shown_as_the_stream_ended_it(gen_client):
    panel = written_summary(gen_client)["oob"]["summary-panel-en"]
    page = gen_client.get(f"{DOC}/summary").text
    assert panel in page and 'id="summary-panel-en"' in page
    assert "data-autosubmit" not in page
    assert 'id="fulltext-block"' in page and f'action="{DOC}/fulltext"' in page


def test_the_serbian_summary_is_kept_apart(gen_client):
    written_summary(gen_client)
    page = gen_client.get(f"{DOC}/summary", params={"language": "sr"}).text
    assert "data-autosubmit" in page and 'name="language" value="sr"' in page
    assert "written in Serbian" in written_summary(gen_client, "sr")["oob"]["summary-panel-sr"]


def test_an_unknown_language_is_refused(gen_client):
    response = gen_client.get(f"{DOC}/summary", params={"language": "fr"})
    assert response.status_code == 400 and response.text == UNKNOWN_LANGUAGE


def test_without_a_language_model_the_tabs_say_so_and_start_nothing(client):
    for tab in ("summary", "terms", "obligations", "fulltext"):
        page = client.get(f"{DOC}/{tab}").text
        assert SUMMARIES_OFF in page and "data-stream" not in page
    assert "Write a note" not in client.get(f"{DOC}/review").text


def test_a_contract_without_clauses_has_nothing_to_generate(make_client, settings):
    client = make_client(gen=EchoTextClient())
    cache = make_cache("Nothing here at all.", settings, {})
    client.app.state.workspace.services.store.save(cache)
    doc = f"/documents/{cache.doc_id}"
    summary = client.get(f"{doc}/summary").text
    assert NOTHING_TO_SUMMARIZE in summary and f'action="{doc}/fulltext"' in summary
    assert NO_TERM_CLAUSES in client.get(f"{doc}/terms").text
    assert NO_OBLIGATION_CLAUSES in client.get(f"{doc}/obligations").text


def test_the_full_text_overview_streams_its_stages_then_the_sections(gen_client):
    before = gen_client.get(f"{DOC}/fulltext").text
    assert f'action="{DOC}/fulltext"' in before and "data-autosubmit" not in before
    lines = frames(gen_client, "POST", f"{DOC}/fulltext")
    statuses = [line["html"] for line in lines if "html" in line]
    assert "Writing the overview · section 1 of 4 · " in statuses[0]
    assert any("merging the sections" in status for status in statuses)
    block = lines[-1]["oob"]["fulltext-block"]
    assert "badge-generated" in block and "4 sections · " in block
    assert "Show in document" in block and "label=Section+1" in block
    after = gen_client.get(f"{DOC}/fulltext").text
    assert block in after and "Write the full-text overview" not in after


def test_the_key_terms_stream_ends_with_the_rows_and_the_dates(gen_client):
    assert "data-autosubmit" in gen_client.get(f"{DOC}/terms").text
    lines = frames(gen_client, "POST", f"{DOC}/terms")
    assert EXTRACTING_TERMS in lines[0]["html"]
    panel = lines[-1]["oob"]["terms-panel"]
    assert '<span class="serif">the Customer</span>' in panel and "from Parties" in panel
    assert f'<em class="muted">{NOT_STATED}</em>' in panel
    assert "Download JSON" in panel and f'href="{DOC}/terms.json"' in panel
    assert "Key dates" in panel and "Agreement date" in panel and "not readable" in panel
    assert panel in gen_client.get(f"{DOC}/terms").text


def test_a_withheld_term_sheet_shows_why_with_the_rows_and_is_not_saved(make_client):
    client = make_client(gen=FakeLLMClient(["no json", "still none"]))
    panel = frames(client, "POST", f"{DOC}/terms")[-1]["oob"]["terms-panel"]
    assert "The term sheet is not shown because" in panel and "not the JSON object" in panel
    assert f'<em class="muted">{NOT_STATED}</em>' in panel
    assert "Download JSON" not in panel and "Key dates" not in panel
    assert client.get(f"{DOC}/terms.json").status_code == 404


def test_the_saved_term_sheet_downloads_as_json(gen_client):
    assert gen_client.get(f"{DOC}/terms.json").status_code == 404
    frames(gen_client, "POST", f"{DOC}/terms")
    response = gen_client.get(f"{DOC}/terms.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "attachment" in response.headers["content-disposition"]
    assert response.json()["doc_id"] == SHORT_DOC_ID
    other = gen_client.get(f"{DOC}/terms.json", params={"low": 0.2, "high": 0.8})
    assert other.status_code == 404


def test_the_review_lists_the_flags_with_their_severity(gen_client):
    ws = gen_client.app.state.workspace
    flags = ws.flags(ws.load(SHORT_DOC_ID), TAU)
    page = gen_client.get(f"{DOC}/review").text
    assert '<span class="badge badge-medium badge-sm">Medium</span>' in page
    assert '<span class="badge badge-low badge-sm">Low</span>' in page
    assert "No cap on liability found" in page and "Cap on Liability (not found)" in page
    assert NO_PASSAGE_ATTACHED in page and "Show passage" in page
    with_quotes = [flag for flag in flags if any(item.citation for item in flag.items)]
    assert page.count(">Write a note<") == len(with_quotes) == len(flags) - 1


def test_a_note_on_a_flag_streams_and_ends_with_the_generated_text(gen_client):
    ws = gen_client.app.state.workspace
    flags = ws.flags(ws.load(SHORT_DOC_ID), TAU)
    index = next(i for i, flag in enumerate(flags) if flag.rule == "auto_renewal")
    lines = frames(
        gen_client, "POST", f"{DOC}/review/{index}/note", data={"rule": "auto_renewal"}
    )
    assert WRITING_FLAG in lines[0]["html"]
    note = lines[-1]["html"]
    assert "badge-generated" in note and "The contract states" in note
    assert "Automatic renewal with a notice window" in note and "not legal advice" in note


def test_a_note_on_a_flag_that_moved_or_has_no_passage_is_refused(gen_client):
    moved = gen_client.post(f"{DOC}/review/0/note", data={"rule": "auto_renewal"})
    assert moved.status_code == 400 and moved.text == FLAG_CHANGED
    assert gen_client.post(f"{DOC}/review/99/note").text == FLAG_CHANGED
    bare = gen_client.post(f"{DOC}/review/0/note", data={"rule": "no_liability_cap"})
    assert bare.status_code == 400 and bare.text == NO_PASSAGE_ATTACHED


def test_the_thresholds_in_the_query_change_the_review_flags(client):
    default = client.get(f"{DOC}/review").text
    stricter = client.get(f"{DOC}/review", params={"low": 0.5, "high": 0.91}).text
    assert "Exclusivity obligation" in default and "Exclusivity obligation" not in stricter
    assert "No insurance obligation found" in stricter
    assert "No insurance obligation found" not in default


def test_the_obligations_stream_ends_with_the_entries_by_party(make_client):
    client = make_client(gen=ObligationClient())
    assert "data-autosubmit" in client.get(f"{DOC}/obligations").text
    lines = frames(client, "POST", f"{DOC}/obligations")
    assert WRITING_OBLIGATIONS in lines[0]["html"]
    panel = lines[-1]["oob"]["obligations-panel"]
    assert 'class="group-title"' in panel and "badge badge-must badge-sm" in panel
    assert "Show in document" in panel
    assert "1 entries of the language model were not in the clauses and were dropped." in panel
    ws = client.app.state.workspace
    sheet = ws.obligations.load(SHORT_DOC_ID, "fake-gen", TAU.low, TAU.high)
    first = sheet.obligations[0]
    assert html.escape(" ".join(first.action.split()), quote=False) in panel
    page = client.get(f"{DOC}/obligations").text
    assert panel in page and "data-autosubmit" not in page


def test_obligations_the_model_cannot_list_are_withheld_and_asked_for_again(gen_client):
    panel = frames(gen_client, "POST", f"{DOC}/obligations")[-1]["oob"]["obligations-panel"]
    assert "The obligations are not shown because" in panel and "not the JSON list" in panel
    assert "data-autosubmit" in gen_client.get(f"{DOC}/obligations").text


def active_chip(page: str) -> str:
    return re.search(r'class="chip chip-active"[^>]*>([^<]+)<', page).group(1)


def test_the_checklist_filters_by_status(client):
    page = client.get(f"{DOC}/checklist").text
    assert active_chip(page) == "All"
    for status in ("found", "check", "not_found"):
        assert f"badge badge-{status} badge-sm" in page
    found = client.get(f"{DOC}/checklist", params={"status": "found"}).text
    assert active_chip(found) == "Found"
    assert "badge badge-found badge-sm" in found
    assert "badge badge-check badge-sm" not in found
    assert "badge badge-not_found badge-sm" not in found
    unknown = client.get(f"{DOC}/checklist", params={"status": "maybe"})
    assert unknown.status_code == 400 and unknown.text == UNKNOWN_FILTER


def test_a_checklist_row_opens_its_top_passage(client):
    cache = client.app.state.workspace.load(SHORT_DOC_ID)
    citation = top_citation(cache, "Governing Law")
    status = tier(citation.conf, TAU.low, TAU.high)
    page = client.get(f"{DOC}/checklist").text
    assert html.escape(view_url(SHORT_DOC_ID, "Governing Law", citation, status)) in page


def test_definitions_link_to_the_document_pane(client):
    ws = client.app.state.workspace
    first = ws.definitions(ws.load(SHORT_DOC_ID))[0]
    url = view_url(SHORT_DOC_ID, DEFINITION_LABEL, definition_citation(first), "definition")
    page = client.get(f"{DOC}/definitions").text
    assert html.escape(url) in page and "Show in document" in page
    assert f'data-label="{DEFINITION_LABEL}"' in client.get(url).text


def test_the_report_builds_from_what_is_saved_and_downloads(gen_client):
    assert gen_client.get(f"{DOC}/report.docx").status_code == 404
    written_summary(gen_client)
    response = gen_client.post(f"{DOC}/report")
    assert response.status_code == 200
    assert (
        "Report built without the key terms, the obligations and the full-text overview."
        in response.text
    )
    assert f'href="{DOC}/report.docx"' in response.text and "data-thresholds" in response.text
    docx = gen_client.get(f"{DOC}/report.docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"
    markdown = gen_client.get(f"{DOC}/report.md").text
    assert "Contract report" in markdown and "The contract states" in markdown


def test_the_report_without_a_language_model_names_every_missing_part(client):
    text = client.post(f"{DOC}/report").text
    assert "the summary, the key terms, the obligations and the full-text overview" in text
    assert client.get(f"{DOC}/report.md").status_code == 200
