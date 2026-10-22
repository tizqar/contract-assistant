import html
import json
import re

import pytest
from markupsafe import escape

from contract_assistant.answer import tier
from contract_assistant.term_sheet import NOT_STATED
from contract_assistant.ui.views import definition_citation, display_name, top_citation
from contract_assistant.web import copy
from contract_assistant.web.core import DEFINITION_LABEL, Thresholds, view_url
from contract_assistant.web.generate import (
    EXTRACTING_TERMS,
    NO_TERM_CLAUSES,
    NOTHING_TO_SUMMARIZE,
    WRITING_SUMMARY,
)
from contract_assistant.web.insights import (
    AI_NEEDED,
    CHECK_HINT,
    FLAG_CHANGED,
    FULLTEXT_BUTTON,
    NO_OBLIGATION_CLAUSES,
    NO_PASSAGE_ATTACHED,
    NO_PASSAGE_SHOWN,
    SECTION_ROUTES,
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
# In the short demo contract: a clause to check, and an extra clause type found but capped.
TO_CHECK = "Parties"
CAPPED = "Dispute Resolution"
# Words of the first version of the insights that a plain reader should not see.
TECHNICAL = ("Written by the language model", "fake-gen", "Verify", "#insight-body")


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


def decide(client, name: str, verdict: str):
    top = client.app.state.workspace.load(SHORT_DOC_ID).clauses[name][0]
    response = client.post(
        f"{DOC}/decide",
        data={"clause_type": name, "verdict": verdict, "start": top.start, "end": top.end},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return top


def clause_row(page: str, name: str) -> str:
    """The HTML of the All clauses row of a clause type."""
    start = page.index(f'<span class="clause-name">{html.escape(display_name(name))}</span>')
    end = page.find('<div class="clause-row', start)
    return page[start : end if end >= 0 else len(page)]


# Overview


def test_the_overview_lists_the_six_sections_with_their_lazy_load_urls(client):
    page = client.get(f"{DOC}/overview").text
    positions = []
    for section, title, line in copy.SECTIONS:
        body = f'id="sec-{section}-body" hx-get="{DOC}/{SECTION_ROUTES[section]}" hx-trigger="load"'
        assert body in page
        assert f'href="#sec-{section}"' in page and f">{title}<" in page
        assert str(escape(line)) in page
        positions.append(page.index(f'id="sec-{section}"'))
    assert positions == sorted(positions)
    assert 'class="section-chips"' in page and 'data-tau="0.06-0.91"' in page
    assert "#insight-body" not in page


def test_the_what_happened_line_carries_the_counts(client):
    entry = client.app.state.workspace.entry(SHORT_DOC_ID)
    total = entry.found + entry.check + entry.not_found
    line = copy.WHAT_HAPPENED.format(
        total=total, found=entry.found, check=entry.check, not_found=entry.not_found
    )
    page = client.get(f"{DOC}/overview").text
    assert f'<p class="happened-line">{line}</p>' in page
    assert CHECK_HINT in page and "You confirmed" not in page
    assert f"Show the {entry.check} passages to check" in page
    card = client.get(f"{DOC}/overview/happened").text
    assert line in card and f'data-refresh="{DOC}/overview/happened"' in card


def test_the_what_happened_line_follows_the_thresholds(client):
    page = client.get(f"{DOC}/overview", params={"low": 0.5, "high": 0.91}).text
    clauses = client.get(f"{DOC}/checklist", params={"low": 0.5, "high": 0.91}).text
    found, check, not_found = (
        re.search(rf"{label}<span class=\"chip-count\">(\d+)<", clauses).group(1)
        for label in ("Found", "Check this", "Not found")
    )
    line = copy.WHAT_HAPPENED.format(
        total=int(found) + int(check) + int(not_found),
        found=found,
        check=check,
        not_found=not_found,
    )
    assert line in page and 'data-tau="0.5-0.91"' in page


# Summary and the overview of the whole text


def test_the_summary_section_starts_writing_when_nothing_is_saved(gen_client):
    page = gen_client.get(f"{DOC}/summary").text
    assert "data-autosubmit" in page and f'action="{DOC}/summary"' in page
    assert 'name="language" value="en"' in page and "data-stop" in page
    assert WRITING_SUMMARY in page


def test_the_summary_stream_shows_the_status_then_the_summary(gen_client):
    lines = frames(gen_client, "POST", f"{DOC}/summary", data={"language": "en"})
    assert "status-line" in lines[0]["html"] and WRITING_SUMMARY in lines[0]["html"]
    assert not re.search(r"\d+ s<", lines[0]["html"])
    assert any("The contract states" in line.get("html", "") for line in lines[:-1])
    assert lines[-1]["done"] is True
    panel = lines[-1]["oob"]["summary-panel-en"]
    assert "badge-generated" in panel and "Based on" in panel
    assert "source-dot-found" in panel and "source-dot-check" in panel
    assert 'hx-target="#sec-summary-body"' in panel
    for word in (*TECHNICAL, "Text in double quotes", "Not validated", "*<"):
        assert word not in panel


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
    panel = written_summary(gen_client, "sr")["oob"]["summary-panel-sr"]
    assert "The contract states" in panel
    assert re.search(r'seg-item seg-item-active"\s+aria-pressed="true"[^>]*>Serbian<', panel)


def test_an_unknown_language_is_refused(gen_client):
    response = gen_client.get(f"{DOC}/summary", params={"language": "fr"})
    assert response.status_code == 400 and response.text == UNKNOWN_LANGUAGE


def test_without_a_language_model_the_sections_say_so_and_start_nothing(client):
    for route in ("summary", "terms", "obligations", "fulltext"):
        page = client.get(f"{DOC}/{route}").text
        assert AI_NEEDED in page and "data-stream" not in page
    assert 'class="seg"' not in client.get(f"{DOC}/summary").text
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


def test_the_whole_text_overview_streams_its_stages_then_the_sections(gen_client):
    before = gen_client.get(f"{DOC}/fulltext").text
    assert f'action="{DOC}/fulltext"' in before and "data-autosubmit" not in before
    assert FULLTEXT_BUTTON in before
    lines = frames(gen_client, "POST", f"{DOC}/fulltext")
    statuses = [line["html"] for line in lines if "html" in line]
    assert "Writing the overview of the whole text: section 1 of 4<" in statuses[0]
    assert any("putting the sections together" in status for status in statuses)
    block = lines[-1]["oob"]["fulltext-block"]
    assert "badge-generated" in block and "Section by section · 4" in block
    assert "Show in document" in block and "label=Section+1" in block
    assert not re.search(r"\d+ s\b", block) and "fake-gen" not in block
    after = gen_client.get(f"{DOC}/fulltext").text
    assert block in after and FULLTEXT_BUTTON not in after


# Key facts


def test_the_key_facts_stream_ends_with_the_rows_and_the_dates(gen_client):
    assert "data-autosubmit" in gen_client.get(f"{DOC}/terms").text
    lines = frames(gen_client, "POST", f"{DOC}/terms")
    assert EXTRACTING_TERMS in lines[0]["html"]
    panel = lines[-1]["oob"]["terms-panel"]
    assert '<span class="serif">the Customer</span>' in panel
    assert 'title="Copied from: Parties"' in panel and "from Parties" not in panel
    assert f'<em class="muted">{NOT_STATED}</em>' in panel
    assert "Download the key facts as JSON" in panel and f'href="{DOC}/terms.json"' in panel
    assert "Key dates" in panel and "Agreement date" in panel and "could not be read" in panel
    assert "Show in document" in panel and "label=Parties&amp;start=" in panel
    assert panel in gen_client.get(f"{DOC}/terms").text


def test_a_withheld_term_sheet_shows_why_with_the_rows_and_is_not_saved(make_client):
    client = make_client(gen=FakeLLMClient(["no json", "still none"]))
    panel = frames(client, "POST", f"{DOC}/terms")[-1]["oob"]["terms-panel"]
    assert "text was held back because" in panel and "not the JSON object" in panel
    assert f'<em class="muted">{NOT_STATED}</em>' in panel
    assert "Download the key facts" not in panel and "Key dates" not in panel
    assert client.get(f"{DOC}/terms.json").status_code == 404


def test_the_saved_key_facts_download_as_json(gen_client):
    assert gen_client.get(f"{DOC}/terms.json").status_code == 404
    frames(gen_client, "POST", f"{DOC}/terms")
    response = gen_client.get(f"{DOC}/terms.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "attachment" in response.headers["content-disposition"]
    assert response.json()["doc_id"] == SHORT_DOC_ID
    other = gen_client.get(f"{DOC}/terms.json", params={"low": 0.2, "high": 0.8})
    assert other.status_code == 404


# Things to check


def test_the_things_to_check_carry_plain_severities(gen_client):
    ws = gen_client.app.state.workspace
    flags = ws.flags(ws.load(SHORT_DOC_ID), TAU)
    page = gen_client.get(f"{DOC}/review").text
    assert '<span class="badge badge-medium badge-sm">Worth a look</span>' in page
    assert '<span class="badge badge-low badge-sm">Minor</span>' in page
    assert ">Medium<" not in page and ">Low<" not in page and "Rests on" not in page
    assert "No cap on liability found" in page
    assert 'title="Based on: Cap on Liability (not found)"' in page
    assert NO_PASSAGE_SHOWN in page and "Show passage" in page
    assert f'data-refresh="{DOC}/review"' in page
    with_quotes = [flag for flag in flags if any(item.citation for item in flag.items)]
    assert page.count(">Write a note<") == len(with_quotes) == len(flags) - 1


def test_a_note_on_a_point_streams_and_ends_with_the_generated_text(gen_client):
    ws = gen_client.app.state.workspace
    flags = ws.flags(ws.load(SHORT_DOC_ID), TAU)
    index = next(i for i, flag in enumerate(flags) if flag.rule == "auto_renewal")
    lines = frames(
        gen_client, "POST", f"{DOC}/review/{index}/note", data={"rule": "auto_renewal"}
    )
    assert WRITING_FLAG in lines[0]["html"]
    note = lines[-1]["html"]
    assert "badge-generated" in note and "The contract states" in note
    assert "not legal advice" not in note and "fake-gen" not in note


def test_a_note_on_a_point_that_moved_or_has_no_passage_is_refused(gen_client):
    moved = gen_client.post(f"{DOC}/review/0/note", data={"rule": "auto_renewal"})
    assert moved.status_code == 400 and moved.text == FLAG_CHANGED
    assert gen_client.post(f"{DOC}/review/99/note").text == FLAG_CHANGED
    bare = gen_client.post(f"{DOC}/review/0/note", data={"rule": "no_liability_cap"})
    assert bare.status_code == 400 and bare.text == NO_PASSAGE_ATTACHED


def test_the_thresholds_in_the_query_change_the_things_to_check(client):
    default = client.get(f"{DOC}/review").text
    stricter = client.get(f"{DOC}/review", params={"low": 0.5, "high": 0.91}).text
    assert "Exclusivity obligation" in default and "Exclusivity obligation" not in stricter
    assert "No insurance obligation found" in stricter
    assert "No insurance obligation found" not in default


# Who must do what


def test_who_must_do_what_ends_with_the_entries_by_party(make_client):
    client = make_client(gen=ObligationClient())
    assert "data-autosubmit" in client.get(f"{DOC}/obligations").text
    lines = frames(client, "POST", f"{DOC}/obligations")
    assert WRITING_OBLIGATIONS in lines[0]["html"]
    panel = lines[-1]["oob"]["obligations-panel"]
    assert 'class="group-title"' in panel and "badge badge-must badge-sm" in panel
    assert "Show in document" in panel
    assert "1 entry could not be matched to the contract and was left out." in panel
    ws = client.app.state.workspace
    sheet = ws.obligations.load(SHORT_DOC_ID, "fake-gen", TAU.low, TAU.high)
    first = sheet.obligations[0]
    assert html.escape(" ".join(first.action.split()), quote=False) in panel
    page = client.get(f"{DOC}/obligations").text
    assert panel in page and "data-autosubmit" not in page


def test_obligations_the_model_cannot_list_are_withheld_and_asked_for_again(gen_client):
    panel = frames(gen_client, "POST", f"{DOC}/obligations")[-1]["oob"]["obligations-panel"]
    assert "text was held back because" in panel and "not the JSON list" in panel
    assert "data-autosubmit" in gen_client.get(f"{DOC}/obligations").text


# All clauses


def active_chip(page: str) -> str:
    return re.search(r'class="chip chip-active"[^>]*>([^<]+)<', page).group(1)


def test_all_clauses_filter_by_status(client):
    page = client.get(f"{DOC}/checklist").text
    assert active_chip(page) == "All"
    for status in ("found", "check", "not_found"):
        assert f'class="badge badge-{status}"' in page
    assert 'hx-target="#sec-clauses-body"' in page and not re.search(r"\d%(<| ·)", page)
    found = client.get(f"{DOC}/checklist", params={"status": "found"}).text
    assert active_chip(found) == "Found"
    assert 'class="badge badge-found"' in found
    assert 'class="badge badge-check"' not in found
    assert 'class="badge badge-not_found"' not in found
    assert f'data-refresh="{DOC}/checklist?status=found"' in found
    unknown = client.get(f"{DOC}/checklist", params={"status": "maybe"})
    assert unknown.status_code == 400 and unknown.text == UNKNOWN_FILTER


def test_a_clause_row_opens_its_top_passage(client):
    cache = client.app.state.workspace.load(SHORT_DOC_ID)
    citation = top_citation(cache, "Governing Law")
    status = tier(citation.conf, TAU.low, TAU.high)
    page = client.get(f"{DOC}/checklist").text
    assert html.escape(view_url(SHORT_DOC_ID, "Governing Law", citation, status)) in page


def test_a_clause_not_found_shows_no_passage(client):
    row = clause_row(client.get(f"{DOC}/checklist").text, "Cap On Liability")
    assert 'class="badge badge-not_found"' in row
    assert "clause-passage" not in row and "Show in document" not in row


def test_a_check_row_in_all_clauses_has_the_confirm_form(client):
    top = client.app.state.workspace.load(SHORT_DOC_ID).clauses[TO_CHECK][0]
    row = clause_row(client.get(f"{DOC}/checklist").text, TO_CHECK)
    assert 'class="badge badge-check"' in row
    assert f'hx-post="{DOC}/decide" hx-target="#document-pane"' in row
    assert f'name="clause_type" value="{TO_CHECK}"' in row
    assert f'name="start" value="{top.start}"' in row and f'name="end" value="{top.end}"' in row
    assert ">Confirm<" in row and ">Not this<" in row


def test_a_confirmed_clause_shows_confirmed_by_you(client):
    before = client.app.state.workspace.entry(SHORT_DOC_ID)
    decide(client, TO_CHECK, "confirmed")
    page = client.get(f"{DOC}/checklist").text
    row = clause_row(page, TO_CHECK)
    assert copy.STATUS["confirmed"] in row and 'class="badge badge-confirmed"' in row
    assert ">Undo<" in row and ">Confirm<" not in row
    assert TO_CHECK in client.get(f"{DOC}/checklist", params={"status": "found"}).text
    card = client.get(f"{DOC}/overview/happened").text
    assert "You confirmed 1 passage." in card
    assert f"{before.found + 1} found, {before.check - 1} to check" in card


def test_a_confirmed_extra_clause_is_not_capped_at_check_this(client):
    capped = clause_row(client.get(f"{DOC}/checklist").text, CAPPED)
    assert 'class="badge badge-check"' in capped
    decide(client, CAPPED, "confirmed")
    row = clause_row(client.get(f"{DOC}/checklist").text, CAPPED)
    assert copy.STATUS["confirmed"] in row and ">Confirm<" not in row


def test_a_rejected_clause_offers_undo(client):
    rejected = decide(client, TO_CHECK, "rejected")
    row = clause_row(client.get(f"{DOC}/checklist").text, TO_CHECK)
    assert "You pressed Not this on a passage here." in row
    assert f'hx-post="{DOC}/decide/undo"' in row and ">Undo<" in row
    assert f'name="start" value="{rejected.start}"' not in row


def test_no_section_shows_the_technical_words(make_client):
    client = make_client(gen=ObligationClient())
    for route in ("summary", "terms", "obligations", "fulltext"):
        frames(client, "POST", f"{DOC}/{route}", data={"language": "en"})
    for route in (*SECTION_ROUTES.values(), "overview", "fulltext"):
        page = client.get(f"{DOC}/{route}").text
        for word in TECHNICAL:
            assert word not in page, (route, word)


# Defined terms and the report


def test_definitions_link_to_the_document_pane(client):
    ws = client.app.state.workspace
    first = ws.definitions(ws.load(SHORT_DOC_ID))[0]
    url = view_url(SHORT_DOC_ID, DEFINITION_LABEL, definition_citation(first), "definition")
    page = client.get(f"{DOC}/definitions").text
    assert html.escape(url) in page and "Show in document" in page
    assert f">{copy.STATUS['definition']}<" in page
    assert f'data-label="{DEFINITION_LABEL}"' in client.get(url).text


def test_the_report_builds_from_what_is_saved_and_downloads(gen_client):
    assert gen_client.get(f"{DOC}/report.docx").status_code == 404
    written_summary(gen_client)
    response = gen_client.post(f"{DOC}/report")
    assert response.status_code == 200
    assert (
        "Your report is ready, but it leaves out the key facts, who must do what and the "
        "overview of the whole text." in response.text
    )
    assert f'href="{DOC}/report.docx"' in response.text and "data-thresholds" in response.text
    assert "Download Word file" in response.text and "Download Markdown" in response.text
    docx = gen_client.get(f"{DOC}/report.docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"
    markdown = gen_client.get(f"{DOC}/report.md").text
    assert "Contract report" in markdown and "The contract states" in markdown


def test_the_report_without_a_language_model_names_every_missing_part(client):
    text = client.post(f"{DOC}/report").text
    assert (
        "It leaves out the summary, the key facts, who must do what and the overview of the "
        "whole text, which need the local AI." in text
    )
    assert client.get(f"{DOC}/report.md").status_code == 200

