import json

import pytest

from contract_assistant.web.core import (
    BAD_THRESHOLDS,
    WorkspaceError,
    frame,
    shorten,
    thresholds_of,
    view_url,
)
from tests.web.conftest import SHORT_DOC_ID


def test_home_opens_the_short_demo_contract(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/documents/{SHORT_DOC_ID}"


def test_the_workspace_page_lists_the_contracts_and_the_tabs(client):
    page = client.get(f"/documents/{SHORT_DOC_ID}").text
    assert "Short supply agreement" in page
    assert "Long outsourcing agreement" in page
    for tab in ("Chat", "Summary", "Key terms", "Review", "Obligations", "Clauses", "Definitions"):
        assert f">{tab}<" in page
    assert 'id="tau-low"' in page and 'value="0.06"' in page


def test_an_unknown_contract_is_not_found(client):
    assert client.get("/documents/sha256:nope").status_code == 404


def test_the_thresholds_come_from_the_query_and_default_to_the_settings(client):
    page = client.get(f"/documents/{SHORT_DOC_ID}?low=0.2&high=0.8").text
    assert 'value="0.2"' in page and 'value="0.8"' in page
    assert "Found from 0.8 · Verify from 0.2" in page


def test_bad_thresholds_are_a_plain_notice(client):
    response = client.get(f"/documents/{SHORT_DOC_ID}?low=0.9&high=0.5")
    assert response.status_code == 400
    assert response.text == BAD_THRESHOLDS


def test_thresholds_of_checks_the_order_and_the_range(settings):
    assert thresholds_of(settings, None, None).params() == {"low": "0.06", "high": "0.91"}
    with pytest.raises(WorkspaceError):
        thresholds_of(settings, 0.5, 0.4)
    with pytest.raises(WorkspaceError):
        thresholds_of(settings, -0.1, 0.4)


def test_the_document_list_counts_found_and_verify(client):
    ws = client.app.state.workspace
    short = next(entry for entry in ws.documents() if entry.doc_id == SHORT_DOC_ID)
    assert short.kind == "txt"
    assert short.size.endswith("words")
    assert short.found > 0 and short.check > 0


def test_frames_carry_only_the_keys_given():
    assert json.loads(frame("<p>x</p>")) == {"html": "<p>x</p>"}
    assert json.loads(frame(notice="no", done=True)) == {"notice": "no", "done": True}
    assert frame().endswith("\n")


def test_view_urls_name_the_span_and_the_label():
    url = view_url("sha256:abc", "Governing Law")
    assert url == "/documents/sha256:abc/view?label=Governing+Law"


def test_shorten_squashes_whitespace_and_cuts_long_text():
    assert shorten("a  b\n c") == "a b c"
    assert shorten("x" * 200, 10) == "x" * 9 + "…"


def test_the_document_pane_highlights_a_span(client):
    ws = client.app.state.workspace
    cache = ws.load(SHORT_DOC_ID)
    top = cache.clauses["Governing Law"][0]
    url = f"/documents/{SHORT_DOC_ID}/view?label=Governing+Law&start={top.start}&end={top.end}"
    html = client.get(url).text
    assert f'<mark class="highlight" data-label="Governing Law">{top.text}</mark>' in html


def test_static_files_are_served(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/htmx.min.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200
