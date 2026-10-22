import csv
import io
import json
import re
from datetime import UTC, datetime

import pytest
from fpdf import FPDF

from contract_assistant.extract.extractor import Extractor
from contract_assistant.pdf import load_document
from contract_assistant.pdf_view import DOCUMENTS_DIR, copy_path
from contract_assistant.store import doc_id_of
from contract_assistant.term_sheet import FIELDS, TermSheet, TermValue
from contract_assistant.web import library
from tests.support.caches import make_cache
from tests.support.fakes import EchoTextClient, FakeQA, slow_fake_qa
from tests.web.conftest import LONG_DOC_ID, MEDIUM_DOC_ID, SHORT_DOC_ID, frames

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PAGE_ONE = "Alpha supplies the goods to Beta."
PAGE_TWO = "This agreement is governed by the laws of Vanuatu."
TEXT = f"Preamble. {PAGE_TWO} Final words."


def law_question(clause_types) -> str:
    return next(c for c in clause_types if c.name == "Governing Law").cuad_question


def write_pdf(path, pages: list[str]):
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.cell(text=text)
    pdf.output(str(path))
    return path


def upload(client, name: str, data: bytes) -> list[dict]:
    return frames(client, "POST", "/upload", files={"file": (name, data)})


def statuses(sent: list[dict]) -> list[str]:
    return [item["html"] for item in sent if item.get("html")]


@pytest.fixture
def pdf_client(make_client, settings, clause_types, tmp_path):
    """A client over a two-page PDF analysed through the upload route, the law on page 2."""
    qa = FakeQA({law_question(clause_types): [(PAGE_TWO, 0.98)]})
    client = make_client(extractor=Extractor(qa, clause_types, settings, "fake"))
    path = write_pdf(tmp_path / "two_pages.pdf", [PAGE_ONE, PAGE_TWO])
    sent = upload(client, "two_pages.pdf", path.read_bytes())
    doc_id = doc_id_of(load_document(path).text)
    assert sent[-1]["redirect"] == f"/documents/{doc_id}"
    client.doc_id = doc_id
    return client


def fresh(client, tmp_path):
    """Hide the demo caches, so that the client starts with no contract, as `serve --fresh`."""
    client.app.state.workspace.services.store.demo_dir = tmp_path / "no-demo"
    return client


def law_span(client, doc_id: str) -> tuple[int, int]:
    top = client.app.state.workspace.load(doc_id).clauses["Governing Law"][0]
    return top.start, top.end


def test_the_text_view_highlights_the_span_and_offers_the_actions(client):
    start, end = law_span(client, SHORT_DOC_ID)
    html = client.get(
        f"/documents/{SHORT_DOC_ID}/view?label=Governing+Law&start={start}&end={end}"
        "&conf=0.97&status=found"
    ).text
    quote = client.app.state.workspace.load(SHORT_DOC_ID).text[start:end]
    assert f'<mark class="highlight" data-label="Governing Law">{quote}</mark>' in html
    assert "Excerpt around the highlighted passage" in html
    assert 'class="card passage-card"' in html and "badge-found" in html
    assert "confidence" not in html and "97%" not in html
    assert html.count(f'action="/documents/{SHORT_DOC_ID}/explain"') == 2
    assert 'name="language" value="en"' in html and 'name="language" value="sr"' in html
    assert '>Explain</button>' in html and '>Explain in Serbian</button>' in html
    assert 'data-target="#chat-body"' in html
    assert "data-copy=" in html and "Copy quote" in html
    assert "Previous page" not in html and 'class="seg"' not in html
    assert "Pick a passage" not in html
    assert "/decide" not in html


def test_a_passage_to_check_offers_confirm_and_not_this(client):
    ws = client.app.state.workspace
    cache = ws.load(SHORT_DOC_ID)
    name = next(n for n in ws.names if cache.clauses.get(n) and cache.clauses[n][0].conf < 0.91)
    top = cache.clauses[name][0]
    html = client.get(
        f"/documents/{SHORT_DOC_ID}/view?label={name}&start={top.start}&end={top.end}"
        "&status=check"
    ).text
    assert ">Check this</span>" in html
    assert f'hx-post="/documents/{SHORT_DOC_ID}/decide"' in html
    assert ">Confirm</button>" in html and ">Not this</button>" in html


def test_the_text_view_without_a_span_shows_the_beginning_and_a_hint(client):
    html = client.get(f"/documents/{SHORT_DOC_ID}/view").text
    assert "Beginning of the contract" in html
    assert "passage-card" not in html
    assert "Pick a passage in the overview or ask a question to see it here" in html


def test_a_span_outside_the_text_is_ignored(client):
    html = client.get(f"/documents/{SHORT_DOC_ID}/view?start=5&end=99999999").text
    assert "Beginning of the contract" in html and "<mark" not in html


def test_the_upload_of_a_pdf_keeps_a_copy_and_the_view_shows_page_one(pdf_client, settings):
    doc_id = pdf_client.doc_id
    assert copy_path(settings.cache_dir / DOCUMENTS_DIR, doc_id).exists()
    html = pdf_client.get(f"/documents/{doc_id}/view").text
    assert f'<img class="doc-page" src="/documents/{doc_id}/page/1.png"' in html
    assert "Page 1 <span" in html and "of 2</span>" in html
    assert 'aria-pressed="true">Page</button>' in html and "mode=text" in html


def test_the_page_view_opens_on_the_page_of_the_span_with_the_box(pdf_client):
    doc_id = pdf_client.doc_id
    start, end = law_span(pdf_client, doc_id)
    html = pdf_client.get(
        f"/documents/{doc_id}/view?label=Governing+Law&start={start}&end={end}&conf=0.98"
        "&status=found"
    ).text
    assert f'src="/documents/{doc_id}/page/2.png?start={start}&amp;end={end}"' in html
    assert '<span class="muted small">Page 2</span>' in html and "confidence" not in html
    assert "with the Governing Law passage boxed" in html


def test_previous_is_disabled_on_the_first_page_and_next_on_the_last(pdf_client):
    doc_id = pdf_client.doc_id
    first = pdf_client.get(f"/documents/{doc_id}/view?page=1").text
    assert re.search(r'aria-label="Previous page"\s+disabled', first)
    assert re.search(r'aria-label="Next page"\s+hx-get="[^"]*page=2', first)
    last = pdf_client.get(f"/documents/{doc_id}/view?page=2").text
    assert re.search(r'aria-label="Previous page"\s+hx-get="[^"]*page=1', last)
    assert re.search(r'aria-label="Next page"\s+disabled', last)
    assert 'page/2.png"' in pdf_client.get(f"/documents/{doc_id}/view?page=9").text


def test_the_text_mode_of_a_pdf_shows_the_page_asked_for(pdf_client):
    doc_id = pdf_client.doc_id
    start, end = law_span(pdf_client, doc_id)
    html = pdf_client.get(
        f"/documents/{doc_id}/view?mode=text&page=2&label=Governing+Law&start={start}&end={end}"
    ).text
    assert "<img" not in html and 'aria-pressed="true">Text</button>' in html
    assert f'<mark class="highlight" data-label="Governing Law">{PAGE_TWO}</mark>' in html


def test_the_page_image_is_a_png_and_is_rendered_once(pdf_client, monkeypatch):
    doc_id = pdf_client.doc_id
    start, end = law_span(pdf_client, doc_id)
    response = pdf_client.get(f"/documents/{doc_id}/page/2.png?start={start}&end={end}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(PNG_SIGNATURE)
    calls = []
    monkeypatch.setattr(library, "page_image", lambda *args: calls.append(args) or b"x")
    pdf_client.get(f"/documents/{doc_id}/page/2.png?start={start}&end={end}")
    assert calls == []


def test_a_missing_page_or_pdf_is_not_found(pdf_client):
    assert pdf_client.get(f"/documents/{pdf_client.doc_id}/page/3.png").status_code == 404
    assert pdf_client.get(f"/documents/{SHORT_DOC_ID}/page/1.png").status_code == 404


def test_the_upload_of_a_text_streams_the_progress_then_opens_the_contract(
    make_client, settings, clause_types
):
    client = make_client(extractor=Extractor(slow_fake_qa(0), clause_types, settings, "fake"))
    sent = upload(client, "new_agreement.txt", TEXT.encode())
    total = len(clause_types)
    shown = statuses(sent)
    assert library.READING_FILE in shown[0]
    finding = [html for html in shown if "Finding the clauses" in html]
    assert len(finding) == total
    assert f"Finding the clauses · {total} of {total} · " in finding[-1]
    assert any(library.SAVING in html for html in shown)
    assert not any(clause.name in html for clause in clause_types for html in finding)
    assert sent[-1]["redirect"] == f"/documents/{doc_id_of(TEXT)}"
    assert sent[-1]["notice"] == library.ANALYSIS_COMPLETE and sent[-1]["done"]
    assert len(list(settings.cache_dir.glob("*.json"))) == 1
    assert not (settings.cache_dir / DOCUMENTS_DIR).exists()


def test_the_upload_starts_the_background_writing_when_generating(
    make_client, settings, clause_types, monkeypatch
):
    calls = []
    monkeypatch.setattr(library, "run_in_background", lambda target, *args: calls.append(target))
    extractor = Extractor(FakeQA(), clause_types, settings, "fake")
    upload(make_client(extractor=extractor, gen=EchoTextClient()), "new.txt", TEXT.encode())
    assert calls == [library.precompute, library.precompute_obligations]


def test_an_unsupported_file_is_a_notice(client):
    sent = upload(client, "notes.docx", b"x")
    assert sent[-1]["notice"] == "unsupported file type: .docx"
    assert "redirect" not in sent[-1]


def test_a_request_without_a_file_is_a_notice(client):
    sent = frames(client, "POST", "/upload", files={"file": ("", b"")})
    assert sent[-1]["notice"] == library.CHOOSE_FILE


def test_an_upload_without_the_analysis_model_says_so(client):
    assert upload(client, "new.txt", TEXT.encode())[-1]["notice"] == library.NO_MODEL


def test_an_analysed_contract_opens_without_running_the_model(make_client, settings):
    qa = FakeQA()
    client = make_client(extractor=Extractor(qa, [], settings, "fake"))
    text = client.app.state.workspace.load(SHORT_DOC_ID).text
    sent = upload(client, "short_supply_agreement.txt", text.encode())
    assert sent[-1]["redirect"] == f"/documents/{SHORT_DOC_ID}"
    assert sent[-1]["notice"] == "Short supply agreement was read before; opening it"
    assert qa.calls == []


def test_an_analysis_missing_clause_types_is_extended(make_client, settings, clause_types):
    older = make_client(extractor=Extractor(FakeQA(), clause_types[:41], settings, "fake"))
    upload(older, "older.txt", TEXT.encode())
    qa = FakeQA()
    full = make_client(extractor=Extractor(qa, clause_types, settings, "fake"))
    sent = upload(full, "older.txt", TEXT.encode())
    assert sent[-1]["notice"] == library.ANALYSIS_UPDATED
    assert len(qa.calls) == len(clause_types) - 41
    assert "Payment Terms" in full.app.state.workspace.load(doc_id_of(TEXT)).clauses


def test_reading_a_contract_again_keeps_the_decisions_apart(make_client, settings, clause_types):
    qa = FakeQA({law_question(clause_types): [(PAGE_TWO, 0.5)]})
    older = make_client(extractor=Extractor(qa, clause_types[:41], settings, "fake"))
    upload(older, "older.txt", TEXT.encode())
    doc_id = doc_id_of(TEXT)
    start, end = law_span(older, doc_id)
    decision = {"clause_type": "Governing Law", "verdict": "confirmed", "start": start, "end": end}
    older.post(f"/documents/{doc_id}/decide", data=decision)
    full = make_client(extractor=Extractor(FakeQA(), clause_types, settings, "fake"))
    assert upload(full, "older.txt", TEXT.encode())[-1]["notice"] == library.ANALYSIS_UPDATED
    ws = full.app.state.workspace
    assert ws.services.store.load(doc_id, settings).clauses["Governing Law"][0].conf == 0.5
    assert ws.load(doc_id).clauses["Governing Law"][0].conf == 1.0


def test_the_compare_page_offers_two_contracts(client):
    html = client.get(f"/compare?left={SHORT_DOC_ID}").text
    assert html.count("<select") == 2
    assert f'name="left" value="{SHORT_DOC_ID}"' in html
    assert 'name="right" value="sha256:' in html and "data-autosubmit" in html
    assert 'id="compare-result"' in html and "Swap the two contracts" in html


def test_the_compare_page_with_one_contract_asks_for_a_second(
    client, settings, tmp_path, monkeypatch
):
    store = client.app.state.workspace.services.store
    monkeypatch.setattr(store, "demo_dir", tmp_path / "no-demo")
    store.save(make_cache(TEXT, settings))
    html = client.get("/compare").text
    assert library.ADD_SECOND in html and "<select" not in html


def test_the_comparison_streams_the_terms_the_flags_and_the_clause_groups(make_client):
    client = make_client(gen=EchoTextClient())
    sent = frames(
        client, "POST", "/compare/result", data={"left": SHORT_DOC_ID, "right": LONG_DOC_ID}
    )
    assert any("Reading the key terms of Short supply agreement" in h for h in statuses(sent))
    html = sent[-1]["html"]
    assert re.search(r'class="kv kv-3 diff"', html)
    assert "Only in Long outsourcing agreement" in html and "Same in both" in html
    assert "Things to check</h2>" in html and "badge-medium" in html
    assert "of 49 clauses differ" in sent[-1]["oob"]["compare-summary"]
    assert "to check" in sent[-1]["oob"]["left-counts"]
    assert 'class="legend-chip legend-diff"' in html and "coloured edge" not in html
    assert "confidence" not in html and "Verify" not in html and "clause types" not in html


def test_the_comparison_without_the_language_model_compares_the_clauses(client):
    sent = frames(
        client, "POST", "/compare/result", data={"left": SHORT_DOC_ID, "right": LONG_DOC_ID}
    )
    html = sent[-1]["html"]
    assert library.TERMS_SKIPPED.format(reason=library.TERMS_NEED_MODEL) in html
    assert "Clauses</h2>" in html
    assert "The key terms are not compared." in sent[-1]["oob"]["compare-summary"]


def test_the_same_contract_on_both_sides_or_an_unknown_one_is_a_notice(client):
    same = {"left": SHORT_DOC_ID, "right": SHORT_DOC_ID}
    unknown = {"left": SHORT_DOC_ID, "right": "sha256:missing"}
    sent = frames(client, "POST", "/compare/result", data=same)
    assert sent[-1]["notice"] == library.CHOOSE_DIFFERENT
    sent = frames(client, "POST", "/compare/result", data=unknown)
    assert sent[-1]["notice"] == library.OTHER_UNAVAILABLE


def test_the_portfolio_lists_the_contracts_and_writes_the_csv(client, settings):
    html = client.get("/portfolio").text
    for name in ("Short supply agreement", "Long outsourcing agreement"):
        assert name in html
    assert f'href="/documents/{SHORT_DOC_ID}"' in html
    assert re.search(r'class="found" style="width: \d+%"', html)
    assert "Next notice deadline" not in html
    assert "to check · " in html and "verify" not in html.lower()
    assert "default thresholds" not in html and "confidence" not in html
    assert (settings.cache_dir / library.PORTFOLIO_CSV).exists()


def test_the_portfolio_csv_downloads(client):
    response = client.get("/portfolio.csv")
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0][:3] == ["Contract", "Size", "Found"]
    assert len(rows) == 1 + len(client.app.state.workspace.documents())


def sheet_of(settings, doc_id: str, **values: str) -> TermSheet:
    terms = {field: [] for field in FIELDS}
    for field, value in values.items():
        terms[field] = [
            TermValue(value=value, clause_type=FIELDS[field][0], page=None, start=0, end=1)
        ]
    return TermSheet(
        doc_id=doc_id,
        model=EchoTextClient.model,
        tau_low=settings.tau_low,
        tau_high=settings.tau_high,
        latency_ms=1.0,
        items=[],
        terms=terms,
        rejected={},
    )


def test_the_portfolio_names_the_next_notice_deadline(make_client, settings):
    client = make_client(gen=EchoTextClient())
    year = datetime.now(tz=UTC).year + 1
    sheet = sheet_of(
        settings,
        SHORT_DOC_ID,
        effective_date=f"1st day of January {year}",
        term="two (2) years",
        notice_period="six (6) months",
        governing_law="the laws of England",
    )
    client.app.state.workspace.term_sheets.save(sheet)
    html = client.get("/portfolio").text
    assert "Next notice deadline" in html
    assert f"1 Jul {year + 1}" in html and "six (6) months before the term ends" in html
    assert "the laws of England" in html


def test_the_technical_details_name_the_extraction_model(client, settings):
    html = client.get("/technical").text
    assert f"<strong>Extraction model:</strong> <code>{settings.extractor_name}</code>" in html
    assert "<li>" in html and "**" not in html


def test_the_analysis_downloads_as_json(client):
    response = client.get(f"/documents/{SHORT_DOC_ID}/analysis.json")
    assert response.status_code == 200
    assert json.loads(response.text)["doc_id"] == SHORT_DOC_ID
    assert "short_supply_agreement.analysis.json" in response.headers["content-disposition"]
    assert client.get("/documents/sha256:nope/analysis.json").status_code == 404


def test_the_workspace_rail_links_the_analysis_and_the_upload(client):
    html = client.get(f"/documents/{SHORT_DOC_ID}").text
    assert f'href="/documents/{SHORT_DOC_ID}/analysis.json"' in html
    assert 'action="/upload"' in html and html.count('id="report-result"') == 1
    rail = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    assert "How answers are made" not in rail and "Found from" not in rail


def test_the_rail_says_found_and_to_check_and_marks_the_open_contract(client):
    ws = client.app.state.workspace
    short = ws.entry(SHORT_DOC_ID)
    html = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    meta = f"Text · {short.size} · {short.found} found · {short.check} to check"
    assert meta in re.sub(r"<[^>]+>", "", html)
    assert re.search(
        rf'class="doc-item doc-item-active" href="/documents/{SHORT_DOC_ID}"\s+aria-current="page"',
        html,
    )
    assert html.count('aria-current="page"') == 1
    assert "verify" not in html.lower() and "confidence" not in html


def test_the_rail_route_renders_the_rail_of_the_page(make_client, settings):
    client = make_client(extractor=Extractor(FakeQA(), [], settings, "fake"))
    page = client.get(f"/documents/{SHORT_DOC_ID}").text
    rail = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    assert rail.strip() and rail.strip() in page


def test_the_rail_offers_the_samples_not_read_yet_when_it_can_read(make_client, settings):
    client = make_client(extractor=Extractor(FakeQA(), [], settings, "fake"))
    html = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    # The three text samples ship analysed; the PDF of the short one is a different text.
    assert "Or try a sample contract" in html
    assert 'action="/samples/short_supply_agreement.pdf"' in html
    assert 'action="/samples/short_supply_agreement.txt"' not in html
    assert "PDF · 5 pages" in html


def test_the_rail_without_the_reading_model_offers_no_samples(client):
    html = client.get(f"/documents/{SHORT_DOC_ID}/rail").text
    assert 'action="/samples/' not in html and "Or try a sample contract" not in html


def test_the_first_page_welcomes_and_lists_the_samples(make_client, settings, tmp_path):
    client = fresh(make_client(extractor=Extractor(FakeQA(), [], settings, "fake")), tmp_path)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200
    html = response.text
    assert "Read a contract with a local assistant" in html
    assert "reads it once on this computer" in html and 'action="/upload"' in html
    assert "Or try a sample contract" in html
    for name in library.SAMPLE_FILES:
        assert f'action="/samples/{name}"' in html
    assert "disabled" not in html and library.NO_MODEL not in html
    assert "data-drawer-toggle" not in html


def test_the_first_page_without_the_reading_model_says_so(client, tmp_path):
    html = fresh(client, tmp_path).get("/").text
    assert library.NO_MODEL in html and library.SAMPLES_NEED_MODEL in html
    assert html.count(f'disabled title="{library.SAMPLES_NEED_MODEL}"') == len(library.SAMPLE_FILES)


def test_a_sample_is_read_like_an_upload_then_opened(
    make_client, settings, clause_types, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(library, "run_in_background", lambda target, *args: calls.append(target))
    extractor = Extractor(FakeQA(), clause_types, settings, "fake")
    client = fresh(make_client(extractor=extractor, gen=EchoTextClient()), tmp_path)
    sent = frames(client, "POST", "/samples/medium_sponsorship_agreement.txt")
    shown = statuses(sent)
    assert library.READING_FILE in shown[0]
    total = len(clause_types)
    assert any(f"Finding the clauses · {total} of {total} · " in html for html in shown)
    assert sent[-1]["redirect"] == f"/documents/{MEDIUM_DOC_ID}"
    assert sent[-1]["notice"] == library.ANALYSIS_COMPLETE and sent[-1]["done"]
    assert calls == [library.precompute, library.precompute_obligations]
    assert client.app.state.workspace.entry(MEDIUM_DOC_ID).name == "Medium sponsorship agreement"
    home = client.get("/", follow_redirects=False)
    assert home.headers["location"] == f"/documents/{MEDIUM_DOC_ID}"
    rail = client.get(f"/documents/{MEDIUM_DOC_ID}/rail").text
    assert 'action="/samples/medium_sponsorship_agreement.txt"' not in rail
    assert 'action="/samples/long_outsourcing_agreement.txt"' in rail


def test_the_pdf_sample_keeps_its_pdf_for_the_page_view(make_client, settings, tmp_path):
    client = fresh(make_client(extractor=Extractor(FakeQA(), [], settings, "fake")), tmp_path)
    sent = frames(client, "POST", "/samples/short_supply_agreement.pdf")
    doc_id = doc_id_of(load_document(library.SAMPLES_DIR / "short_supply_agreement.pdf").text)
    assert sent[-1]["redirect"] == f"/documents/{doc_id}"
    assert copy_path(settings.cache_dir / DOCUMENTS_DIR, doc_id).exists()
    assert client.app.state.workspace.entry(doc_id).kind == "pdf"


def test_an_analysed_sample_opens_with_a_notice(make_client, settings):
    qa = FakeQA()
    client = make_client(extractor=Extractor(qa, [], settings, "fake"))
    sent = frames(client, "POST", "/samples/short_supply_agreement.txt")
    assert sent[-1]["redirect"] == f"/documents/{SHORT_DOC_ID}"
    assert sent[-1]["notice"] == "Short supply agreement was read before; opening it"
    assert qa.calls == []


def test_a_sample_without_the_reading_model_is_a_notice(client, tmp_path):
    sent = frames(fresh(client, tmp_path), "POST", "/samples/long_outsourcing_agreement.txt")
    assert sent[-1]["notice"] == library.NO_MODEL and "redirect" not in sent[-1]


def test_only_the_listed_samples_can_be_read(client):
    for name in ("notes.txt", "demo_documents.json", "README.md", "..%2Fpyproject.toml"):
        response = client.post(f"/samples/{name}")
        assert response.status_code == 404
