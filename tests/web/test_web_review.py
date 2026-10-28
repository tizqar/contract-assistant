"""A contract's review from the workspace: the steps by level and status, the bar, the history."""

import json
import re

import pytest

from contract_assistant import access, audit, files
from contract_assistant.access import ReviewStep
from contract_assistant.db.models import APPROVED, EDIT, IN_REVIEW, SUBMITTED, VIEW
from contract_assistant.web.review import DONE, DONE_EVENT, HISTORY_PAGE
from tests.support.db import add_user
from tests.web.conftest import SHORT_DOC_ID, seed_demo, sign_in

DOC = f"/documents/{SHORT_DOC_ID}"
HTMX = {"HX-Request": "true"}
NOTE = "Check the notice period in section 9."
# The steps a status reaches from In review, and who takes each.
PATH_TO = {IN_REVIEW: (), SUBMITTED: (ReviewStep.SUBMIT,)}
PATH_TO[APPROVED] = (ReviewStep.SUBMIT, ReviewStep.APPROVE)
STEP_ACTION = re.compile(r'action="/documents/[^"]+/review/(\w+)"')


@pytest.fixture
def deals(database, settings, reader):
    """The reader's folder "Deals" with the short contract in it."""
    seed_demo(database, settings, reader)
    with database.session() as session:
        folder = files.create_folder(session, reader, None, "Deals")
        files.move_contract(session, reader, SHORT_DOC_ID, folder.id)
    return folder


@pytest.fixture
def people(database, reader, admin, deals):
    """Everyone a test signs in as, by the part they play: owner, edit, view, admin, stranger."""
    edit = add_user(database, "editor@example.com", name="Editor", user_id="editor")
    view = add_user(database, "viewer@example.com", name="Viewer", user_id="viewer")
    stranger = add_user(database, "stranger@example.com", name="Stranger", user_id="stranger")
    with database.session() as session:
        files.share_folder(session, reader, deals.id, edit.email, EDIT)
        files.share_folder(session, reader, deals.id, view.email, VIEW)
    return {"owner": reader, "edit": edit, "view": view, "admin": admin, "stranger": stranger}


@pytest.fixture
def as_(make_client, database, people):
    """Return a client signed in as one of the people."""

    def make(who: str):
        return sign_in(make_client(seed=False), database, people[who])

    return make


def reach(database, people, status: str) -> None:
    """Take the contract's review from In review to a status, the admin approving."""
    for step in PATH_TO[status]:
        actor = people["admin"] if step is ReviewStep.APPROVE else people["owner"]
        with database.session() as session:
            files.review(session, actor, SHORT_DOC_ID, step)


def status_of(database, people) -> str:
    with database.session() as session:
        return files.contract(session, people["admin"], SHORT_DOC_ID).review_status


def step(client, name: str, htmx: bool = False, **data):
    return client.post(
        f"{DOC}/review/{name}", data=data, headers=HTMX if htmx else {}, follow_redirects=False
    )


def bar(client) -> str:
    """The review bar of the contract's page."""
    html = client.get(DOC).text
    start = html.index('<section class="review-bar')
    return html[start : html.index("</section>", start)]


def to_check(client) -> tuple[str, int, int]:
    """Return a clause type of the short contract with a passage to check, and its span."""
    cache = client.ws.load(SHORT_DOC_ID)
    names = client.ws.names
    name = next(n for n in names if cache.clauses.get(n) and cache.clauses[n][0].conf < 0.91)
    top = cache.clauses[name][0]
    return name, top.start, top.end


def decide(client, verdict: str = "confirmed"):
    name, start, end = to_check(client)
    data = {"clause_type": name, "verdict": verdict, "start": start, "end": end}
    return client.post(f"{DOC}/decide", data=data, follow_redirects=False)


# Who may take which step from which status

ALLOWED = {
    # (step, status the contract is at): the people who may take it
    ("submit", IN_REVIEW): {"owner", "edit", "admin"},
    ("approve", SUBMITTED): {"admin"},
    ("return", SUBMITTED): {"admin"},
    ("reopen", APPROVED): {"admin"},
}
EVERYONE = ("owner", "edit", "view", "admin", "stranger")


@pytest.mark.parametrize(("name", "status"), list(ALLOWED))
@pytest.mark.parametrize("who", EVERYONE)
def test_each_step_is_taken_only_by_the_level_it_needs(
    as_, database, people, name, status, who
):
    reach(database, people, status)
    response = step(as_(who), name, note=NOTE)
    if who in ALLOWED[(name, status)]:
        assert response.status_code == 303
        assert response.headers["location"] == DOC
        assert status_of(database, people) == access.REVIEW_STEPS[ReviewStep(name)][1]
    else:
        expected = 404 if who == "stranger" else 403
        message = access.NOT_FOUND if who == "stranger" else access.NOT_ALLOWED
        assert (response.status_code, response.text) == (expected, message)
        assert status_of(database, people) == status


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("submit", SUBMITTED),
        ("submit", APPROVED),
        ("approve", IN_REVIEW),
        ("approve", APPROVED),
        ("return", IN_REVIEW),
        ("return", APPROVED),
        ("reopen", IN_REVIEW),
        ("reopen", SUBMITTED),
    ],
)
def test_a_step_from_the_wrong_status_is_refused(as_, database, people, name, status):
    reach(database, people, status)
    response = step(as_("admin"), name, htmx=True, note=NOTE)
    assert (response.status_code, response.text) == (409, access.WRONG_STEP)
    assert "HX-Redirect" not in response.headers
    assert status_of(database, people) == status


def test_an_unknown_step_is_not_found(as_, people):
    assert step(as_("owner"), "publish").status_code == 404


def test_sending_back_needs_a_note_in_the_readers_language(as_, database, people):
    reach(database, people, SUBMITTED)
    client = as_("admin")
    refused = step(client, "return", htmx=True, note="   ")
    assert (refused.status_code, refused.text) == (400, files.NOTE_NEEDED)
    client.cookies.set("lang", "sr")
    serbian = step(client, "return", htmx=True)
    assert serbian.status_code == 400
    assert serbian.text == "Napišite napomenu o tome šta treba promeniti."
    assert status_of(database, people) == SUBMITTED


def test_a_note_too_long_is_refused(as_, database, people):
    reach(database, people, SUBMITTED)
    response = step(as_("admin"), "return", note="x" * (files.MAX_NOTE_CHARS + 1))
    assert (response.status_code, response.text) == (400, files.NOTE_TOO_LONG)


# Where a step goes next


def test_from_the_workspace_a_step_reloads_the_contract_page_with_a_notice(as_, people):
    response = step(as_("owner"), "submit", htmx=True)
    assert response.status_code == 200 and response.text == ""
    assert response.headers["HX-Redirect"] == DOC
    trigger = json.loads(response.headers["HX-Trigger"])
    assert trigger == {DONE_EVENT: DONE[ReviewStep.SUBMIT]}


def test_the_notice_after_a_step_is_in_the_readers_language(as_, people):
    client = as_("owner")
    client.cookies.set("lang", "sr")
    response = step(client, "submit", htmx=True)
    header = response.headers["HX-Trigger"]
    assert header.isascii()
    assert json.loads(header)[DONE_EVENT].startswith("Poslato na odobrenje.")


def test_the_reload_keeps_thresholds_that_are_not_the_defaults(as_, people):
    client = as_("owner")
    response = client.post(
        f"{DOC}/review/submit?low=0.2&high=0.95", headers=HTMX, follow_redirects=False
    )
    assert response.headers["HX-Redirect"] == f"{DOC}?low=0.2&high=0.95"


@pytest.mark.parametrize("htmx", [False, True])
def test_a_local_next_page_is_where_a_step_goes(as_, database, people, htmx):
    reach(database, people, SUBMITTED)
    response = step(as_("admin"), "approve", htmx=htmx, next="/admin/approvals?page=2")
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/approvals?page=2"
    assert status_of(database, people) == APPROVED


@pytest.mark.parametrize(
    "elsewhere", ["https://example.com/x", "//example.com/x", "/\\example.com", "admin"]
)
def test_a_next_page_on_another_site_is_ignored(as_, database, people, elsewhere):
    reach(database, people, SUBMITTED)
    response = step(as_("admin"), "approve", next=elsewhere)
    assert response.status_code == 303 and response.headers["location"] == DOC


# The lock on decisions, through the routes


def test_sending_for_approval_locks_the_decisions(as_, people):
    owner = as_("edit")
    assert step(owner, "submit").status_code == 303
    refused = decide(owner)
    assert (refused.status_code, refused.text) == (409, access.LOCKED)


@pytest.mark.parametrize("unlock", ["return", "reopen"])
def test_sending_back_or_reopening_opens_the_decisions_again(as_, people, unlock):
    owner, admin = as_("owner"), as_("admin")
    step(owner, "submit")
    if unlock == "reopen":
        step(admin, "approve")
        assert decide(owner).status_code == 409
    assert step(admin, unlock, note=NOTE).status_code == 303
    assert decide(owner).status_code == 303


# The bar


BUTTONS = {
    # (who, status): the steps the bar offers
    ("owner", IN_REVIEW): {"submit"},
    ("edit", IN_REVIEW): {"submit"},
    ("view", IN_REVIEW): set(),
    ("admin", IN_REVIEW): {"submit"},
    ("owner", SUBMITTED): set(),
    ("edit", SUBMITTED): set(),
    ("view", SUBMITTED): set(),
    ("admin", SUBMITTED): {"approve", "return"},
    ("owner", APPROVED): set(),
    ("edit", APPROVED): set(),
    ("view", APPROVED): set(),
    ("admin", APPROVED): {"reopen"},
}


@pytest.mark.parametrize(("who", "status"), list(BUTTONS))
def test_the_bar_offers_only_the_steps_the_reader_may_take(
    as_, database, people, who, status
):
    reach(database, people, status)
    html = bar(as_(who))
    assert set(STEP_ACTION.findall(html)) == BUTTONS[(who, status)]
    labels = {IN_REVIEW: "In review", SUBMITTED: "Sent for approval", APPROVED: "Approved"}
    assert f">{labels[status]}</span>" in html
    assert ">History</button>" in html


def test_the_bar_says_who_changed_the_review_and_when(as_, database, people):
    client = as_("view")
    assert "review-changed" not in bar(client)
    reach(database, people, SUBMITTED)
    assert "Sent for approval by Reader, " in bar(client)
    step(as_("admin"), "approve")
    assert "Approved by Admin, " in bar(client)
    step(as_("admin"), "reopen")
    html = bar(client)
    assert "Reopened by Admin, " in html and "review-note" not in html
    assert "Passages can be confirmed or rejected; send it for approval when done." in html


def test_the_note_of_a_contract_sent_back_is_shown_until_it_is_sent_again(as_, database, people):
    reach(database, people, SUBMITTED)
    step(as_("admin"), "return", note="Check the <notice> period.")
    for who in ("owner", "edit", "view"):
        html = bar(as_(who))
        assert "Sent back by Admin, " in html and ">In review</span>" in html
        assert "What to change" in html and "Check the &lt;notice&gt; period." in html
    step(as_("edit"), "submit")
    html = bar(as_("owner"))
    assert "What to change" not in html and "Sent for approval by Editor, " in html


def test_the_bar_alone_is_a_partial(as_, people):
    client = as_("view")
    response = client.get(f"{DOC}/review/bar")
    assert response.status_code == 200
    assert response.text.lstrip().startswith('<section class="review-bar review-in_review"')
    assert as_("stranger").get(f"{DOC}/review/bar").status_code == 404


def test_the_bar_speaks_serbian(as_, database, people):
    reach(database, people, SUBMITTED)
    client = as_("admin")
    client.cookies.set("lang", "sr")
    html = bar(client)
    assert ">Poslat na odobrenje</span>" in html and "Poslat na odobrenje: Reader, " in html
    assert ">Odobri</button>" in html and ">Vrati na doradu</summary>" in html


# The history


def test_the_history_lists_decisions_and_steps_with_their_actors(as_, database, people):
    editor = as_("edit")
    decide(editor)
    step(editor, "submit")
    step(as_("admin"), "return", note=NOTE)
    html = as_("view").get(f"{DOC}/history").text
    sentences = re.findall(r'<p class="history-sentence">([^<]+)</p>', html)
    name = "short_supply_agreement.txt"
    assert sentences[0] == f"Admin sent {name} back: {NOTE}"
    assert sentences[1] == f"Editor sent {name} for approval"
    assert sentences[2].startswith("Editor confirmed the ") and sentences[2].endswith(name)
    assert sentences[-1] == f"Reader added {name}"
    assert "history-review" in html and "Older" not in html


def test_the_history_is_only_for_readers_of_the_contract(as_, people):
    assert as_("view").get(f"{DOC}/history").status_code == 200
    assert as_("stranger").get(f"{DOC}/history").status_code == 404


def test_the_history_comes_a_page_at_a_time(as_, database, people):
    with database.session() as session:
        for number in range(HISTORY_PAGE + 5):
            audit.record(
                session,
                audit.CONTRACT_EXPORTED,
                actor_id="reader",
                contract_id=SHORT_DOC_ID,
                details={"what": f"copy {number}"},
            )
    client = as_("owner")
    first = client.get(f"{DOC}/history").text
    assert first.count('class="history-row') == HISTORY_PAGE
    assert "copy 34" in first and "copy 5" in first and "copy 4" not in first
    older = re.search(r'hx-get="([^"]+/history\?before=(\d+))"', first)
    assert older and ">Older</button>" in first
    rest = client.get(older.group(1)).text
    assert '<ol class="history-list">' not in rest
    assert rest.count('class="history-row') == 7 and "Older" not in rest
    assert "copy 4" in rest and "copy 0" in rest and "Reader added" in rest


def test_the_workspace_has_the_history_panel(as_, people):
    html = as_("owner").get(DOC).text
    assert '<dialog class="history-dialog" id="history-dialog"' in html
    assert f'data-history-open="{DOC}/history"' in html
