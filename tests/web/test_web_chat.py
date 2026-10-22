from markupsafe import escape

from contract_assistant.explain import MISQUOTE_REASON
from contract_assistant.feedback import FEEDBACK_FILE, read_feedback
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.router.clients import LLMError
from contract_assistant.ui.views import NOT_FOUND_TEXT
from contract_assistant.web import chat, copy
from contract_assistant.web.chat import (
    AI_OFF,
    ANSWER_GONE,
    BASED_ON,
    BASED_ON_SHOWN,
    CHAT_BODY,
    EMPTY_TEXT,
    EMPTY_TITLE,
    FEEDBACK_THANKS,
    NO_MATCH,
    NOTHING_TO_EXPLAIN,
    PASSAGES_INTRO,
    PENDING_ID,
    SOMETHING_WRONG,
    THINKING,
    TYPE_A_QUESTION,
    WRITING_ANSWER,
)
from contract_assistant.web.core import SUMMARIES_OFF
from contract_assistant.web.generate import WRITING_EXPLANATION, WRITING_SUMMARY
from tests.support.fakes import EchoTextClient, FakeLLMClient, FixedRouter
from tests.web.conftest import SHORT_DOC_ID, frames

CHAT = f"/documents/{SHORT_DOC_ID}/chat"
ASK = f"/documents/{SHORT_DOC_ID}/ask"
EXPLAIN = f"/documents/{SHORT_DOC_ID}/explain"
FEEDBACK = f"/documents/{SHORT_DOC_ID}/feedback"
CLEAR = f"/documents/{SHORT_DOC_ID}/chat/clear"
DECIDE = f"/documents/{SHORT_DOC_ID}/decide"
LAW_QUESTION = "Which law governs the contract?"
# Routed to Parties, which the short contract has as a passage to check.
PARTIES_QUESTION = "Who are the parties?"
# Matches no clause type of the fake router, and the fake encoder finds passages close to it.
SHIPPING_QUESTION = "How are the goods shipped and delivered to the customer?"
MARS_QUESTION = "What about the weather on Mars?"
# What the first version of the chat showed and the reader no longer sees. The AI badge's
# tooltip says "Written by a language model"; the lines under the texts named the model.
TECHNICAL = ("confidence", "Not validated", "highlighted", "Written by the language model")


def route(*names: str) -> RouteResult:
    return RouteResult(clause_types=list(names), kind="explicit", latency_ms=0)


def ask(client, question: str) -> list[dict]:
    return frames(client, "POST", ASK, data={"question": question})


def body(reply: dict) -> str:
    """Return the chat a frame draws."""
    return reply["oob"][CHAT_BODY]


def top(client, clause_type: str):
    return client.app.state.workspace.load(SHORT_DOC_ID).clauses[clause_type][0]


def conversation(client):
    return client.app.state.workspace.conversation(SHORT_DOC_ID)


def pending_texts(replies: list[dict]) -> list[str]:
    return [reply["oob"][PENDING_ID] for reply in replies if PENDING_ID in reply.get("oob", {})]


def plain(html: str) -> bool:
    return not any(word in html for word in TECHNICAL)


def decide(client, clause_type: str, verdict: str):
    candidate = top(client, clause_type)
    data = {
        "clause_type": clause_type,
        "verdict": verdict,
        "start": candidate.start,
        "end": candidate.end,
    }
    response = client.post(DECIDE, data=data, follow_redirects=False)
    assert response.status_code == 303
    return candidate


def test_the_chat_starts_empty(client):
    html = client.get(CHAT).text
    assert EMPTY_TITLE in html and EMPTY_TEXT in html
    assert 'id="chat-messages"' in html and "Clear chat" not in html


def test_a_routed_question_shows_a_plain_found_card(client):
    replies = ask(client, LAW_QUESTION)
    first, last = replies[0], replies[-1]
    assert "done" not in first and "activate" not in first
    assert LAW_QUESTION in body(first) and THINKING in body(first)
    html = body(last)
    assert last["done"] and "activate" not in last and "load" not in last
    assert "badge-found" in html and copy.STATUS["found"] in html
    assert "<strong>Governing Law</strong>" in html
    assert str(escape(top(client, "Governing Law").text)) in html
    assert copy.SHOW_IN_DOCUMENT in html and 'id="feedback-1"' in html
    assert plain(html) and "%" not in html


def test_the_document_follows_the_passage_without_leaving_the_overview(client):
    html = body(ask(client, LAW_QUESTION)[-1])
    law = top(client, "Governing Law")
    assert 'data-follow="/documents/' in html
    assert "label=Governing+Law" in html and f"start={law.start}" in html
    assert "status=found" in html
    # Kept in the chat, the reply does not move the document again.
    assert "data-follow" not in client.get(CHAT).text


def test_the_chat_lists_the_messages_oldest_first(client):
    ask(client, LAW_QUESTION)
    ask(client, PARTIES_QUESTION)
    html = client.get(CHAT).text
    assert html.index(LAW_QUESTION) < html.index(PARTIES_QUESTION)
    assert "Clear chat" in html and 'hx-target="#chat-body"' in html
    assert EMPTY_TITLE not in html


def test_a_clause_not_found_says_so_and_shows_no_passage(make_client):
    client = make_client(router=FixedRouter({"q": route("Most Favored Nation")}))
    html = body(ask(client, "q")[-1])
    assert "badge-not_found" in html and NOT_FOUND_TEXT in html
    assert "data-follow" not in html and copy.SHOW_IN_DOCUMENT not in html
    assert 'id="feedback-1"' in html


def test_a_check_answer_offers_confirm_and_not_this(client):
    html = body(ask(client, PARTIES_QUESTION)[-1])
    parties = top(client, "Parties")
    assert "badge-check" in html and copy.STATUS["check"] in html
    assert f'hx-post="{DECIDE}"' in html and 'hx-target="#document-pane"' in html
    assert 'name="clause_type" value="Parties"' in html
    assert f'name="start" value="{parties.start}"' in html
    assert 'value="confirmed"' in html and copy.CONFIRM in html
    assert 'value="rejected"' in html and copy.REJECT in html


def test_a_found_answer_offers_no_decision(client):
    html = body(ask(client, LAW_QUESTION)[-1])
    assert "/decide" not in html and copy.CONFIRM not in html


def test_a_confirmed_clause_shows_confirmed_by_you_in_the_chat(client):
    ask(client, PARTIES_QUESTION)
    decide(client, "Parties", "confirmed")
    last = ask(client, PARTIES_QUESTION)[-1]
    html = body(last)
    assert html.count("badge-confirmed") == 2 and copy.STATUS["confirmed"] in html
    assert "badge-check" not in html and 'value="confirmed"' not in html
    assert f'hx-post="{DECIDE}/undo"' in html and copy.UNDO in html
    # The document opens at the passage with the reader's verdict.
    assert "status=confirmed" in html and "status=check" not in html


def test_an_earlier_card_follows_a_decision_taken_later(client):
    ask(client, PARTIES_QUESTION)
    decide(client, "Parties", "rejected")
    html = client.get(CHAT).text
    assert "badge-rejected" in html and copy.STATUS["rejected"] in html
    assert f'hx-post="{DECIDE}/undo"' in html and copy.REJECT not in html
    client.post(f"{DECIDE}/undo", data={"clause_type": "Parties"}, follow_redirects=False)
    html = client.get(CHAT).text
    assert "badge-check" in html and copy.CONFIRM in html


def test_a_follow_up_passes_the_earlier_turns_to_the_router(make_client):
    router = FixedRouter({"q1": route("Governing Law"), "q2": route()})
    client = make_client(router=router)
    ask(client, "q1")
    ask(client, "q2")
    assert router.turns == [[], [Turn(question="q1", clause_types=["Governing Law"])]]


def test_a_definition_question_quotes_the_defining_sentence(make_client):
    client = make_client(gen=EchoTextClient())
    html = body(ask(client, "What is the Customer?")[-1])
    assert "badge-definition" in html and copy.STATUS["definition"] in html
    assert "quote quote-plain" in html and "<strong>Customer</strong>" in html
    assert f">{copy.EXPLAIN}<" in html and copy.EXPLAIN_SR not in html
    assert "label=Definition" in html and "status=definition" in html
    assert "/decide" not in html and plain(html)
    assert conversation(client).last.definition.term == "Customer"


def test_an_out_of_scope_question_streams_an_answer_from_the_passages(make_client):
    gen = EchoTextClient()
    client = make_client(router=FixedRouter({SHIPPING_QUESTION: route()}), gen=gen)
    replies = ask(client, SHIPPING_QUESTION)
    assert any(WRITING_ANSWER in text for text in pending_texts(replies))
    html = body(replies[-1])
    assert "badge-generated" in html and html.index("badge-generated") < html.index("badge-related")
    assert copy.STATUS["related"] in html and str(escape(PASSAGES_INTRO)) in html
    assert copy.SHOW_IN_DOCUMENT in html and f'action="{EXPLAIN}"' not in html
    assert "status=related" in html
    assert gen.model not in html and plain(html)
    assert conversation(client).last.generated.text


def test_a_failed_answer_from_the_passages_keeps_the_passages(make_client):
    gen = FakeLLMClient([LLMError("down"), LLMError("down")])
    client = make_client(router=FixedRouter({SHIPPING_QUESTION: route()}), gen=gen)
    html = body(ask(client, SHIPPING_QUESTION)[-1])
    assert "badge-related" in html and "badge-generated" not in html


def test_an_empty_route_without_passages_lists_the_topics(make_client):
    client = make_client(router=FixedRouter({MARS_QUESTION: route()}))
    html = body(ask(client, MARS_QUESTION)[-1])
    assert NO_MATCH in html
    assert "General terms" in html and "Other common clauses" in html
    assert 'data-fill="Who are the parties?"' in html
    assert "data-follow" not in html and 'id="feedback-1"' in html


def test_explain_that_after_a_found_answer_streams_an_explanation(make_client):
    gen = EchoTextClient()
    client = make_client(gen=gen)
    ask(client, LAW_QUESTION)
    replies = ask(client, "explain that")
    assert any(WRITING_EXPLANATION in text for text in pending_texts(replies))
    assert not any(" s<" in text or " s " in text for text in pending_texts(replies))
    html = body(replies[-1])
    assert html.rindex("badge-generated") > html.rindex("explain that")
    assert "<strong>Governing Law in plain English</strong>" in html
    assert gen.model not in html and plain(html)
    assert "label=Governing+Law" in html[html.rindex("explain that") :]


def test_explain_that_in_serbian_names_the_language(make_client):
    client = make_client(gen=EchoTextClient())
    ask(client, LAW_QUESTION)
    assert "Governing Law in plain Serbian" in body(ask(client, "Explain that in Serbian")[-1])


def test_a_held_back_explanation_says_why(make_client):
    gen = FakeLLMClient(['The clause says "nothing like this is in the contract".'])
    client = make_client(gen=gen)
    ask(client, LAW_QUESTION)
    html = body(ask(client, "explain that")[-1])
    withheld = copy.AI_WITHHELD.format(reason=MISQUOTE_REASON)
    assert str(escape(withheld)) in html and gen.model not in html


def test_explain_that_before_any_answer_says_what_to_ask(make_client):
    client = make_client(gen=EchoTextClient())
    assert str(escape(NOTHING_TO_EXPLAIN)) in body(ask(client, "explain that")[-1])


def test_explain_that_without_a_model_says_so_in_plain_words(client):
    ask(client, LAW_QUESTION)
    last = ask(client, "explain that")[-1]
    assert AI_OFF in body(last) and SUMMARIES_OFF not in body(last)
    assert "notice" not in last


def test_answers_offer_explanations_only_with_a_model(client, make_client):
    assert copy.EXPLAIN_SR not in body(ask(client, LAW_QUESTION)[-1])
    generating = make_client(gen=EchoTextClient())
    html = body(ask(generating, LAW_QUESTION)[-1])
    assert f'action="{EXPLAIN}"' in html and copy.EXPLAIN_SR in html
    assert 'data-target="#chat-body"' in html and "data-activate" not in html


def test_summarize_streams_a_summary_then_reuses_it(make_client):
    gen = EchoTextClient()
    client = make_client(gen=gen)
    replies = ask(client, "summarize")
    assert any(WRITING_SUMMARY in text for text in pending_texts(replies))
    html = body(replies[-1])
    assert "badge-generated" in html and BASED_ON in html
    assert "chat-source-found" in html and "chat-source-check" in html
    assert html.count('class="chat-source ') == BASED_ON_SHOWN and " more</span>" in html
    assert gen.model not in html and plain(html)
    calls = len(gen.text_calls)
    again = ask(client, "summarize")
    assert len(again) == 2 and len(gen.text_calls) == calls
    assert body(again[-1]).count("<strong>Summary</strong>") == 2


def test_summarize_without_a_model_says_so_in_plain_words(client):
    assert AI_OFF in body(ask(client, "summarize")[-1])


def test_the_explain_route_restates_a_passage_of_the_document(make_client):
    client = make_client(gen=EchoTextClient())
    law = top(client, "Governing Law")
    data = {
        "label": "Governing Law",
        "quote": law.text,
        "start": law.start,
        "end": law.end,
        "status": "found",
    }
    replies = frames(client, "POST", EXPLAIN, data=data)
    assert "Explain: Governing Law" in body(replies[0])
    html = body(replies[-1])
    assert "Governing Law in plain English" in html and "activate" not in replies[-1]
    assert f"start={law.start}" in html and copy.SHOW_IN_DOCUMENT in html
    assert conversation(client).last is None


def test_an_inline_explanation_fills_only_the_pending_place_at_first(make_client):
    client = make_client(gen=EchoTextClient())
    data = {"label": "Governing Law", "quote": top(client, "Governing Law").text, "inline": "1"}
    data["language"] = "sr"
    replies = frames(client, "POST", EXPLAIN, data=data)
    assert CHAT_BODY not in replies[0]["oob"] and "html" not in replies[0]
    assert "Explain in Serbian: Governing Law" in replies[0]["oob"][PENDING_ID]
    html = body(replies[-1])
    assert "Governing Law in plain Serbian" in html
    assert "data-follow" not in html


def test_the_explain_route_refuses_an_unknown_language(make_client):
    client = make_client(gen=EchoTextClient())
    data = {"label": "Governing Law", "quote": "Some text.", "language": "de"}
    replies = frames(client, "POST", EXPLAIN, data=data)
    assert len(replies) == 1 and replies[0]["done"]
    assert "English or Serbian" in replies[0]["notice"]
    assert conversation(client).messages == []


def test_feedback_writes_a_line_and_the_chat_keeps_the_thanks(client, settings):
    ask(client, LAW_QUESTION)
    response = client.post(FEEDBACK, data={"index": "1", "liked": "1"})
    assert response.status_code == 200 and FEEDBACK_THANKS in response.text
    [verdict] = read_feedback(settings.cache_dir / FEEDBACK_FILE)
    assert verdict.liked and verdict.question == LAW_QUESTION
    assert verdict.clause_types == ["Governing Law"] and verdict.doc_id == SHORT_DOC_ID
    html = client.get(CHAT).text
    assert FEEDBACK_THANKS in html and 'id="feedback-1"' not in html


def test_feedback_on_a_message_without_an_answer_is_refused(client, settings):
    ask(client, LAW_QUESTION)
    response = client.post(FEEDBACK, data={"index": "0", "liked": "0"})
    assert response.status_code == 400 and response.text == ANSWER_GONE
    assert read_feedback(settings.cache_dir / FEEDBACK_FILE) == []


def test_clear_empties_the_chat(client):
    ask(client, LAW_QUESTION)
    html = client.post(CLEAR).text
    assert EMPTY_TITLE in html
    chat = conversation(client)
    assert chat.messages == [] and chat.turns == [] and chat.last is None


def test_bad_thresholds_in_the_answer_drop_the_question_with_a_notice(client, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("tau_low is above tau_high")

    monkeypatch.setattr(chat, "answer", refuse)
    last = ask(client, LAW_QUESTION)[-1]
    assert last["notice"] == "tau_low is above tau_high" and last["done"]
    assert LAW_QUESTION not in body(last) and conversation(client).messages == []


def test_a_failure_while_answering_ends_with_a_message_in_the_chat(make_client):
    client = make_client(router=FixedRouter({}))
    last = ask(client, "q")[-1]
    assert last["done"] and SOMETHING_WRONG in body(last)
    assert len(conversation(client).messages) == 2


def test_an_empty_question_is_a_notice(client):
    assert ask(client, "   ") == [{"notice": TYPE_A_QUESTION, "done": True}]
    assert conversation(client).messages == []


def test_a_question_about_an_unknown_contract_is_not_found(client):
    response = client.post("/documents/sha256:nope/ask", data={"question": "Who?"})
    assert response.status_code == 404
