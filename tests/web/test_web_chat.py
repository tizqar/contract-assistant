from markupsafe import escape

from contract_assistant.feedback import FEEDBACK_FILE, read_feedback
from contract_assistant.router.base import RouteResult, Turn
from contract_assistant.router.clients import LLMError
from contract_assistant.ui.views import EMPTY_ROUTE_TEXT, NOT_FOUND_TEXT, PASSAGES_TEXT
from contract_assistant.web import chat
from contract_assistant.web.chat import (
    ANSWER_GONE,
    FEEDBACK_THANKS,
    NOTHING_TO_EXPLAIN,
    PENDING_ID,
    THINKING,
    TYPE_A_QUESTION,
    WRITING_ANSWER,
)
from contract_assistant.web.core import SUMMARIES_OFF, UNEXPECTED_ERROR
from contract_assistant.web.generate import WRITING_EXPLANATION, WRITING_SUMMARY
from tests.support.fakes import EchoTextClient, FakeLLMClient, FixedRouter
from tests.web.conftest import SHORT_DOC_ID, frames

CHAT = f"/documents/{SHORT_DOC_ID}/chat"
ASK = f"/documents/{SHORT_DOC_ID}/ask"
EXPLAIN = f"/documents/{SHORT_DOC_ID}/explain"
FEEDBACK = f"/documents/{SHORT_DOC_ID}/feedback"
CLEAR = f"/documents/{SHORT_DOC_ID}/chat/clear"
LAW_QUESTION = "Which law governs the contract?"
# Matches no clause type of the fake router, and the fake encoder finds passages close to it.
SHIPPING_QUESTION = "How are the goods shipped and delivered to the customer?"
MARS_QUESTION = "What about the weather on Mars?"


def route(*names: str) -> RouteResult:
    return RouteResult(clause_types=list(names), kind="explicit", latency_ms=0)


def ask(client, question: str) -> list[dict]:
    return frames(client, "POST", ASK, data={"question": question})


def law_clause(client):
    return client.app.state.workspace.load(SHORT_DOC_ID).clauses["Governing Law"][0]


def conversation(client):
    return client.app.state.workspace.conversation(SHORT_DOC_ID)


def pending_texts(replies: list[dict]) -> list[str]:
    return [reply["oob"][PENDING_ID] for reply in replies if "oob" in reply]


def test_the_chat_starts_empty(client):
    html = client.get(CHAT).text
    assert "Ask a question about this contract" in html
    assert 'id="chat-messages"' in html and "Clear chat" not in html


def test_a_routed_question_shows_a_found_card_and_the_passage(client):
    replies = ask(client, LAW_QUESTION)
    first, last = replies[0], replies[-1]
    assert first["activate"] == "chat" and "done" not in first
    assert LAW_QUESTION in first["html"] and THINKING in first["html"]
    assert last["done"] and last["activate"] == "chat"
    assert "badge-found" in last["html"]
    assert str(escape(law_clause(client).text)) in last["html"]
    assert "Show in document" in last["html"] and 'id="feedback-1"' in last["html"]
    pane = last["load"]["document-pane"]
    assert "label=Governing+Law" in pane and "status=found" in pane
    assert f"start={law_clause(client).start}" in pane


def test_the_chat_lists_the_messages_oldest_first(client):
    ask(client, LAW_QUESTION)
    ask(client, "Who are the parties?")
    html = client.get(CHAT).text
    assert html.index(LAW_QUESTION) < html.index("Who are the parties?")
    assert "Clear chat" in html and "Ask a question about this contract" not in html


def test_a_clause_not_found_says_so_and_shows_no_passage(make_client):
    client = make_client(router=FixedRouter({"q": route("Most Favored Nation")}))
    last = ask(client, "q")[-1]
    assert "badge-not_found" in last["html"] and NOT_FOUND_TEXT in last["html"]
    assert "load" not in last and "Show in document" not in last["html"]


def test_a_follow_up_passes_the_earlier_turns_to_the_router(make_client):
    router = FixedRouter({"q1": route("Governing Law"), "q2": route()})
    client = make_client(router=router)
    ask(client, "q1")
    ask(client, "q2")
    assert router.turns == [[], [Turn(question="q1", clause_types=["Governing Law"])]]


def test_a_definition_question_quotes_the_defining_sentence(client):
    last = ask(client, "What is the Customer?")[-1]
    assert "badge-definition" in last["html"] and "quote quote-plain" in last["html"]
    assert "<strong>Customer</strong>" in last["html"]
    assert "The sentence of the contract that defines the term" in last["html"]
    pane = last["load"]["document-pane"]
    assert "label=Definition" in pane and "status=definition" in pane
    assert conversation(client).last.definition.term == "Customer"


def test_an_out_of_scope_question_streams_an_answer_from_the_passages(make_client):
    client = make_client(router=FixedRouter({SHIPPING_QUESTION: route()}), gen=EchoTextClient())
    replies = ask(client, SHIPPING_QUESTION)
    assert any(WRITING_ANSWER in text for text in pending_texts(replies))
    html = replies[-1]["html"]
    assert "badge-generated" in html and "Answer from the passages" in html
    assert "badge-related" in html and str(escape(PASSAGES_TEXT)) in html
    assert "status=related" in replies[-1]["load"]["document-pane"]
    assert conversation(client).last.generated.text


def test_a_failed_answer_from_the_passages_keeps_the_passages(make_client):
    gen = FakeLLMClient([LLMError("down"), LLMError("down")])
    client = make_client(router=FixedRouter({SHIPPING_QUESTION: route()}), gen=gen)
    html = ask(client, SHIPPING_QUESTION)[-1]["html"]
    assert "badge-related" in html and "badge-generated" not in html


def test_an_empty_route_without_passages_lists_the_clause_groups(make_client):
    client = make_client(router=FixedRouter({MARS_QUESTION: route()}))
    last = ask(client, MARS_QUESTION)[-1]
    assert EMPTY_ROUTE_TEXT in last["html"]
    assert "General terms" in last["html"] and "Other common clauses" in last["html"]
    assert 'data-fill="Who are the parties?"' in last["html"]
    assert "load" not in last


def test_explain_that_after_a_found_answer_streams_an_explanation(make_client):
    client = make_client(gen=EchoTextClient())
    ask(client, LAW_QUESTION)
    replies = ask(client, "explain that")
    assert any(WRITING_EXPLANATION in text for text in pending_texts(replies))
    html = replies[-1]["html"]
    assert html.rindex("badge-generated") > html.rindex("explain that")
    assert "<strong>Governing Law</strong> · in plain English" in html
    assert "This is a restatement, not a quote" in html
    assert "label=Governing+Law" in replies[-1]["load"]["document-pane"]


def test_explain_that_in_serbian_names_the_language(make_client):
    client = make_client(gen=EchoTextClient())
    ask(client, LAW_QUESTION)
    assert "in plain Serbian" in ask(client, "Explain that in Serbian")[-1]["html"]


def test_explain_that_before_any_answer_says_what_to_ask(make_client):
    client = make_client(gen=EchoTextClient())
    assert str(escape(NOTHING_TO_EXPLAIN)) in ask(client, "explain that")[-1]["html"]


def test_explain_that_without_a_model_says_so_in_the_chat(client):
    ask(client, LAW_QUESTION)
    last = ask(client, "explain that")[-1]
    assert SUMMARIES_OFF in last["html"] and "notice" not in last


def test_answers_offer_explanations_only_with_a_model(client, make_client):
    assert "Explain in Serbian" not in ask(client, LAW_QUESTION)[-1]["html"]
    generating = make_client(gen=EchoTextClient())
    html = ask(generating, LAW_QUESTION)[-1]["html"]
    assert f'action="{EXPLAIN}"' in html and "Explain in Serbian" in html


def test_summarize_streams_a_summary_then_reuses_it(make_client):
    gen = EchoTextClient()
    client = make_client(gen=gen)
    replies = ask(client, "summarize")
    assert any(WRITING_SUMMARY in text for text in pending_texts(replies))
    html = replies[-1]["html"]
    assert "badge-generated" in html and "Based on" in html
    assert "Text in double quotes is verified word for word" in html
    calls = len(gen.text_calls)
    again = ask(client, "summarize")
    assert len(again) == 2 and len(gen.text_calls) == calls
    assert again[-1]["html"].count("<strong>Summary</strong>") == 2


def test_summarize_without_a_model_says_so_in_the_chat(client):
    assert SUMMARIES_OFF in ask(client, "summarize")[-1]["html"]


def test_the_explain_route_restates_a_passage_of_the_document(make_client):
    client = make_client(gen=EchoTextClient())
    law = law_clause(client)
    data = {
        "label": "Governing Law",
        "quote": law.text,
        "start": law.start,
        "end": law.end,
        "status": "found",
    }
    replies = frames(client, "POST", EXPLAIN, data=data)
    assert "Explain: Governing Law" in replies[0]["html"]
    assert "in plain English" in replies[-1]["html"] and replies[-1]["activate"] == "chat"
    assert f"start={law.start}" in replies[-1]["load"]["document-pane"]
    assert conversation(client).last is None


def test_an_inline_explanation_fills_only_the_pending_place_at_first(make_client):
    client = make_client(gen=EchoTextClient())
    data = {"label": "Governing Law", "quote": law_clause(client).text, "inline": "1"}
    data["language"] = "sr"
    replies = frames(client, "POST", EXPLAIN, data=data)
    assert "html" not in replies[0]
    assert "Explain in Serbian: Governing Law" in replies[0]["oob"][PENDING_ID]
    assert "in plain Serbian" in replies[-1]["html"]
    assert "load" not in replies[-1]


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
    assert "Ask a question about this contract" in html
    chat = conversation(client)
    assert chat.messages == [] and chat.turns == [] and chat.last is None


def test_bad_thresholds_in_the_answer_drop_the_question_with_a_notice(client, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("tau_low is above tau_high")

    monkeypatch.setattr(chat, "answer", refuse)
    last = ask(client, LAW_QUESTION)[-1]
    assert last["notice"] == "tau_low is above tau_high" and last["done"]
    assert LAW_QUESTION not in last["html"] and conversation(client).messages == []


def test_a_failure_while_answering_ends_with_a_message_in_the_chat(make_client):
    client = make_client(router=FixedRouter({}))
    last = ask(client, "q")[-1]
    assert last["done"] and UNEXPECTED_ERROR in last["html"]
    assert len(conversation(client).messages) == 2


def test_an_empty_question_is_a_notice(client):
    assert ask(client, "   ") == [{"notice": TYPE_A_QUESTION, "done": True}]
    assert conversation(client).messages == []


def test_a_question_about_an_unknown_contract_is_not_found(client):
    response = client.post("/documents/sha256:nope/ask", data={"question": "Who?"})
    assert response.status_code == 404
