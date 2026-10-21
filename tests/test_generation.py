import logging

import pytest

from contract_assistant.generation import (
    EMPTY_REASON,
    MISQUOTE_REASON,
    NUMBERS_REASON,
    REJECTION,
    GeneratedRecord,
    GeneratedStore,
    Generation,
    Partial,
    complete_sentences,
    faithfulness_reason,
    generate,
    generate_stream,
)
from contract_assistant.router.clients import LLMError
from tests.support.fakes import FakeLLMClient

SYSTEM = "system prompt"
USER = "user message"
MAX_TOKENS = 50
SOURCE = "passages"
REASON = "it is wrong"
QUOTE = "The Term is five years."
DOC_ID = "sha256:abcdef"
MODEL = "fake-llm"
TAU_LOW = 0.2
TAU_HIGH = 0.8


class Note(GeneratedRecord):
    prompt_version: int = 1
    body: str = ""


def accept(text: str) -> str | None:
    return None


def reject(text: str) -> str | None:
    return REASON


def reject_unless_good(text: str) -> str | None:
    return None if text == "good." else REASON


def run(client, check=accept, **kwargs):
    return generate(client, SYSTEM, USER, MAX_TOKENS, check, source=SOURCE, **kwargs)


def make_note(**changes) -> Note:
    values = {
        "doc_id": DOC_ID,
        "model": MODEL,
        "tau_low": TAU_LOW,
        "tau_high": TAU_HIGH,
        "latency_ms": 1.0,
        "body": "hello",
    }
    return Note(**(values | changes))


def make_store(tmp_path, kind="note", version=1) -> GeneratedStore[Note]:
    return GeneratedStore(tmp_path / "notes", kind, version, Note)


def test_generate_with_no_attempts_raises_before_any_call():
    client = FakeLLMClient([])
    with pytest.raises(ValueError, match="attempts"):
        run(client, attempts=0)
    assert client.text_calls == []


def test_generate_accepted_reply_is_the_stripped_text():
    client = FakeLLMClient(["  Fine.  \n"])
    result = run(client)
    assert (result.text, result.reply, result.reason, result.attempts) == (
        "Fine.",
        "Fine.",
        None,
        1,
    )
    assert client.text_calls == [(SYSTEM, USER, MAX_TOKENS)]


def test_generate_asks_again_naming_the_reason_the_reply_and_the_source():
    client = FakeLLMClient(["bad.", "good."])
    run(client, reject_unless_good, attempts=2)
    (system1, _, _), (system2, user2, _) = client.text_calls
    assert system1 == system2 == SYSTEM
    assert user2.startswith(USER)
    assert user2 == USER + REJECTION.format(reason=REASON, text="bad.", source=SOURCE)
    assert REASON in user2 and "bad." in user2 and SOURCE in user2


def test_generate_stops_after_the_attempts_with_the_last_reply_and_reason():
    client = FakeLLMClient(["one.", "two.", "three."])
    result = run(client, reject, attempts=2)
    assert (result.text, result.reply, result.reason, result.attempts) == (
        None,
        "two.",
        REASON,
        2,
    )
    assert len(client.text_calls) == 2


def test_generate_gives_the_second_reply_when_the_first_is_rejected():
    result = run(FakeLLMClient(["bad.", "good."]), reject_unless_good, attempts=3)
    assert (result.text, result.attempts) == ("good.", 2)


def test_generate_warning_names_the_label_and_the_model(caplog):
    client = FakeLLMClient(["bad."], model="model-x")
    with caplog.at_level(logging.WARNING, logger="contract_assistant.generation"):
        run(client, reject, label="summary of doc")
    assert "summary of doc" in caplog.text
    assert "model-x" in caplog.text


def test_generate_cuts_an_unfinished_sentence_before_the_check():
    seen = []

    def check(text: str) -> str | None:
        seen.append(text)
        return None

    result = run(FakeLLMClient(["First. Second is cut"]), check)
    assert seen == ["First."]
    assert result.text == "First."


def test_generate_keeps_an_unfinished_sentence_without_trim():
    result = run(FakeLLMClient(["First. Second is cut"]), trim=False)
    assert result.text == "First. Second is cut"


def test_generate_latency_is_not_negative():
    assert run(FakeLLMClient(["Fine."])).latency_ms >= 0


def test_generate_propagates_llm_errors():
    with pytest.raises(LLMError):
        run(FakeLLMClient([LLMError("down")]))


def test_faithfulness_reason_of_empty_text_is_empty():
    assert faithfulness_reason("", [QUOTE], SOURCE) == EMPTY_REASON


def test_faithfulness_reason_of_blank_text_is_empty():
    assert faithfulness_reason("   ", [QUOTE], SOURCE) == EMPTY_REASON


def test_faithfulness_reason_of_a_misquote_names_the_source():
    reason = faithfulness_reason('It says "The Term is ten years."', [QUOTE], SOURCE)
    assert reason == MISQUOTE_REASON.format(source=SOURCE)


def test_faithfulness_reason_of_an_unsupported_number_lists_the_numbers():
    reason = faithfulness_reason("The fee is 500 or 700.", [QUOTE], SOURCE)
    assert reason == NUMBERS_REASON.format(source=SOURCE, numbers="500, 700")


def test_faithfulness_reason_puts_a_misquote_before_a_number():
    reason = faithfulness_reason('It says "ten years" for 500.', [QUOTE], SOURCE)
    assert reason == MISQUOTE_REASON.format(source=SOURCE)


def test_faithfulness_reason_of_faithful_text_is_none():
    assert faithfulness_reason('It says "five years".', [QUOTE], SOURCE) is None


def test_complete_sentences_cuts_an_unfinished_tail():
    assert complete_sentences("One. Two! Three is un") == "One. Two!"


def test_complete_sentences_keeps_a_closing_quote():
    assert complete_sentences('He said "Stop." and left') == 'He said "Stop."'


def test_complete_sentences_keeps_text_without_a_sentence_end():
    assert complete_sentences("no end here") == "no end here"


def test_complete_sentences_keeps_a_finished_text():
    assert complete_sentences("Done.") == "Done."


def test_complete_sentences_of_empty_text_is_empty():
    assert complete_sentences("") == ""


def test_store_round_trips_a_record(tmp_path):
    store = make_store(tmp_path)
    note = make_note()
    store.save(note)
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) == note


def test_store_path_of_is_the_saved_path_under_the_directory(tmp_path):
    store = make_store(tmp_path)
    note = make_note()
    path = store.save(note)
    assert store.path_of(note) == path
    assert path.parent == store.directory


def test_store_file_name_starts_with_the_doc_id_without_prefix(tmp_path):
    path = make_store(tmp_path).path_of(make_note())
    assert path.name.startswith("abcdef-")
    assert "sha256:" not in path.name
    assert path.suffix == ".json"


def test_store_kinds_give_different_paths_for_the_same_keys(tmp_path):
    note = make_note()
    assert make_store(tmp_path, "a").path_of(note) != make_store(tmp_path, "b").path_of(note)


def test_store_other_version_gives_none(tmp_path):
    make_store(tmp_path, version=1).save(make_note())
    assert make_store(tmp_path, version=2).load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None


def test_store_misses_for_another_model(tmp_path):
    store = make_store(tmp_path)
    store.save(make_note())
    assert store.load(DOC_ID, "other", TAU_LOW, TAU_HIGH) is None


def test_store_misses_for_other_thresholds(tmp_path):
    store = make_store(tmp_path)
    store.save(make_note())
    assert store.load(DOC_ID, MODEL, 0.1, TAU_HIGH) is None
    assert store.load(DOC_ID, MODEL, TAU_LOW, 0.9) is None


def test_store_invalid_file_gives_none_and_a_warning_naming_the_kind(tmp_path, caplog):
    store = make_store(tmp_path, kind="special-kind")
    path = store.path_of(make_note())
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="contract_assistant.generation"):
        assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None
    assert "special-kind" in caplog.text


def test_store_record_saved_under_the_path_of_other_keys_gives_none(tmp_path):
    store = make_store(tmp_path)
    path = store.path_of(make_note())
    path.parent.mkdir(parents=True)
    path.write_text(make_note(model="other").model_dump_json(), encoding="utf-8")
    assert store.load(DOC_ID, MODEL, TAU_LOW, TAU_HIGH) is None


def test_store_save_creates_the_directory(tmp_path):
    store = make_store(tmp_path)
    assert not store.directory.exists()
    store.save(make_note())
    assert store.directory.is_dir()


def stream(client, check=accept, **kwargs):
    return generate_stream(client, SYSTEM, USER, MAX_TOKENS, check, source=SOURCE, **kwargs)


def test_stream_yields_growing_partials_of_the_first_attempt_then_the_generation():
    items = list(stream(FakeLLMClient(["One two three."])))
    *partials, final = items
    assert [p.text for p in partials] == ["One ", "One two ", "One two three."]
    assert all(isinstance(p, Partial) and p.attempt == 1 for p in partials)
    assert all(p.previous_reason is None for p in partials)
    assert isinstance(final, Generation)


def test_stream_names_the_first_reason_in_the_partials_of_the_second_attempt():
    client = FakeLLMClient(["bad one.", "good."])
    items = list(stream(client, reject_unless_good, attempts=2))
    *partials, final = items
    second = [p for p in partials if p.attempt == 2]
    assert second
    assert all(p.previous_reason == REASON for p in second)
    assert final.attempts == 2
    assert final.text == "good."


def test_stream_checks_the_stripped_and_trimmed_full_reply():
    seen: list[str] = []

    def record(text: str) -> str | None:
        seen.append(text)
        return None

    list(stream(FakeLLMClient(["  First done. Second cut"]), record))
    assert seen == ["First done."]


def test_stream_without_trim_keeps_an_unfinished_sentence():
    *_, final = stream(FakeLLMClient(["First done. Second cut"]), trim=False)
    assert final.text == "First done. Second cut"


def test_stream_with_fewer_than_one_attempt_raises_before_any_call():
    client = FakeLLMClient(["unused"])
    with pytest.raises(ValueError, match="attempts"):
        next(stream(client, attempts=0))
    assert client.text_calls == []


def test_stream_of_a_rejected_only_run_ends_with_no_text_and_the_last_reply():
    *_, final = stream(FakeLLMClient(["one.", "two."]), reject, attempts=2)
    assert final.text is None
    assert final.reply == "two."
    assert final.reason == REASON
    assert final.attempts == 2


def test_stream_propagates_an_llm_error_raised_mid_stream():
    class Breaking:
        model = MODEL

        def stream_text(self, system, user, max_tokens):
            yield "first "
            raise LLMError("broke")

    consumer = stream(Breaking())
    assert isinstance(next(consumer), Partial)
    with pytest.raises(LLMError, match="broke"):
        next(consumer)


def test_generate_returns_the_generation_the_stream_ends_with():
    replies = ["bad one.", "good."]
    expected = list(stream(FakeLLMClient(replies), reject_unless_good, attempts=2))[-1]
    outcome = run(FakeLLMClient(replies), reject_unless_good, attempts=2)
    assert outcome.model_dump(exclude={"latency_ms"}) == expected.model_dump(
        exclude={"latency_ms"}
    )


def test_generate_makes_as_many_requests_as_the_stream():
    streamed, generated = FakeLLMClient(["a.", "b.", "c."]), FakeLLMClient(["a.", "b.", "c."])
    list(stream(streamed, reject, attempts=3))
    run(generated, reject, attempts=3)
    assert len(generated.text_calls) == len(streamed.text_calls) == 3
