import pytest

from cdm_rag import generate


def test_answer_sends_system_prompt_context_and_question_then_returns_reply(monkeypatch):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return "the answer"

    monkeypatch.setattr(generate.llm_client, "chat", fake_chat)

    chunks = [
        {"text": "Branch has a many-to-one relationship to Bank.", "metadata": {"chunk_type": "relationship"}},
        {"text": "Account is an entity in the banking layer.", "metadata": {"chunk_type": "entity"}},
    ]
    result = generate.answer("How does Branch relate to Bank?", chunks)

    assert result == "the answer"
    assert len(seen["messages"]) == 2
    system, user = seen["messages"]
    assert system["role"] == "system" and system["content"] == generate.SYSTEM_PROMPT
    assert "answer only from" in system["content"].lower()
    assert "no direct relationship, but x is related via y" in system["content"].lower()
    # every relevant near-miss must be enumerated, not just the first one found
    assert "enumerate all" in system["content"].lower()
    assert "not only the first or most obvious" in system["content"].lower()
    # an infrastructure/tenant note must be surfaced when present, not silently dropped
    assert "note" in system["content"].lower() and "infrastructure/tenant" in system["content"].lower()
    assert user["role"] == "user"
    assert "Branch has a many-to-one relationship to Bank." in user["content"]
    assert "Account is an entity in the banking layer." in user["content"]
    assert "How does Branch relate to Bank?" in user["content"]


def test_answer_accepts_searchresult_like_objects_not_just_dicts(monkeypatch):
    class FakeSearchResult:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    seen = {}
    monkeypatch.setattr(generate.llm_client, "chat", lambda messages: seen.setdefault("messages", messages) or "ok")

    chunks = [FakeSearchResult("Contact can have a parent Account.", {"chunk_type": "relationship"})]
    generate.answer("q", chunks)
    assert "Contact can have a parent Account." in seen["messages"][1]["content"]


def test_empty_chunk_list_still_calls_the_llm_with_a_says_so_context(monkeypatch):
    seen = {}
    monkeypatch.setattr(generate.llm_client, "chat", lambda messages: seen.setdefault("messages", messages) or "ok")

    generate.answer("anything", [])
    assert "no chunks were retrieved" in seen["messages"][1]["content"].lower()


# --- truncation retry (see llm_client.ResponseTruncatedError) ---------------------------------


def test_answer_retries_once_with_doubled_max_tokens_after_truncation(monkeypatch):
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise generate.llm_client.ResponseTruncatedError("cut off", max_tokens=100)
        return "complete answer"

    monkeypatch.setattr(generate.llm_client, "chat", fake_chat)

    result = generate.answer("q", [])

    assert result == "complete answer"
    assert len(calls) == 2
    assert "max_tokens" not in calls[0]  # first attempt uses llm_client's own default, unspecified here
    assert calls[1]["max_tokens"] == 200  # doubled from the raised exception's max_tokens=100


def test_answer_raises_a_clear_error_when_the_retry_also_truncates(monkeypatch):
    # A second truncation must never be swallowed or served as a partial answer -- it propagates.
    def fake_chat(messages, **kwargs):
        raise generate.llm_client.ResponseTruncatedError("still cut off", max_tokens=100)

    monkeypatch.setattr(generate.llm_client, "chat", fake_chat)

    with pytest.raises(generate.llm_client.ResponseTruncatedError):
        generate.answer("q", [])


def test_answer_retry_clears_stale_usage_before_the_successful_retrys_own_usage(monkeypatch):
    calls = []

    def fake_chat(messages, capture_usage=None, max_tokens=None):
        calls.append(max_tokens)
        if len(calls) == 1:
            if capture_usage is not None:
                capture_usage.update(completion_tokens=999)  # stale: must not survive into `timing`
            raise generate.llm_client.ResponseTruncatedError("cut off", max_tokens=100)
        if capture_usage is not None:
            capture_usage.update(completion_tokens=50)
        return "complete answer"

    monkeypatch.setattr(generate.llm_client, "chat", fake_chat)

    timing: dict = {}
    result = generate.answer("q", [], timing=timing)

    assert result == "complete answer"
    assert calls == [None, 200]
    assert timing["completion_tokens"] == 50


def test_no_module_other_than_llm_client_imports_groq():
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "cdm_rag"
    offenders = []
    for f in src.glob("*.py"):
        if f.name == "llm_client.py":
            continue
        text = f.read_text(encoding="utf-8")
        if re.search(r"^\s*(import groq|from groq)", text, re.MULTILINE):
            offenders.append(f.name)
    assert offenders == []
