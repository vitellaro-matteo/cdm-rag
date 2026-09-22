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
