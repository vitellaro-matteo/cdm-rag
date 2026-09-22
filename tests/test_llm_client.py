import pytest

from cdm_rag import llm_client


def test_missing_llm_model_raises_immediately_without_touching_the_client(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(llm_client, "_client", lambda: pytest.fail("_client() must not be called"))
    with pytest.raises(RuntimeError, match="LLM_MODEL.*\\.env\\.example"):
        llm_client.chat([{"role": "user", "content": "hi"}])


def test_empty_llm_model_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "   ")
    monkeypatch.setattr(llm_client, "_client", lambda: pytest.fail("_client() must not be called"))
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        llm_client.chat([{"role": "user", "content": "hi"}])


def test_missing_groq_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm_client._client.cache_clear()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY.*\\.env\\.example"):
        llm_client.chat([{"role": "user", "content": "hi"}])


def test_chat_passes_model_and_messages_through_and_returns_reply_text(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")

    class FakeMessage:
        content = "the reply"

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletions:
        def create(self, model, messages):
            self.seen = (model, messages)
            return type("R", (), {"choices": [FakeChoice()]})()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    fake = FakeClient()
    monkeypatch.setattr(llm_client, "_client", lambda: fake)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    result = llm_client.chat(messages)

    assert result == "the reply"
    assert fake.chat.completions.seen == ("test-model", messages)
