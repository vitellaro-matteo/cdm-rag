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
        def create(self, model, messages, timeout=None, max_tokens=None):
            self.seen = (model, messages)
            self.seen_timeout = timeout
            self.seen_max_tokens = max_tokens
            return type("R", (), {"choices": [FakeChoice()], "usage": None})()

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
    assert fake.chat.completions.seen_timeout == llm_client.REQUEST_TIMEOUT_SECONDS
    assert fake.chat.completions.seen_max_tokens == llm_client.MAX_COMPLETION_TOKENS


def test_chat_retries_once_on_timeout_then_succeeds(monkeypatch):
    from groq import APITimeoutError

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)

    class FakeMessage:
        content = "the reply"

    class FakeChoice:
        message = FakeMessage()

    calls = []

    class FakeCompletions:
        def create(self, model, messages, timeout=None, max_tokens=None):
            calls.append(1)
            if len(calls) == 1:
                raise APITimeoutError(request=object())
            return type("R", (), {"choices": [FakeChoice()], "usage": None})()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm_client, "_client", lambda: FakeClient())

    result = llm_client.chat([{"role": "user", "content": "hi"}])

    assert result == "the reply"
    assert len(calls) == 2


def test_chat_raises_after_second_timeout(monkeypatch):
    from groq import APITimeoutError

    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)

    class FakeCompletions:
        def create(self, model, messages, timeout=None, max_tokens=None):
            raise APITimeoutError(request=object())

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm_client, "_client", lambda: FakeClient())

    with pytest.raises(APITimeoutError):
        llm_client.chat([{"role": "user", "content": "hi"}])
