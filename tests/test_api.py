import sys

import pytest
from fastapi.testclient import TestClient

from cdm_rag import api, demo_guard, router
from cdm_rag.graph import banking_seeds, build_graph
from cdm_rag.inheritance import Corpus


@pytest.fixture(autouse=True)
def _clean_demo_guard_state(monkeypatch):
    """Every test starts with both demo_guard env vars unset and a clean rate-limit window, so
    one test's opt-in doesn't leak into the next -- and so every test written before demo_guard
    existed keeps running exactly as before by default."""
    monkeypatch.delenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv(demo_guard.DEMO_RATE_LIMIT_ENV_VAR, raising=False)
    demo_guard.reset_rate_limit_state()
    yield
    demo_guard.reset_rate_limit_state()


@pytest.fixture(scope="module")
def real_graph(corpus_root):
    corpus = Corpus(corpus_root)
    return build_graph(corpus, banking_seeds(corpus))


@pytest.fixture
def client():
    """No ``with``: the lifespan never runs, so nothing is loaded until a request needs it
    (see api.py's module docstring) -- exactly what keeps /health and /entities tests fast."""
    return TestClient(api.app)


@pytest.fixture
def client_with_fake_deps(real_graph):
    """/ask tests: inject the real (cheap) graph and a stand-in collection, so router.answer
    being mocked is what it takes to skip the real Chroma/embedding/Groq stack entirely."""
    api.app.dependency_overrides[api.get_graph] = lambda: real_graph
    api.app.dependency_overrides[api.get_collection] = lambda: object()
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


# --- /health ---------------------------------------------------------------------------------


def test_health_returns_ok_and_touches_neither_graph_nor_collection(client, monkeypatch):
    monkeypatch.setattr(api, "_graph", None)
    monkeypatch.setattr(api, "_collection", None)

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert api._graph is None
    assert api._collection is None


# --- /entities/{name} -------------------------------------------------------------------------


def test_get_entity_account_returns_own_attributes_correctly(client, monkeypatch):
    monkeypatch.setattr(api, "_collection", None)

    r = client.get("/entities/Account")

    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "Account"
    assert data["layer"] == "banking"  # most specific layer wins among the 4 Account nodes
    assert data["display_name"] == "Account (banking)"

    own_names = {a["name"] for a in data["own_attributes"]}
    assert {"annualReviewDate", "availableLimit", "daysPastDue"} <= own_names
    assert data["attribute_counts"] == {"own": 28, "inherited": 92, "standard": 15, "total": 135}
    assert len(data["own_attributes"]) == 28
    assert len(data["inherited_attributes"]) == 92
    assert len(data["standard_attributes"]) == 15

    assert data["parent_chain"] == ["Account (CRM base)", "Account (Foundation)", "Account (Core)", "CdsStandard (CDS standard)"]

    outgoing = [r_ for r_ in data["relationships"] if r_["direction"] == "outgoing"]
    incoming = [r_ for r_ in data["relationships"] if r_["direction"] == "incoming"]
    assert any(r_["attribute"] == "SLA" and r_["other_entities"] == ["SLA (Core)"] for r_ in outgoing)
    assert any(
        r_["attribute"] == "Account" and r_["other_entities"] == ["BusinessCheckingAccount (banking)"] for r_ in incoming
    )
    assert not any(r_["is_audit"] for r_ in data["relationships"] if "is_audit" in r_)  # audit edges never appear here

    # pure graph lookup: the vector store is never touched
    assert api._collection is None


def test_get_entity_is_case_insensitive(client):
    exact = client.get("/entities/Account").json()
    lower = client.get("/entities/account").json()
    assert lower["entity_id"] == exact["entity_id"]


def test_get_entity_unknown_name_returns_404(client):
    r = client.get("/entities/NoSuchEntityXYZ")
    assert r.status_code == 404


# --- /ask --------------------------------------------------------------------------------------


def test_ask_returns_answer_sources_and_context_with_router_answer_mocked(client_with_fake_deps, real_graph, monkeypatch):
    (contact,) = [n for n in real_graph.find("Contact") if n.layer == "banking"]
    fake_context = [
        {"text": "Organization has no non-audit relationships.", "metadata": {"chunk_type": "note", "source": "note", "entity": "Organization"}},
        {
            "text": "Contact relates to Account via employer.",
            "metadata": {
                "chunk_type": "relationship", "source": "near_miss", "attribute": "employer",
                "from_id": contact.entity_id, "is_audit": False,
            },
        },
    ]
    seen = {}

    def fake_answer(question, graph, collection, k=router.DEFAULT_K):
        seen["args"] = (question, graph, collection, k)
        return router.AnswerResult(answer="There is no direct relationship.", context=fake_context)

    monkeypatch.setattr(router, "answer", fake_answer)

    r = client_with_fake_deps.post("/ask", json={"question": "How does Contact relate to Organization?"})

    assert r.status_code == 200
    assert seen["args"][0] == "How does Contact relate to Organization?"
    data = r.json()
    assert data["answer"] == "There is no direct relationship."
    assert data["context_used"] == [
        "Organization has no non-audit relationships.",
        "Contact relates to Account via employer.",
    ]
    assert data["sources"] == [
        {"entity": "Organization", "chunk_type": "note", "source": "note", "attribute": None, "is_audit": None},
        {"entity": "Contact", "chunk_type": "relationship", "source": "near_miss", "attribute": "employer", "is_audit": False},
    ]


def test_ask_empty_question_returns_400_and_never_calls_router(client_with_fake_deps, monkeypatch):
    monkeypatch.setattr(router, "answer", lambda *a, **k: pytest.fail("router.answer must not be called for an empty question"))

    r = client_with_fake_deps.post("/ask", json={"question": "   "})

    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_ask_missing_question_field_is_a_validation_error(client_with_fake_deps):
    r = client_with_fake_deps.post("/ask", json={})
    assert r.status_code == 422


# --- demo_guard: shared-secret access key ---------------------------------------------------


def test_ask_works_normally_with_no_demo_key_header_when_demo_access_key_is_unset(client_with_fake_deps, monkeypatch):
    # DEMO_ACCESS_KEY is unset by the autouse fixture -- current behavior, completely unaffected.
    monkeypatch.setattr(router, "answer", lambda *a, **k: router.AnswerResult(answer="ok", context=[]))

    r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})

    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


def test_ask_is_blocked_with_a_missing_key_when_demo_access_key_is_set(client_with_fake_deps, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setattr(router, "answer", lambda *a, **k: pytest.fail("router.answer must not be called when auth fails"))

    r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})

    assert r.status_code == 401
    assert "X-Demo-Key" in r.json()["detail"]


def test_ask_is_blocked_with_a_wrong_key_when_demo_access_key_is_set(client_with_fake_deps, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setattr(router, "answer", lambda *a, **k: pytest.fail("router.answer must not be called when auth fails"))

    r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"}, headers={"X-Demo-Key": "wrong"})

    assert r.status_code == 401


def test_ask_succeeds_with_the_correct_key_when_demo_access_key_is_set(client_with_fake_deps, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setattr(router, "answer", lambda *a, **k: router.AnswerResult(answer="ok", context=[]))

    r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"}, headers={"X-Demo-Key": "secret123"})

    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


# --- demo_guard: rate limit --------------------------------------------------------------------


def test_ask_rate_limit_does_not_trigger_when_demo_rate_limit_is_unset(client_with_fake_deps, monkeypatch):
    monkeypatch.setattr(router, "answer", lambda *a, **k: router.AnswerResult(answer="ok", context=[]))

    for _ in range(20):  # well beyond any real cap -- proves the limiter is a true no-op when off
        r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})
        assert r.status_code == 200


def test_ask_rate_limit_triggers_once_the_cap_is_reached(client_with_fake_deps, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_RATE_LIMIT_ENV_VAR, "2")
    monkeypatch.setattr(router, "answer", lambda *a, **k: router.AnswerResult(answer="ok", context=[]))

    first = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})
    second = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})
    third = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "rate limit" in third.json()["detail"].lower()


def test_ask_rate_limit_does_not_count_requests_rejected_for_a_bad_key(client_with_fake_deps, monkeypatch):
    # A request that fails auth must not consume the shared rate-limit budget -- otherwise an
    # unauthenticated caller could exhaust it for everyone with repeated wrong-key attempts.
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setenv(demo_guard.DEMO_RATE_LIMIT_ENV_VAR, "1")
    monkeypatch.setattr(router, "answer", lambda *a, **k: router.AnswerResult(answer="ok", context=[]))

    for _ in range(5):
        r = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"}, headers={"X-Demo-Key": "wrong"})
        assert r.status_code == 401

    ok = client_with_fake_deps.post("/ask", json={"question": "How does Branch relate to Bank?"}, headers={"X-Demo-Key": "secret123"})
    assert ok.status_code == 200  # the budget of 1 is still intact despite 5 failed attempts


# --- demo_guard never affects /health or /entities ----------------------------------------------


def test_health_is_unaffected_by_either_demo_guard_mechanism(client, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setenv(demo_guard.DEMO_RATE_LIMIT_ENV_VAR, "0")  # 0 -> treated as off, but even a real value wouldn't apply here

    for _ in range(5):
        r = client.get("/health")
        assert r.status_code == 200


def test_get_entity_is_unaffected_by_either_demo_guard_mechanism(client, monkeypatch):
    monkeypatch.setenv(demo_guard.DEMO_ACCESS_KEY_ENV_VAR, "secret123")
    monkeypatch.setenv(demo_guard.DEMO_RATE_LIMIT_ENV_VAR, "1")

    for _ in range(5):  # no X-Demo-Key header at all, still succeeds every time
        r = client.get("/entities/Account")
        assert r.status_code == 200


# --- works without GROQ_API_KEY, except /ask --------------------------------------------------


def test_import_does_not_require_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    sys.modules.pop("cdm_rag.api", None)
    import importlib

    fresh = importlib.import_module("cdm_rag.api")
    assert fresh.app is not None


def test_health_and_entities_work_without_groq_api_key(monkeypatch, client):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert client.get("/health").status_code == 200
    assert client.get("/entities/Account").status_code == 200


def test_ask_fails_clearly_without_groq_api_key(client_with_fake_deps, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # real router/graph/generate/llm_client path; only the vector-search leg is stubbed, so this
    # stays fast without needing a real Chroma collection
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    r = client_with_fake_deps.post("/ask", json={"question": "How does Contact relate to Organization?"})

    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]
