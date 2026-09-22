import sys

import pytest
from fastapi.testclient import TestClient

from cdm_rag import api, router
from cdm_rag.graph import banking_seeds, build_graph
from cdm_rag.inheritance import Corpus


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
