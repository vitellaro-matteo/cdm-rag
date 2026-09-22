import pytest

from cdm_rag import generate, router
from cdm_rag.embeddings import load_model
from cdm_rag.graph import EntityNode, Graph, banking_seeds, build_graph, entity_id
from cdm_rag.inheritance import Corpus, EntityRef
from cdm_rag.store import build_index


def _tiny_graph(*names: str) -> Graph:
    """A graph with one bare node per name, no edges -- enough to test entity-name detection
    without needing the real corpus."""
    graph = Graph()
    for name in names:
        ref = EntityRef(f"{name}.cdm.json", name)
        eid = entity_id(ref)
        graph.nodes[eid] = EntityNode(eid, ref, name, "test", name, None)
    return graph


@pytest.fixture(scope="module")
def real_graph(corpus_root):
    corpus = Corpus(corpus_root)
    return build_graph(corpus, banking_seeds(corpus))


@pytest.fixture(scope="module")
def real_index(real_graph, tmp_path_factory):
    """A real, persisted Chroma index -- used by the tests below that must exercise the actual
    store_query path, not a mock of it. A test that only ever mocks store_query away can pass
    even when the real vector-search + dedup interaction is broken; see the tests marked
    "real store_query" below for the bug class this is meant to catch."""
    load_model()  # amortize the slow first-call JIT warmup across this module's tests
    return build_index(real_graph, persist_dir=tmp_path_factory.mktemp("router_chroma"))


# --- entity-pair detection -----------------------------------------------------


def test_detect_entity_pair_finds_two_names_in_question_order():
    graph = _tiny_graph("Account", "Bank", "Contact")
    assert router.detect_entity_pair("How does Contact relate to Bank?", graph) == ("Contact", "Bank")


def test_detect_entity_pair_is_whole_word_not_substring():
    graph = _tiny_graph("Product", "FinancialProduct")
    # "Product" must not match inside "FinancialProduct"
    pair = router.detect_entity_pair("Tell me about FinancialProduct.", graph)
    assert pair is None  # only one distinct entity named


def test_detect_entity_pair_returns_none_for_a_single_entity_question():
    graph = _tiny_graph("Account", "Bank")
    assert router.detect_entity_pair("What are the attributes of Account?", graph) is None


def test_detect_entity_pair_is_case_insensitive():
    graph = _tiny_graph("Account", "Bank")
    assert router.detect_entity_pair("how does account relate to bank?", graph) == ("Account", "Bank")


# --- routing behavior -----------------------------------------------------------


def test_two_entity_question_calls_relations_between(monkeypatch):
    from cdm_rag.graph import EntityPairRelations

    graph = _tiny_graph("Contact", "Organization")
    calls = []

    def fake_relations_between(a, b):
        calls.append((a, b))
        return EntityPairRelations(a_name=a, b_name=b, a_ids=(), b_ids=(), edges=(), has_edges=False)

    monkeypatch.setattr(graph, "relations_between", fake_relations_between)
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    router.retrieve("How does Contact relate to Organization?", graph, collection=object())

    assert calls == [("Contact", "Organization")]


def test_single_entity_question_does_not_call_relations_between_and_uses_vector_search(monkeypatch):
    graph = _tiny_graph("Account", "Bank")

    def fail_relations_between(a, b):
        pytest.fail("relations_between must not be called for a single-entity question")

    monkeypatch.setattr(graph, "relations_between", fail_relations_between)

    queries = []

    class FakeHit:
        def __init__(self, text):
            self.text = text
            self.metadata = {"chunk_type": "entity"}

    def fake_store_query(collection, text, k=router.DEFAULT_K):
        queries.append((text, k))
        return [FakeHit("Account is an entity in the banking layer.")]

    monkeypatch.setattr(router, "store_query", fake_store_query)

    context = router.retrieve("What are the attributes of Account?", graph, collection=object())

    assert queries == [("What are the attributes of Account?", router.DEFAULT_K)]
    assert context == [{"text": "Account is an entity in the banking layer.", "metadata": {"chunk_type": "entity"}}]


# --- context content, real corpus, mocked store_query ---------------------------
#
# These check relations_context()/retrieve() in isolation from vector search: fast, but they
# only prove relations_between's own output is right, not that it survives being combined with
# real vector-search hits and deduplicated. See the "real store_query" tests below for that.


def _items_by_attribute(context, attribute):
    return [c for c in context if c["metadata"].get("attribute") == attribute]


def test_contact_organization_context_has_both_near_misses_as_distinct_items(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("How does Contact relate to Organization?", real_graph, collection=object())

    employer = _items_by_attribute(context, "employer")
    parent_customer = _items_by_attribute(context, "parentCustomer")
    notes = [c for c in context if c["metadata"].get("chunk_type") == "note"]
    assert len(employer) == 1  # each near-miss is its own context item, not merged with another
    assert len(parent_customer) == 1
    assert len(notes) == 1
    assert parent_customer[0]["metadata"]["is_polymorphic"] is True
    assert "Account" in employer[0]["text"]
    # the infrastructure note explaining *why* Organization has no relationship
    assert "no non-audit (business) relationships" in notes[0]["text"]
    assert "infrastructure/tenant" in notes[0]["text"]


def test_branch_bank_context_has_the_real_edge_and_no_near_misses(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])
    context = router.retrieve("How does Branch relate to Bank?", real_graph, collection=object())

    rel_items = [c for c in context if c["metadata"].get("chunk_type") == "relationship"]
    assert len(rel_items) == 1
    assert rel_items[0]["metadata"]["attribute"] == "bank"
    assert rel_items[0]["metadata"]["is_audit"] is False
    assert "Branch" in rel_items[0]["text"] and "Bank" in rel_items[0]["text"]
    assert not any(c["metadata"].get("chunk_type") == "note" for c in context)


# --- context content, real corpus, real store_query ------------------------------
#
# Exercises the actual retrieve() path end to end: real vector search, real dedup-by-text, real
# message assembly -- the path the router-fix report checked only with store_query mocked away,
# which is exactly why it didn't catch that the message the *real* pipeline sends is fine (both
# near-misses ARE present, as separate items) while the model still under-used it.


def test_contact_organization_real_pipeline_keeps_both_near_misses_as_separate_message_items(real_graph, real_index):
    question = "How does Contact relate to Organization?"
    context = router.retrieve(question, real_graph, real_index)

    employer = _items_by_attribute(context, "employer")
    parent_customer = _items_by_attribute(context, "parentCustomer")
    notes = [c for c in context if c["metadata"].get("chunk_type") == "note"]
    assert len(employer) == 1
    assert len(parent_customer) == 1
    assert len(notes) == 1

    block = generate._context_block(context)
    employer_line = f"(relationship) {employer[0]['text']}"
    parent_line = f"(relationship) {parent_customer[0]['text']}"
    note_line = f"(note) {notes[0]['text']}"

    # each is present verbatim, and as its own blank-line-delimited numbered item -- not just a
    # substring buried inside a different, merged entry (the class of bug a plain "x in message"
    # check on the full string would miss)
    items = [segment for segment in block.split("\n\n") if segment.strip()]
    for line in (employer_line, parent_line, note_line):
        matches = [segment for segment in items if line in segment]
        assert len(matches) == 1, f"expected exactly one context item containing: {line!r}"
    assert not any(employer_line in segment and parent_line in segment for segment in items)


def test_contact_organization_real_pipeline_message_reaches_the_llm_call(real_graph, real_index, monkeypatch):
    seen = {}
    monkeypatch.setattr(generate.llm_client, "chat", lambda messages: seen.setdefault("messages", messages) or "ok")

    router.answer("How does Contact relate to Organization?", real_graph, real_index)

    user_content = seen["messages"][1]["content"]
    assert "via attribute employer" in user_content
    assert "via attribute parentCustomer" in user_content
    assert "infrastructure/tenant" in user_content
