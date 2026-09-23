import pytest

from cdm_rag import generate, router
from cdm_rag.embeddings import load_model
from cdm_rag.graph import EntityNode, Graph, banking_seeds, build_graph, entity_id
from cdm_rag.inheritance import Corpus, EntityRef
from cdm_rag.store import build_index


def _tiny_graph(*names: str) -> Graph:
    """A graph with one bare, attribute-less, edge-less node per name -- enough to test
    entity-name detection and routing without needing the real corpus. ``graph.attributes`` is
    populated (empty) so ``entity_detail()`` doesn't KeyError on these synthetic nodes."""
    graph = Graph()
    for name in names:
        ref = EntityRef(f"{name}.cdm.json", name)
        eid = entity_id(ref)
        graph.nodes[eid] = EntityNode(eid, ref, name, "test", name, None)
        graph.attributes[eid] = ()
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


def test_detect_entity_pair_finds_the_first_two_of_three_named_entities():
    graph = _tiny_graph("Account", "Bank", "Contact")
    # detect_single_entity must never fire alongside this: 3 names named is still "not one"
    pair = router.detect_entity_pair("Account, Bank, and Contact are all entities.", graph)
    assert pair == ("Account", "Bank")


def test_detect_single_entity_finds_the_one_named_entity():
    graph = _tiny_graph("Account", "Bank")
    assert router.detect_single_entity("What are the attributes of Account?", graph) == "Account"


def test_detect_single_entity_is_case_insensitive():
    graph = _tiny_graph("Account", "Bank")
    assert router.detect_single_entity("what does account inherit from?", graph) == "Account"


def test_detect_single_entity_returns_none_for_zero_names():
    graph = _tiny_graph("Account", "Bank")
    assert router.detect_single_entity("What is the weather today?", graph) is None


def test_detect_single_entity_returns_none_for_two_or_more_names():
    graph = _tiny_graph("Account", "Bank", "Contact")
    assert router.detect_single_entity("How does Account relate to Bank?", graph) is None
    assert router.detect_single_entity("Account, Bank, and Contact are all entities.", graph) is None


def test_detect_single_entity_is_whole_word_not_substring():
    graph = _tiny_graph("Product", "FinancialProduct")
    assert router.detect_single_entity("Tell me about FinancialProduct.", graph) == "FinancialProduct"


# --- attribute detection, real corpus (needs real resolved attributes) ---------


def test_detect_attribute_finds_the_plain_name(real_graph):
    assert router.detect_attribute("What can the regardingObject attribute point to?", real_graph) == "regardingObject"


def test_detect_attribute_finds_the_fk_column_name(real_graph):
    assert router.detect_attribute("What does the bankId attribute point to?", real_graph) == "bank"


def test_detect_attribute_is_case_insensitive(real_graph):
    assert router.detect_attribute("what does the bankid attribute point to?", real_graph) == "bank"


def test_detect_attribute_returns_none_for_an_unknown_attribute(real_graph):
    assert router.detect_attribute("What does the frobnicate attribute do?", real_graph) is None


def test_detect_attribute_returns_none_for_two_named_attributes(real_graph):
    assert router.detect_attribute("How do bank and customer differ?", real_graph) is None


def test_detect_attribute_does_not_by_itself_defer_to_entity_names(real_graph):
    # detect_attribute alone doesn't know about entity precedence -- "Account" is both a real
    # entity name and a real attribute name in this corpus (BusinessCheckingAccount.Account ->
    # Account). retrieve() is what enforces the precedence, by only calling detect_attribute
    # after both entity checks come up empty; see the routing tests below.
    assert router.detect_attribute("Tell me about Account.", real_graph) == "Account"


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


def test_multi_hop_fallback_fires_only_when_relations_between_found_nothing_at_all(monkeypatch):
    from cdm_rag.graph import EntityPairRelations

    graph = _tiny_graph("Collateral", "Bank")
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])
    monkeypatch.setattr(
        graph,
        "relations_between",
        lambda a, b: EntityPairRelations(a_name=a, b_name=b, a_ids=(), b_ids=(), edges=(), has_edges=False),
    )

    class FakeEdge:
        attribute = "financialProduct"

    calls = []

    def fake_find_path(a, b, max_hops=4):
        calls.append((a, b))
        return (router.PathHop(edge=FakeEdge(), from_id="collateral#1", to_id="bank#1"),)

    monkeypatch.setattr(graph, "find_path", fake_find_path)

    context = router.retrieve("What's the path from Collateral to Bank?", graph, collection=object())

    assert calls == [("Collateral", "Bank")]  # only called because has_edges and both near-misses were empty
    path_items = [c for c in context if c["metadata"].get("source") == router.SOURCE_MULTI_HOP_PATH]
    assert len(path_items) == 1
    assert "NOT a direct relationship" in path_items[0]["text"]


def test_multi_hop_fallback_never_fires_when_a_near_miss_already_exists(monkeypatch):
    from cdm_rag.graph import Edge, EdgeTarget, EntityPairRelations

    graph = _tiny_graph("Contact", "Organization")
    (contact_id,) = [n.entity_id for n in graph.nodes.values() if n.name == "Contact"]
    fake_edge = Edge(
        edge_id="e1", from_id=contact_id, attribute="employer", fk_name="employerId",
        targets=(EdgeTarget(None, "Account", "accountId"),),  # unresolved target: fine, still renders
        is_audit=False, is_polymorphic=False, fk_inferred=False, inherited_from=None,
    )
    monkeypatch.setattr(
        graph,
        "relations_between",
        lambda a, b: EntityPairRelations(
            a_name=a, b_name=b, a_ids=(), b_ids=(), edges=(), has_edges=False, a_outgoing_near_misses=(fake_edge,)
        ),
    )
    monkeypatch.setattr(graph, "find_path", lambda a, b, max_hops=4: pytest.fail("find_path must not be called"))
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    router.retrieve("How does Contact relate to Organization?", graph, collection=object())  # must not raise


def test_question_naming_no_known_entity_falls_back_to_plain_vector_search(monkeypatch):
    graph = _tiny_graph("Account", "Bank")

    def fail_relations_between(a, b):
        pytest.fail("relations_between must not be called when no entity is named")

    monkeypatch.setattr(graph, "relations_between", fail_relations_between)

    queries = []

    class FakeHit:
        def __init__(self, text):
            self.text = text
            self.metadata = {"chunk_type": "entity"}

    def fake_store_query(collection, text, k=router.DEFAULT_K):
        queries.append((text, k))
        return [FakeHit("Some retrieved chunk.")]

    monkeypatch.setattr(router, "store_query", fake_store_query)

    context = router.retrieve("What is the weather today?", graph, collection=object())

    assert queries == [("What is the weather today?", router.DEFAULT_K)]
    assert context == [
        {"text": "Some retrieved chunk.", "metadata": {"chunk_type": "entity", "source": router.SOURCE_VECTOR_SEARCH}}
    ]


def test_single_entity_question_does_not_call_relations_between_but_does_a_full_lookup(monkeypatch):
    graph = _tiny_graph("Account", "Bank")

    def fail_relations_between(a, b):
        pytest.fail("relations_between must not be called for a single-entity question")

    monkeypatch.setattr(graph, "relations_between", fail_relations_between)
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What are the attributes of Account?", graph, collection=object())

    lookup_items = [c for c in context if c["metadata"].get("source") == router.SOURCE_ENTITY_LOOKUP]
    assert len(lookup_items) == 1
    assert lookup_items[0]["metadata"]["entity"] == "Account"
    assert lookup_items[0]["metadata"]["chunk_type"] == "entity_detail"
    assert "Account" in lookup_items[0]["text"]


# --- layer-duplicate filtering (Option A) + over-fetch, synthetic --------------


class _FakeHit:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


def test_layer_filter_overfetches_so_k_useful_items_still_come_back(monkeypatch):
    graph = _tiny_graph("Account", "Bank")
    monkeypatch.setattr(graph, "relations_between", lambda a, b: pytest.fail("must not fire"))

    # 3 "other layer" Account duplicates (mirrors the real corpus's worst case: Account exists in
    # 4 layers) plus exactly 5 genuinely different hits -- if the filter starved the budget, fewer
    # than 5 would come back; if over-fetch didn't happen, the duplicates would eat into the 5.
    raw_hits = [
        _FakeHit("dup-core", {"chunk_type": "entity", "name": "Account", "layer": "Core"}),
        _FakeHit("dup-foundation", {"chunk_type": "entity", "name": "Account", "layer": "Foundation"}),
        _FakeHit("dup-crmbase", {"chunk_type": "entity", "name": "Account", "layer": "CRM base"}),
        _FakeHit("keep-1", {"chunk_type": "attribute", "attribute": "Account"}),
        _FakeHit("keep-2", {"chunk_type": "entity", "name": "User", "layer": "Core"}),
        _FakeHit("keep-3", {"chunk_type": "entity", "name": "Account", "layer": "test"}),  # resolved layer (_tiny_graph nodes are all layer="test"): kept
        _FakeHit("keep-4", {"chunk_type": "attribute", "attribute": "address"}),
        _FakeHit("keep-5", {"chunk_type": "entity", "name": "BusinessCheckingAccount", "layer": "banking"}),
    ]
    seen_k = []

    def fake_store_query(collection, text, k=router.DEFAULT_K):
        seen_k.append(k)
        return raw_hits[:k]

    monkeypatch.setattr(router, "store_query", fake_store_query)

    context = router.retrieve("What are the attributes of Account?", graph, collection=object())

    assert seen_k == [router.DEFAULT_K + router.LAYER_FILTER_OVERFETCH]  # over-fetched, not plain k
    vector_items = [c for c in context if c["metadata"].get("source") == router.SOURCE_VECTOR_SEARCH]
    assert len(vector_items) == router.DEFAULT_K  # still k genuinely useful items, not fewer
    kept_texts = {c["text"] for c in vector_items}
    assert kept_texts == {"keep-1", "keep-2", "keep-3", "keep-4", "keep-5"}
    assert kept_texts.isdisjoint({"dup-core", "dup-foundation", "dup-crmbase"})


def test_layer_filter_is_a_no_op_when_the_entity_has_no_other_layers(monkeypatch):
    """Control case: an entity that exists in only one layer must behave exactly as before --
    same items, same count, nothing accidentally dropped by a filter that has nothing to do."""
    graph = _tiny_graph("Bank")  # single layer in this synthetic graph, like the real Bank
    raw_hits = [
        _FakeHit(f"hit-{i}", {"chunk_type": "entity", "name": n, "layer": "test"})
        for i, n in enumerate(["Branch", "Syndicates", "FinancialProduct", "Limit", "RequestedFacility"])
    ]

    def fake_store_query(collection, text, k=router.DEFAULT_K):
        return raw_hits[:k]

    monkeypatch.setattr(router, "store_query", fake_store_query)

    context = router.retrieve("What entities reference Bank?", graph, collection=object())

    vector_items = [c for c in context if c["metadata"].get("source") == router.SOURCE_VECTOR_SEARCH]
    assert len(vector_items) == router.DEFAULT_K
    assert {c["text"] for c in vector_items} == {h.text for h in raw_hits}


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


def test_contact_organization_context_never_includes_a_multi_hop_path_item(real_graph, monkeypatch):
    # The non-negotiable regression check: Contact already has real near-misses against
    # Organization (employer, parentCustomer -- see the test above), so find_path's fallback must
    # never even be attempted here, and this context must be identical to before that fallback
    # existed. Asserted two ways: the fallback's own trigger condition is false (checked directly
    # against the real graph, not assumed), and no multi_hop_path item is present in the context
    # a real question actually produces.
    relations = real_graph.relations_between("Contact", "Organization")
    assert relations.has_edges is False
    assert relations.a_outgoing_near_misses != ()  # this is what keeps the fallback from firing

    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])
    context = router.retrieve("How does Contact relate to Organization?", real_graph, collection=object())

    assert not any(c["metadata"].get("source") == router.SOURCE_MULTI_HOP_PATH for c in context)


def test_branch_bank_context_has_the_real_edge_and_no_near_misses(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])
    context = router.retrieve("How does Branch relate to Bank?", real_graph, collection=object())

    rel_items = [c for c in context if c["metadata"].get("chunk_type") == "relationship"]
    assert len(rel_items) == 1
    assert rel_items[0]["metadata"]["attribute"] == "bank"
    assert rel_items[0]["metadata"]["is_audit"] is False
    assert "Branch" in rel_items[0]["text"] and "Bank" in rel_items[0]["text"]
    assert not any(c["metadata"].get("chunk_type") == "note" for c in context)


def _entity_lookup_item(context):
    (item,) = [c for c in context if c["metadata"].get("source") == router.SOURCE_ENTITY_LOOKUP]
    return item


def test_account_core_attributes_question_gets_the_full_own_attribute_list_and_the_right_layer(real_graph, monkeypatch):
    # The word "core" in the question must not cause the *Core*-layer Account to be selected
    # over the banking one -- this is exactly the failure an eval run found: vector search alone
    # retrieved "Account (Core)" (80 attrs) for this question, not the intended banking Account.
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What are the core attributes of the Account entity?", real_graph, collection=object())

    item = _entity_lookup_item(context)
    assert item["metadata"]["layer"] == "banking"
    assert item["metadata"]["entity"] == "Account"

    (node,) = [n for n in real_graph.find("Account") if n.layer == "banking"]
    own_names = {a.name for a in real_graph.attributes[node.entity_id] if a.origin.value == "own"}
    assert len(own_names) == 28  # every one of the 28 own attributes, not a truncated sample
    assert all(name in item["text"] for name in own_names)
    # spot-check names known to sit outside any short "first few" sample a chunk would show
    assert "annualReviewDate" in item["text"]
    assert "availableLimit" in item["text"]
    assert "daysPastDue" in item["text"]


def test_banking_account_inherit_question_gets_the_full_ancestor_chain_and_real_attribute_names(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What does the banking Account inherit from?", real_graph, collection=object())

    item = _entity_lookup_item(context)
    text = item["text"]
    # the full 4-level chain, not just the immediate parent
    assert "Account (CRM base) -> Account (Foundation) -> Account (Core) -> CdsStandard (CDS standard)" in text
    # real inherited attribute names are spelled out, never collapsed to "+N more"
    assert "more)" not in text
    assert "accountId" in text  # one of the 92 real inherited attribute names
    (node,) = [n for n in real_graph.find("Account") if n.layer == "banking"]
    inherited_names = {a.name for a in real_graph.attributes[node.entity_id] if a.origin.value == "inherited"}
    assert len(inherited_names) == 92
    assert all(name in text for name in inherited_names)
    # type + description too, not just names -- this is what makes filtering the ancestors'
    # duplicate entity chunks out of secondary search (see the layer-filter tests) lose nothing:
    # their type/description detail is now here instead.
    assert "accountId (entityId): Unique identifier of the account." in text
    assert "(listLookup)" in text  # a real CDM data type, not just bare names


def test_entity_lookup_inherited_attributes_have_type_and_description_grouped_by_ancestor(real_graph):
    items = router.entity_lookup_context(real_graph, "Account")
    text = items[0]["text"]
    assert "- from Account (Core) (80):" in text  # ancestor heading, count only -- not the names list
    # the old rendering put this on the heading line itself ("... (1): defaultPriceLevel"); now
    # each attribute is its own indented, typed line underneath.
    assert "- from Account (Foundation) (1): defaultPriceLevel" not in text
    assert "- from Account (Foundation) (1):\n  - defaultPriceLevel (reference to PriceList)" in text


def test_account_core_attributes_header_warns_against_the_coincidental_core_layer_match(real_graph, monkeypatch):
    """The Q1 fix: the entity_lookup block is correct and first, but the model still pulled its
    answer from a Core-layer chunk elsewhere in context, because "core" in the question happens
    to match the "Core" layer label. The header now says explicitly not to make that inference."""
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What are the core attributes of the Account entity?", real_graph, collection=object())

    (header,) = [c for c in context if c["metadata"].get("source") == router.SOURCE_SECTION_HEADER]
    assert "authoritative" in header["text"].lower()
    assert "coincidental" in header["text"].lower()
    assert "layer" in header["text"].lower()


# --- ancestor-layer lexical collision (2nd Q1 attempt) --------------------------


def test_ancestor_layer_collision_detected_for_core(real_graph):
    from cdm_rag.graph import entity_detail, find_entity

    node = find_entity(real_graph, "Account")
    detail = entity_detail(real_graph, node)
    assert router._ancestor_layer_collision("What are the core attributes of the Account entity?", detail) == ("core", "Core")


def test_ancestor_layer_collision_none_when_no_word_matches(real_graph):
    from cdm_rag.graph import entity_detail, find_entity

    node = find_entity(real_graph, "Contact")
    detail = entity_detail(real_graph, node)
    assert router._ancestor_layer_collision("What are the attributes of Contact?", detail) is None


def test_route_adds_the_ambiguity_instruction_only_when_a_collision_exists(real_graph):
    """Fast, real_index-free check of _route() itself (no vector search involved): the
    instruction is only added for the colliding question, not for an ordinary single-entity one."""
    _, _, _, extra_core = router._route("What are the core attributes of the Account entity?", real_graph)
    assert extra_core is not None
    assert "Ambiguity check" in extra_core
    assert '"core"' in extra_core and '"Core"' in extra_core

    _, _, _, extra_contact = router._route("What are the attributes of Contact?", real_graph)
    assert extra_contact is None


def test_account_core_attributes_prompt_includes_the_ambiguity_instruction(real_graph, real_index, monkeypatch):
    seen = {}
    monkeypatch.setattr(generate.llm_client, "chat", lambda messages: seen.setdefault("messages", messages) or "ok")

    router.answer("What are the core attributes of the Account entity?", real_graph, real_index)

    system_content = seen["messages"][0]["content"]
    assert "Ambiguity check" in system_content
    assert '"core"' in system_content
    assert '"Core"' in system_content
    assert "own attributes" in system_content


def test_contact_attributes_prompt_control_gets_no_ambiguity_instruction(real_graph, real_index, monkeypatch):
    """Control question naming a single entity (Contact) whose ancestor layer names don't
    collide with any word in the question -- the prompt must be unchanged, no regression."""
    seen = {}
    monkeypatch.setattr(generate.llm_client, "chat", lambda messages: seen.setdefault("messages", messages) or "ok")

    router.answer("What are the attributes of Contact?", real_graph, real_index)

    system_content = seen["messages"][0]["content"]
    assert "Ambiguity check" not in system_content
    assert system_content == generate.SYSTEM_PROMPT


def _attribute_lookup_item(context):
    (item,) = [c for c in context if c["metadata"].get("source") == router.SOURCE_ATTRIBUTE_LOOKUP]
    return item


def test_regarding_object_question_gets_a_direct_attribute_lookup_with_real_targets(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What can the regardingObject attribute point to?", real_graph, collection=object())

    item = _attribute_lookup_item(context)
    assert item["metadata"]["attribute"] == "regardingObject"
    assert item["metadata"]["is_polymorphic"] is True
    for target in ("Account", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord"):
        assert target in item["text"]


def test_bankid_question_gets_a_direct_attribute_lookup_the_same_way(real_graph, monkeypatch):
    """A non-polymorphic attribute question, named by its FK column form ("bankId") rather than
    its plain name ("bank"), routes the same way."""
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("What does the bankId attribute point to?", real_graph, collection=object())

    item = _attribute_lookup_item(context)
    assert item["metadata"]["attribute"] == "bank"
    assert item["metadata"]["is_polymorphic"] is False
    assert "Bank" in item["text"]


def test_attribute_lookup_never_fires_when_a_known_entity_is_also_named(real_graph, monkeypatch):
    monkeypatch.setattr(router, "store_query", lambda collection, text, k=router.DEFAULT_K: [])

    context = router.retrieve("How does the bank attribute on Branch relate to Bank?", real_graph, collection=object())

    assert not any(c["metadata"].get("source") == router.SOURCE_ATTRIBUTE_LOOKUP for c in context)
    assert any(c["metadata"].get("source") == router.SOURCE_DIRECT_EDGE for c in context)


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


def test_collateral_bank_real_pipeline_gets_the_multi_hop_path_as_a_labeled_fallback(real_graph, real_index):
    # Q16 from the harder eval set. Ground truth (verified directly against the graph): no direct
    # Collateral-Bank edge and no near-miss either (Collateral's only edge is to FinancialProduct,
    # which doesn't lexically or structurally match "Bank"), so this exercises find_path as a real
    # last resort, not relations_between's own logic -- confirmed explicitly, not just inferred
    # from the context this produces.
    relations = real_graph.relations_between("Collateral", "Bank")
    assert relations.has_edges is False
    assert relations.a_outgoing_near_misses == ()
    assert relations.b_incoming_near_misses == ()

    question = "What's the path from Collateral to Bank?"
    context = router.retrieve(question, real_graph, real_index)

    (path_item,) = [c for c in context if c["metadata"].get("source") == router.SOURCE_MULTI_HOP_PATH]
    assert path_item["metadata"]["hops"] == 3
    assert "NOT a direct relationship" in path_item["text"]
    assert "Collateral" in path_item["text"]
    assert "Bank" in path_item["text"]
    assert "FinancialProduct" in path_item["text"]
    assert "Branch" in path_item["text"]
    assert "via financialProduct" in path_item["text"]
    assert "via branch" in path_item["text"]
    assert "via bank" in path_item["text"]
    # exactly one compact item for the whole path, not one per hop (see path_context's docstring)
    assert sum(1 for c in context if c["metadata"].get("source") == router.SOURCE_MULTI_HOP_PATH) == 1


def test_account_core_attributes_real_pipeline_excludes_other_layer_duplicates(real_graph, real_index):
    """The actual Q1 bug, end to end: real vector search for this question does surface Account
    (Core)/(Foundation)/(CRM base) among its raw top hits (verified separately against the live
    index); this confirms the router's filter actually keeps them out of what reaches the model,
    and that k genuinely useful secondary items still come back despite the filtering."""
    context = router.retrieve("What are the core attributes of the Account entity?", real_graph, real_index)

    other_layer_duplicates = [
        c
        for c in context
        if c["metadata"].get("chunk_type") == "entity" and c["metadata"].get("name") == "Account" and c["metadata"].get("layer") != "banking"
    ]
    assert other_layer_duplicates == []

    vector_items = [c for c in context if c["metadata"].get("source") == router.SOURCE_VECTOR_SEARCH]
    assert len(vector_items) == router.DEFAULT_K  # the filter didn't starve secondary context
