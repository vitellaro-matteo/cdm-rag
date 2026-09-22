import pytest

from cdm_rag.chunks import (
    MAX_ENTITY_TOKENS,
    build_entity_chunks,
    build_relationship_chunks,
    estimate_tokens,
)
from cdm_rag.config import INDEX_EXCLUDED_ENTITY_NAMES
from cdm_rag.graph import banking_seeds, build_graph
from cdm_rag.inheritance import Corpus, EntityRef

@pytest.fixture(scope="module")
def corpus(corpus_root):
    return Corpus(corpus_root)


@pytest.fixture(scope="module")
def graph(corpus):
    return build_graph(corpus, banking_seeds(corpus))


@pytest.fixture(scope="module")
def entity_chunks(graph):
    return {c.entity_id: c for c in build_entity_chunks(graph)}


@pytest.fixture(scope="module")
def rel_chunks(graph):
    return build_relationship_chunks(graph)


def banking_id(graph, name):
    (node,) = [n for n in graph.find(name) if n.layer == "banking"]
    return node.entity_id


def rel_chunk(chunks, graph, entity, attribute):
    (chunk,) = [c for c in chunks if c.from_id == banking_id(graph, entity) and c.attribute == attribute]
    return chunk


# --- entity chunks -----------------------------------------------------------


def test_banking_account_entity_chunk(graph, entity_chunks):
    acct = banking_id(graph, "Account")
    chunk = entity_chunks[acct]
    attrs = graph.attributes[acct]

    for own in ("annualReviewDate", "availableLimit", "daysPastDue"):
        assert own in chunk.text
    assert "Account (CRM base)" in chunk.text  # the parent chain names the CRM base parent
    assert chunk.text.count("Standard audit fields") == 1

    # a summary, not a dump: only a small fraction of the 135 names appear
    assert len(attrs) == 135
    mentioned = [a.name for a in attrs if a.name in chunk.text]
    assert len(mentioned) < 40
    # ...while the structured lookup still answers "list all attributes" exactly
    assert {a.name for a in attrs} - set(mentioned)
    assert chunk.attribute_counts == {"own": 28, "inherited": 92, "standard": 15, "total": 135}
    assert estimate_tokens(chunk.text) <= MAX_ENTITY_TOKENS


def test_entity_chunk_metadata(graph, entity_chunks):
    acct = banking_id(graph, "Account")
    meta = entity_chunks[acct].to_dict()["metadata"]
    assert meta["chunk_type"] == "entity" and meta["name"] == "Account" and meta["layer"] == "banking"
    assert meta["entity_id"] == acct and meta["parent_id"] == graph.nodes[acct].parent_id
    assert meta["source_file"] == "Account.cdm.json" and meta["document"].endswith("banking/Account.cdm.json")
    assert set(meta["attribute_count"]) == {"own", "inherited", "standard", "total"}


def test_one_entity_chunk_per_node_within_token_budget(graph, entity_chunks):
    # One chunk per node, except the infrastructure entities excluded from the index (below).
    assert set(entity_chunks) == set(graph.nodes) - {n.entity_id for n in graph.nodes.values() if n.name in INDEX_EXCLUDED_ENTITY_NAMES}
    assert len(entity_chunks) == 51
    assert max(estimate_tokens(c.text) for c in entity_chunks.values()) <= MAX_ENTITY_TOKENS
    for c in entity_chunks.values():
        assert c.text.count("Standard audit fields") <= 1


def test_infrastructure_entities_are_excluded_from_the_index_but_stay_in_the_graph(graph, entity_chunks):
    for name in INDEX_EXCLUDED_ENTITY_NAMES:
        (node,) = graph.find(name)
        assert node.entity_id in graph.nodes  # still a graph node: exact-name lookup and parent chains work
        assert node.entity_id not in entity_chunks  # but no entity chunk is built for it
    # CampaignResponse's parent chain still walks through the excluded ActivitySystem/ActivityCommon
    (campaign_response,) = graph.find("CampaignResponse")
    chain_names = [n.name for n in graph.chain(campaign_response.entity_id)]
    assert {"ActivitySystem", "ActivityCommon", "CdsStandard"} <= set(chain_names)


def test_process_entity_with_inlined_audit_fields_collapses_them_and_omits_empty_sections(graph, entity_chunks):
    text = entity_chunks[banking_id(graph, "BusinessCheckingAccount")].text
    assert "Inherited attributes" not in text and "It has no parent entity." in text
    assert text.count("Standard audit fields") == 1  # inlined audit fields still collapse into one line
    assert "Related entities: Account (banking), Opportunity (banking)." in text  # organization (audit) is excluded
    assert "createdBy (" not in text and "organization (" in text  # own attribute list is unfiltered


# --- relationship chunks -----------------------------------------------------


def test_branch_to_bank_mentions_both_directions(graph, rel_chunks):
    chunk = rel_chunk(rel_chunks, graph, "Branch", "bank")
    assert "Branch has a many-to-one relationship to Bank" in chunk.text
    assert "via attribute bank (foreign key bankId)" in chunk.text
    assert "Reverse: a Bank can have many Branch records" in chunk.text
    meta = chunk.to_dict()["metadata"]
    assert meta["chunk_type"] == "relationship" and meta["to_ids"] == [banking_id(graph, "Bank")]
    assert meta["fk_name"] == "bankId" and not meta["fk_inferred"] and not meta["is_polymorphic"]
    assert meta["inherited_from"] is None


def test_polymorphic_edge_lists_all_targets_in_one_sentence(graph, rel_chunks):
    chunk = rel_chunk(rel_chunks, graph, "FinancialProduct", "customer")
    assert "customer can refer to Account (banking) or Contact (banking)" in chunk.text
    assert chunk.is_polymorphic and len(chunk.to_ids) == 2


def test_self_edge_reads_as_a_qualified_reference_not_a_tautology(graph, rel_chunks):
    # The seed-layer rule (see graph.py) makes Account.parentAccount and Account.master point
    # back at Account (banking) itself; the plain template would read "Account has a
    # many-to-one relationship to Account", so self-edges get their own phrasing.
    account_id = banking_id(graph, "Account")
    parent = rel_chunk(rel_chunks, graph, "Account", "parentAccount")
    assert parent.from_id == account_id and parent.to_ids == (account_id,)
    assert "Account (banking) has a many-to-one relationship to Account (banking)" not in parent.text
    assert "can have a parent Account (banking), via attribute parentAccount" in parent.text
    assert "can be the parent of many Account (banking) records" in parent.text
    assert parent.text[0].isupper()

    master = rel_chunk(rel_chunks, graph, "Account", "master")
    assert master.from_id == account_id and master.to_ids == (account_id,)
    assert "can have a master Account (banking), via attribute master" in master.text
    assert master.text[0].isupper()


def test_four_way_polymorphic_edge_from_core(corpus):
    ref = EntityRef("core/applicationCommon/Appointment.cdm.json", "Appointment")
    g = build_graph(corpus, [ref])
    (chunk,) = [c for c in build_relationship_chunks(g) if c.attribute == "regardingObject"]
    assert "regardingObject can refer to Account, Contact, KnowledgeArticle or KnowledgeBaseRecord" in chunk.text
    assert len(chunk.to_ids) == 4 and chunk.is_polymorphic


def test_inferred_fk_name_is_not_stated_as_fact(graph, rel_chunks):
    inferred = rel_chunk(rel_chunks, graph, "Contact", "parentCustomer")
    assert inferred.fk_inferred
    assert "conventionally named parentCustomerId" in inferred.text
    assert "foreign key parentCustomerId" not in inferred.text
    assert "inherited from" in inferred.text and inferred.inherited_from

    declared = rel_chunk(rel_chunks, graph, "Branch", "bank")
    assert "conventionally" not in declared.text


def test_no_chunk_for_any_audit_edge(graph, rel_chunks):
    audit = [e for e in graph.edges.values() if e.is_audit]
    assert audit, "audit edges must remain in the graph"
    chunk_ids = {c.chunk_id for c in rel_chunks}
    assert not any(f"rel:{e.edge_id}" in chunk_ids for e in audit)
    assert not any(c.fk_name in {"createdBy", "modifiedBy", "ownerId", "organizationId", "transactionCurrencyId"} for c in rel_chunks)
    assert len(rel_chunks) == len(graph.edges) - len(audit)


def test_chunk_count_summary(graph, entity_chunks, rel_chunks):
    summary = {
        "entity_chunks": len(entity_chunks),
        "relationship_chunks": len(rel_chunks),
        "edges_total": len(graph.edges),
        "edges_audit_no_chunk": sum(e.is_audit for e in graph.edges.values()),
    }
    assert summary == {
        "entity_chunks": 51, "relationship_chunks": 92, "edges_total": 133, "edges_audit_no_chunk": 41}
    assert len({c.chunk_id for c in rel_chunks}) == len(rel_chunks)  # ids are unique
