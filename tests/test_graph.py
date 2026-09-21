from collections import Counter
import pytest

from cdm_rag.graph import BANKING_DIR, banking_seeds, build_graph, entity_id, layer_of
from cdm_rag.inheritance import Corpus, EntityRef

@pytest.fixture(scope="module")
def corpus(corpus_root):
    return Corpus(corpus_root)


@pytest.fixture(scope="module")
def graph(corpus):
    return build_graph(corpus, banking_seeds(corpus))


def banking(graph, name):
    (node,) = [n for n in graph.find(name) if n.layer == "banking"]
    return node


def test_banking_scope_and_node_count(graph, corpus):
    seeds = banking_seeds(corpus)
    assert len(seeds) == 24
    assert {n.name for n in graph.nodes.values() if n.layer == "banking"} == {r.name for r in seeds}
    assert len(graph.nodes) == 54
    assert Counter(n.layer for n in graph.nodes.values()) == {
        "banking": 24, "CRM base": 13, "Core": 8, "Foundation": 6, "CDS standard": 3}


def test_only_unversioned_banking_files_are_seeds(corpus):
    assert all(r.document.startswith(BANKING_DIR) for r in banking_seeds(corpus))
    assert not any(r.document.count(".") > 2 for r in banking_seeds(corpus))  # Account.1.0.cdm.json etc.


def test_same_named_entities_are_distinct_nodes_with_a_parent_chain(graph):
    accounts = graph.find("Account")
    assert sorted(n.layer for n in accounts) == ["CRM base", "Core", "Foundation", "banking"]
    assert len({n.entity_id for n in accounts}) == 4
    chain = graph.chain(banking(graph, "Account").entity_id)
    assert [n.display_name for n in chain] == [
        "Account (banking)", "Account (CRM base)", "Account (Foundation)", "Account (Core)", "CdsStandard (CDS standard)"]
    assert chain[0].parent_id == chain[1].entity_id


def test_display_names_and_ids_are_unique_and_stable(graph, corpus):
    assert len({n.display_name for n in graph.nodes.values()}) == len(graph.nodes)
    ref = EntityRef(BANKING_DIR + "Bank.cdm.json", "Bank")
    assert entity_id(ref) == BANKING_DIR + "Bank.cdm.json#Bank" in graph.nodes


def test_every_parent_is_a_node(graph):
    assert all(n.parent_id is None or n.parent_id in graph.nodes for n in graph.nodes.values())


def test_layer_labels():
    assert layer_of(BANKING_DIR + "Account.cdm.json") == "banking"
    assert layer_of("core/applicationCommon/foundationCommon/crmCommon/Account.cdm.json") == "CRM base"
    assert layer_of("core/applicationCommon/foundationCommon/Account.cdm.json") == "Foundation"
    assert layer_of("core/applicationCommon/Account.cdm.json") == "Core"


def test_edge_totals_and_no_unresolved_targets(graph):
    assert len(graph.edges) == 133
    assert sum(e.is_audit for e in graph.edges.values()) == 41
    assert sum(e.is_polymorphic for e in graph.edges.values()) == 5
    assert graph.unresolved_targets == []


def test_reverse_index_bank_is_referenced_by_branch(graph):
    bank, branch = banking(graph, "Bank"), banking(graph, "Branch")
    hits = [e for e in graph.incoming(bank.entity_id) if e.from_id == branch.entity_id and e.attribute == "bank"]
    assert len(hits) == 1 and hits[0].fk_name == "bankId"
    assert hits[0] in graph.outgoing(branch.entity_id)  # forward and reverse agree


def test_audit_edges_stay_in_graph_but_are_hidden_by_default(graph):
    acct = banking(graph, "Account").entity_id
    everything = graph.outgoing(acct, include_audit=True)
    assert any(e.is_audit for e in everything)
    assert not any(e.is_audit for e in graph.outgoing(acct))


def test_polymorphic_edge_keeps_every_target(graph):
    fp = banking(graph, "FinancialProduct")
    (edge,) = [e for e in graph.outgoing(fp.entity_id) if e.attribute == "customer"]
    assert edge.is_polymorphic and len(edge.targets) == 2
    assert {graph.nodes[i].name for i in edge.to_ids} == {"Account", "Contact"}
    # Bare names resolve from the document that wrote them (a shared group), i.e. to the Core
    # entities, not to banking's own Account/Contact. Documented limitation in graph.py.
    assert {graph.nodes[i].layer for i in edge.to_ids} == {"Core"}


def test_full_attribute_lookup_is_exact(graph):
    attrs = graph.attributes[banking(graph, "Account").entity_id]
    assert len(attrs) == 135 and len({a.name for a in attrs}) == 135
