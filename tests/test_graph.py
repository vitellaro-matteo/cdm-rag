import json
from collections import Counter

import pytest

from cdm_rag.graph import (
    BANKING_DIR,
    NEAR_MISS_LIMIT,
    attribute_detail,
    attribute_details,
    banking_seeds,
    build_graph,
    entity_id,
    known_attribute_names,
    layer_of,
)
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
    # The shared group that wrote the reference sits above banking, so strict import order would
    # pick the Core entities; the seed-layer rule points at banking's own Account/Contact instead.
    assert {graph.nodes[i].layer for i in edge.to_ids} == {"banking"}
    assert {graph.nodes[t.import_order_id].layer for t in edge.targets} == {"Core"}


def test_banking_account_and_contact_have_incoming_edges(graph):
    # Strict import order left both with none (all 17 / 16 references landed on the Core entities).
    account, contact = banking(graph, "Account"), banking(graph, "Contact")
    for node, total in ((account, 17), (contact, 16)):
        assert len(graph.incoming(node.entity_id, include_audit=True)) == total
        assert len(graph.incoming(node.entity_id)) == total  # none of them is an audit edge
        for other in graph.find(node.name):
            if other.layer != "banking":
                assert graph.incoming(other.entity_id, include_audit=True) == []

    sources = {(graph.nodes[e.from_id].name, e.attribute) for e in graph.incoming(account.entity_id)}
    assert {("FinancialProduct", "customer"), ("KYC", "customer"), ("Contact", "employer")} <= sources
    sources = {(graph.nodes[e.from_id].name, e.attribute) for e in graph.incoming(contact.entity_id)}
    assert {("FinancialProduct", "customer"), ("KYC", "customer"), ("Account", "primaryContact")} <= sources


def test_resolved_by_keeps_the_strict_import_order_result_visible(graph):
    targets = [(e, t) for e in graph.edges.values() for t in e.targets]
    assert all(t.resolved_by in {"seed_layer", "import_order"} for _, t in targets)
    # every target that was not overridden equals the strict import-order result
    assert all(t.entity_id == t.import_order_id for _, t in targets if t.resolved_by == "import_order")
    deviating = Counter(t.name for _, t in targets if t.entity_id != t.import_order_id)
    assert deviating == {"Account": 17, "Contact": 16, "Product": 7, "Lead": 4, "Opportunity": 1}
    assert all(t.resolved_by == "seed_layer" for _, t in targets if t.entity_id != t.import_order_id)
    assert not any(e.is_audit for e, t in targets if t.entity_id != t.import_order_id)

    (edge,) = [e for e in graph.outgoing(banking(graph, "KYC").entity_id) if e.attribute == "customer"]
    assert {t.resolved_by for t in edge.targets} == {"seed_layer"}
    assert [graph.nodes[t.import_order_id].layer for t in edge.targets] == ["Core", "Core"]

    (bank,) = [e for e in graph.outgoing(banking(graph, "Branch").entity_id) if e.attribute == "bank"]
    (target,) = bank.targets  # seed rule and import order agree here
    assert target.entity_id == target.import_order_id


# --- seed-layer resolution rule, synthetic -----------------------------------


def _write(root, path, definitions, imports=()):
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"imports": [{"corpusPath": p, **({"moniker": m} if m else {})} for p, m in imports],
                             "definitions": definitions}))
    return path


def _fk(name, target, fk):
    ref = {"entityReference": target}
    return {"entity": ref, "name": name, "resolutionGuidance": {"entityByReference": {
        "allowReference": True, "foreignKeyAttribute": {"name": fk, "dataType": "entityId"}}}}


def _synthetic(tmp_path, extra_seed_party=False):
    """lib/Party and seed/Party share a name; Order imports only lib, so strict import order picks lib."""
    _write(tmp_path, "lib/Party.cdm.json", [{"entityName": "Party", "hasAttributes": []}])
    _write(tmp_path, "lib/Widget.cdm.json", [{"entityName": "Widget", "hasAttributes": []}])
    _write(tmp_path, "seed/Party.cdm.json", [{"entityName": "Party", "hasAttributes": []}])
    _write(tmp_path, "other/Party.cdm.json", [{"entityName": "Party", "hasAttributes": []}])
    _write(tmp_path, "seed/Order.cdm.json", [{"entityName": "Order", "hasAttributes": [
        _fk("party", "Party", "partyId"), _fk("widget", "Widget", "widgetId"), _fk("viaMoniker", "lib/Party", "monikerId")]}],
        imports=[("/lib/Party.cdm.json", None), ("/lib/Widget.cdm.json", None), ("/lib/Party.cdm.json", "lib")])
    seeds = [EntityRef("seed/Order.cdm.json", "Order"), EntityRef("seed/Party.cdm.json", "Party")]
    if extra_seed_party:
        seeds.append(EntityRef("other/Party.cdm.json", "Party"))
    return Corpus(tmp_path), seeds


def _target(graph, attribute):
    (edge,) = [e for e in graph.edges.values() if e.attribute == attribute]
    (target,) = edge.targets
    return target


def test_seed_layer_entity_wins_over_import_order(tmp_path):
    corpus, seeds = _synthetic(tmp_path)
    t = _target(build_graph(corpus, seeds), "party")
    assert (t.entity_id, t.resolved_by, t.import_order_id) == ("seed/Party.cdm.json#Party", "seed_layer", "lib/Party.cdm.json#Party")


def test_falls_back_to_import_order_when_no_seed_has_the_name(tmp_path):
    corpus, seeds = _synthetic(tmp_path)
    t = _target(build_graph(corpus, seeds), "widget")
    assert (t.entity_id, t.resolved_by, t.import_order_id) == ("lib/Widget.cdm.json#Widget", "import_order", "lib/Widget.cdm.json#Widget")


def test_ambiguous_seed_names_fall_back_to_import_order(tmp_path):
    corpus, seeds = _synthetic(tmp_path, extra_seed_party=True)
    t = _target(build_graph(corpus, seeds), "party")
    assert (t.entity_id, t.resolved_by) == ("lib/Party.cdm.json#Party", "import_order")


def test_monikered_names_are_never_overridden(tmp_path):
    corpus, seeds = _synthetic(tmp_path)
    t = _target(build_graph(corpus, seeds), "viaMoniker")
    assert (t.entity_id, t.resolved_by) == ("lib/Party.cdm.json#Party", "import_order")


def test_full_attribute_lookup_is_exact(graph):
    attrs = graph.attributes[banking(graph, "Account").entity_id]
    assert len(attrs) == 135 and len({a.name for a in attrs}) == 135


# --- relations_between ---------------------------------------------------------


def test_no_edges_between_contact_and_organization_surfaces_near_misses(graph):
    # The headline "how does Contact relate to Organization?" case: Contact has no organizationId
    # and Organization has no outgoing edges in this graph (only seed entities are expanded), so
    # the only edges into Organization at all are 11 audit organizationId edges from banking
    # entities, none of which is Contact. There is no direct link in either direction.
    r = graph.relations_between("Contact", "Organization")
    assert r.has_edges is False
    assert r.edges == ()
    assert r.a_ids == tuple(sorted(n.entity_id for n in graph.find("Contact")))
    assert len(r.a_ids) == 4  # banking, CRM base, Foundation, Core
    assert r.b_ids == tuple(n.entity_id for n in graph.find("Organization"))

    near = {(graph.nodes[e.from_id].name, e.attribute): [graph.nodes[i].name for i in e.to_ids] for e in r.a_outgoing_near_misses}
    assert near[("Contact", "employer")] == ["Account"]
    assert near[("Contact", "parentCustomer")] == ["Account", "Contact"]
    assert not any(e.is_audit for e in r.a_outgoing_near_misses)  # near-misses are non-audit only

    # Contact (banking) actually has 14 non-audit outgoing edges (SLA, SLAInvoked, address,
    # defaultChargeAccount, defaultPriceLevel, employer, enrollmentBranch, master, originatingLead,
    # parentCustomer, preferredBranch, preferredEquipment, preferredService, preferredSystemUser).
    # Only 3 score any relevance to "Organization" (employer/parentCustomer -> Account, master ->
    # Contact -- Account and Contact are this graph's only polymorphic "party" targets, see
    # _polymorphic_target_names); the other 11 are dropped outright, not merely pushed down.
    assert {e.attribute for e in r.a_outgoing_near_misses} == {"employer", "parentCustomer", "master"}
    assert len(r.a_outgoing_near_misses) <= NEAR_MISS_LIMIT

    # Organization has no non-audit incoming edges either: its only incoming edges are audit.
    assert r.b_incoming_near_misses == ()
    assert graph.incoming(r.b_ids[0], include_audit=True) != ()
    assert all(e.is_audit for e in graph.incoming(r.b_ids[0], include_audit=True))


def test_direct_edges_between_branch_and_bank_include_audit_and_both_directions(graph):
    r = graph.relations_between("Branch", "Bank")
    assert r.has_edges is True
    assert r.a_outgoing_near_misses == () and r.b_incoming_near_misses == ()  # only populated when has_edges is False
    non_audit = [e for e in r.edges if not e.is_audit]
    assert len(non_audit) == 1
    edge = non_audit[0]
    assert graph.nodes[edge.from_id].name == "Branch" and edge.attribute == "bank"
    assert [graph.nodes[i].name for i in edge.to_ids] == ["Bank"]
    # order-independent: same result querying "Bank", "Branch"
    reverse = graph.relations_between("Bank", "Branch")
    assert {e.edge_id for e in reverse.edges} == {e.edge_id for e in r.edges}


# --- find_path: BFS multi-hop fallback (see router._route's narrow use of it) ------------------


def test_find_path_collateral_to_bank_goes_through_financialproduct_and_branch(graph):
    # Real ground truth, verified against this graph directly: no direct Collateral-Bank edge,
    # but a genuine 3-hop path -- Collateral's only edge is to FinancialProduct, which has its
    # own direct edge to Branch (independent of FinancialProduct's separate "customer" edge to
    # Account/Contact, which does NOT lead to Bank), and Branch has the direct edge to Bank.
    r = graph.relations_between("Collateral", "Bank")
    assert r.has_edges is False  # confirms this exercises the fallback, not the direct-edge path

    path = graph.find_path("Collateral", "Bank")
    assert path is not None
    assert [h.edge.attribute for h in path] == ["financialProduct", "branch", "bank"]
    hop_names = [graph.nodes[path[0].from_id].name] + [graph.nodes[h.to_id].name for h in path]
    assert hop_names == ["Collateral", "FinancialProduct", "Branch", "Bank"]
    # each hop's to_id chains into the next hop's from_id -- a real connected walk, not
    # independently-found edges stitched together after the fact
    assert path[0].to_id == path[1].from_id
    assert path[1].to_id == path[2].from_id


def test_find_path_contact_to_organization_is_none_at_any_reachable_hop_count(graph):
    # The project's headline "no relationship" result: confirms multi-hop doesn't change it.
    # Organization has zero non-audit edges at all (in or out) anywhere in this graph -- so it is
    # genuinely unreachable via find_path's non-audit BFS, not merely "not found within 4 hops."
    assert graph.find_path("Contact", "Organization") is None
    assert graph.find_path("Contact", "Organization", max_hops=50) is None  # not a cap artifact


def test_find_path_branch_to_bank_is_the_existing_direct_edge_not_duplicated(graph):
    # Multi-hop must not break or re-derive the simple, already-working one-hop case.
    path = graph.find_path("Branch", "Bank")
    assert path is not None
    assert len(path) == 1
    assert path[0].edge.attribute == "bank"
    assert graph.nodes[path[0].from_id].name == "Branch"
    assert graph.nodes[path[0].to_id].name == "Bank"


def test_find_path_respects_the_max_hops_cap(graph):
    # The real Collateral->Bank path is 3 hops; capping below that must return None, not a
    # truncated/wrong path -- confirms the cap is a real search boundary, not decorative.
    assert graph.find_path("Collateral", "Bank", max_hops=2) is None
    assert graph.find_path("Collateral", "Bank", max_hops=3) is not None


def test_find_path_unknown_entity_name_returns_none(graph):
    assert graph.find_path("Collateral", "NoSuchEntity") is None
    assert graph.find_path("NoSuchEntity", "Bank") is None


def _poly_fk(name, targets, fk):
    return {
        "entity": {"entityReference": {"entityName": "Alt", "hasAttributes": [
            {"entity": {"entityReference": t}, "name": t + "Option"} for t in targets
        ]}},
        "name": name,
        "resolutionGuidance": {"entityByReference": {
            "allowReference": True, "foreignKeyAttribute": {"name": fk, "dataType": "entityId"}}},
    }


def _near_miss_scoring_corpus(tmp_path):
    """Widget has 11 non-audit edges against missing_name="Target": 6 lexical matches (attribute
    name contains "target"), 1 structural-only match (points at Hub, a polymorphic "party" target
    elsewhere), and 4 with neither signal. NEAR_MISS_LIMIT=5 < 7 relevant candidates, so this
    also exercises the cap dropping a genuinely-relevant item (hubRef), not just the irrelevant ones."""
    for name in ("Hub", "Decoy", "Noise", "Target"):
        _write(tmp_path, f"seed/{name}.cdm.json", [{"entityName": name, "hasAttributes": []}])
    attrs = [_poly_fk("party", ["Hub", "Decoy"], "partyId")]  # registers Hub as a polymorphic target
    attrs += [_fk(f"targetLike{i}", "Noise", f"targetLike{i}Id") for i in range(6)]  # lexical
    attrs.append(_fk("hubRef", "Hub", "hubRefId"))  # structural only
    attrs += [_fk(f"noise{i}", "Noise", f"noise{i}Id") for i in range(4)]  # zero signal
    _write(tmp_path, "seed/Widget.cdm.json", [{"entityName": "Widget", "hasAttributes": attrs}])
    corpus = Corpus(tmp_path)
    seeds = [EntityRef("seed/Widget.cdm.json", "Widget")]
    return build_graph(corpus, seeds)


def test_near_misses_drop_zero_signal_edges_and_cap_the_rest(tmp_path):
    g = _near_miss_scoring_corpus(tmp_path)
    r = g.relations_between("Widget", "Target")

    assert len(r.a_outgoing_near_misses) == NEAR_MISS_LIMIT  # capped, even though 7 edges scored
    attrs = {e.attribute for e in r.a_outgoing_near_misses}
    assert not any(a.startswith("noise") for a in attrs)  # zero-signal edges dropped outright
    # lexical matches outrank the structural-only one; with 6 lexical candidates for 5 slots,
    # the (relevant, but lower-ranked) structural-only "hubRef" is capped out entirely
    assert attrs <= {f"targetLike{i}" for i in range(6)}
    assert "hubRef" not in attrs


def test_near_misses_keep_a_structural_only_match_when_it_fits_the_cap(tmp_path):
    tmp_path2 = tmp_path / "b"
    tmp_path2.mkdir()
    for name in ("Hub", "Decoy", "Noise", "Target"):
        _write(tmp_path2, f"seed/{name}.cdm.json", [{"entityName": name, "hasAttributes": []}])
    attrs = [
        _poly_fk("party", ["Hub", "Decoy"], "partyId"),
        _fk("hubRef", "Hub", "hubRefId"),  # structural only: no lexical overlap with "Target"
        _fk("noise0", "Noise", "noise0Id"),  # zero signal
    ]
    _write(tmp_path2, "seed/Widget.cdm.json", [{"entityName": "Widget", "hasAttributes": attrs}])
    corpus = Corpus(tmp_path2)
    g = build_graph(corpus, [EntityRef("seed/Widget.cdm.json", "Widget")])

    r = g.relations_between("Widget", "Target")
    # "party" is itself a polymorphic edge to {Hub, Decoy}, so it also scores structural (its own
    # targets are in the polymorphic-target set by construction); "hubRef" scores structural via
    # that same set. Both survive since 2 <= NEAR_MISS_LIMIT; "noise0" (zero signal) does not.
    assert {e.attribute for e in r.a_outgoing_near_misses} == {"hubRef", "party"}


# --- attribute_detail, real corpus ----------------------------------------------


def test_attribute_details_finds_regarding_object_with_its_real_polymorphic_targets(graph):
    # regardingObject has no Edge at all: CampaignResponse (which declares it) isn't a seed
    # entity, so its own FKs are never parsed into edges. attribute_details reads resolved
    # attributes directly, independent of edges, and finds it anyway.
    detail = attribute_detail(graph, "regardingObject")
    assert detail is not None
    assert detail.fk_name == "regardingObjectId"
    assert detail.is_polymorphic is True
    assert set(detail.targets) == {
        "Account", "BookableResourceBooking", "BookableResourceBookingHeader", "Campaign",
        "CampaignActivity", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord", "Lead", "QuickCampaign",
    }
    assert detail.declared_by == ("CampaignResponse (CRM base)",)
    assert detail.inherited_by == ()


def test_attribute_details_finds_a_non_polymorphic_attribute(graph):
    detail = attribute_detail(graph, "bank")
    assert detail.fk_name == "bankId"
    assert detail.is_polymorphic is False
    assert detail.targets == ("Bank",)
    assert set(detail.declared_by) == {"Branch (banking)", "Syndicates (banking)"}


def test_attribute_details_excludes_audit_and_standard_fields(graph):
    names = set(attribute_details(graph))
    # createdBy/modifiedBy/organizationId/... (see relationships.AUDIT_ATTRIBUTES) and any
    # CdsStandard-origin attribute never get their own attribute chunk/lookup -- same fields
    # build_relationship_chunks already excludes via Edge.is_audit, reached here a different way
    # since some of these have no edge at all.
    assert names.isdisjoint({"createdBy", "modifiedBy", "organization", "transactionCurrency", "owner"})
    assert attribute_detail(graph, "createdBy") is None


def test_known_attribute_names_maps_both_plain_and_fk_column_forms(graph):
    tokens = known_attribute_names(graph)
    assert tokens["bank"] == "bank"
    assert tokens["bankId"] == "bank"
    assert tokens["regardingObject"] == "regardingObject"
    assert tokens["regardingObjectId"] == "regardingObject"


def test_no_attribute_name_maps_to_two_different_target_sets(graph):
    # The assumption this whole feature depends on: one chunk per attribute *name* is only
    # correct if no two entities declare that name with a different meaning (a different target
    # set). Verified true for the real corpus by directly re-deriving each name's target set from
    # every node's resolved attributes (not going through attribute_details' own aggregation, so
    # this doesn't just check attribute_details agrees with itself) -- see
    # test_attribute_name_collision_is_a_known_limitation below for what happens if it's ever not.
    from collections import defaultdict

    from cdm_rag.inheritance import Origin
    from cdm_rag.relationships import AUDIT_ATTRIBUTES

    by_name: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for node_id, attrs in graph.attributes.items():
        for a in attrs:
            if a.is_fk and a.fk_targets and not a.fk_is_placeholder and a.origin is not Origin.STANDARD and a.fk_name not in AUDIT_ATTRIBUTES:
                by_name[a.name].add(tuple(sorted(t.entity for t in a.fk_targets)))

    collisions = {name: sets for name, sets in by_name.items() if len(sets) > 1}
    assert collisions == {}

    details = attribute_details(graph)
    assert len(details) == 33
    assert set(details) == set(by_name)


# --- attribute_detail, synthetic: documents the collision limitation -----------


def test_attribute_name_collision_is_a_known_limitation(tmp_path):
    """If two unrelated entities declare an attribute with the SAME name but DIFFERENT targets,
    attribute_details currently merges them into one entry with the UNION of both target sets --
    it does not detect or flag the collision. Checked against the real corpus (no such collision
    exists there today, see test_no_attribute_name_maps_to_two_different_target_sets), but this
    pins the actual behavior for a hypothetical corpus where it does, so it can't silently change
    (or silently start mattering) without a test noticing."""
    for name in ("Bank", "Hotel"):
        _write(tmp_path, f"seed/{name}.cdm.json", [{"entityName": name, "hasAttributes": []}])
    attrs_a = [_fk("manager", "Bank", "managerId")]
    attrs_b = [_fk("manager", "Hotel", "managerId")]
    _write(tmp_path, "seed/A.cdm.json", [{"entityName": "A", "hasAttributes": attrs_a}])
    _write(tmp_path, "seed/B.cdm.json", [{"entityName": "B", "hasAttributes": attrs_b}])
    corpus = Corpus(tmp_path)
    g = build_graph(corpus, [EntityRef("seed/A.cdm.json", "A"), EntityRef("seed/B.cdm.json", "B")])

    detail = attribute_detail(g, "manager")
    # merged, not split: both Bank and Hotel appear as if "manager" were one polymorphic concept
    assert set(detail.targets) == {"Bank", "Hotel"}
    assert detail.is_polymorphic is True
    assert {n.split(" (")[0] for n in detail.declared_by} == {"A", "B"}
