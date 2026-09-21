import json

import pytest

from cdm_rag.config import corpus_path
from cdm_rag.inheritance import Corpus, EntityRef
from cdm_rag.relationships import (
    Relationship,
    Target,
    canonicalize_placeholders,
    derived_relationships,
    is_entity_document,
    merge,
    parse_entity_directory,
    parse_entity_document,
    parse_manifest,
)

CORPUS = corpus_path()
BANKING = CORPUS / "core/applicationCommon/foundationCommon/crmCommon/accelerators/financialServices/banking"
APP_COMMON = CORPUS / "core/applicationCommon"
MANIFEST = APP_COMMON / "applicationCommon.manifest.cdm.json"


def fk_attr(name, target, fk, polymorphic_targets=None):
    if not polymorphic_targets:
        ref = {"entityReference": target, "appliedTraits": [
            {"traitReference": "is.identifiedBy", "arguments": [f"{target}/(resolvedAttributes)/{target.lower()}Id"]}]}
    else:
        ref = {"entityReference": {"entityName": "Alt", "hasAttributes": [
            {"entity": {"entityReference": t}, "name": t + "Option"} for t in polymorphic_targets]}}
    return {"entity": ref, "name": name, "resolutionGuidance": {"entityByReference": {
        "allowReference": True, "foreignKeyAttribute": {"name": fk, "dataType": "entityId"}}}}


def write_doc(tmp_path, name, entity, attrs):
    path = tmp_path / name
    path.write_text(json.dumps({"definitions": [{"entityName": entity, "hasAttributes": attrs}]}))
    return path


# --- synthetic ---------------------------------------------------------------


def test_simple_fk_edge(tmp_path):
    p = write_doc(tmp_path, "Branch.cdm.json", "Branch", [fk_attr("bank", "Bank", "bankId")])
    (rel,) = parse_entity_document(p, tmp_path)
    assert (rel.from_entity, rel.from_attribute) == ("Branch", "bankId")
    assert rel.targets == {Target("Bank", "bankId")}
    assert not rel.is_polymorphic and not rel.is_audit


def test_inline_polymorphic_targets_are_a_set(tmp_path):
    p = write_doc(tmp_path, "A.cdm.json", "A", [fk_attr("regardingObject", None, "id", ["Account", "Contact", "Case"])])
    (rel,) = parse_entity_document(p, tmp_path)
    assert rel.to_entities == {"Account", "Contact", "Case"}
    assert rel.is_polymorphic
    assert rel.from_attribute == "regardingObject"  # placeholder "id" replaced


def test_nested_attribute_groups_are_walked(tmp_path):
    grp = {"attributeGroupReference": {"attributeGroupName": "g", "members": [fk_attr("x", "Bank", "bankId")]}}
    p = write_doc(tmp_path, "N.cdm.json", "N", [grp])
    assert len(parse_entity_document(p, tmp_path)) == 1


def test_entity_attribute_without_entity_by_reference_is_not_an_edge(tmp_path):
    attr = {"entity": {"entityReference": "Contact"}, "name": "contactOption"}
    p = write_doc(tmp_path, "N.cdm.json", "N", [attr])
    assert parse_entity_document(p, tmp_path) == []


@pytest.mark.parametrize("attr,audit", [("createdBy", True), ("ownerId", True), ("bankId", False)])
def test_is_audit_flag(tmp_path, attr, audit):
    p = write_doc(tmp_path, "N.cdm.json", "N", [fk_attr("x", "SystemUser", attr)])
    assert parse_entity_document(p, tmp_path)[0].is_audit is audit


def test_merge_unions_targets_traits_sources():
    a = Relationship("d", "E", "fk", frozenset({Target("A", None)}), frozenset({"t1"}), frozenset({"entity"}))
    b = Relationship("d", "E", "fk", frozenset({Target("B", "id")}), frozenset({"t2"}), frozenset({"manifest"}))
    c = Relationship("d", "E", "other", frozenset({Target("C", None)}))
    merged = {r.key: r for r in merge([a], [b, c])}
    assert len(merged) == 2
    m = merged[("d", "E", "fk")]
    assert m.to_entities == {"A", "B"} and m.traits == {"t1", "t2"} and m.sources == {"entity", "manifest"}


def test_manifest_paths_relative_and_absolute(tmp_path):
    (tmp_path / "sub").mkdir()
    m = tmp_path / "sub" / "x.manifest.cdm.json"
    m.write_text(json.dumps({"relationships": [
        {"fromEntity": "Account.cdm.json/Account", "fromEntityAttribute": "ownerId",
         "toEntity": "/core/SystemUser.cdm.json/SystemUser", "toEntityAttribute": "systemUserId",
         "exhibitsTraits": ["is.CDS.owner"]}]}))
    (rel,) = parse_manifest(m, tmp_path)
    assert rel.from_document == "sub/Account.cdm.json"
    assert rel.targets == {Target("SystemUser", "systemUserId")}
    assert rel.traits == {"is.CDS.owner"} and rel.is_audit


def test_is_entity_document_filters():
    assert is_entity_document("Account.cdm.json")
    assert not is_entity_document("Account.1.5.cdm.json")
    assert not is_entity_document("Account.0.9.cdm.json")
    assert not is_entity_document("_allImports.cdm.json")
    assert not is_entity_document("foo.manifest.cdm.json")


# --- real corpus -------------------------------------------------------------


@pytest.fixture(scope="module")
def banking(corpus_root):
    return parse_entity_directory(BANKING, CORPUS)


def test_banking_edge_count_matches_raw_entity_by_reference_count(banking):
    raw = 0
    for f in BANKING.glob("*.cdm.json"):
        if is_entity_document(f):
            raw += f.read_text(encoding="utf-8").count('"entityByReference"')
    assert raw == 78 == len(banking)


def test_banking_audit_split(banking):
    # 35 of 78 are createdBy/modifiedBy/organizationId/transactionCurrencyId boilerplate.
    assert not any(r.is_polymorphic for r in banking)
    assert sum(r.is_audit for r in banking) == 35
    assert sum(not r.is_audit for r in banking) == 43


def test_banking_known_edge(banking):
    hit = [r for r in banking if r.from_entity == "Collateral" and r.from_attribute == "financialProductId"]
    assert len(hit) == 1
    assert hit[0].targets == {Target("FinancialProduct", "financialProductId")}


@pytest.mark.usefixtures("corpus_root")
def test_core_activity_regarding_is_polymorphic():
    rels = parse_entity_document(APP_COMMON / "Activity.cdm.json", CORPUS)
    regarding = [r for r in rels if r.from_attribute == "regardingObject"]
    assert len(regarding) == 1
    assert {"Account", "Contact"} <= regarding[0].to_entities


@pytest.mark.usefixtures("corpus_root")
def test_manifest_is_audit_dominated_and_polymorphic_present():
    rels = merge(parse_manifest(MANIFEST, CORPUS))
    assert len(rels) > 0
    audit = sum(r.is_audit for r in rels)
    assert audit / len(rels) > 0.5
    assert any(r.is_polymorphic for r in rels)


# --- placeholder FK names ----------------------------------------------------


@pytest.fixture(scope="module")
def corpus(corpus_root):
    return Corpus(corpus_root)


@pytest.mark.usefixtures("corpus_root")
def test_placeholder_edge_is_flagged_before_canonicalization():
    rels = parse_entity_document(APP_COMMON / "Activity.cdm.json", CORPUS)
    (edge,) = [r for r in rels if r.from_attribute == "regardingObject"]
    assert edge.placeholder and edge.is_polymorphic


def test_regarding_object_alias_merges_entity_and_manifest_rows(corpus):
    doc = "core/applicationCommon/Activity.cdm.json"
    entity_rels = canonicalize_placeholders(parse_entity_document(CORPUS / doc, CORPUS), corpus)
    merged = merge(entity_rels, parse_manifest(MANIFEST, CORPUS))
    (edge,) = [r for r in merged if r.key == (doc, "Activity", "regardingObjectId")]
    assert edge.sources == {"entity", "manifest"}
    assert edge.to_entities == {"Account", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord"}
    assert not edge.placeholder and not edge.fk_inferred  # declared on ActivityCommon
    assert not any(r.from_attribute == "regardingObject" for r in merged if r.from_document == doc)


def test_inferred_fk_names_match_manifest_rows(corpus):
    manifest_keys = {r.key for r in parse_manifest(MANIFEST, CORPUS)}
    entity_rels = []
    for f in sorted(APP_COMMON.glob("*.cdm.json")):
        if is_entity_document(f):
            entity_rels += parse_entity_document(f, CORPUS)
    fixed = canonicalize_placeholders(entity_rels, corpus)
    assert not any(r.placeholder for r in fixed)
    inferred = [r for r in fixed if r.fk_inferred]
    # Where the manifest independently names the FK, "<attribute>Id" is what it says.
    assert len(inferred) >= 10
    assert all(r.key in manifest_keys for r in inferred)


# --- inherited / group-declared / placeholder edges --------------------------


def write_entities(root, path, definitions, imports=()):
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"imports": list(imports), "definitions": definitions}))


def test_child_gets_edge_for_fk_declared_on_parent(tmp_path):
    write_entities(tmp_path, "parent.cdm.json", [
        {"entityName": "Parent", "hasAttributes": [fk_attr("owner", "User", "ownerUserId")]}])
    write_entities(tmp_path, "child.cdm.json", [
        {"entityName": "Child", "extendsEntity": "Parent", "hasAttributes": []}],
        [{"corpusPath": "parent.cdm.json"}])
    (edge,) = derived_relationships([EntityRef("child.cdm.json", "Child")], Corpus(tmp_path))
    assert (edge.from_entity, edge.from_attribute, edge.attribute_name) == ("Child", "ownerUserId", "owner")
    assert edge.inherited_from == EntityRef("parent.cdm.json", "Parent")
    assert edge.sources == {"inherited"}
    assert edge.to_entities == {"User"}


def test_own_fk_already_parsed_is_not_duplicated_and_standard_fks_are_skipped(tmp_path):
    write_entities(tmp_path, "e.cdm.json", [{"entityName": "E", "hasAttributes": [fk_attr("bank", "Bank", "bankId")]}])
    corpus = Corpus(tmp_path)
    direct = parse_entity_document(tmp_path / "e.cdm.json", tmp_path)
    assert derived_relationships([EntityRef("e.cdm.json", "E")], corpus, direct) == []


def test_single_target_fk_with_placeholder_or_missing_name_is_inferred(tmp_path):
    no_name = fk_attr("account", "Account", "x")
    del no_name["resolutionGuidance"]["entityByReference"]["foreignKeyAttribute"]
    write_entities(tmp_path, "e.cdm.json", [{"entityName": "E", "hasAttributes": [
        fk_attr("contact", "Contact", "id"), no_name]}])
    rels = canonicalize_placeholders(parse_entity_document(tmp_path / "e.cdm.json", tmp_path), Corpus(tmp_path))
    assert {(r.from_attribute, r.fk_inferred) for r in rels} == {("contactId", True), ("accountId", True)}
    assert not any(r.from_attribute == "id" for r in rels)


def test_appointment_regarding_object_id_is_an_own_edge_not_inherited(corpus):
    # Appointment extends ActivityCommon (not Activity) and declares regardingObject itself,
    # so the edge already exists and there is nothing to inherit.
    doc = "core/applicationCommon/Appointment.cdm.json"
    direct = canonicalize_placeholders(parse_entity_document(CORPUS / doc, CORPUS), corpus)
    (edge,) = [r for r in direct if r.from_attribute == "regardingObjectId"]
    assert edge.inherited_from is None and edge.is_polymorphic
    assert edge.to_entities == {"Account", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord"}
    derived = derived_relationships([EntityRef(doc, "Appointment")], corpus, direct)
    assert not any(r.from_attribute == "regardingObjectId" for r in derived)


def test_banking_contact_inherits_polymorphic_parent_customer(corpus):
    doc = BANKING.relative_to(CORPUS).as_posix() + "/Contact.cdm.json"
    direct = canonicalize_placeholders(parse_entity_document(CORPUS / doc, CORPUS), corpus)
    edges = derived_relationships([EntityRef(doc, "Contact")], corpus, direct)
    (edge,) = [r for r in edges if r.from_attribute == "parentCustomerId"]
    assert edge.inherited_from is not None and edge.inherited_from.name == "Contact"
    assert edge.inherited_from.document != doc
    assert edge.to_entities == {"Account", "Contact"} and edge.is_polymorphic
    assert edge.fk_inferred and edge.sources == {"inherited"}


def test_banking_group_declared_and_inherited_edge_totals(corpus):
    seeds = []
    for f in sorted(BANKING.glob("*.cdm.json")):
        if is_entity_document(f):
            doc = f.relative_to(CORPUS).as_posix()
            seeds += [EntityRef(doc, e) for e in corpus.document(doc).entities]
    direct = canonicalize_placeholders(parse_entity_directory(BANKING, CORPUS), corpus)
    derived = derived_relationships(seeds, corpus, direct)
    assert len(seeds) == 24 and len(direct) == 78
    assert sum(r.sources == {"inherited"} for r in derived) == 53
    group = [r for r in derived if r.sources == {"group"}]
    # FKs pulled in from the shared customerIdAttribute group: invisible to a raw entityByReference count
    assert {(r.from_entity, r.from_attribute) for r in group} == {("FinancialProduct", "customerId"), ("KYC", "customerId")}
    assert all(r.to_entities == {"Account", "Contact"} for r in group)

