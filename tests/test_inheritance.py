import json
from collections import Counter
import pytest

from cdm_rag.inheritance import Corpus, EntityRef, Origin

BANKING = "core/applicationCommon/foundationCommon/crmCommon/accelerators/financialServices/banking/"


# --- synthetic corpus --------------------------------------------------------


def attr(name, **extra):
    return {"name": name, "dataType": "string", **extra}


def fk(name, fk_name, target="Thing"):
    return {
        "entity": {"entityReference": target},
        "name": name,
        "resolutionGuidance": {"entityByReference": {"foreignKeyAttribute": {"name": fk_name}}},
    }


def scope(*members):
    return {"attributeGroupReference": {"attributeGroupName": "attributesAddedAtThisScope", "members": list(members)}}


def write(root, path, definitions, imports=()):
    f = root / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"imports": list(imports), "definitions": definitions}))


@pytest.fixture
def corpus(tmp_path):
    """
    std.cdm.json      CdsStandard -> group audit -> group stamps (nested, with a cycle back to audit)
    a/Thing.cdm.json  Thing extends CdsStandard
    b/Thing.cdm.json  a *different* Thing
    all.cdm.json      unaliased import declaring the moniker base_Thing -> a/Thing
    b/Child.cdm.json  Child extends base_Thing/Thing
    b/Leaf.cdm.json   Leaf extends Child; inlines createdBy; redeclares 'label'
    """
    write(tmp_path, "std.cdm.json", [
        {"entityName": "CdsStandard", "extendsEntity": "CdmEntity", "hasAttributes": ["audit"]},
        {"attributeGroupName": "audit", "members": [attr("createdBy"), "stamps"]},
        {"attributeGroupName": "stamps", "members": [attr("createdOn"), "audit"]},  # cycle
    ])
    write(tmp_path, "a/Thing.cdm.json",
          [{"entityName": "Thing", "extendsEntity": "CdsStandard",
            "hasAttributes": [scope(attr("label", description="from a"))]}],
          [{"corpusPath": "/std.cdm.json"}])
    write(tmp_path, "b/Thing.cdm.json", [{"entityName": "Thing", "hasAttributes": [attr("wrong")]}])
    write(tmp_path, "all.cdm.json", [], [{"corpusPath": "/a/Thing.cdm.json", "moniker": "base_Thing"}])
    write(tmp_path, "b/Child.cdm.json",
          [{"entityName": "Child", "extendsEntity": "base_Thing/Thing", "hasAttributes": [scope(attr("childOnly"))]}],
          [{"corpusPath": "/all.cdm.json"}, {"corpusPath": "/std.cdm.json"}])
    write(tmp_path, "b/Leaf.cdm.json",
          [{"entityName": "Leaf", "extendsEntity": "Child",
            "hasAttributes": [scope(attr("label", description="from leaf"), attr("createdBy"))]}],
          [{"corpusPath": "Child.cdm.json"}, {"corpusPath": "/std.cdm.json"}])
    return Corpus(tmp_path)


def by_name(attrs):
    return {a.name: a for a in attrs}


def test_moniker_declared_in_unaliased_import_resolves_to_the_right_document(corpus):
    ref = corpus.resolve_entity("b/Child.cdm.json", "base_Thing/Thing")
    assert ref == EntityRef("a/Thing.cdm.json", "Thing")  # not b/Thing


def test_chain_follows_monikers_and_stops_at_root(corpus):
    names = [(r.document, r.name) for r in corpus.chain("b/Leaf.cdm.json", "Leaf")]
    assert names == [
        ("b/Leaf.cdm.json", "Leaf"),
        ("b/Child.cdm.json", "Child"),
        ("a/Thing.cdm.json", "Thing"),
        ("std.cdm.json", "CdsStandard"),
    ]


def test_origins_and_declaring_entity(corpus):
    attrs = by_name(corpus.resolve_attributes("b/Child.cdm.json", "Child"))
    assert attrs["childOnly"].origin is Origin.OWN
    assert attrs["label"].origin is Origin.INHERITED
    assert attrs["label"].declared_in == EntityRef("a/Thing.cdm.json", "Thing")
    assert "wrong" not in attrs  # the other Thing must not leak in


def test_attribute_groups_expand_transitively_and_cycles_terminate(corpus):
    attrs = by_name(corpus.resolve_attributes("a/Thing.cdm.json", "Thing"))
    assert {"createdBy", "createdOn"} <= attrs.keys()
    assert attrs["createdOn"].group_path == ("audit", "stamps")
    assert attrs["createdOn"].origin is Origin.STANDARD
    assert attrs["createdOn"].declared_in.name == "CdsStandard"


def test_inlined_standard_field_is_tagged_standard(corpus):
    a = by_name(corpus.resolve_attributes("b/Leaf.cdm.json", "Leaf"))["createdBy"]
    assert a.origin is Origin.STANDARD and a.inlined


def test_most_derived_declaration_wins_and_order_is_ancestors_first(corpus):
    attrs = corpus.resolve_attributes("b/Leaf.cdm.json", "Leaf")
    names = [a.name for a in attrs]
    label = by_name(attrs)["label"]
    assert (label.origin, label.description) == (Origin.OWN, "from leaf")
    assert names.count("label") == 1
    assert names.index("childOnly") < names.index("label")  # derived attributes come last


def test_placeholder_fk_takes_real_name_from_base(tmp_path):
    write(tmp_path, "base.cdm.json", [{"entityName": "Base", "hasAttributes": [fk("regarding", "regardingId")]}])
    write(tmp_path, "d.cdm.json",
          [{"entityName": "Derived", "extendsEntity": "Base", "hasAttributes": [fk("regarding", "id")]}],
          [{"corpusPath": "base.cdm.json"}])
    c = Corpus(tmp_path)
    a = by_name(c.resolve_attributes("d.cdm.json", "Derived"))["regarding"]
    assert (a.fk_name, a.origin) == ("regardingId", Origin.OWN)
    assert c.canonical_fk_name("d.cdm.json", "Derived", "regarding") == "regardingId"


def test_canonical_fk_name_is_none_without_a_declared_base(tmp_path):
    write(tmp_path, "d.cdm.json", [{"entityName": "D", "hasAttributes": [fk("object", "id")]}])
    assert Corpus(tmp_path).canonical_fk_name("d.cdm.json", "D", "object") is None


def test_unresolvable_parent_is_recorded_not_raised(tmp_path):
    write(tmp_path, "d.cdm.json", [{"entityName": "D", "extendsEntity": "Nope", "hasAttributes": [attr("x")]}])
    c = Corpus(tmp_path)
    assert [a.name for a in c.resolve_attributes("d.cdm.json", "D")] == ["x"]
    assert ("d.cdm.json", "Nope") in c.unresolved


# --- real corpus -------------------------------------------------------------


@pytest.fixture(scope="module")
def real(corpus_root):
    return Corpus(corpus_root)


def origins(attrs):
    return dict(Counter(a.origin.value for a in attrs))


def test_banking_account_walks_four_same_named_entities(real):
    chain = real.chain(BANKING + "Account.cdm.json", "Account")
    assert [r.name for r in chain] == ["Account"] * 4 + ["CdsStandard"]
    assert len({r.document for r in chain}) == 5
    assert chain[1].document.endswith("crmCommon/Account.cdm.json")  # via the base_Account moniker
    attrs = real.resolve_attributes(BANKING + "Account.cdm.json", "Account")
    assert origins(attrs) == {"own": 28, "inherited": 92, "standard": 15}


def test_banking_process_entity_has_inlined_standard_fields(real):
    attrs = real.resolve_attributes(BANKING + "BusinessCheckingAccount.cdm.json", "BusinessCheckingAccount")
    assert origins(attrs) == {"own": 11, "standard": 10}
    assert all(a.inlined for a in attrs if a.origin is Origin.STANDARD)


def test_standard_fields_arrive_through_cdsstandard(real):
    attrs = real.resolve_attributes(BANKING + "Collateral.cdm.json", "Collateral")
    assert origins(attrs) == {"own": 18, "standard": 15}
    assert not any(a.inlined for a in attrs if a.origin is Origin.STANDARD)


def test_all_banking_entities_resolve_without_gaps(real, corpus_root):
    for f in sorted((corpus_root / BANKING).glob("*.cdm.json")):
        if f.name.startswith("_") or f.name.count(".") > 2:  # skip _allImports and versioned files
            continue
        for entity in real.document(BANKING + f.name).entities:
            assert real.resolve_attributes(BANKING + f.name, entity), entity
    assert real.unresolved == set()


def test_activity_regarding_object_id_comes_from_activitycommon(real):
    doc = "core/applicationCommon/Activity.cdm.json"
    assert [r.name for r in real.chain(doc, "Activity")] == ["Activity", "ActivityCommon", "CdsStandard"]
    a = by_name(real.resolve_attributes(doc, "Activity"))["regardingObject"]
    assert a.fk_name == "regardingObjectId" and not a.fk_is_placeholder


def test_audit_attributes_agree_with_cdsstandard(real):
    from cdm_rag.relationships import AUDIT_ATTRIBUTES

    standard = real.standard_names(BANKING + "Collateral.cdm.json")
    # generic, but declared per entity rather than provided by CdsStandard
    assert AUDIT_ATTRIBUTES - standard == {"organizationId", "transactionCurrencyId"}
