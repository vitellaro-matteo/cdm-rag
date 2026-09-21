"""Extract relationships (foreign-key edges) from CDM schema documents.

Two sources encode relationships, and neither is complete on its own:

* entity documents: an entity-typed attribute carrying
  ``resolutionGuidance.entityByReference`` (the only source for banking's
  domain edges such as Branch -> Bank);
* manifests: a flat ``relationships`` array (the only source for the
  CRM-core edges inherited via attribute groups, and for polymorphic FKs).

Both are normalised into ``Relationship`` objects keyed by
(document, entity, FK attribute). Targets are a set, so a polymorphic FK
(``Activity.regardingObjectId`` -> Account, Contact, ...) is one edge.

Limitation: entity documents reference targets by bare name (resolved via
imports/monikers at load time), so ``Target.entity`` is a bare entity name for
every source; the target's document is not tracked.
"""

from __future__ import annotations

import json
import logging
import posixpath
import re
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from cdm_rag.inheritance import PLACEHOLDER_FK, EntityRef, Origin, Target, entity_targets

if TYPE_CHECKING:
    from cdm_rag.inheritance import Corpus

log = logging.getLogger(__name__)

# FK attributes present on (nearly) every entity. They dominate the manifest graph
# (~94% of edges) and carry no domain meaning, so graph expansion should skip them
# by default. Eight come from the CdsStandard entity; organizationId and
# transactionCurrencyId do NOT (they are declared per entity), but are equally generic.
AUDIT_ATTRIBUTES = frozenset(
    {
        "createdBy",
        "modifiedBy",
        "createdOnBehalfBy",
        "modifiedOnBehalfBy",
        "ownerId",
        "owningUser",
        "owningTeam",
        "owningBusinessUnit",
        "transactionCurrencyId",
        "organizationId",
    }
)

# Versioned snapshots (Account.1.5.cdm.json) duplicate the unversioned file.
_VERSIONED = re.compile(r"\.\d+\.\d+(\.\d+)?\.cdm\.json$")

ENTITY_SOURCE = "entity"
MANIFEST_SOURCE = "manifest"
INHERITED_SOURCE = "inherited"
GROUP_SOURCE = "group"  # FK pulled in from a named attribute group in another document


@dataclass(frozen=True)
class Relationship:
    from_document: str  # corpus-relative posix path
    from_entity: str
    from_attribute: str  # the FK attribute name, e.g. "financialProductId"
    targets: frozenset[Target]
    traits: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    # True when from_attribute is the attribute *name* standing in for an FK whose real
    # name (e.g. regardingObjectId) is declared on a base; see canonicalize_placeholders().
    placeholder: bool = False
    # True when the FK name was not declared anywhere but derived as "<attribute>Id",
    # CDM's default FK naming (every manifest row that can be checked agrees).
    fk_inferred: bool = False
    # Name of the entity attribute carrying the FK ("bank" for FK "bankId"). Empty when
    # unknown (manifest rows only name the FK).
    attribute_name: str = ""
    # Set on edges re-emitted for a child entity: the ancestor that declares the FK.
    inherited_from: EntityRef | None = None
    # Document that bare target names must be resolved from (where the reference is written;
    # for an FK from a shared attribute group that is the group's file). Empty = from_document.
    resolve_from: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.from_document, self.from_entity, self.from_attribute)

    @property
    def is_audit(self) -> bool:
        return self.from_attribute in AUDIT_ATTRIBUTES

    @property
    def to_entities(self) -> frozenset[str]:
        return frozenset(t.entity for t in self.targets)

    @property
    def is_polymorphic(self) -> bool:
        return len(self.to_entities) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_document": self.from_document,
            "from_entity": self.from_entity,
            "from_attribute": self.from_attribute,
            "targets": [
                {"entity": t.entity, "attribute": t.attribute}
                for t in sorted(self.targets, key=lambda t: (t.entity, t.attribute or ""))
            ],
            "traits": sorted(self.traits),
            "sources": sorted(self.sources),
            "is_audit": self.is_audit,
            "is_polymorphic": self.is_polymorphic,
            "placeholder": self.placeholder,
            "fk_inferred": self.fk_inferred,
            "attribute_name": self.attribute_name,
            "inherited_from": str(self.inherited_from) if self.inherited_from else None,
            "resolve_from": self.resolve_from,
        }


def is_entity_document(path: str | Path) -> bool:
    name = Path(path).name
    return (
        name.endswith(".cdm.json")
        and not name.endswith(".manifest.cdm.json")
        and name != "_allImports.cdm.json"
        and not _VERSIONED.search(name)
    )


# --- entity documents -------------------------------------------------------


def _fk_name(node: dict[str, Any], by_reference: dict[str, Any]) -> tuple[str | None, bool]:
    """(name, is_placeholder). A missing FK name or the placeholder "id" (polymorphic or not)
    means the real name is declared on a base attribute of the same name, or defaults to
    "<attribute>Id"; return the attribute name for canonicalize_placeholders() to resolve."""
    fk = (by_reference.get("foreignKeyAttribute") or {}).get("name")
    if fk in (None, PLACEHOLDER_FK):
        return node.get("name"), True
    return fk, False


def _walk(node: Any, document: str, entity: str, out: list[Relationship]) -> None:
    if isinstance(node, dict):
        ref = node.get("entity")
        guidance = node.get("resolutionGuidance") or {}
        if isinstance(ref, dict) and "entityReference" in ref and "entityByReference" in guidance:
            targets = entity_targets(ref)
            attribute, placeholder = _fk_name(node, guidance["entityByReference"])
            if not targets or not attribute:
                log.warning("%s: %s has an unresolvable entityReference; skipped", document, entity)
            else:
                out.append(
                    Relationship(
                        from_document=document,
                        from_entity=entity,
                        from_attribute=attribute,
                        targets=targets,
                        sources=frozenset({ENTITY_SOURCE}),
                        placeholder=placeholder,
                        attribute_name=node.get("name") or "",
                        resolve_from=document,
                    )
                )
        for value in node.values():
            _walk(value, document, entity, out)
    elif isinstance(node, list):
        for value in node:
            _walk(value, document, entity, out)


def parse_entity_document(path: str | Path, corpus_root: str | Path) -> list[Relationship]:
    """Edges declared by entity-typed FK attributes in one entity document."""
    path, corpus_root = Path(path), Path(corpus_root)
    document = path.resolve().relative_to(corpus_root.resolve()).as_posix()
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[Relationship] = []
    for definition in data.get("definitions") or []:
        if isinstance(definition, dict) and "entityName" in definition:
            _walk(definition.get("hasAttributes") or [], document, definition["entityName"], out)
    return out


def parse_entity_directory(directory: str | Path, corpus_root: str | Path) -> list[Relationship]:
    """Parse every unversioned entity document under ``directory`` (recursive)."""
    files = sorted(p for p in Path(directory).rglob("*.cdm.json") if is_entity_document(p))
    return [rel for f in files for rel in parse_entity_document(f, corpus_root)]


# --- manifests --------------------------------------------------------------


def _split_reference(ref: str, manifest_dir: str) -> tuple[str, str]:
    """"foo/Account.cdm.json/Account" -> (corpus-relative document, "Account")."""
    document, _, entity = ref.rpartition("/")
    if document.startswith("/"):
        document = document.lstrip("/")
    else:
        document = posixpath.normpath(posixpath.join(manifest_dir, document))
    return document, entity


def parse_manifest(path: str | Path, corpus_root: str | Path) -> list[Relationship]:
    """One single-target edge per row of the manifest's ``relationships`` array."""
    path, corpus_root = Path(path), Path(corpus_root)
    manifest_dir = posixpath.dirname(path.resolve().relative_to(corpus_root.resolve()).as_posix())
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[Relationship] = []
    for row in data.get("relationships") or []:
        from_doc, from_entity = _split_reference(row["fromEntity"], manifest_dir)
        _, to_entity = _split_reference(row["toEntity"], manifest_dir)
        out.append(
            Relationship(
                from_document=from_doc,
                from_entity=from_entity,
                from_attribute=row["fromEntityAttribute"],
                targets=frozenset({Target(to_entity, row.get("toEntityAttribute"))}),
                traits=frozenset(row.get("exhibitsTraits") or []),
                sources=frozenset({MANIFEST_SOURCE}),
            )
        )
    return out


# --- merging ----------------------------------------------------------------


def merge(*groups: Iterable[Relationship]) -> list[Relationship]:
    """Collapse rows sharing a key into one edge: union of targets, traits, sources."""
    merged: dict[tuple[str, str, str], Relationship] = {}
    for rel in chain.from_iterable(groups):
        prev = merged.get(rel.key)
        merged[rel.key] = (
            rel
            if prev is None
            else replace(
                prev,
                targets=prev.targets | rel.targets,
                traits=prev.traits | rel.traits,
                sources=prev.sources | rel.sources,
                fk_inferred=prev.fk_inferred and rel.fk_inferred,
                attribute_name=prev.attribute_name or rel.attribute_name,
                inherited_from=prev.inherited_from or rel.inherited_from,
                resolve_from=prev.resolve_from or rel.resolve_from,
            )
        )
    return list(merged.values())


def canonicalize_placeholders(rels: Iterable[Relationship], corpus: "Corpus") -> list[Relationship]:
    """Rename placeholder edges to their real FK name so they merge with manifest rows.

    ``Activity.regardingObject`` (polymorphic, FK declared as "id") becomes
    ``regardingObjectId``, taken from the same-named attribute the entity inherits
    from ``ActivityCommon``. Where no base declares the name (``Note.object``), fall
    back to "<attribute>Id" and set ``fk_inferred``.
    """
    out = []
    for rel in rels:
        if rel.placeholder:
            declared = corpus.canonical_fk_name(rel.from_document, rel.from_entity, rel.from_attribute)
            rel = replace(
                rel,
                from_attribute=declared or rel.from_attribute + "Id",
                placeholder=False,
                fk_inferred=declared is None,
            )
        out.append(rel)
    return out


def derived_relationships(
    entities: Iterable[EntityRef], corpus: "Corpus", direct: Iterable[Relationship] = ()
) -> list[Relationship]:
    """Edges that ``parse_entity_document`` cannot see, built from resolved attributes.

    * FKs inherited from an ancestor (``inherited_from`` names it): banking ``Contact``
      gets ``parentCustomer`` from the CRM ``Contact``.
    * FKs an entity pulls in from a named attribute group defined in another document
      (``FinancialProduct.customer`` from ``customerIdAttribute``); ``sources`` is
      ``{"group"}``. ``direct`` (the parsed edges) is used to skip FKs already covered.

    Limitations: CdsStandard's own FKs (createdBy, ownerId, ...) are skipped; they live in
    groups and are audit edges anyway. If a polymorphic FK's real name is not declared
    anywhere it is inferred as "<attribute>Id" (``fk_inferred``), as in
    ``canonicalize_placeholders``.
    """
    covered = {(r.from_document, r.from_entity, r.attribute_name) for r in direct}
    out: list[Relationship] = []
    for entity in entities:
        for attr in corpus.resolve_attributes(entity.document, entity.name):
            if not attr.is_fk or not attr.fk_targets or attr.origin is Origin.STANDARD:
                continue
            inherited = attr.origin is Origin.INHERITED
            if not inherited and (entity.document, entity.name, attr.name) in covered:
                continue
            placeholder = attr.fk_is_placeholder
            out.append(
                Relationship(
                    from_document=entity.document,
                    from_entity=entity.name,
                    from_attribute=attr.name + "Id" if placeholder else attr.fk_name,
                    targets=frozenset(attr.fk_targets),
                    sources=frozenset({INHERITED_SOURCE if inherited else GROUP_SOURCE}),
                    fk_inferred=placeholder,
                    attribute_name=attr.name,
                    inherited_from=attr.declared_in if inherited else None,
                    resolve_from=attr.document,
                )
            )
    return out
