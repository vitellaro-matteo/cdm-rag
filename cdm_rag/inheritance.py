"""Resolve an entity's full attribute list across ``extendsEntity`` and attribute groups.

Every attribute is tagged with where it came from:

* ``OWN``       declared by the entity itself;
* ``INHERITED`` declared by an ancestor (``declared_in`` names it);
* ``STANDARD``  a CdsStandard audit/bookkeeping field (createdOn, ownerId, ...).
  These arrive either through the ``CdsStandard`` entity in the ``extendsEntity``
  chain or, in some banking process entities, inlined verbatim (``inlined=True``).

Name resolution follows CDM's rules closely enough for this corpus:

* ``extendsEntity`` / group names are looked up in the referencing document, then
  in its imports (transitively, in declaration order). Names are NOT unique across
  the corpus (there are several ``Account`` entities), so a global index would be wrong.
* ``moniker/Name`` (e.g. ``base_Account/Account``) is looked up only in the
  document imported under that moniker, plus that document's own imports. The
  moniker is usually declared in an unaliased import (``_allImports.cdm.json``),
  so it is searched for transitively.
* ``CdmEntity`` is the built-in root and ends the chain.

Known simplifications: import-order priority stands in for CDM's full priority
rules; attributes are de-duplicated by name (most derived wins); traits and
resolution-guidance directives are not applied.
"""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, NamedTuple

log = logging.getLogger(__name__)

ROOT_ENTITY = "CdmEntity"
STANDARD_ENTITY = "CdsStandard"
PLACEHOLDER_FK = "id"  # FK name used by polymorphic attributes; the real one comes from a base


class Target(NamedTuple):
    entity: str  # bare entity name as written in the document (not resolved through imports)
    attribute: str | None


def _identified_by(entity_ref: dict[str, Any]) -> str | None:
    """Target attribute from ``is.identifiedBy`` ("Bank/(resolvedAttributes)/bankId")."""
    for trait in entity_ref.get("appliedTraits") or []:
        if isinstance(trait, dict) and trait.get("traitReference") == "is.identifiedBy":
            args = trait.get("arguments") or []
            if args and isinstance(args[0], str):
                return args[0].rsplit("/", 1)[-1]
    return None


def entity_targets(ref: dict[str, Any]) -> frozenset[Target]:
    """Targets of an ``entity`` node: one for a bare name, several for an inline entity
    whose attributes are the polymorphic alternatives (empty if there are none)."""
    target = ref.get("entityReference")
    if isinstance(target, str):
        return frozenset({Target(target, _identified_by(ref))})
    found: set[Target] = set()
    if isinstance(target, dict):
        for attr in target.get("hasAttributes") or []:
            inner = attr.get("entity") if isinstance(attr, dict) else None
            if isinstance(inner, dict):
                found |= entity_targets(inner)
    return frozenset(found)


class Origin(str, Enum):
    OWN = "own"
    INHERITED = "inherited"
    STANDARD = "standard"


class EntityRef(NamedTuple):
    document: str  # corpus-relative posix path
    name: str

    def __str__(self) -> str:
        return f"{self.document}/{self.name}"


@dataclass(frozen=True)
class ResolvedAttribute:
    name: str
    origin: Origin
    declared_in: EntityRef
    group_path: tuple[str, ...] = ()
    data_type: str | None = None
    display_name: str | None = None
    description: str | None = None
    fk_name: str | None = None  # set for entity-typed (foreign key) attributes
    fk_targets: tuple[Target, ...] = ()  # several when the FK is polymorphic
    document: str = ""  # where the attribute node is written (an entity file or a shared group's file)
    inlined: bool = False  # STANDARD attribute copied into the entity rather than inherited

    @property
    def is_fk(self) -> bool:
        return self.fk_name is not None

    @property
    def fk_is_placeholder(self) -> bool:
        return self.fk_name == PLACEHOLDER_FK

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin.value,
            "declared_in": str(self.declared_in),
            "group_path": list(self.group_path),
            "data_type": self.data_type,
            "display_name": self.display_name,
            "description": self.description,
            "fk_name": self.fk_name,
            "fk_targets": [t.entity for t in self.fk_targets],
            "inlined": self.inlined,
        }


@dataclass
class _Document:
    path: str
    imports: list[tuple[str, str | None]]  # (corpus-relative path, moniker)
    entities: dict[str, dict[str, Any]]
    groups: dict[str, dict[str, Any]]


class Corpus:
    """Lazily loads ``.cdm.json`` documents under ``root`` and resolves inheritance."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._docs: dict[str, _Document | None] = {}
        self._attr_cache: dict[EntityRef, tuple[ResolvedAttribute, ...]] = {}
        self.unresolved: set[tuple[str, str]] = set()  # (document, name) that could not be resolved

    # --- documents ----------------------------------------------------------

    @staticmethod
    def _join(base_document: str, corpus_path: str) -> str:
        if corpus_path.startswith("/"):
            return posixpath.normpath(corpus_path.lstrip("/"))
        return posixpath.normpath(posixpath.join(posixpath.dirname(base_document), corpus_path))

    def document(self, path: str) -> _Document | None:
        if path not in self._docs:
            self._docs[path] = self._load(path)
        return self._docs[path]

    def _load(self, path: str) -> _Document | None:
        file = self.root / path
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("cannot load %s: %s", path, exc)
            return None
        imports = [
            (self._join(path, i["corpusPath"]), i.get("moniker"))
            for i in data.get("imports") or []
            if isinstance(i, dict) and "corpusPath" in i
        ]
        entities: dict[str, dict[str, Any]] = {}
        groups: dict[str, dict[str, Any]] = {}
        for d in data.get("definitions") or []:
            if not isinstance(d, dict):
                continue
            if "entityName" in d:
                entities[d["entityName"]] = d
            elif "attributeGroupName" in d:
                groups[d["attributeGroupName"]] = d
        return _Document(path, imports, entities, groups)

    # --- name resolution ----------------------------------------------------

    def _find(self, document: str, name: str, kind: str, seen: set[str]) -> tuple[str, dict[str, Any]] | None:
        if document in seen:
            return None
        seen.add(document)
        doc = self.document(document)
        if doc is None:
            return None
        found = getattr(doc, kind).get(name)
        if found is not None:
            return document, found
        for path, moniker in doc.imports:
            if moniker is None and (hit := self._find(path, name, kind, seen)):
                return hit
        return None

    def _moniker_target(self, document: str, moniker: str, seen: set[str]) -> str | None:
        """Document imported under ``moniker``, declared in ``document`` or (usually) in an
        unaliased import such as _allImports.cdm.json."""
        if document in seen:
            return None
        seen.add(document)
        doc = self.document(document)
        if doc is None:
            return None
        for path, m in doc.imports:
            if m == moniker:
                return path
        for path, m in doc.imports:
            if m is None and (hit := self._moniker_target(path, moniker, seen)):
                return hit
        return None

    def _lookup(self, document: str, name: str, kind: str) -> tuple[str, dict[str, Any]] | None:
        if "/" in name:
            moniker, _, rest = name.partition("/")
            target = self._moniker_target(document, moniker, set())
            return self._find(target, rest, kind, set()) if target else None
        return self._find(document, name, kind, set())

    def resolve_entity(self, document: str, name: str) -> EntityRef | None:
        hit = self._lookup(document, name, "entities")
        return EntityRef(hit[0], hit[1]["entityName"]) if hit else None

    def resolve_group(self, document: str, name: str) -> tuple[str, dict[str, Any]] | None:
        return self._lookup(document, name, "groups")

    # --- inheritance --------------------------------------------------------

    def parent(self, ref: EntityRef) -> EntityRef | None:
        doc = self.document(ref.document)
        definition = doc.entities.get(ref.name) if doc else None
        extends = definition.get("extendsEntity") if definition else None
        if isinstance(extends, dict):
            extends = extends.get("entityReference")
        if not isinstance(extends, str) or extends == ROOT_ENTITY:
            return None
        parent = self.resolve_entity(ref.document, extends)
        if parent is None:
            self.unresolved.add((ref.document, extends))
        return parent

    def chain(self, document: str, entity: str) -> list[EntityRef]:
        """[entity, parent, grandparent, ...] up to (not including) CdmEntity."""
        out: list[EntityRef] = []
        seen: set[EntityRef] = set()
        cur: EntityRef | None = EntityRef(document, entity)
        while cur is not None and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = self.parent(cur)
        return out

    # --- attributes ---------------------------------------------------------

    def _expand(
        self,
        items: Iterable[Any],
        document: str,
        declared_in: EntityRef,
        origin: Origin,
        path: tuple[str, ...],
        stack: frozenset[tuple[str, str]],
    ) -> Iterable[ResolvedAttribute]:
        """Flatten attribute-group structure into attributes, expanding named groups transitively."""
        for item in items:
            if isinstance(item, str):
                yield from self._expand_named(item, document, declared_in, origin, path, stack)
            elif isinstance(item, dict) and "attributeGroupReference" in item:
                ref = item["attributeGroupReference"]
                if isinstance(ref, str):
                    yield from self._expand_named(ref, document, declared_in, origin, path, stack)
                elif isinstance(ref, dict) and "members" in ref:
                    yield from self._expand(
                        ref["members"], document, declared_in, origin, path + (ref.get("attributeGroupName", "?"),), stack
                    )
                elif isinstance(ref, dict) and "attributeGroupName" in ref:
                    yield from self._expand_named(ref["attributeGroupName"], document, declared_in, origin, path, stack)
            elif isinstance(item, dict) and "name" in item:
                yield self._attribute(item, document, declared_in, origin, path)

    def _expand_named(self, name, document, declared_in, origin, path, stack):
        hit = self.resolve_group(document, name)
        if hit is None:
            self.unresolved.add((document, name))
            return
        group_doc, group = hit
        key = (group_doc, name)
        if key in stack:
            return
        yield from self._expand(group.get("members") or [], group_doc, declared_in, origin, path + (name,), stack | {key})

    @staticmethod
    def _attribute(
        node: dict[str, Any], document: str, declared_in: EntityRef, origin: Origin, path: tuple[str, ...]
    ) -> ResolvedAttribute:
        guidance = (node.get("resolutionGuidance") or {}).get("entityByReference")
        fk_name = None
        if isinstance(guidance, dict) and isinstance(node.get("entity"), dict):
            # No declared FK name, or the placeholder "id": the real name comes from a base
            # declaration of the same attribute, else CDM's default "<attribute>Id".
            fk_name = (guidance.get("foreignKeyAttribute") or {}).get("name") or PLACEHOLDER_FK
        data_type = node.get("dataType")
        if isinstance(data_type, dict):
            data_type = data_type.get("dataTypeReference")
        if data_type is None and isinstance(node.get("entity"), dict):
            data_type = "entity"
        return ResolvedAttribute(
            name=node["name"],
            origin=origin,
            declared_in=declared_in,
            group_path=path,
            data_type=data_type if isinstance(data_type, str) else None,
            display_name=node.get("displayName"),
            description=node.get("description"),
            fk_name=fk_name,
            document=document,
            fk_targets=tuple(sorted(entity_targets(node["entity"]))) if fk_name and isinstance(node.get("entity"), dict) else (),
        )

    def definition(self, ref: EntityRef) -> dict[str, Any] | None:
        """The raw entity definition (for description and other document-level fields)."""
        doc = self.document(ref.document)
        return doc.entities.get(ref.name) if doc else None

    def standard_names(self, document: str) -> frozenset[str]:
        """Names (and FK names) of the fields CdsStandard provides, as visible from ``document``."""
        ref = self.resolve_entity(document, STANDARD_ENTITY)
        if ref is None:
            return frozenset()
        names: set[str] = set()
        for a in self._own_attributes(ref, Origin.STANDARD):
            names.add(a.name)
            if a.fk_name:
                names.add(a.fk_name)
        return frozenset(names)

    def _own_attributes(self, ref: EntityRef, origin: Origin) -> list[ResolvedAttribute]:
        doc = self.document(ref.document)
        definition = doc.entities.get(ref.name) if doc else None
        if definition is None:
            return []
        return list(self._expand(definition.get("hasAttributes") or [], ref.document, ref, origin, (), frozenset()))

    def resolve_attributes(self, document: str, entity: str) -> tuple[ResolvedAttribute, ...]:
        """All attributes of ``entity``, ancestors first, de-duplicated by name (most derived wins)."""
        key = EntityRef(document, entity)
        if key in self._attr_cache:
            return self._attr_cache[key]

        chain = self.chain(document, entity)
        standard = self.standard_names(document)
        by_name: dict[str, ResolvedAttribute] = {}
        for depth, ref in reversed(list(enumerate(chain))):
            if ref.name == STANDARD_ENTITY:
                origin = Origin.STANDARD
            else:
                origin = Origin.OWN if depth == 0 else Origin.INHERITED
            for attr in self._own_attributes(ref, origin):
                if origin is not Origin.STANDARD and (attr.name in standard or attr.fk_name in standard):
                    attr = replace(attr, origin=Origin.STANDARD, inlined=True)
                prev = by_name.pop(attr.name, None)  # pop + reinsert keeps derived attributes last
                if prev is not None and attr.fk_is_placeholder and prev.fk_name and not prev.fk_is_placeholder:
                    # e.g. Activity.regardingObject: the polymorphic declaration carries the
                    # placeholder FK name "id"; the real one (regardingObjectId) comes from the base.
                    attr = replace(attr, fk_name=prev.fk_name, description=attr.description or prev.description)
                by_name[attr.name] = attr

        result = tuple(by_name.values())
        self._attr_cache[key] = result
        return result

    def canonical_fk_name(self, document: str, entity: str, attribute: str) -> str | None:
        """Real FK name for ``attribute`` on ``entity`` (None if unknown or still a placeholder)."""
        for a in self.resolve_attributes(document, entity):
            if a.name == attribute and a.fk_name and not a.fk_is_placeholder:
                return a.fk_name
        return None
