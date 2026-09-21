"""Entity nodes and the relationship graph (the source of truth; chunks are only search handles).

Every (document, entity) pair is its own node. Same-named entities are never merged:
banking ``Account`` -> CRM-base ``Account`` -> Foundation ``Account`` -> Core ``Account``
are four nodes linked by ``parent_id``.

Scope of ``build_graph``: the seed entities (default: the unversioned banking files),
their ancestors, and the entities their FK edges point to (one hop) plus those
entities' ancestors.

Known limitations
* Only seed entities get outgoing edges. Ancestors and one-hop targets are nodes with
  attributes and (incoming) edges, but their own FKs are not expanded; doing so would
  pull in most of the CRM core.
* Bare target names are resolved with a rule that DEVIATES from strict CDM import order.
  Literal CDM resolution walks the imports of the document where the reference is written
  and takes the first same-named entity; for banking documents that lands all 17 "Account"
  and 16 "Contact" references on the *Core* entities (while "Opportunity" and "Lead" land on
  the banking ones), leaving banking Account/Contact with no incoming edges. Instead, when a
  bare name matches exactly one seed entity, the edge points at that seed entity
  (``resolved_by="seed_layer"``); otherwise import order decides (``"import_order"``). The
  rule applies to every edge because only seed entities have outgoing edges, including edges
  inherited from an ancestor that wrote the reference in its own document. Names written
  with a moniker ("base_Account/...") are never overridden. ``EdgeTarget.import_order_id``
  always holds what strict import order would have picked, so the deviation stays visible.
  Whether CDM intends the seed-layer specialisation is unverified; it is a modelling choice.
* The applicationCommon manifest is not merged in; entity files already carry every edge.
"""

from __future__ import annotations

import logging
import posixpath
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

from cdm_rag.inheritance import Corpus, EntityRef, ResolvedAttribute
from cdm_rag.relationships import (
    Relationship,
    canonicalize_placeholders,
    derived_relationships,
    is_entity_document,
    merge,
    parse_entity_document,
)

log = logging.getLogger(__name__)

BANKING_DIR = "core/applicationCommon/foundationCommon/crmCommon/accelerators/financialServices/banking/"

# First match wins. Labels appear in display names: "Account (CRM base)".
_LAYERS = (
    (BANKING_DIR, "banking"),
    ("core/applicationCommon/foundationCommon/crmCommon/accelerators/", "CRM accelerator"),
    ("core/applicationCommon/foundationCommon/crmCommon/", "CRM base"),
    ("core/applicationCommon/foundationCommon/", "Foundation"),
    ("core/applicationCommon/", "Core"),
    ("core/wellKnownCDSAttributeGroups", "CDS standard"),
)


def layer_of(document: str) -> str:
    for prefix, label in _LAYERS:
        if document.startswith(prefix):
            return label
    return posixpath.dirname(document) or "root"


def entity_id(ref: EntityRef) -> str:
    """Stable, unique, human-readable: "<corpus-relative document>#<entity name>"."""
    return f"{ref.document}#{ref.name}"


@dataclass(frozen=True)
class EntityNode:
    entity_id: str
    ref: EntityRef
    name: str
    layer: str
    display_name: str  # "Account (banking)"; unique within a graph
    parent_id: str | None
    description: str | None = None

    @property
    def document(self) -> str:
        return self.ref.document


RESOLVED_BY_SEED_LAYER = "seed_layer"
RESOLVED_BY_IMPORT_ORDER = "import_order"


class EdgeTarget(NamedTuple):
    entity_id: str | None  # None when the name could not be resolved to an entity
    name: str
    attribute: str | None
    resolved_by: str = RESOLVED_BY_IMPORT_ORDER  # "seed_layer" | "import_order"
    import_order_id: str | None = None  # what strict CDM import order picks (== entity_id unless deviated)


@dataclass(frozen=True)
class Edge:
    edge_id: str
    from_id: str
    attribute: str  # entity attribute carrying the FK ("bank")
    fk_name: str  # "bankId"
    targets: tuple[EdgeTarget, ...]
    is_audit: bool
    is_polymorphic: bool
    fk_inferred: bool
    inherited_from: str | None  # entity_id of the ancestor declaring the FK
    traits: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @property
    def to_ids(self) -> tuple[str, ...]:
        return tuple(t.entity_id for t in self.targets if t.entity_id)


@dataclass
class Graph:
    nodes: dict[str, EntityNode] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    # Full resolved attribute lists, keyed by entity_id. Chunk text only samples these;
    # exact "list all attributes of X" answers must come from here.
    attributes: dict[str, tuple[ResolvedAttribute, ...]] = field(default_factory=dict)
    outgoing_index: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    incoming_index: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unresolved_targets: list[tuple[str, str]] = field(default_factory=list)  # (edge_id, name)

    def add_edge(self, edge: Edge) -> None:
        self.edges[edge.edge_id] = edge
        self.outgoing_index[edge.from_id].append(edge.edge_id)
        for to_id in dict.fromkeys(edge.to_ids):
            self.incoming_index[to_id].append(edge.edge_id)

    def outgoing(self, entity_id: str, include_audit: bool = False) -> list[Edge]:
        return [e for e in map(self.edges.get, self.outgoing_index.get(entity_id, [])) if include_audit or not e.is_audit]

    def incoming(self, entity_id: str, include_audit: bool = False) -> list[Edge]:
        return [e for e in map(self.edges.get, self.incoming_index.get(entity_id, [])) if include_audit or not e.is_audit]

    def chain(self, entity_id: str) -> list[EntityNode]:
        """[node, parent, grandparent, ...]"""
        out, seen = [], set()
        cur = self.nodes.get(entity_id)
        while cur is not None and cur.entity_id not in seen:
            out.append(cur)
            seen.add(cur.entity_id)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return out

    def find(self, name: str, layer: str | None = None) -> list[EntityNode]:
        return [n for n in self.nodes.values() if n.name == name and (layer is None or n.layer == layer)]


def entities_in(corpus: Corpus, directory: str) -> list[EntityRef]:
    """Every entity defined in the unversioned entity documents directly inside ``directory``."""
    refs = []
    for f in sorted((corpus.root / directory).glob("*.cdm.json")):
        if is_entity_document(f):
            doc = f"{directory.rstrip('/')}/{f.name}"
            refs += [EntityRef(doc, name) for name in corpus.document(doc).entities]  # type: ignore[union-attr]
    return refs


def banking_seeds(corpus: Corpus) -> list[EntityRef]:
    return entities_in(corpus, BANKING_DIR)


def _display_names(refs: Iterable[EntityRef]) -> dict[EntityRef, str]:
    refs = list(refs)
    base = {r: f"{r.name} ({layer_of(r.document)})" for r in refs}
    counts = defaultdict(int)
    for name in base.values():
        counts[name] += 1
    out = {}
    for r, name in base.items():
        if counts[name] > 1:  # same name in the same layer: add the folder to keep names unique
            folder = posixpath.basename(posixpath.dirname(r.document))
            name = f"{r.name} ({layer_of(r.document)}: {folder})"
        out[r] = name
    return out


def _resolve_targets(
    corpus: Corpus, seeds_by_name: dict[str, list[EntityRef]], rel: Relationship
) -> list[tuple[str, str | None, EntityRef | None, str, EntityRef | None]]:
    """(name, attribute, chosen ref, resolved_by, strict import-order ref) for each target of ``rel``."""
    base = rel.resolve_from or rel.from_document
    row = []
    for t in sorted(rel.targets):
        literal = corpus.resolve_entity(base, t.entity)
        ref, method = literal, RESOLVED_BY_IMPORT_ORDER
        if len(seeds_by_name.get(t.entity, ())) == 1:  # a monikered name ("lib/Party") never matches a seed name
            ref, method = seeds_by_name[t.entity][0], RESOLVED_BY_SEED_LAYER
        row.append((t.entity, t.attribute, ref, method, literal))
    return row


def build_graph(corpus: Corpus, seeds: Iterable[EntityRef]) -> Graph:
    seeds = list(dict.fromkeys(seeds))
    seed_set = set(seeds)

    # 1. edges for the seed entities: parsed + inherited/group-declared
    direct: list[Relationship] = []
    for document in sorted({s.document for s in seeds}):
        direct += [
            r
            for r in parse_entity_document(corpus.root / document, corpus.root)
            if EntityRef(document, r.from_entity) in seed_set
        ]
    direct = canonicalize_placeholders(direct, corpus)
    relationships = merge(direct, derived_relationships(seeds, corpus, direct))

    # 2. resolve bare target names: the unique seed entity of that name if there is one,
    #    else through imports of the document that wrote the reference (see module docstring)
    seeds_by_name: dict[str, list[EntityRef]] = defaultdict(list)
    for s in seeds:
        seeds_by_name[s.name].append(s)
    resolved = {rel.key: _resolve_targets(corpus, seeds_by_name, rel) for rel in relationships}
    refs: set[EntityRef] = set(seeds)
    for row in resolved.values():
        for _, _, ref, _, literal in row:
            refs.update(r for r in (ref, literal) if r)

    # 3. close over ancestors, so every node's parent is a node
    for ref in list(refs):
        refs.update(corpus.chain(ref.document, ref.name))

    display = _display_names(refs)
    graph = Graph()
    for ref in sorted(refs):
        parent = corpus.parent(ref)
        definition = corpus.definition(ref) or {}
        graph.nodes[entity_id(ref)] = EntityNode(
            entity_id=entity_id(ref),
            ref=ref,
            name=ref.name,
            layer=layer_of(ref.document),
            display_name=display[ref],
            parent_id=entity_id(parent) if parent else None,
            description=definition.get("description"),
        )
        graph.attributes[entity_id(ref)] = corpus.resolve_attributes(ref.document, ref.name)

    # 4. edges
    for rel in sorted(relationships, key=lambda r: r.key):
        from_id = entity_id(EntityRef(rel.from_document, rel.from_entity))
        edge = Edge(
            edge_id=f"{from_id}::{rel.from_attribute}",
            from_id=from_id,
            attribute=rel.attribute_name or rel.from_attribute,
            fk_name=rel.from_attribute,
            targets=tuple(
                EdgeTarget(
                    entity_id(ref) if ref else None,
                    name,
                    attribute,
                    method,
                    entity_id(literal) if literal else None,
                )
                for name, attribute, ref, method, literal in resolved[rel.key]
            ),
            is_audit=rel.is_audit,
            is_polymorphic=rel.is_polymorphic,
            fk_inferred=rel.fk_inferred,
            inherited_from=entity_id(rel.inherited_from) if rel.inherited_from else None,
            traits=tuple(sorted(rel.traits)),
            sources=tuple(sorted(rel.sources)),
        )
        graph.add_edge(edge)
        graph.unresolved_targets += [(edge.edge_id, t.name) for t in edge.targets if t.entity_id is None]
    return graph
