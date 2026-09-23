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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

from cdm_rag.inheritance import Corpus, EntityRef, Origin, ResolvedAttribute
from cdm_rag.relationships import (
    AUDIT_ATTRIBUTES,
    Relationship,
    canonicalize_placeholders,
    derived_relationships,
    is_entity_document,
    merge,
    parse_entity_document,
)

log = logging.getLogger(__name__)

BANKING_DIR = "core/applicationCommon/foundationCommon/crmCommon/accelerators/financialServices/banking/"

# Cap on relations_between()'s near-misses per side (see Graph._ranked_near_misses). An entity
# can have a dozen-plus unrelated edges; passing all of them as "near misses" to an LLM buries
# the couple of actually relevant ones in noise instead of surfacing them.
NEAR_MISS_LIMIT = 5

# Default hop cap for Graph.find_path -- small enough that a question with no real answer
# doesn't search the entire graph, generous enough for every real path this corpus has been
# found to need (the longest verified so far, Collateral -> Bank, is 3 hops).
DEFAULT_MAX_HOPS = 4

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


class PathHop(NamedTuple):
    """One edge of a ``Graph.find_path`` result, oriented in the direction the path actually
    travels (``from_id`` -> ``to_id``) -- which may be against the edge's own FK direction, since
    ``find_path`` treats the graph as undirected (see its docstring). ``edge`` is the underlying
    ``Edge``; render a hop's label from ``edge.attribute``, not from ``edge.from_id``/``to_ids``,
    since those describe the edge's storage direction, not this hop's traversal direction."""

    edge: "Edge"
    from_id: str
    to_id: str


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

    def relations_between(self, name_a: str, name_b: str) -> "EntityPairRelations":
        """Direct edges between every node named ``name_a`` and every node named ``name_b``
        (both directions, audit edges included, since "is there any link at all" is the
        question). When there are none, also return near-misses: ``name_a``'s own non-audit
        outgoing edges and ``name_b``'s own non-audit incoming edges, ranked against the *other*
        name and capped to ``NEAR_MISS_LIMIT`` (see ``_ranked_near_misses``) — structured data a
        caller (e.g. an LLM prompt) can turn into "no direct relationship, but X is related via
        Y", without burying the couple of real candidates in every other edge the entity has."""
        a_ids = tuple(sorted(n.entity_id for n in self.find(name_a)))
        b_ids = tuple(sorted(n.entity_id for n in self.find(name_b)))
        a_set, b_set = set(a_ids), set(b_ids)
        edges = tuple(
            e
            for e in self.edges.values()
            if (e.from_id in a_set and b_set.intersection(e.to_ids)) or (e.from_id in b_set and a_set.intersection(e.to_ids))
        )
        a_near = b_near = ()
        if not edges:
            a_all = tuple(e for i in a_ids for e in self.outgoing(i))
            b_all = tuple(e for i in b_ids for e in self.incoming(i))
            a_near = self._ranked_near_misses(a_all, name_b)
            b_near = self._ranked_near_misses(b_all, name_a)
        return EntityPairRelations(
            a_name=name_a,
            b_name=name_b,
            a_ids=a_ids,
            b_ids=b_ids,
            edges=edges,
            has_edges=bool(edges),
            a_outgoing_near_misses=a_near,
            b_incoming_near_misses=b_near,
            a_note=self._infrastructure_note(name_a, a_ids),
            b_note=self._infrastructure_note(name_b, b_ids),
        )

    def _polymorphic_target_names(self) -> set[str]:
        """Entity names used as a target of any polymorphic edge, anywhere in this graph. This
        schema's polymorphic edges are exactly the places it says "this could be any of several
        general-purpose party/counterparty types" (here: every one targets {Account, Contact});
        it's the closest structural analog this graph has to a generic "party" or "organization"
        concept, and the signal that catches a near-miss a pure text match would not."""
        names: set[str] = set()
        for e in self.edges.values():
            if e.is_polymorphic:
                names.update(t.name for t in e.targets)
        return names

    def _near_miss_score(self, edge: Edge, missing_name: str) -> tuple[bool, bool]:
        """(lexical, structural) -- both booleans, sorted True-first. Lexical: does
        ``missing_name`` appear in the edge's own attribute name or a target entity name (e.g.
        asking about "Bank" would favor an attribute or target literally containing "bank").
        Structural: is a target entity itself one of this graph's polymorphic "party" types (see
        ``_polymorphic_target_names``) -- catches a near-miss with no textual overlap at all,
        such as Contact.employer -> Account having none with "Organization"."""
        missing_lower = missing_name.lower()
        target_names = {t.name for t in edge.targets}
        lexical = missing_lower in edge.attribute.lower() or any(missing_lower in t.lower() for t in target_names)
        structural = bool(target_names & self._polymorphic_target_names())
        return (lexical, structural)

    def _ranked_near_misses(self, edges: tuple[Edge, ...], missing_name: str) -> tuple[Edge, ...]:
        """``edges`` (an entity's own non-audit edges), scored against ``missing_name``, with
        anything that scores no signal at all (neither lexical nor structural, see
        ``_near_miss_score``) dropped outright -- not padded back in to hit a quota -- and the
        remainder capped to the top ``NEAR_MISS_LIMIT``. An entity can have a dozen-plus edges
        with nothing to do with the entity actually asked about; passing all of them as "near
        misses" buries the couple that are in noise rather than surfacing them. Ties among
        equally-scored edges keep their original (insertion) order, since Python's sort is stable."""
        scored = sorted(edges, key=lambda e: self._near_miss_score(e, missing_name), reverse=True)
        relevant = [e for e in scored if any(self._near_miss_score(e, missing_name))]
        return tuple(relevant[:NEAR_MISS_LIMIT])

    def _path_neighbors(self, node_id: str) -> list[tuple[Edge, str]]:
        """(edge, neighbor_id) for every non-audit edge touching ``node_id``, from either side --
        outgoing (this node is ``from_id``, possibly several targets for a polymorphic edge) and
        incoming (this node is one of ``to_ids``, the neighbor is ``from_id``). ``find_path``
        treats the graph as undirected: an edge's own FK direction is a storage detail (which
        side happened to declare the foreign key), not a statement about which direction a path
        is allowed to use it -- e.g. Branch declares the FK to Bank, but a path from Bank back to
        Branch is exactly as real a connection."""
        out = [(e, to_id) for e in self.outgoing(node_id) for to_id in dict.fromkeys(e.to_ids)]
        out += [(e, e.from_id) for e in self.incoming(node_id)]
        return out

    def find_path(self, name_a: str, name_b: str, max_hops: int = DEFAULT_MAX_HOPS) -> tuple["PathHop", ...] | None:
        """Shortest chain of non-audit edges connecting any node named ``name_a`` to any node
        named ``name_b`` (same audit exclusion as ``relations_between``'s default), or ``None``
        if no such chain exists within ``max_hops`` edges. Breadth-first search, not Dijkstra or
        A*: every edge here is equally meaningful -- there is no real notion of "distance"
        between two entities in this schema -- so a weighted-shortest-path algorithm would add
        complexity without adding any real capability; BFS already finds the shortest chain by
        hop count, which is the only ordering that means anything here. ``max_hops`` bounds the
        search so a question with no real answer doesn't walk the entire graph -- reaching it
        without finding ``name_b`` returns ``None``, the same result as if no path existed at all
        (it's a caller-visible cap, not silently swapped for "no path"; see its default's
        comment). This is a last-resort fallback for the router (see ``router._route``), not a
        replacement for ``relations_between``'s direct-edge/near-miss logic -- it only ever
        matters when that logic has already come up completely empty."""
        a_ids = {n.entity_id for n in self.find(name_a)}
        b_ids = {n.entity_id for n in self.find(name_b)}
        if not a_ids or not b_ids:
            return None
        if a_ids & b_ids:
            return ()  # the same entity named on both sides -- a zero-hop "path"

        visited = set(a_ids)
        parent: dict[str, tuple[str, Edge]] = {}
        queue: deque[tuple[str, int]] = deque((i, 0) for i in a_ids)
        while queue:
            node_id, depth = queue.popleft()
            if node_id in b_ids:
                hops: list[PathHop] = []
                cur = node_id
                while cur in parent:
                    prev, edge = parent[cur]
                    hops.append(PathHop(edge=edge, from_id=prev, to_id=cur))
                    cur = prev
                hops.reverse()
                return tuple(hops)
            if depth >= max_hops:
                continue
            for edge, neighbor_id in self._path_neighbors(node_id):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                parent[neighbor_id] = (node_id, edge)
                queue.append((neighbor_id, depth + 1))
        return None

    def _infrastructure_note(self, name: str, ids: tuple[str, ...]) -> str | None:
        """A plain-language note when every node named ``name`` has zero non-audit (business)
        edges, in or out, anywhere in this graph -- not just within one relations_between() pair.
        Organization is the concrete case: it has 11 incoming audit ``organizationId`` edges and
        nothing else, so it never has a business relationship to explain, to anything."""
        if not ids:
            return None
        business = sum(len(self.outgoing(i)) + len(self.incoming(i)) for i in ids)
        if business:
            return None
        audit = sum(len(self.outgoing(i, include_audit=True)) + len(self.incoming(i, include_audit=True)) for i in ids)
        if audit:
            return (
                f"{name} has no non-audit (business) relationships anywhere in this graph -- only "
                f"{audit} audit edge(s) (e.g. organizationId, createdBy, ownerId) touch it. It "
                "functions here as infrastructure/tenant metadata, not a business entity with its "
                "own relationships, so no relationship to another business entity should be expected."
            )
        return f"{name} has no relationships at all (audit or otherwise) anywhere in this graph."


@dataclass(frozen=True)
class EntityPairRelations:
    """Result of ``Graph.relations_between``. ``a_ids``/``b_ids`` are every node (any layer)
    matching that name; near-misses are populated only when ``has_edges`` is False."""

    a_name: str
    b_name: str
    a_ids: tuple[str, ...]
    b_ids: tuple[str, ...]
    edges: tuple[Edge, ...]
    has_edges: bool
    a_outgoing_near_misses: tuple[Edge, ...] = ()
    b_incoming_near_misses: tuple[Edge, ...] = ()
    a_note: str | None = None  # set when name_a is infrastructure-like (see _infrastructure_note)
    b_note: str | None = None


# --- entity_detail: the single source of "everything known about one entity" ---------------------
#
# Used by both api.py's GET /entities/{name} (wrapped into that endpoint's Pydantic response
# model) and router.py's single-entity context (wrapped into LLM-facing text) -- one assembly of
# the attribute-resolution + relationship-lookup logic, two renderings, not two implementations.


_LAYER_PRIORITY = [label for _, label in _LAYERS]  # banking first: preferred when a name is ambiguous


def find_entity(graph: Graph, name: str) -> EntityNode | None:
    """Exact name match if one exists, else case-insensitive; when several nodes share that name
    (an entity re-declared across layers, e.g. Account), the most specific layer wins -- banking
    first, then CRM accelerator, CRM base, Foundation, Core, CDS standard (see ``_LAYERS`` above)."""
    candidates = [n for n in graph.nodes.values() if n.name == name]
    if not candidates:
        lowered = name.lower()
        candidates = [n for n in graph.nodes.values() if n.name.lower() == lowered]
    if not candidates:
        return None

    def rank(node: EntityNode) -> int:
        return _LAYER_PRIORITY.index(node.layer) if node.layer in _LAYER_PRIORITY else len(_LAYER_PRIORITY)

    return min(candidates, key=rank)


@dataclass(frozen=True)
class AttributeInfo:
    name: str
    origin: str  # "own" | "inherited" | "standard"
    data_type: str | None = None
    fk_name: str | None = None
    description: str | None = None
    declared_in: str | None = None  # display name of the ancestor that declares it; None for "own"


@dataclass(frozen=True)
class RelationshipInfo:
    direction: str  # "outgoing" | "incoming"
    attribute: str
    fk_name: str
    other_entities: tuple[str, ...]  # display name(s) on the other end; several when polymorphic
    is_polymorphic: bool
    fk_inferred: bool
    inherited_from: str | None = None


@dataclass(frozen=True)
class EntityDetail:
    """Everything known about one entity node: full (untruncated) own/inherited/standard
    attributes, the complete ancestor chain, and every non-audit relationship, both directions.
    Unlike an ``EntityChunk``'s text (deliberately summarized for embedding similarity), this is
    meant to be read once an entity is already identified, when completeness matters more than
    compactness."""

    entity_id: str
    name: str
    display_name: str
    layer: str
    document: str
    description: str | None
    parent_chain: tuple[str, ...]  # display names, immediate parent first
    own_attributes: tuple[AttributeInfo, ...]
    inherited_attributes: tuple[AttributeInfo, ...]
    standard_attributes: tuple[AttributeInfo, ...]
    attribute_counts: dict[str, int]
    relationships: tuple[RelationshipInfo, ...]
    other_layers: tuple[str, ...] = ()  # display names of other nodes sharing this bare name, if any


def _attribute_info(graph: Graph, attr: ResolvedAttribute) -> AttributeInfo:
    declared_in = None
    if attr.origin is not Origin.OWN:
        node = graph.nodes.get(entity_id(attr.declared_in))
        declared_in = node.display_name if node else str(attr.declared_in)
    return AttributeInfo(
        name=attr.name, origin=attr.origin.value, data_type=attr.type_label, fk_name=attr.fk_name,
        description=attr.description, declared_in=declared_in,
    )


def _relationship_info(graph: Graph, edge: Edge, direction: str) -> RelationshipInfo:
    if direction == "outgoing":
        others = tuple(graph.nodes[t.entity_id].display_name if t.entity_id in graph.nodes else t.name for t in edge.targets)
    else:
        node = graph.nodes.get(edge.from_id)
        others = (node.display_name if node else edge.from_id,)
    inherited_from = None
    if edge.inherited_from and edge.inherited_from in graph.nodes:
        inherited_from = graph.nodes[edge.inherited_from].display_name
    return RelationshipInfo(
        direction=direction, attribute=edge.attribute, fk_name=edge.fk_name, other_entities=others,
        is_polymorphic=edge.is_polymorphic, fk_inferred=edge.fk_inferred, inherited_from=inherited_from,
    )


def entity_detail(graph: Graph, node: EntityNode) -> EntityDetail:
    """Full structured data for ``node``: the same assembly ``GET /entities/{name}`` and the
    router's single-entity context both build on."""
    attrs = graph.attributes[node.entity_id]
    counts = {o.value: 0 for o in Origin}
    for a in attrs:
        counts[a.origin.value] += 1
    counts["total"] = len(attrs)

    relationships = tuple(_relationship_info(graph, e, "outgoing") for e in graph.outgoing(node.entity_id))
    relationships += tuple(_relationship_info(graph, e, "incoming") for e in graph.incoming(node.entity_id))

    other_layers = tuple(n.display_name for n in graph.find(node.name) if n.entity_id != node.entity_id)

    return EntityDetail(
        entity_id=node.entity_id,
        name=node.name,
        display_name=node.display_name,
        layer=node.layer,
        document=node.document,
        description=node.description,
        parent_chain=tuple(n.display_name for n in graph.chain(node.entity_id)[1:]),
        own_attributes=tuple(_attribute_info(graph, a) for a in attrs if a.origin is Origin.OWN),
        inherited_attributes=tuple(_attribute_info(graph, a) for a in attrs if a.origin is Origin.INHERITED),
        standard_attributes=tuple(_attribute_info(graph, a) for a in attrs if a.origin is Origin.STANDARD),
        attribute_counts=counts,
        relationships=relationships,
        other_layers=other_layers,
    )


# --- attribute_detail: "everything known about one FK attribute NAME" ----------------------------
#
# A separate lookup from entity_detail/relations_between, because an attribute like
# CampaignResponse's ``regardingObject`` has real, resolved polymorphic targets but no
# corresponding ``Edge``: only seed entities get their FKs turned into edges (see the module
# docstring), and CampaignResponse isn't one. This reads every node's *resolved attributes*
# instead (``graph.attributes``, already computed for every node in the graph), grouped by
# attribute name across every entity that declares or inherits it -- independent of edges.


@dataclass(frozen=True)
class AttributeDetail:
    """Everything known about one FK attribute name (e.g. "regardingObject", "customer"),
    aggregated across every entity in the graph that declares or inherits it. Audit/standard
    fields (createdBy, ownerId, transactionCurrencyId, ...) are excluded -- same fields
    ``build_relationship_chunks`` already excludes via ``Edge.is_audit``, just reached here via
    ``Origin.STANDARD`` and ``AUDIT_ATTRIBUTES`` since these attributes may have no edge at all."""

    name: str
    fk_name: str
    targets: tuple[str, ...]  # bare target entity names, as written (not resolved to a node)
    declared_by: tuple[str, ...]  # display names of entities where this is an OWN attribute
    inherited_by: tuple[str, ...]  # display names of entities that inherit it
    is_polymorphic: bool


def _attribute_groups(graph: Graph) -> dict[str, list[tuple[EntityNode, ResolvedAttribute]]]:
    groups: dict[str, list[tuple[EntityNode, ResolvedAttribute]]] = defaultdict(list)
    for node_id, attrs in graph.attributes.items():
        node = graph.nodes[node_id]
        for a in attrs:
            if not a.is_fk or not a.fk_targets or a.fk_is_placeholder:
                continue
            if a.origin is Origin.STANDARD or a.fk_name in AUDIT_ATTRIBUTES:
                continue
            groups[a.name].append((node, a))
    return groups


def attribute_details(graph: Graph) -> dict[str, AttributeDetail]:
    """Every non-audit, non-standard FK attribute name in the graph, keyed by that name. Checked
    against the real corpus: no two entities declare the same attribute name with different
    target sets, so one entry per name is a safe aggregation here -- not a guarantee for every
    possible corpus (see ``_attribute_groups``'s per-name grouping if that ever needs revisiting)."""
    out: dict[str, AttributeDetail] = {}
    for name, items in _attribute_groups(graph).items():
        targets = tuple(sorted({t.entity for _, a in items for t in a.fk_targets}))
        declared = tuple(sorted({n.display_name for n, a in items if a.origin is Origin.OWN}))
        inherited = tuple(sorted({n.display_name for n, a in items if a.origin is Origin.INHERITED}))
        fk_names = sorted({a.fk_name for _, a in items})
        out[name] = AttributeDetail(
            name=name,
            fk_name=fk_names[0] if fk_names else "",
            targets=targets,
            declared_by=declared,
            inherited_by=inherited,
            is_polymorphic=len(targets) > 1,
        )
    return out


def attribute_detail(graph: Graph, name: str) -> AttributeDetail | None:
    return attribute_details(graph).get(name)


def known_attribute_names(graph: Graph) -> dict[str, str]:
    """Every token that could name a known attribute in a question -- its plain name ("bank")
    and, when different, its FK column name ("bankId") -- mapped to the canonical (plain) name,
    so either form resolves to the same ``AttributeDetail``."""
    tokens: dict[str, str] = {}
    for name, detail in attribute_details(graph).items():
        tokens[name] = name
        if detail.fk_name and detail.fk_name != name:
            tokens[detail.fk_name] = name
    return tokens


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
