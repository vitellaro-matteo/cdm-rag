"""Chunks for retrieval: one per entity, one per non-audit relationship edge.

The chunk text is a *search handle* (what gets embedded). It samples attributes and
names relationships; anything that must be exact (the full attribute list, edge
structure) is read from the ``Graph``, never parsed back out of chunk text.

Known limitations
* Token counts are an estimate (see ``estimate_tokens``), not a specific model's tokenizer.
* "Representative" inherited attribute names are the first few in declaration order
  (minus currency ``*Base`` shadows and ``*_display`` helpers), not ranked by importance.
* "Related entities" lists outgoing non-audit targets only; incoming edges are reachable
  through the graph's reverse index and through the relationship chunks.
* Relationship cardinality is always many-to-one from the FK side; CDM does not encode
  1:1 or optionality in these references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cdm_rag.graph import Edge, EntityNode, Graph
from cdm_rag.inheritance import Origin, ResolvedAttribute

MAX_ENTITY_TOKENS = 300
MAX_REPRESENTATIVE = 8
MAX_RELATED = 12
_DESCRIPTION_CHARS = 48

_PIECES = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Rough BPE-style token count: words split at camelCase/digits, punctuation counted
    individually. Within ~15% of common tokenizers on this corpus's identifier-heavy text."""
    return len(_PIECES.findall(text))


@dataclass(frozen=True)
class EntityChunk:
    chunk_id: str
    text: str
    entity_id: str
    name: str
    layer: str
    document: str
    parent_id: str | None
    attribute_counts: dict[str, int]
    source_file: str
    chunk_type: str = field(default="entity", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": {
                "chunk_type": self.chunk_type,
                "entity_id": self.entity_id,
                "name": self.name,
                "layer": self.layer,
                "document": self.document,
                "parent_id": self.parent_id,
                "attribute_count": dict(self.attribute_counts),
                "source_file": self.source_file,
            },
        }


@dataclass(frozen=True)
class RelationshipChunk:
    chunk_id: str
    text: str
    from_id: str
    to_ids: tuple[str, ...]
    attribute: str
    fk_name: str
    is_polymorphic: bool
    fk_inferred: bool
    inherited_from: str | None
    chunk_type: str = field(default="relationship", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": {
                "chunk_type": self.chunk_type,
                "from_id": self.from_id,
                "to_ids": list(self.to_ids),
                "attribute": self.attribute,
                "fk_name": self.fk_name,
                "is_polymorphic": self.is_polymorphic,
                "fk_inferred": self.fk_inferred,
                "inherited_from": self.inherited_from,
            },
        }


# --- helpers -----------------------------------------------------------------


def _label(graph: Graph, node: EntityNode) -> str:
    """Plain name unless several nodes share it, then the layered display name."""
    return node.display_name if len(graph.find(node.name)) > 1 else node.name


def _target_labels(graph: Graph, edge: Edge) -> list[str]:
    return [
        _label(graph, graph.nodes[t.entity_id]) if t.entity_id else t.name
        for t in edge.targets
    ]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _join_or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _short(text: str | None, limit: int = _DESCRIPTION_CHARS) -> str:
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    if len(first) <= limit:
        return first
    cut = first[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(",;:") + "…"


def _type_of(a: ResolvedAttribute) -> str:
    if a.is_fk:
        return "reference to " + "/".join(t.entity for t in a.fk_targets) if a.fk_targets else "reference"
    return a.data_type or "unspecified"


def _is_shadow(a: ResolvedAttribute) -> bool:
    """Currency "*Base" value copies and "*_display" helpers add no meaning to a summary."""
    return a.name.endswith("Base") or a.name.endswith("_display")


def _representative(attrs: list[ResolvedAttribute], limit: int) -> list[str]:
    useful = [a.name for a in attrs if not _is_shadow(a)]
    return (useful or [a.name for a in attrs])[:limit]


def _listed_own(graph: Graph, node: EntityNode) -> list[ResolvedAttribute]:
    own = [a for a in graph.attributes[node.entity_id] if a.origin is Origin.OWN]
    return [a for a in own if not _is_shadow(a)] or own


def _field_name(a: ResolvedAttribute) -> str:
    """Column-level name: the FK name for references (createdBy), else the attribute name."""
    return a.fk_name if a.fk_name and not a.fk_is_placeholder else a.name


# --- entity chunks -------------------------------------------------------------


def _related_line(graph: Graph, node: EntityNode) -> str:
    seen: dict[str, None] = {}
    for edge in graph.outgoing(node.entity_id):  # non-audit only
        for label in _target_labels(graph, edge):
            seen.setdefault(label)
    names = list(seen)
    if not names:
        return ""
    more = f", and {len(names) - MAX_RELATED} more" if len(names) > MAX_RELATED else ""
    return "Related entities: " + ", ".join(names[:MAX_RELATED]) + more + "."


def _entity_text(graph: Graph, node: EntityNode, own_cap: int, own_described: int, inherited_names: int) -> str:
    attrs = graph.attributes[node.entity_id]
    own = [a for a in attrs if a.origin is Origin.OWN]
    inherited = [a for a in attrs if a.origin is Origin.INHERITED]
    standard = [a for a in attrs if a.origin is Origin.STANDARD]

    lines = [f"{node.display_name} is an entity in the {node.layer} layer of the Microsoft Common Data Model."]
    if node.description:
        lines[0] += " " + _short(node.description, 200)

    chain = graph.chain(node.entity_id)[1:]
    lines.append(
        "Parent chain: " + " → ".join(n.display_name for n in chain) + "." if chain else "It has no parent entity."
    )

    if own:
        listed = _listed_own(graph, node)
        shown = []
        for i, a in enumerate(listed[:own_cap]):
            desc = _short(a.description) if i < own_described else ""
            shown.append(f"{a.name} ({_type_of(a)})" + (f": {desc}" if desc else ""))
        rest = f"; and {len(own) - len(shown)} more" if len(own) > len(shown) else ""
        lines.append(f"Own attributes ({len(own)}): " + "; ".join(shown) + rest + ".")
    else:
        lines.append("Own attributes: none.")

    if inherited:
        by_parent: dict[str, list[ResolvedAttribute]] = {}
        for a in inherited:
            by_parent.setdefault(a.declared_in.name + "|" + a.declared_in.document, []).append(a)
        parts = []
        for group in by_parent.values():
            parent = graph.nodes[f"{group[0].declared_in.document}#{group[0].declared_in.name}"]
            names = _representative(group, inherited_names)
            more = f", +{len(group) - len(names)} more" if len(group) > len(names) else ""
            parts.append(f"{len(group)} from {parent.display_name} ({', '.join(names)}{more})")
        lines.append(f"Inherited attributes ({len(inherited)}): " + "; ".join(parts) + ".")

    if standard:
        sample = ", ".join(_field_name(a) for a in standard[:4])
        lines.append(f"Standard audit fields: {len(standard)} common CDS system fields such as {sample}.")

    related = _related_line(graph, node)
    if related:
        lines.append(related)
    return "\n".join(lines)


def _fit(graph: Graph, node: EntityNode) -> str:
    """Densest text within the token budget. Names of the entity's own attributes matter most,
    so keep them broad (>= 8 when the entity has that many), describe a few, and shrink the
    inherited sample (8 -> 5 -> 3 names per parent) before giving up either."""
    n = len(_listed_own(graph, node))
    caps = [c for c in (n, 12, 8) if c <= n] or [n]
    configs = [
        (names, cap, described)
        for described in (4, 2, 0)
        for names in (MAX_REPRESENTATIVE, 5, 3)
        for cap in sorted(set(caps), reverse=True)
    ] + [(3, cap, 0) for cap in (5, 3)]
    text = ""
    for names, cap, described in configs:
        text = _entity_text(graph, node, min(cap, n), min(described, n), names)
        if estimate_tokens(text) <= MAX_ENTITY_TOKENS:
            return text
    return text  # over budget only if the fixed parts alone exceed it


def build_entity_chunk(graph: Graph, node: EntityNode) -> EntityChunk:
    text = _fit(graph, node)
    counts = {o.value: 0 for o in Origin}
    for a in graph.attributes[node.entity_id]:
        counts[a.origin.value] += 1
    counts["total"] = sum(counts.values())
    return EntityChunk(
        chunk_id=f"entity:{node.entity_id}",
        text=text,
        entity_id=node.entity_id,
        name=node.name,
        layer=node.layer,
        document=node.document,
        parent_id=node.parent_id,
        attribute_counts=counts,
        source_file=node.document.rsplit("/", 1)[-1],
    )


def build_entity_chunks(graph: Graph) -> list[EntityChunk]:
    return [build_entity_chunk(graph, n) for n in sorted(graph.nodes.values(), key=lambda n: n.entity_id)]


# --- relationship chunks -------------------------------------------------------


def _relationship_text(graph: Graph, edge: Edge) -> str:
    source = graph.nodes[edge.from_id]
    frm = _label(graph, source)
    targets = _target_labels(graph, edge)
    fk = (
        f"foreign key conventionally named {edge.fk_name}"
        if edge.fk_inferred
        else f"foreign key {edge.fk_name}"
    )

    if edge.is_polymorphic:
        text = (
            f"{frm} has a many-to-one polymorphic relationship via attribute {edge.attribute} ({fk}): "
            f"{edge.attribute} can refer to {_join_or(targets)}. "
            f"Each {frm} record refers to one record of one of these types. "
            f"Reverse: {_article(targets[0])} {_join_or(targets)} can be referred to by many {frm} records."
        )
    else:
        to = targets[0]
        text = (
            f"{frm} has a many-to-one relationship to {to}: each {frm} refers to one {to}, "
            f"via attribute {edge.attribute} ({fk}). "
            f"Reverse: {_article(to)} {to} can have many {frm} records."
        )

    if edge.inherited_from:
        text += f" This relationship is inherited from {graph.nodes[edge.inherited_from].display_name}."
    described = next((a for a in graph.attributes[edge.from_id] if a.name == edge.attribute and a.description), None)
    if described:
        text += f" Attribute description: {_short(described.description, 160)}"
    return text


def build_relationship_chunk(graph: Graph, edge: Edge) -> RelationshipChunk:
    return RelationshipChunk(
        chunk_id=f"rel:{edge.edge_id}",
        text=_relationship_text(graph, edge),
        from_id=edge.from_id,
        to_ids=edge.to_ids,
        attribute=edge.attribute,
        fk_name=edge.fk_name,
        is_polymorphic=edge.is_polymorphic,
        fk_inferred=edge.fk_inferred,
        inherited_from=edge.inherited_from,
    )


def build_relationship_chunks(graph: Graph) -> list[RelationshipChunk]:
    """One chunk per non-audit edge. Audit edges (createdBy, ownerId, ...) stay in the graph only."""
    return [build_relationship_chunk(graph, e) for e in graph.edges.values() if not e.is_audit]
