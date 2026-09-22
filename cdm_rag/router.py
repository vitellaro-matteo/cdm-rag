"""Route a question to a structural lookup, plain vector search, or both, then hand the
result to ``generate.answer``.

Why the two-entity path exists: a manual smoke test asked "How does Contact relate to
Organization?" and the model correctly reported the ``employer`` near-miss but missed the
``parentCustomer`` one -- not a prompt problem, a context problem. Vector top-k only returns
what embeds closest to the query, so it can silently drop a real near-miss edge.
``Graph.relations_between`` (graph.py) is exhaustive by construction: when a question visibly
names two known entities, this module calls it directly and folds its *entire* output (every
direct edge, every near-miss, any infrastructure note) into the context, on top of whatever
vector search also finds.

Why the single-entity path exists: a later eval run found "what are the core attributes of
Account" and "what does the banking Account inherit from" answered wrong or incompletely even
though vector search *did* retrieve an Account chunk -- because an ``EntityChunk``'s text is
deliberately summarized for embedding similarity (own attributes in full, inherited attributes
truncated to "+N more", only the immediate parent named). That's the right shape for finding the
entity; it's the wrong shape for answering about it once found. When a question names exactly
one known entity, this module instead builds the context from ``graph.entity_detail`` -- the
same full, untruncated attribute/relationship/parent-chain data ``GET /entities/{name}`` serves
-- so completeness, not search-compactness, is what the model reads from.

Why the attribute path exists: the same eval run asked "what can the regardingObject attribute
point to" and got "not in the context" -- not wrong, just unlucky. ``regardingObject`` is
declared on ``CampaignResponse``, which isn't a seed entity, so it has no ``Edge`` at all (see
graph.py: only seed entities get their FKs turned into edges); the only place its real,
polymorphic targets exist is in ``CampaignResponse``'s own resolved attributes, and vector search
never ranked that one entity chunk high enough for a question that never names CampaignResponse.
When a question names a known FK attribute (by its plain name or its FK column name, e.g.
"regardingObject" or "bankId") but no known entity, this module looks it up directly via
``graph.attribute_detail`` -- independent of which entity happens to declare or inherit it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cdm_rag import generate
from cdm_rag.chunks import build_attribute_chunk, build_relationship_chunk
from cdm_rag.graph import (
    Edge,
    EntityDetail,
    EntityPairRelations,
    Graph,
    RelationshipInfo,
    attribute_detail,
    entity_detail,
    find_entity,
    known_attribute_names,
)
from cdm_rag.store import query as store_query

DEFAULT_K = 5


def known_entity_names(graph: Graph) -> list[str]:
    """Every distinct bare entity name (``node.name``, not the layered display name) in the
    graph -- what a question would plausibly say, e.g. "Account", not "Account (banking)"."""
    return sorted({n.name for n in graph.nodes.values()})


def _distinct_matched_names(question: str, names: list[str]) -> list[str]:
    """Every known entity name that appears whole-word and case-insensitive in ``question``,
    in the order it first appears (deduplicated). Whole-word matching means "Product" won't
    match inside "FinancialProduct"; a name is one word here, never containing whitespace, so
    \\b works without a manual longest-match pass."""
    hits = []
    for name in names:
        m = re.search(r"\b" + re.escape(name) + r"\b", question, re.IGNORECASE)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    ordered: list[str] = []
    for _, name in hits:
        if name not in ordered:
            ordered.append(name)
    return ordered


def detect_entity_pair(question: str, graph: Graph, names: list[str] | None = None) -> tuple[str, str] | None:
    """The first two distinct known entity names named in ``question`` (see
    ``_distinct_matched_names``), in the order they appear -- or None if fewer than two are named."""
    names = names if names is not None else known_entity_names(graph)
    ordered = _distinct_matched_names(question, names)
    return (ordered[0], ordered[1]) if len(ordered) >= 2 else None


def detect_single_entity(question: str, graph: Graph, names: list[str] | None = None) -> str | None:
    """The one known entity name in ``question``, when exactly one is named -- None if zero or
    two-or-more are (a two-entity question is ``detect_entity_pair``'s to handle; this never
    fires alongside it)."""
    names = names if names is not None else known_entity_names(graph)
    ordered = _distinct_matched_names(question, names)
    return ordered[0] if len(ordered) == 1 else None


def detect_attribute(question: str, graph: Graph) -> str | None:
    """The one known FK attribute named in ``question`` -- by its plain name ("bank") or its FK
    column name ("bankId"), see ``graph.known_attribute_names`` -- when exactly one is named.
    Callers must check this only after ``detect_entity_pair``/``detect_single_entity`` both come
    up empty: a few attribute names (e.g. "Account") collide with entity names, and a question
    naming a known entity should always get the entity treatment, never the attribute one."""
    tokens = known_attribute_names(graph)
    matched = _distinct_matched_names(question, list(tokens))
    return tokens[matched[0]] if len(matched) == 1 else None


#: What produced a context item, in ``metadata["source"]`` -- lets a caller (e.g. the API's
#: "sources" field) say *why* each item is there without re-deriving it from item order/text.
SOURCE_DIRECT_EDGE = "direct_edge"
SOURCE_NEAR_MISS = "near_miss"
SOURCE_NOTE = "note"
SOURCE_ENTITY_LOOKUP = "entity_lookup"
SOURCE_ATTRIBUTE_LOOKUP = "attribute_lookup"
SOURCE_VECTOR_SEARCH = "vector_search"
SOURCE_SECTION_HEADER = "section_header"


def _edge_items(graph: Graph, edges: tuple[Edge, ...], source: str) -> list[dict[str, Any]]:
    items = []
    for edge in edges:
        chunk = build_relationship_chunk(graph, edge)
        meta = dict(chunk.to_dict()["metadata"])
        meta["is_audit"] = edge.is_audit  # not on RelationshipChunk: only non-audit edges get one
        meta["source"] = source
        items.append({"text": chunk.text, "metadata": meta})
    return items


def relations_context(graph: Graph, relations: EntityPairRelations) -> list[dict[str, Any]]:
    """``relations_between``'s full output, reformatted as generation-ready {"text", "metadata"}
    items: every direct edge (incl. audit), every near-miss, and any infrastructure note."""
    items = _edge_items(graph, relations.edges, SOURCE_DIRECT_EDGE)
    items += _edge_items(graph, relations.a_outgoing_near_misses, SOURCE_NEAR_MISS)
    items += _edge_items(graph, relations.b_incoming_near_misses, SOURCE_NEAR_MISS)
    for entity_name, note in ((relations.a_name, relations.a_note), (relations.b_name, relations.b_note)):
        if note:
            items.append({"text": note, "metadata": {"chunk_type": "note", "source": SOURCE_NOTE, "entity": entity_name}})
    return items


def _own_attributes_lines(detail: EntityDetail) -> list[str]:
    if not detail.own_attributes:
        return ["Own attributes: none."]
    lines = [f"Own attributes, full list ({len(detail.own_attributes)}):"]
    for a in detail.own_attributes:
        desc = f": {a.description}" if a.description else ""
        lines.append(f"- {a.name} ({a.data_type or 'unspecified'}){desc}")
    return lines


def _inherited_attributes_lines(detail: EntityDetail) -> list[str]:
    if not detail.inherited_attributes:
        return ["Inherited attributes: none."]
    by_ancestor: dict[str, list[str]] = {}
    for a in detail.inherited_attributes:
        by_ancestor.setdefault(a.declared_in or "an ancestor", []).append(a.name)
    lines = [f"Inherited attributes, full list by ancestor ({len(detail.inherited_attributes)} total):"]
    lines += [f"- from {ancestor} ({len(names)}): {', '.join(names)}" for ancestor, names in by_ancestor.items()]
    return lines


def _relationship_line(r: RelationshipInfo) -> str:
    verb = "refers to" if r.direction == "outgoing" else "is referred to by"
    poly = " (polymorphic: any one of these)" if r.is_polymorphic else ""
    inherited = f" [inherited from {r.inherited_from}]" if r.inherited_from else ""
    return f"- ({r.direction}) {verb} {', '.join(r.other_entities)} via attribute {r.attribute} (foreign key {r.fk_name}){poly}{inherited}"


def _relationships_lines(detail: EntityDetail) -> list[str]:
    if not detail.relationships:
        return ["Relationships: none."]
    return [f"Relationships, non-audit, both directions ({len(detail.relationships)}):"] + [
        _relationship_line(r) for r in detail.relationships
    ]


def _entity_lookup_text(detail: EntityDetail) -> str:
    """Full structured rendering of ``detail``: own/inherited/standard attributes in full (not
    the entity chunk's truncated "+N more" summary), the complete ancestor chain, and every
    non-audit relationship in both directions. Meant to be read once the entity is already
    identified, when completeness matters more than the compactness an embedded chunk needs."""
    lines = [f"{detail.display_name}, a {detail.layer}-layer entity."]
    if detail.description:
        lines.append(detail.description)
    if detail.other_layers:
        lines.append(
            f"Note: {detail.name} also exists in other layers of this schema -- {', '.join(detail.other_layers)} -- "
            f"this is the {detail.layer} layer version, the most specific one."
        )
    lines.append(
        "Full parent chain (immediate ancestor first): "
        + (" -> ".join(detail.parent_chain) if detail.parent_chain else "(none; this entity has no parent)")
        + "."
    )
    lines += _own_attributes_lines(detail)
    lines += _inherited_attributes_lines(detail)
    if detail.standard_attributes:
        lines.append(
            f"Standard audit fields ({len(detail.standard_attributes)}): "
            + ", ".join(a.name for a in detail.standard_attributes) + "."
        )
    lines += _relationships_lines(detail)
    return "\n".join(lines)


def entity_lookup_context(graph: Graph, name: str) -> list[dict[str, Any]]:
    """The full ``entity_detail`` for the (single) entity named ``name``, as one generation-ready
    {"text", "metadata"} item -- empty if ``name`` doesn't resolve to a node (shouldn't happen for
    a name that came from ``detect_single_entity``, since that only matches known node names)."""
    node = find_entity(graph, name)
    if node is None:
        return []
    detail = entity_detail(graph, node)
    meta = {
        "chunk_type": "entity_detail",
        "source": SOURCE_ENTITY_LOOKUP,
        "entity": detail.name,
        "entity_id": detail.entity_id,
        "layer": detail.layer,
    }
    return [{"text": _entity_lookup_text(detail), "metadata": meta}]


def attribute_lookup_context(graph: Graph, name: str) -> list[dict[str, Any]]:
    """The full ``attribute_detail`` for the FK attribute named ``name``, as one generation-ready
    {"text", "metadata"} item -- the same text an indexed attribute chunk would have (see
    ``chunks.build_attribute_chunk``), reused directly rather than reformatted a second time.
    Empty if ``name`` doesn't resolve (shouldn't happen for a name that came from
    ``detect_attribute``, which only matches known attribute/FK-column names)."""
    detail = attribute_detail(graph, name)
    if detail is None:
        return []
    chunk = build_attribute_chunk(detail)
    meta = dict(chunk.to_dict()["metadata"])
    meta["source"] = SOURCE_ATTRIBUTE_LOOKUP
    return [{"text": chunk.text, "metadata": meta}]


def _section_header(text: str) -> dict[str, Any]:
    return {"text": text, "metadata": {"chunk_type": "section_header", "source": SOURCE_SECTION_HEADER}}


def retrieve(question: str, graph: Graph, collection: Any, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Structured context for ``question``, checked in this order (first match wins):

    * Two known entities named -> ``relations_between``'s full output (see ``relations_context``).
    * Exactly one known entity named -> that entity's full ``entity_detail``
      (see ``entity_lookup_context``) -- the same complete data ``GET /entities/{name}`` serves,
      not the summarized entity chunk vector search would otherwise return.
    * No known entity, but one known FK attribute named (plain or FK-column form) ->
      that attribute's full ``attribute_detail`` (see ``attribute_lookup_context``) -- independent
      of which entity happens to declare or inherit it, and of whether that entity even has an
      edge (some don't; see the module docstring).
    * None of the above -> nothing beyond vector search.

    In the first three cases the structural lookup is always followed by top-``k`` vector search
    for secondary context, deduplicated by text, under its own header -- never interleaved, and
    always second: these lookups are the highest-confidence, most-targeted facts for a question
    that names something known, and an undifferentiated wall of context is exactly what let the
    model skim past a real near-miss earlier."""
    relation_items: list[dict[str, Any]] = []
    header_label: str | None = None

    pair = detect_entity_pair(question, graph)
    if pair:
        relations = graph.relations_between(*pair)
        relation_items = relations_context(graph, relations)
        header_label = f"Direct schema lookup for {pair[0]} and {pair[1]} (the highest-confidence facts for this question):"
    else:
        single = detect_single_entity(question, graph)
        if single:
            relation_items = entity_lookup_context(graph, single)
            header_label = (
                f"Full schema lookup for {single} -- this is the authoritative, complete record "
                f"for the specific entity named in this question. Prefer it over anything below, "
                f"even if a word in the question happens to coincidentally match a layer name, "
                f"attribute name, or other label appearing elsewhere in this context (a "
                f"coincidental wording match does not mean a different entity or layer was meant):"
            )
        else:
            attr = detect_attribute(question, graph)
            if attr:
                relation_items = attribute_lookup_context(graph, attr)
                header_label = (
                    f"Direct schema lookup for the attribute `{attr}` -- this is the "
                    f"authoritative, complete record for the specific attribute named in this "
                    f"question, independent of which entity happens to declare or inherit it. "
                    f"Prefer it over anything below:"
                )

    seen_text = {item["text"] for item in relation_items}
    vector_items: list[dict[str, Any]] = []
    for hit in store_query(collection, question, k=k):
        if hit.text not in seen_text:
            vector_items.append({"text": hit.text, "metadata": {**dict(hit.metadata), "source": SOURCE_VECTOR_SEARCH}})
            seen_text.add(hit.text)

    if relation_items and vector_items:
        return [_section_header(header_label), *relation_items, _section_header("Additional context from search:"), *vector_items]
    if relation_items:
        return [_section_header(header_label), *relation_items]
    return vector_items


@dataclass(frozen=True)
class AnswerResult:
    """Result of ``answer()``: the generated text plus the exact context it was grounded in,
    so a caller (the API, a script) can show its work -- which entities/relationships/notes
    were actually used -- rather than exposing only the final string."""

    answer: str
    context: list[dict[str, Any]] = field(default_factory=list)


def answer(question: str, graph: Graph, collection: Any, k: int = DEFAULT_K) -> AnswerResult:
    """Route, retrieve, and generate: the single entry point tying the pipeline together."""
    context = retrieve(question, graph, collection, k=k)
    return AnswerResult(answer=generate.answer(question, context), context=context)
