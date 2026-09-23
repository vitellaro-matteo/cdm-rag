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
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from cdm_rag import generate
from cdm_rag.chunks import build_attribute_chunk, build_relationship_chunk
from cdm_rag.graph import (
    AttributeInfo,
    Edge,
    EntityDetail,
    EntityPairRelations,
    Graph,
    PathHop,
    RelationshipInfo,
    attribute_detail,
    entity_detail,
    find_entity,
    known_attribute_names,
)
from cdm_rag.store import query as store_query

DEFAULT_K = 5


@contextmanager
def _measure(timing: dict[str, float] | None, stage: str) -> Iterator[None]:
    """Add elapsed wall-clock seconds under ``timing[stage]`` (accumulating, since e.g.
    "detection" can run more than once per call -- up to three regex checks before falling
    through to plain vector search). A no-op, at the cost of one function call, when ``timing``
    is None -- the default, and the only behavior every existing caller sees."""
    if timing is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing[stage] = timing.get(stage, 0.0) + (time.perf_counter() - t0)


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
SOURCE_MULTI_HOP_PATH = "multi_hop_path"


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


def _format_path(graph: Graph, hops: tuple[PathHop, ...]) -> str:
    def label(node_id: str) -> str:
        node = graph.nodes.get(node_id)
        return node.display_name if node else node_id

    parts = [label(hops[0].from_id)]
    for hop in hops:
        parts.append(f"-> {label(hop.to_id)} (via {hop.edge.attribute})")
    return " ".join(parts)


def path_context(graph: Graph, name_a: str, name_b: str, hops: tuple[PathHop, ...]) -> list[dict[str, Any]]:
    """``graph.find_path``'s result as ONE generation-ready {"text", "metadata"} item -- a single
    compact sentence naming every hop, not one item per hop (that's exactly the context-flooding
    problem ``_ranked_near_misses`` was built to fix once already, see the module docstring's
    near-miss rationale; a multi-hop path is a last-resort fallback, not a reason to reintroduce
    it). Explicitly labeled as a multi-hop path, not a direct relationship, so generation doesn't
    present indirection as if it were a direct edge."""
    text = (
        f"No direct relationship or near-miss connects {name_a} and {name_b}, but a multi-hop "
        f"path does, {len(hops)} hop(s): {_format_path(graph, hops)}. This is NOT a direct "
        f"relationship -- state that plainly, then describe this path as the indirect connection."
    )
    meta = {
        "chunk_type": "multi_hop_path",
        "source": SOURCE_MULTI_HOP_PATH,
        "entity_a": name_a,
        "entity_b": name_b,
        "hops": len(hops),
    }
    return [{"text": text, "metadata": meta}]


def _own_attributes_lines(detail: EntityDetail) -> list[str]:
    if not detail.own_attributes:
        return ["Own attributes: none."]
    lines = [f"Own attributes, full list ({len(detail.own_attributes)}):"]
    for a in detail.own_attributes:
        desc = f": {a.description}" if a.description else ""
        lines.append(f"- {a.name} ({a.data_type or 'unspecified'}){desc}")
    return lines


def _inherited_attribute_line(a: AttributeInfo) -> str:
    desc = f": {a.description}" if a.description else ""
    return f"  - {a.name} ({a.data_type or 'unspecified'}){desc}"


def _inherited_attributes_lines(detail: EntityDetail) -> list[str]:
    """Full detail (type + description, same as ``_own_attributes_lines``), grouped by
    ancestor -- not just names. Names alone left the *only* place an inherited attribute's type
    or description existed as the ancestor's own (now filtered-out, see ``retrieve()``) entity
    chunk; this is what makes filtering that chunk out lose nothing."""
    if not detail.inherited_attributes:
        return ["Inherited attributes: none."]
    by_ancestor: dict[str, list[AttributeInfo]] = {}
    for a in detail.inherited_attributes:
        by_ancestor.setdefault(a.declared_in or "an ancestor", []).append(a)
    lines = [f"Inherited attributes, full list by ancestor ({len(detail.inherited_attributes)} total):"]
    for ancestor, attrs in by_ancestor.items():
        lines.append(f"- from {ancestor} ({len(attrs)}):")
        lines += [_inherited_attribute_line(a) for a in attrs]
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


#: Extra items to over-fetch from vector search when filtering same-entity-name-other-layer
#: duplicates out of the secondary results (see retrieve()), so the filter eating into the
#: budget doesn't leave fewer than ``k`` genuinely useful secondary items. Generous relative to
#: the real corpus's worst case (an entity re-declared across 4 layers, i.e. 3 duplicates).
LAYER_FILTER_OVERFETCH = 10


def _is_other_layer_duplicate(meta: dict[str, Any], name: str, layer: str) -> bool:
    """True for a vector hit that's another layer's entity chunk for the same entity name
    already resolved via entity_lookup -- e.g. Account (Core) when the question resolved to
    Account (banking). Its content is already subsumed by entity_lookup's own parent chain and
    (now-enriched) inherited-attributes section, so keeping it around only invites the model to
    read the wrong layer's chunk instead of the authoritative one (see the Q1 eval finding this
    is built to fix). Scoped to entity chunks only, and only *other* layers -- the resolved
    entity's own indexed chunk, and unrelated entities, are never touched by this filter."""
    return meta.get("chunk_type") == "entity" and meta.get("name") == name and meta.get("layer") != layer


def _ancestor_layer_collision(question: str, detail: EntityDetail) -> tuple[str, str] | None:
    """(the question's own word, the colliding ancestor layer name) for the first ancestor in
    ``detail.parent_chain`` whose layer label shares a whole word, case-insensitive, with
    ``question`` -- e.g. "core attributes" vs. the real "Core"-layer ancestor genuinely present
    in Account's own chain. Not a retrieval mistake: the layer name is real, correctly-sourced
    data; this exists to have the model flag the ambiguity instead of silently picking a reading
    (see the Q1 eval finding this is built to address). None if there's no such collision.
    Short (<=2 char) words are skipped to avoid flagging on trivial overlaps."""
    for ancestor_display_name in detail.parent_chain:
        if "(" not in ancestor_display_name:
            continue
        layer = ancestor_display_name.rsplit("(", 1)[-1].rstrip(")")
        for word in layer.split():
            if len(word) <= 2:
                continue
            m = re.search(r"\b" + re.escape(word) + r"\b", question, re.IGNORECASE)
            if m:
                return question[m.start() : m.end()], layer
    return None


def _ambiguity_instruction(question_word: str, layer: str, entity_display_name: str) -> str:
    return (
        f'Ambiguity check: the word "{question_word}" in the question could mean either an '
        f'ordinary English word (e.g. the entity\'s own primary attributes) or this schema\'s '
        f'"{layer}" layer, which is genuinely one of {entity_display_name}\'s real ancestors in '
        f"the context below. Do not silently pick one reading. Start your answer with one "
        f'sentence naming this ambiguity, then address BOTH: {entity_display_name}\'s own '
        f'attributes, and, separately, what it inherits from the "{layer}"-layer ancestor.'
    )


def _route(
    question: str, graph: Graph, timing: dict[str, float] | None = None
) -> tuple[list[dict[str, Any]], str | None, tuple[str, str] | None, str | None]:
    """(relation_items, header_label, exclude_other_layers_of, extra_instructions) for
    ``question`` -- the structural part of the routing decision, split out to keep the caller's
    own complexity down. ``exclude_other_layers_of`` is (name, resolved layer) when a single
    entity resolved, for filtering its other-layer duplicates out of secondary vector search.
    ``extra_instructions`` is set only when ``_ancestor_layer_collision`` finds a genuine
    lexical ambiguity -- see ``generate.answer``'s ``extra_instructions`` parameter.

    ``timing``, when given a dict, accumulates wall-clock seconds under ``"detection"`` (the
    regex name-matching calls: 1-3 of them run depending on how many come up empty before a
    match, or all 3 for a question naming nothing known) and ``"graph_lookup"`` (whichever of
    ``relations_between``/``entity_detail``/``attribute_detail`` actually fires)."""
    with _measure(timing, "detection"):
        pair = detect_entity_pair(question, graph)
    if pair:
        with _measure(timing, "graph_lookup"):
            relations = graph.relations_between(*pair)
            items = relations_context(graph, relations)
            # Last-resort fallback, kept narrow: only attempted when relations_between's own
            # direct-edge and (already ranked/filtered) near-miss logic came up completely empty
            # on both sides -- never when either found something, so this can only ever add
            # context, never compete with or override what already works (see graph.find_path).
            if not relations.has_edges and not relations.a_outgoing_near_misses and not relations.b_incoming_near_misses:
                hops = graph.find_path(*pair)
                if hops:
                    items = items + path_context(graph, pair[0], pair[1], hops)
        header = f"Direct schema lookup for {pair[0]} and {pair[1]} (the highest-confidence facts for this question):"
        return items, header, None, None

    with _measure(timing, "detection"):
        single = detect_single_entity(question, graph)
    if single:
        header = (
            f"Full schema lookup for {single} -- this is the authoritative, complete record "
            f"for the specific entity named in this question. Prefer it over anything below, "
            f"even if a word in the question happens to coincidentally match a layer name, "
            f"attribute name, or other label appearing elsewhere in this context (a "
            f"coincidental wording match does not mean a different entity or layer was meant):"
        )
        with _measure(timing, "graph_lookup"):
            node = find_entity(graph, single)
            exclude = None
            extra_instructions = None
            if node is not None:
                exclude = (node.name, node.layer)
                collision = _ancestor_layer_collision(question, entity_detail(graph, node))
                if collision:
                    extra_instructions = _ambiguity_instruction(*collision, node.display_name)
            items = entity_lookup_context(graph, single)
        return items, header, exclude, extra_instructions

    with _measure(timing, "detection"):
        attr = detect_attribute(question, graph)
    if attr:
        header = (
            f"Direct schema lookup for the attribute `{attr}` -- this is the authoritative, "
            f"complete record for the specific attribute named in this question, independent of "
            f"which entity happens to declare or inherit it. Prefer it over anything below:"
        )
        with _measure(timing, "graph_lookup"):
            items = attribute_lookup_context(graph, attr)
        return items, header, None, None

    return [], None, None, None


def _secondary_vector_items(
    question: str,
    collection: Any,
    k: int,
    seen_text: set[str],
    exclude_other_layers_of: tuple[str, str] | None,
    timing: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Top-``k`` vector hits not already in ``seen_text``, over-fetching when
    ``exclude_other_layers_of`` is set so the filter eating into results doesn't leave fewer than
    ``k`` genuinely useful items (see ``LAYER_FILTER_OVERFETCH``)."""
    items: list[dict[str, Any]] = []
    fetch_k = k + LAYER_FILTER_OVERFETCH if exclude_other_layers_of else k
    with _measure(timing, "vector_query"):
        hits = store_query(collection, question, k=fetch_k)
    for hit in hits:
        if len(items) >= k:
            break
        if hit.text in seen_text:
            continue
        if exclude_other_layers_of and _is_other_layer_duplicate(hit.metadata, *exclude_other_layers_of):
            continue
        items.append({"text": hit.text, "metadata": {**dict(hit.metadata), "source": SOURCE_VECTOR_SEARCH}})
        seen_text.add(hit.text)
    return items


def retrieve(question: str, graph: Graph, collection: Any, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Structured context for ``question``, checked in this order (first match wins; see
    ``_route``):

    * Two known entities named -> ``relations_between``'s full output (see ``relations_context``).
    * Exactly one known entity named -> that entity's full ``entity_detail``
      (see ``entity_lookup_context``) -- the same complete data ``GET /entities/{name}`` serves,
      not the summarized entity chunk vector search would otherwise return. Secondary vector
      search then excludes other layers' chunks for that same entity name (see
      ``_is_other_layer_duplicate``); the model reading the wrong layer's duplicate over the
      correctly-resolved one, even when explicitly told not to, is what this removes.
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
    return _retrieve_with_instructions(question, graph, collection, k)[0]


def _assemble(
    relation_items: list[dict[str, Any]], header_label: str | None, vector_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if relation_items and vector_items:
        return [_section_header(header_label), *relation_items, _section_header("Additional context from search:"), *vector_items]
    if relation_items:
        return [_section_header(header_label), *relation_items]
    return vector_items


def _retrieve_with_instructions(
    question: str, graph: Graph, collection: Any, k: int, timing: dict[str, float] | None = None
) -> tuple[list[dict[str, Any]], str | None]:
    """(context, extra_instructions) -- what ``retrieve()`` and ``answer()`` both need, computed
    once so ``answer()`` doesn't have to re-run routing and vector search a second time just to
    also get the ambiguity instruction ``_route`` may have produced."""
    relation_items, header_label, exclude_other_layers_of, extra_instructions = _route(question, graph, timing)
    seen_text = {item["text"] for item in relation_items}
    vector_items = _secondary_vector_items(question, collection, k, seen_text, exclude_other_layers_of, timing)
    with _measure(timing, "context_assembly"):
        context = _assemble(relation_items, header_label, vector_items)
    return context, extra_instructions


@dataclass(frozen=True)
class AnswerResult:
    """Result of ``answer()``: the generated text plus the exact context it was grounded in,
    so a caller (the API, a script) can show its work -- which entities/relationships/notes
    were actually used -- rather than exposing only the final string."""

    answer: str
    context: list[dict[str, Any]] = field(default_factory=list)


def answer(
    question: str, graph: Graph, collection: Any, k: int = DEFAULT_K, timing: dict[str, float] | None = None
) -> AnswerResult:
    """Route, retrieve, and generate: the single entry point tying the pipeline together.

    ``timing``, when given a dict, is populated with a wall-clock breakdown -- ``detection``,
    ``graph_lookup``, ``vector_query``, ``context_assembly`` (all from routing/retrieval, see
    ``_route``/``_retrieve_with_instructions``), ``prompt_build`` and ``llm_call`` (from
    ``generate.answer``), Groq's own reported token/server-timing stats (from
    ``llm_client.chat``'s ``capture_usage``), and ``total`` (this function's own wall-clock time,
    end to end). Purely diagnostic: omitted (the default), this call is identical to before the
    parameter existed -- every ``_measure`` call above is a no-op without a dict to write into."""
    t0 = time.perf_counter()
    context, extra_instructions = _retrieve_with_instructions(question, graph, collection, k, timing)
    text = generate.answer(question, context, extra_instructions=extra_instructions, timing=timing)
    if timing is not None:
        timing["total"] = time.perf_counter() - t0
    return AnswerResult(answer=text, context=context)
