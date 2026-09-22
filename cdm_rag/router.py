"""Route a question to a structural lookup, plain vector search, or both, then hand the
result to ``generate.answer``.

Why this exists: a manual smoke test asked "How does Contact relate to Organization?" and the
model correctly reported the ``employer`` near-miss but missed the ``parentCustomer`` one --
not a prompt problem, a context problem. Vector top-k only returns what embeds closest to the
query, so it can silently drop a real near-miss edge. ``Graph.relations_between`` (graph.py) is
exhaustive by construction: when a question visibly names two known entities, this module calls
it directly and folds its *entire* output (every direct edge, every near-miss, any
infrastructure note) into the context, on top of whatever vector search also finds.
"""

from __future__ import annotations

import re
from typing import Any

from cdm_rag import generate
from cdm_rag.chunks import build_relationship_chunk
from cdm_rag.graph import Edge, EntityPairRelations, Graph
from cdm_rag.store import query as store_query

DEFAULT_K = 5


def known_entity_names(graph: Graph) -> list[str]:
    """Every distinct bare entity name (``node.name``, not the layered display name) in the
    graph -- what a question would plausibly say, e.g. "Account", not "Account (banking)"."""
    return sorted({n.name for n in graph.nodes.values()})


def detect_entity_pair(question: str, graph: Graph, names: list[str] | None = None) -> tuple[str, str] | None:
    """The first two distinct known entity names that appear, whole-word and case-insensitive,
    in ``question``, in the order they appear -- or None if fewer than two are named. Whole-word
    matching means "Product" won't match inside "FinancialProduct"; a name is one word here,
    never containing whitespace, so \\b works without a manual longest-match pass."""
    names = names if names is not None else known_entity_names(graph)
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
        if len(ordered) == 2:
            break
    return (ordered[0], ordered[1]) if len(ordered) == 2 else None


def _edge_items(graph: Graph, edges: tuple[Edge, ...]) -> list[dict[str, Any]]:
    items = []
    for edge in edges:
        chunk = build_relationship_chunk(graph, edge)
        meta = dict(chunk.to_dict()["metadata"])
        meta["is_audit"] = edge.is_audit  # not on RelationshipChunk: only non-audit edges get one
        items.append({"text": chunk.text, "metadata": meta})
    return items


def relations_context(graph: Graph, relations: EntityPairRelations) -> list[dict[str, Any]]:
    """``relations_between``'s full output, reformatted as generation-ready {"text", "metadata"}
    items: every direct edge (incl. audit), every near-miss, and any infrastructure note."""
    items = _edge_items(graph, relations.edges)
    items += _edge_items(graph, relations.a_outgoing_near_misses)
    items += _edge_items(graph, relations.b_incoming_near_misses)
    for note in (relations.a_note, relations.b_note):
        if note:
            items.append({"text": note, "metadata": {"chunk_type": "note"}})
    return items


def _section_header(text: str) -> dict[str, Any]:
    return {"text": text, "metadata": {"chunk_type": "section_header"}}


def retrieve(question: str, graph: Graph, collection: Any, k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Structured context for ``question``. When it names two known entities, this is
    ``relations_between``'s full output plus top-``k`` vector search, deduplicated by text
    (a real edge that's also indexed would otherwise show up twice); otherwise it's plain
    top-``k`` vector search alone.

    When both parts are present, the relations lookup always comes first, never interleaved with
    the vector-search items, and each part gets its own labeled header -- these are the
    highest-confidence, most-targeted facts for a two-entity question, and a long, unlabeled,
    undifferentiated context is exactly what let the model skim past a real near-miss earlier."""
    relation_items: list[dict[str, Any]] = []
    pair = detect_entity_pair(question, graph)
    if pair:
        relations = graph.relations_between(*pair)
        relation_items = relations_context(graph, relations)

    seen_text = {item["text"] for item in relation_items}
    vector_items: list[dict[str, Any]] = []
    for hit in store_query(collection, question, k=k):
        if hit.text not in seen_text:
            vector_items.append({"text": hit.text, "metadata": dict(hit.metadata)})
            seen_text.add(hit.text)

    if relation_items and vector_items:
        return [
            _section_header(f"Direct schema lookup for {pair[0]} and {pair[1]} (the highest-confidence facts for this question):"),
            *relation_items,
            _section_header("Additional context from search:"),
            *vector_items,
        ]
    if relation_items:
        return [
            _section_header(f"Direct schema lookup for {pair[0]} and {pair[1]} (the highest-confidence facts for this question):"),
            *relation_items,
        ]
    return vector_items


def answer(question: str, graph: Graph, collection: Any, k: int = DEFAULT_K) -> str:
    """Route, retrieve, and generate: the single entry point tying the pipeline together."""
    return generate.answer(question, retrieve(question, graph, collection, k=k))
