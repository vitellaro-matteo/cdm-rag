"""Turn retrieved chunks into a grounded answer, via ``llm_client.chat`` (the only LLM call in
this module -- see llm_client.py for the provider boundary).

The system prompt is deliberately restrictive: answer only from the given context, say plainly
when nothing relevant was found rather than guessing, and name which entities/relationships were
used. This matters concretely for a question like "how does Contact relate to Organization?"
(see ``Graph.relations_between`` in graph.py): there is no direct edge between them, but Contact
does connect to Account via `employer` and `parentCustomer`. Retrieval will surface those Contact
chunks even though the query mentions Organization; the prompt is written so the model reports
the absence plainly and then names that near miss ("no direct relationship, but X is related via
Y"), instead of either a bare "no relationship" or, worse, inventing one.
"""

from __future__ import annotations

from typing import Any

from cdm_rag import llm_client

SYSTEM_PROMPT = """You are answering questions about the Microsoft Common Data Model (CDM) schema \
using ONLY the context chunks given below. Each chunk describes one entity or one relationship \
(foreign-key edge) from the schema graph, or is a note about an entity's role.

Rules:
1. Answer only from the provided context. Do not use outside knowledge of CDM, Dynamics, or any \
other schema, and do not invent entities, attributes, or relationships that are not in the context.
2. If the context contains no relevant relationship or attribute for the question, say so plainly \
instead of guessing -- never imply a relationship exists when it does not. If the context shows a \
near miss (no direct edge between the two entities asked about, but one of them connects to a \
third entity), state clearly that there is no direct relationship, then describe the near miss, \
in the form "no direct relationship, but X is related via Y".
3. The context can contain several relevant relationships or near-misses at once, not just one \
-- for example, two different near-miss edges to the same third entity, via two different \
attributes. Enumerate ALL of them that bear on the question, not only the first or most obvious \
one; do not stop after finding a single match. If more than one near-miss chunk is present, name \
every one of them, each with its own attribute/foreign-key name.
4. If a chunk is a note (about an entity having no business relationships, or acting as \
infrastructure/tenant metadata), you must include what it says in your answer whenever that \
entity is part of the question -- it explains *why* there is no direct relationship, not just \
that there is none.
5. Name the specific entities and relationships (attribute or foreign-key names) you used, so the \
answer is traceable back to the schema.

Example (format only -- these entities are illustrative, not from the real schema):
Context: [1] (relationship) Order has a many-to-one relationship to Customer: each Order refers \
to one Customer, via attribute customer (foreign key customerId). [2] (relationship) Order has \
a many-to-one relationship to Warehouse: each Order refers to one Warehouse, via attribute \
shipFrom (foreign key shipFromId).
Question: How does Order relate to Region?
Answer: There is no direct relationship between Order and Region. However, Order relates to \
Customer via attribute customer (foreign key customerId). Order also relates to Warehouse via \
attribute shipFrom (foreign key shipFromId)."""


def _text_and_metadata(chunk: Any) -> tuple[str, dict[str, Any]]:
    """Accepts a plain {"text": ..., "metadata": ...} dict or an object with those attributes
    (e.g. ``store.SearchResult``)."""
    if isinstance(chunk, dict):
        return chunk["text"], chunk.get("metadata") or {}
    return chunk.text, getattr(chunk, "metadata", None) or {}


def _context_block(chunks: list[Any]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        text, meta = _text_and_metadata(chunk)
        label = meta.get("chunk_type", "chunk")
        lines.append(f"[{i}] ({label}) {text}")
    return "\n\n".join(lines)


def answer(question: str, chunks: list[Any], extra_instructions: str | None = None) -> str:
    """Ask the configured LLM to answer ``question`` using only ``chunks`` -- each a
    {"text", "metadata"} dict or a ``store.SearchResult``, typically the output of
    ``store.query()``. An empty ``chunks`` list still calls the LLM, with context that says so,
    so rule 2 above applies uniformly rather than needing a special "nothing retrieved" path.

    ``extra_instructions``, when given, is appended to the system prompt for this call only --
    e.g. router.py uses it to flag a genuine lexical ambiguity it detected between a word in the
    question and a real ancestor layer name in the retrieved entity_lookup data (see
    ``router._ancestor_layer_collision``), rather than baking a rule for that into every call's
    prompt when it doesn't apply."""
    system_prompt = f"{SYSTEM_PROMPT}\n\n{extra_instructions}" if extra_instructions else SYSTEM_PROMPT
    context = _context_block(chunks) if chunks else "(no chunks were retrieved for this question)"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm_client.chat(messages)
