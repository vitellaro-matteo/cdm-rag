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
(foreign-key edge) from the schema graph.

Rules:
1. Answer only from the provided context. Do not use outside knowledge of CDM, Dynamics, or any \
other schema, and do not invent entities, attributes, or relationships that are not in the context.
2. If the context contains no relevant relationship or attribute for the question, say so plainly \
instead of guessing -- never imply a relationship exists when it does not. If the context shows a \
near miss (no direct edge between the two entities asked about, but one of them connects to a \
third entity), state clearly that there is no direct relationship, then describe the near miss, \
in the form "no direct relationship, but X is related via Y".
3. Name the specific entities and relationships (attribute or foreign-key names) you used, so the \
answer is traceable back to the schema."""


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


def answer(question: str, chunks: list[Any]) -> str:
    """Ask the configured LLM to answer ``question`` using only ``chunks`` -- each a
    {"text", "metadata"} dict or a ``store.SearchResult``, typically the output of
    ``store.query()``. An empty ``chunks`` list still calls the LLM, with context that says so,
    so rule 2 above applies uniformly rather than needing a special "nothing retrieved" path."""
    context = _context_block(chunks) if chunks else "(no chunks were retrieved for this question)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm_client.chat(messages)
