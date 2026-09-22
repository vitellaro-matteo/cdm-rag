"""Chroma vector store: build/persist an index of chunks, and query it.

Local, file-based, no server (``chromadb.PersistentClient``). Embeddings always come from
``cdm_rag.embeddings`` -- the collection is created with ``embedding_function=None`` so nothing
ever falls back to Chroma's own default embedding model, which would silently mix vector
spaces with whatever ``config.embedding_model()`` names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from cdm_rag.chunks import (
    AttributeChunk,
    EntityChunk,
    RelationshipChunk,
    build_attribute_chunks,
    build_entity_chunks,
    build_relationship_chunks,
)
from cdm_rag.config import PROJECT_ROOT, embedding_model
from cdm_rag.embeddings import embed
from cdm_rag.graph import Graph

DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "cdm_chunks"

Chunk = EntityChunk | RelationshipChunk | AttributeChunk


def chunks_for_index(graph: Graph) -> list[Chunk]:
    """Every chunk that belongs in the vector index: entity chunks (infrastructure entities
    already excluded, see ``config.INDEX_EXCLUDED_ENTITY_NAMES`` and ``chunks.build_entity_chunks``),
    every relationship chunk (audit edges already excluded, see ``chunks.build_relationship_chunks``),
    and every attribute chunk (audit/standard names already excluded, see ``chunks.build_attribute_chunks``)."""
    return [*build_entity_chunks(graph), *build_relationship_chunks(graph), *build_attribute_chunks(graph)]


def _metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool/list-of-those, and a list must be
    non-empty. ``None`` (e.g. a root entity's ``parent_id``) and ``[]`` (e.g. an attribute with
    no inheritors) are both dropped rather than coerced, so their absence is the signal, not a
    stray empty value; ``attribute_count`` (a dict) is JSON-encoded. Each chunk's own ``to_dict()``
    already turns tuple fields into lists, so this only needs to handle None/dict/empty-list."""
    out = {}
    for k, v in chunk.to_dict()["metadata"].items():
        if v is None or v == []:
            continue
        if isinstance(v, dict):
            v = json.dumps(v)
        out[k] = v
    return out


def build_index(
    graph: Graph,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str | None = None,
) -> chromadb.api.models.Collection.Collection:
    """(Re)build a persisted collection from every indexable chunk in ``graph``. Any existing
    collection of the same name is dropped first, so this is always a full rebuild, not a merge."""
    chunks = chunks_for_index(graph)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model_name or embedding_model()},
    )
    if chunks:
        vectors = embed([c.text for c in chunks], name=model_name)
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
    return collection


def open_index(
    persist_dir: Path | str = DEFAULT_PERSIST_DIR, collection_name: str = COLLECTION_NAME
) -> chromadb.api.models.Collection.Collection:
    return chromadb.PersistentClient(path=str(persist_dir)).get_collection(collection_name, embedding_function=None)


def open_or_build_index(
    graph: Graph,
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str | None = None,
) -> chromadb.api.models.Collection.Collection:
    """Open the persisted collection at ``persist_dir`` if one already exists there; otherwise
    build it from ``graph`` (a one-time cost for a fresh clone/environment; a machine that has
    already built one -- e.g. from earlier dev or test runs -- just reopens it). Used by the API
    at startup so the embedding model and a full corpus embed aren't paid on every process start."""
    try:
        return open_index(persist_dir=persist_dir, collection_name=collection_name)
    except chromadb.errors.NotFoundError:
        return build_index(graph, persist_dir=persist_dir, collection_name=collection_name, model_name=model_name)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def query(
    collection: chromadb.api.models.Collection.Collection,
    text: str,
    k: int = 5,
    model_name: str | None = None,
) -> list[SearchResult]:
    """Top-``k`` chunks by embedding similarity to ``text``, nearest first."""
    vector = embed([text], name=model_name, is_query=True)[0]
    result = collection.query(query_embeddings=[vector], n_results=k)
    ids, docs, metas, dists = result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    return [SearchResult(i, d, m, dist) for i, d, m, dist in zip(ids, docs, metas, dists)]
