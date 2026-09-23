"""FastAPI service over the graph, vector store, and router.

Two independent dependencies, loaded once at startup (see ``lifespan`` below), never per request:

* the **graph** (``get_graph``) -- pure Python/JSON, no ML, always needed, cheap and fast
  (a fraction of a second). Requires the CDM corpus to exist at ``config.corpus_path()``.
* the **Chroma collection** (``get_collection``) -- opens the persisted index if one already
  exists at ``store.DEFAULT_PERSIST_DIR``, else builds it (loads the embedding model, embeds
  every chunk: a one-time cost for a fresh clone, not paid again once the index is persisted).
  Only used by ``/ask``.

Neither one touches Groq. The LLM client (``llm_client.py``) checks ``GROQ_API_KEY``/``LLM_MODEL``
lazily, only inside ``chat()`` -- so only an actual ``/ask`` call can fail for a missing key, never
app startup, ``/health``, or ``/entities``. See ``README``/the module docstring on each endpoint
for exactly which env vars matter when.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from cdm_rag import demo_guard, router, store
from cdm_rag.config import corpus_path
from cdm_rag.graph import Graph, banking_seeds, build_graph, entity_detail, find_entity
from cdm_rag.inheritance import Corpus

# --- graph / collection: loaded once, cached, overridable in tests via dependency_overrides ----

_graph: Graph | None = None
_collection: Any | None = None


def _build_graph() -> Graph:
    corpus = Corpus(corpus_path())
    return build_graph(corpus, banking_seeds(corpus))


def get_graph() -> Graph:
    """The graph, built once and cached. A FastAPI dependency so tests can override it."""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def get_collection() -> Any:
    """The Chroma collection, opened/built once and cached. A FastAPI dependency so tests can
    override it (e.g. with a stand-in, when the endpoint under test never actually queries it)."""
    global _collection
    if _collection is None:
        _collection = store.open_or_build_index(get_graph())
    return _collection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_graph()  # warm both caches once, at startup, not on the first request
    get_collection()
    yield


app = FastAPI(title="cdm-rag", lifespan=lifespan)


# --- /health --------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Trivial liveness check. Touches no dependency -- not the graph, not the collection, not
    Groq -- so it always responds as long as the process is up, regardless of corpus or API key."""
    return HealthResponse(status="ok")


# --- /entities/{name} -------------------------------------------------------------------------


class AttributeOut(BaseModel):
    name: str
    origin: str = Field(description='"own" | "inherited" | "standard"')
    data_type: str | None = None
    fk_name: str | None = None
    description: str | None = None
    declared_in: str | None = Field(default=None, description="Display name of the ancestor that declares it; None for 'own'.")


class RelationshipOut(BaseModel):
    direction: str = Field(description='"outgoing" | "incoming"')
    attribute: str
    fk_name: str
    other_entities: list[str] = Field(description="Display name(s) on the other end (several when polymorphic).")
    is_polymorphic: bool
    fk_inferred: bool
    inherited_from: str | None = None


class EntityDetail(BaseModel):
    entity_id: str
    name: str
    display_name: str
    layer: str
    document: str
    description: str | None = None
    parent_chain: list[str] = Field(description="Display names, immediate parent first.")
    own_attributes: list[AttributeOut]
    inherited_attributes: list[AttributeOut]
    standard_attributes: list[AttributeOut]
    attribute_counts: dict[str, int]
    relationships: list[RelationshipOut]
    other_layers: list[str] = Field(default_factory=list, description="Display names of other nodes sharing this bare name, if any.")


def _to_response(detail: Any) -> EntityDetail:
    """Adapt a ``graph.EntityDetail`` (framework-agnostic dataclass) into this endpoint's Pydantic
    response model. The data assembly itself -- attribute resolution, relationship lookup, parent
    chain -- lives once in ``graph.entity_detail``; this is just a shape conversion."""
    return EntityDetail(
        entity_id=detail.entity_id,
        name=detail.name,
        display_name=detail.display_name,
        layer=detail.layer,
        document=detail.document,
        description=detail.description,
        parent_chain=list(detail.parent_chain),
        own_attributes=[AttributeOut(**vars(a)) for a in detail.own_attributes],
        inherited_attributes=[AttributeOut(**vars(a)) for a in detail.inherited_attributes],
        standard_attributes=[AttributeOut(**vars(a)) for a in detail.standard_attributes],
        attribute_counts=detail.attribute_counts,
        relationships=[RelationshipOut(**{**vars(r), "other_entities": list(r.other_entities)}) for r in detail.relationships],
        other_layers=list(detail.other_layers),
    )


@app.get("/entities/{name}", response_model=EntityDetail)
def get_entity(name: str, graph: Graph = Depends(get_graph)) -> EntityDetail:
    """Pure graph lookup -- no vector store, no LLM. Own/inherited/standard attributes, the
    parent chain, and every non-audit relationship (both directions) for the entity named
    ``name`` (exact match preferred, else case-insensitive; see ``graph.find_entity``). The same
    assembly (``graph.entity_detail``) backs the router's single-entity context for ``/ask``."""
    node = find_entity(graph, name)
    if node is None:
        raise HTTPException(status_code=404, detail=f"no entity named {name!r}")
    return _to_response(entity_detail(graph, node))


# --- /ask -------------------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    entity: str | None = None
    chunk_type: str
    source: str = Field(description='"direct_edge" | "near_miss" | "note" | "vector_search" | "section_header"')
    attribute: str | None = None
    is_audit: bool | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    context_used: list[str] = Field(description="Raw text of every context item sent to the LLM, in order.")


def _source_item(graph: Graph, item: dict[str, Any]) -> SourceItem:
    meta = item.get("metadata", {})
    chunk_type = meta.get("chunk_type", "chunk")
    entity_name = meta.get("entity") or meta.get("name")  # notes carry "entity"; entity chunks carry "name"
    if entity_name is None and "from_id" in meta and meta["from_id"] in graph.nodes:
        entity_name = graph.nodes[meta["from_id"]].name  # relationship chunks: the entity that owns the edge
    return SourceItem(
        entity=entity_name,
        chunk_type=chunk_type,
        source=meta.get("source", "unknown"),
        attribute=meta.get("attribute"),
        is_audit=meta.get("is_audit"),
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    _access: None = Depends(demo_guard.require_demo_access),
    _rate: None = Depends(demo_guard.enforce_rate_limit),
    graph: Graph = Depends(get_graph),
    collection: Any = Depends(get_collection),
) -> AskResponse:
    """Answer ``question`` via ``router.answer`` (routing -> retrieval -> generation), and
    return the answer alongside exactly which entities/relationships/notes it was grounded in
    (``sources``) and the raw context text sent to the model (``context_used``), so the answer
    is traceable rather than an opaque string. Requires ``GROQ_API_KEY``/``LLM_MODEL`` to be
    configured (see .env.example) -- that is the only thing in this app that does.

    Also gated by ``demo_guard.require_demo_access``/``enforce_rate_limit`` -- both true no-ops
    unless their respective env var (``DEMO_ACCESS_KEY``/``DEMO_RATE_LIMIT_PER_HOUR``) is set; see
    ``demo_guard.py``. Declared before the graph/collection dependencies so a rejected request
    never reaches ``router.answer`` (and so never touches Groq)."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        result = router.answer(question, graph, collection)
    except RuntimeError as exc:
        # llm_client raises RuntimeError for a missing/empty GROQ_API_KEY or LLM_MODEL
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        sources=[_source_item(graph, item) for item in result.context],
        context_used=[item["text"] for item in result.context],
    )
