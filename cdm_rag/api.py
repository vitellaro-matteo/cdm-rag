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

from cdm_rag import router, store
from cdm_rag.config import corpus_path
from cdm_rag.graph import _LAYERS, EntityNode, Graph, banking_seeds, build_graph, entity_id
from cdm_rag.inheritance import Corpus, Origin, ResolvedAttribute

# --- graph / collection: loaded once, cached, overridable in tests via dependency_overrides ----

_graph: Graph | None = None
_collection: Any | None = None

_LAYER_PRIORITY = [label for _, label in _LAYERS]  # banking first: preferred when a name is ambiguous


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


# --- shared lookup helper -----------------------------------------------------------------------


def find_entity(graph: Graph, name: str) -> EntityNode | None:
    """Exact name match if one exists, else case-insensitive; when several nodes share that name
    (an entity re-declared across layers, e.g. Account), the most specific layer wins -- banking
    first, then CRM accelerator, CRM base, Foundation, Core, CDS standard (see graph.py's ``_LAYERS``)."""
    candidates = [n for n in graph.nodes.values() if n.name == name]
    if not candidates:
        lowered = name.lower()
        candidates = [n for n in graph.nodes.values() if n.name.lower() == lowered]
    if not candidates:
        return None

    def rank(node: EntityNode) -> int:
        return _LAYER_PRIORITY.index(node.layer) if node.layer in _LAYER_PRIORITY else len(_LAYER_PRIORITY)

    return min(candidates, key=rank)


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


def _attribute_out(graph: Graph, attr: ResolvedAttribute) -> AttributeOut:
    declared_in = None
    if attr.origin is not Origin.OWN:
        node = graph.nodes.get(entity_id(attr.declared_in))
        declared_in = node.display_name if node else str(attr.declared_in)
    return AttributeOut(
        name=attr.name, origin=attr.origin.value, data_type=attr.data_type, fk_name=attr.fk_name,
        description=attr.description, declared_in=declared_in,
    )


def _relationship_out(graph: Graph, edge, direction: str) -> RelationshipOut:
    if direction == "outgoing":
        others = [graph.nodes[t.entity_id].display_name if t.entity_id in graph.nodes else t.name for t in edge.targets]
    else:
        node = graph.nodes.get(edge.from_id)
        others = [node.display_name if node else edge.from_id]
    inherited_from = None
    if edge.inherited_from and edge.inherited_from in graph.nodes:
        inherited_from = graph.nodes[edge.inherited_from].display_name
    return RelationshipOut(
        direction=direction, attribute=edge.attribute, fk_name=edge.fk_name, other_entities=others,
        is_polymorphic=edge.is_polymorphic, fk_inferred=edge.fk_inferred, inherited_from=inherited_from,
    )


@app.get("/entities/{name}", response_model=EntityDetail)
def get_entity(name: str, graph: Graph = Depends(get_graph)) -> EntityDetail:
    """Pure graph lookup -- no vector store, no LLM. Own/inherited/standard attributes, the
    parent chain, and every non-audit relationship (both directions) for the entity named
    ``name`` (exact match preferred, else case-insensitive; see ``find_entity``)."""
    node = find_entity(graph, name)
    if node is None:
        raise HTTPException(status_code=404, detail=f"no entity named {name!r}")

    attrs = graph.attributes[node.entity_id]
    counts = {o.value: 0 for o in Origin}
    for a in attrs:
        counts[a.origin.value] += 1
    counts["total"] = len(attrs)

    relationships = [_relationship_out(graph, e, "outgoing") for e in graph.outgoing(node.entity_id)]
    relationships += [_relationship_out(graph, e, "incoming") for e in graph.incoming(node.entity_id)]

    return EntityDetail(
        entity_id=node.entity_id,
        name=node.name,
        display_name=node.display_name,
        layer=node.layer,
        document=node.document,
        description=node.description,
        parent_chain=[n.display_name for n in graph.chain(node.entity_id)[1:]],
        own_attributes=[_attribute_out(graph, a) for a in attrs if a.origin is Origin.OWN],
        inherited_attributes=[_attribute_out(graph, a) for a in attrs if a.origin is Origin.INHERITED],
        standard_attributes=[_attribute_out(graph, a) for a in attrs if a.origin is Origin.STANDARD],
        attribute_counts=counts,
        relationships=relationships,
    )


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
def ask(payload: AskRequest, graph: Graph = Depends(get_graph), collection: Any = Depends(get_collection)) -> AskResponse:
    """Answer ``question`` via ``router.answer`` (routing -> retrieval -> generation), and
    return the answer alongside exactly which entities/relationships/notes it was grounded in
    (``sources``) and the raw context text sent to the model (``context_used``), so the answer
    is traceable rather than an opaque string. Requires ``GROQ_API_KEY``/``LLM_MODEL`` to be
    configured (see .env.example) -- that is the only thing in this app that does."""
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
