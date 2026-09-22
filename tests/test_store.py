import pytest

from cdm_rag.config import INDEX_EXCLUDED_ENTITY_NAMES
from cdm_rag.embeddings import load_model
from cdm_rag.graph import banking_seeds, build_graph
from cdm_rag.inheritance import Corpus
from cdm_rag.store import build_index, chunks_for_index, open_index, query


@pytest.fixture(scope="module", autouse=True)
def _warm_model():
    load_model()  # amortize the slow first-call JIT warmup across this module's tests


@pytest.fixture(scope="module")
def graph(corpus_root):
    corpus = Corpus(corpus_root)
    return build_graph(corpus, banking_seeds(corpus))


@pytest.fixture(scope="module")
def index(graph, tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma")
    return build_index(graph, persist_dir=persist_dir), persist_dir


def test_index_has_one_entry_per_indexable_chunk(graph, index):
    collection, _ = index
    assert collection.count() == len(chunks_for_index(graph))


def test_no_indexed_chunk_is_an_excluded_infrastructure_entity(index):
    collection, _ = index
    got = collection.get(include=["metadatas"])
    names = {m.get("name") for m in got["metadatas"] if m.get("chunk_type") == "entity"}
    assert not names & INDEX_EXCLUDED_ENTITY_NAMES


def test_metadata_round_trips_including_list_and_dropped_none_fields(graph, index):
    collection, _ = index
    (bank,) = [n for n in graph.find("Bank") if n.layer == "banking"]
    got = collection.get(ids=[f"entity:{bank.entity_id}"], include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["entity_id"] == bank.entity_id and meta["layer"] == "banking"
    assert "attribute_count" in meta and isinstance(meta["attribute_count"], str)  # dict -> JSON string
    if bank.parent_id is None:
        assert "parent_id" not in meta  # None is dropped, not coerced to a stray value


def test_query_attributes_of_account_surfaces_banking_account_in_top_3(index):
    collection, _ = index
    results = query(collection, "attributes of Account", k=3)
    assert len(results) == 3
    hit = [r for r in results if r.metadata.get("chunk_type") == "entity" and r.metadata.get("name") == "Account"]
    assert hit, [r.chunk_id for r in results]
    assert any(r.metadata.get("layer") == "banking" for r in hit)
    assert all(0 <= r.distance <= 2 for r in results)  # cosine distance range
    assert [r.distance for r in results] == sorted(r.distance for r in results)  # nearest first


def test_reopening_the_persisted_index_returns_the_same_data(graph, index):
    _, persist_dir = index
    reopened = open_index(persist_dir=persist_dir)
    assert reopened.count() == len(chunks_for_index(graph))
    results = query(reopened, "attributes of Account", k=3)
    assert any(r.metadata.get("name") == "Account" for r in results)


def test_attribute_chunks_are_indexed_with_full_metadata_and_empty_lists_dropped(index):
    collection, _ = index
    got = collection.get(ids=["attribute:regardingObject"], include=["metadatas", "documents"])
    meta = got["metadatas"][0]
    assert meta["chunk_type"] == "attribute" and meta["attribute"] == "regardingObject"
    assert meta["is_polymorphic"] is True
    assert set(meta["targets"]) >= {"Account", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord"}
    assert meta["declared_by"] == ["CampaignResponse (CRM base)"]
    assert "inherited_by" not in meta  # empty list dropped, not stored as []
    assert got["documents"][0].startswith("The attribute `regardingObject`")
