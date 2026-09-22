import logging
import math

import pytest

from cdm_rag.embeddings import count_tokens, embed, load_model, max_seq_length


@pytest.fixture(scope="module", autouse=True)
def _warm_model():
    """Load the real model once for this module; encoding is slow to JIT-warm on first call."""
    load_model()


def test_count_tokens_uses_the_real_tokenizer_not_the_heuristic():
    # "bank" is one subword token to a BPE/WordPiece tokenizer, plus [CLS]/[SEP].
    assert count_tokens("bank") == 3


def test_max_seq_length_is_the_models_own_512():
    assert max_seq_length() == 512


def test_embed_returns_normalized_384_dim_vectors():
    vectors = embed(["Branch has a many-to-one relationship to Bank.", "Account is an entity."])
    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 384
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, abs=1e-3)


def test_query_instruction_is_applied_only_for_queries():
    text = "attributes of Account"
    as_passage = embed([text], is_query=False)[0]
    as_query = embed([text], is_query=True)[0]
    assert as_passage != as_query  # the query instruction prefix changes the embedding


def test_over_length_text_is_reported_not_silently_truncated(caplog):
    huge = "word " * 1000  # far more than 512 tokens
    with caplog.at_level(logging.WARNING, logger="cdm_rag.embeddings"):
        embed([huge])
    assert any("over" in r.message and "512" in r.message for r in caplog.records)


def test_normal_chunk_length_text_gives_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="cdm_rag.embeddings"):
        embed(["Branch has a many-to-one relationship to Bank."])
    assert caplog.records == []
