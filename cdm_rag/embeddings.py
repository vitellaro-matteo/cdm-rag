"""Embedding chunk text with the configured sentence-transformers model (default:
``BAAI/bge-small-en-v1.5``, see ``config.embedding_model``).

Loading a model downloads/reads real weights, so it happens lazily on first use, not at
import time, and is cached per model name.

BGE quirk (documented on the model card, not something sentence-transformers does for you):
retrieval quality is noticeably better when a fixed instruction is prepended to *queries* only,
not to the passages being searched. ``embed(..., is_query=True)`` does this; chunk text should
always be embedded with ``is_query=False`` (the default).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from cdm_rag.config import embedding_model

log = logging.getLogger(__name__)

# From the BAAI/bge-small-en-v1.5 model card: prepend to queries (not to indexed passages) for
# retrieval. This build's `model.prompts` ships as empty strings, so sentence-transformers will
# not add it on its own -- embed() does it explicitly.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=None)
def load_model(name: str | None = None) -> Any:
    """The ``SentenceTransformer`` for ``name`` (default: ``config.embedding_model()``), cached."""
    from sentence_transformers import SentenceTransformer  # heavy import, deferred until needed

    return SentenceTransformer(name or embedding_model())


def max_seq_length(name: str | None = None) -> int:
    return load_model(name).max_seq_length


def count_tokens(text: str, name: str | None = None) -> int:
    """Real subword token count for ``text`` under the model's own tokenizer, including the
    special tokens (``[CLS]``/``[SEP]``) it actually adds at encode time."""
    return len(load_model(name).tokenizer(text)["input_ids"])


def embed(
    texts: list[str],
    name: str | None = None,
    is_query: bool = False,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed ``texts`` with the configured model, L2-normalized (BGE is trained for cosine/dot
    similarity on normalized vectors). Any text whose real token count exceeds the model's max
    sequence length is logged as a warning before encoding -- sentence-transformers truncates
    silently otherwise, which would quietly drop the tail of an over-long chunk rather than fail
    loudly or resize it."""
    model = load_model(name)
    limit = model.max_seq_length
    model_name = name or embedding_model()
    for i, text in enumerate(texts):
        n = count_tokens(text, name)
        if n > limit:
            log.warning(
                "text %d is %d tokens, over %s's max sequence length of %d; sentence-transformers "
                "will silently truncate it before encoding",
                i,
                n,
                model_name,
                limit,
            )
    inputs = [QUERY_INSTRUCTION + t for t in texts] if is_query else list(texts)
    vectors = model.encode(inputs, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()
