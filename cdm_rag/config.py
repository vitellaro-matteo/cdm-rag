"""Environment-driven configuration: where the corpus lives, what gets indexed, what embeds it.

The corpus is a separate, read-only checkout of the CDM repository. Point
``CDM_CORPUS_PATH`` at its ``schemaDocuments`` directory; when unset, the sibling
checkout ``../CDM/schemaDocuments`` (relative to this project's root) is used.
"""

from __future__ import annotations

import os
from pathlib import Path

CORPUS_ENV_VAR = "CDM_CORPUS_PATH"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = (PROJECT_ROOT.parent / "CDM" / "schemaDocuments").resolve()


def corpus_path() -> Path:
    """The corpus directory: ``$CDM_CORPUS_PATH`` if set and non-empty, else the default."""
    value = os.environ.get(CORPUS_ENV_VAR, "").strip()
    return Path(value).expanduser().resolve() if value else DEFAULT_CORPUS_PATH


# Entity names that are pure CDS/CRM infrastructure (mixins with no business meaning of their
# own; their content is already folded into every entity chunk's "Standard audit fields" line
# or into the inheriting entity's "Inherited attributes" section). They stay in the graph and in
# exact-name lookup, but the vector index skips their entity chunks so retrieval isn't spent on
# generic system-field boilerplate. Recommendation from the corpus scan: these three currently
# have no incoming or outgoing edges (audit or otherwise) in the banking-seeded graph.
INDEX_EXCLUDED_ENTITY_NAMES: frozenset[str] = frozenset({"CdsStandard", "ActivityCommon", "ActivitySystem"})


EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL"
# A small local sentence-transformers model, not a hosted API: it runs offline, costs nothing
# per call, and can't be deprecated or rate-limited out from under this project the way a hosted
# embedding endpoint can. 384-dim output, 512-token context.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def embedding_model() -> str:
    """Sentence-transformers model name: ``$EMBEDDING_MODEL`` if set and non-empty, else the default."""
    value = os.environ.get(EMBEDDING_MODEL_ENV_VAR, "").strip()
    return value or DEFAULT_EMBEDDING_MODEL
