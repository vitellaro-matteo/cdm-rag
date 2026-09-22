"""Build the Chroma index from the banking-seeded graph and persist it.

Used by the Docker build (see ../Dockerfile) to bake the index in at build time -- so a fresh
container starts instantly instead of paying the first-build cost (model load + embedding every
chunk) live, on the first request, during a demo. Safe to run by hand too, e.g. after a corpus
update, to rebuild the persisted index at ``cdm_rag.store.DEFAULT_PERSIST_DIR``:

    python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdm_rag.config import corpus_path  # noqa: E402
from cdm_rag.graph import banking_seeds, build_graph  # noqa: E402
from cdm_rag.inheritance import Corpus  # noqa: E402
from cdm_rag.store import DEFAULT_PERSIST_DIR, build_index  # noqa: E402


def main() -> None:
    t0 = time.time()
    corpus = Corpus(corpus_path())
    graph = build_graph(corpus, banking_seeds(corpus))
    print(f"graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges ({time.time() - t0:.1f}s)")

    t0 = time.time()
    collection = build_index(graph, persist_dir=DEFAULT_PERSIST_DIR)
    print(f"index: {collection.count()} chunks at {DEFAULT_PERSIST_DIR} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
