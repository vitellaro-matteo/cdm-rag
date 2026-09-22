"""Export the minimal schemaDocuments subset build_graph() actually reads.

The corpus loader (inheritance.Corpus) is fully lazy: it only ever reads a file when an import,
moniker, or extendsEntity chain explicitly asks for it (see Corpus.document/_load); nothing in
the production path (graph.py, relationships.py) scans the whole schemaDocuments tree. This
script proves that by tracing every path Corpus._load() actually touches while building the real
banking-seeded graph, then copies just those files elsewhere -- not a hand-maintained list, so it
can't silently go stale if a future change to the seeds or resolution logic needs a new file: the
``verify`` step below rebuilds the graph from the trimmed copy and fails loudly if it differs.

On the real corpus (2026-09) this is ~200 files / ~11MB, against the full corpus's ~58,000 files
/ ~970MB -- used by the Docker build (see ../Dockerfile) to avoid shipping the full corpus. Not
used at runtime by the app itself; CDM_CORPUS_PATH there just points at whichever directory (full
or trimmed) has what build_graph() needs.

Usage: python scripts/export_corpus_subset.py <source_schemaDocuments> <dest_dir>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdm_rag.graph import Graph, banking_seeds, build_graph  # noqa: E402
from cdm_rag.inheritance import Corpus  # noqa: E402


def build_and_trace(source: Path) -> tuple[Graph, set[str]]:
    """The real graph from ``source``, plus every corpus-relative path reading it touched."""
    corpus = Corpus(source)
    touched: set[str] = set()
    original_load = corpus._load

    def traced_load(path: str):
        touched.add(path)
        return original_load(path)

    corpus._load = traced_load  # type: ignore[method-assign]
    graph = build_graph(corpus, banking_seeds(corpus))
    return graph, touched


def copy_subset(source: Path, dest: Path, files: set[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel in files:
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst_file)


def verify(dest: Path, full_graph: Graph) -> None:
    """Rebuild from the trimmed copy; fail loudly if it doesn't match the full-corpus graph --
    catches a subset that's silently missing a file the resolver actually needed."""
    corpus = Corpus(dest)
    trimmed = build_graph(corpus, banking_seeds(corpus))
    if len(trimmed.nodes) != len(full_graph.nodes) or len(trimmed.edges) != len(full_graph.edges):
        raise SystemExit(
            f"corpus subset at {dest} is incomplete: rebuilding from it gives "
            f"{len(trimmed.nodes)} nodes / {len(trimmed.edges)} edges, "
            f"but the full corpus gives {len(full_graph.nodes)} / {len(full_graph.edges)}"
        )
    if trimmed.unresolved_targets != full_graph.unresolved_targets:
        raise SystemExit(f"corpus subset at {dest} resolves targets differently than the full corpus")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_corpus_subset.py <source_schemaDocuments> <dest_dir>")
    source, dest = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()

    full_graph, files = build_and_trace(source)
    copy_subset(source, dest, files)
    verify(dest, full_graph)
    print(f"exported {len(files)} files to {dest} (verified: matches the full-corpus graph)")


if __name__ == "__main__":
    main()
