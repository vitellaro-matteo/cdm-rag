# Build context is this directory (cdm-rag/) alone. The CDM corpus lives OUTSIDE this repo, so
# it's supplied as a separate named build context rather than by widening the primary context:
#
#   docker build --build-context corpus=../CDM/schemaDocuments -t cdm-rag:latest .
#
# (run from inside cdm-rag/; requires BuildKit, the default since Docker 23 -- `docker buildx
# version` should print something if you're unsure). See the "Running with Docker" section in
# README.md for the full command, including how to pass GROQ_API_KEY at runtime (never baked
# into the image).
#
# Corpus: build_graph() is fully lazy -- it only reads a file when an import/moniker/
# extendsEntity chain asks for it (see cdm_rag/inheritance.py's Corpus), never scans the whole
# schemaDocuments tree. Traced for real (scripts/export_corpus_subset.py): the banking-seeded
# graph touches ~200 of the corpus's ~58,000 files (~11MB of ~970MB). Stage "corpus_export"
# derives and copies exactly that subset -- not a hand-maintained list, verified against a
# full-corpus rebuild -- so the full corpus is never in the final image (or even in "builder").

# syntax=docker/dockerfile:1

########## Stage 1: derive the minimal corpus subset (stdlib only, no pip installs) ##########
# Named "corpus_export" (not "corpus") to avoid colliding with the external named build context
# also called "corpus" (--build-context corpus=...), which stage 2 copies the result from below.
FROM python:3.10-slim AS corpus_export
WORKDIR /export
COPY cdm_rag/__init__.py cdm_rag/inheritance.py cdm_rag/relationships.py cdm_rag/graph.py ./cdm_rag/
COPY scripts/export_corpus_subset.py ./scripts/
COPY --from=corpus . /full_corpus
RUN python scripts/export_corpus_subset.py /full_corpus /subset

########## Stage 2: install deps and bake the Chroma index ##########
FROM python:3.10-slim AS builder
WORKDIR /app
# Same cache location as the final stage, so the embedding model downloaded here (once, at
# build time) is reused at runtime instead of being re-downloaded on the first /ask request.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

COPY requirements.txt .
# requirements.txt already carries "--extra-index-url .../whl/cpu" ahead of sentence-transformers
# (see the comment there), so this alone installs CPU-only torch, not the much larger CUDA build.
RUN pip install --no-cache-dir -r requirements.txt

COPY cdm_rag ./cdm_rag
COPY scripts/build_index.py ./scripts/build_index.py
COPY --from=corpus_export /subset ./schemaDocuments
ENV CDM_CORPUS_PATH=/app/schemaDocuments
# Builds the graph and embeds every chunk now, at build time -- not on first request.
RUN python scripts/build_index.py

########## Stage 3: runtime image ##########
FROM python:3.10-slim AS final
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    CDM_CORPUS_PATH=/app/schemaDocuments \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/schemaDocuments ./schemaDocuments
COPY --from=builder /app/chroma_db ./chroma_db
COPY cdm_rag ./cdm_rag

# GROQ_API_KEY / LLM_MODEL are intentionally NOT set here -- pass them at `docker run` time
# (-e or --env-file). See README.md. Everything except POST /ask works without them.
EXPOSE 8000
CMD ["uvicorn", "cdm_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
