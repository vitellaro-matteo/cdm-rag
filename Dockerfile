# Build context is this directory (cdm-rag/) alone -- a single, standard `docker build .`, with
# nothing external required. That was not always true: build_graph() only ever reads the ~200
# corpus files (~12MB of the full corpus's ~970MB) an import/moniker/extendsEntity chain actually
# asks for (see cdm_rag/inheritance.py's Corpus; it never scans the whole schemaDocuments tree),
# and this was previously exploited by deriving that subset from an *external* second build
# context (--build-context corpus=../CDM/schemaDocuments) at build time. That approach works with
# local `docker build` but not with Render's standard single-context Docker builds, so the
# derived subset is instead committed directly into this repo at corpus_subset/ (201 files,
# ~12MB -- small enough to commit, and static: it only needs re-deriving if the CDM schema itself
# changes). Re-derive and re-verify it with:
#
#   python scripts/export_corpus_subset.py ../CDM/schemaDocuments corpus_subset
#
# which rebuilds the graph from the trimmed copy and fails loudly if it doesn't produce identical
# node/edge ids to the full corpus -- not a hand-maintained file list. See the "Running with
# Docker" / "Deploying to Render" sections in README.md for the full build/run commands and how
# to pass GROQ_API_KEY at runtime (never baked into the image).

# syntax=docker/dockerfile:1

########## Stage 1: install deps and bake the Chroma index ##########
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
COPY corpus_subset ./schemaDocuments
ENV CDM_CORPUS_PATH=/app/schemaDocuments
# Builds the graph and embeds every chunk now, at build time -- not on first request.
RUN python scripts/build_index.py

########## Stage 2: runtime image ##########
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
