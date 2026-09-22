<!-- Rest of the README to be written separately; this is just the Docker section. -->

## Running with Docker

The image is self-contained at runtime: it ships a pre-built Chroma index and the embedding
model's weights, so a container starts instantly with no first-request cold-start. The CDM
corpus lives *outside* this repo, so it's supplied at **build** time only, as a separate named
build context — nothing about it is needed to *run* the image afterward.

### Build

Requires BuildKit (the default since Docker 23; check with `docker buildx version` if unsure)
and a local checkout of the CDM corpus as a sibling of this repo (`../CDM/schemaDocuments`,
same layout `CDM_CORPUS_PATH` defaults to elsewhere in this project).

```bash
docker build --build-context corpus=../CDM/schemaDocuments -t cdm-rag:latest .
```

Run from inside this directory (`cdm-rag/`). The build reads the full corpus only to derive and
bake in the ~200 files (~11MB) `build_graph()` actually needs (see `Dockerfile`'s header comment
and `scripts/export_corpus_subset.py`) — the full corpus is never copied into any image layer.
It also downloads the embedding model and builds the vector index, so the build needs network
access and takes a couple of minutes; the resulting image does not need either at runtime.

### Run

`GROQ_API_KEY` (and optionally `LLM_MODEL`) must be passed at **run** time — they are never
baked into the image. Everything except `POST /ask` works without them.

```bash
docker run --rm -p 8000:8000 --env-file .env cdm-rag:latest
# or explicitly:
docker run --rm -p 8000:8000 -e GROQ_API_KEY=... -e LLM_MODEL=openai/gpt-oss-120b cdm-rag:latest
```

Or with Compose (reads `.env` automatically for both build and run):

```bash
docker compose build --build-context corpus=../CDM/schemaDocuments
docker compose up
```

### Verify

```bash
curl http://localhost:8000/health
curl http://localhost:8000/entities/Account
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Contact relate to Organization?"}'
```
