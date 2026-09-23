# cdm-rag

A FastAPI service that answers questions about the Microsoft Common Data Model (CDM) banking accelerator schema, its entities, attributes, and relationships, grounded only in the schema itself.

## 1. Overview

This is a RAG (retrieval-augmented generation) API over the Microsoft CDM banking model plus the common/foundation objects it references. It was built as a technical assignment: given the CDM schema documents, expose a way to ask natural-language questions about entities, attributes, and relationships. 

It's worth being precise about what "RAG" means here, because this project uses two different retrieval strategies feeding the same generation step, not just one. Classic RAG is one half of this system (`cdm_rag/store.py`, vector search over chunks). The other half is retrieval by direct, structured lookup against a real in-memory graph (`cdm_rag/graph.py`) whenever the question can be pinned to specific, known entities or attributes. Both are retrieval; the graph half exists because similarity search alone cannot prove a negative ("there is no connection between X and Y") or answer precisely from data that was deliberately compressed for embedding. Section 3 below is about why that split exists and what it took to get it right.

It's a FastAPI app with three endpoints:

- `GET /health` — liveness/readiness.
- `GET /entities/{name}` — the full resolved record for one entity (own/inherited/standard attributes, ancestor chain, relationships both directions) --> no LLM call.
- `POST /ask` — the RAG endpoint: routes a free-text question, retrieves grounding context, and generates an answer via an LLM.

## 2. Architecture

```
CDM schema documents (.cdm.json)
        │  cdm_rag/inheritance.py - parse entity docs, resolve extendsEntity /
        │  attributeGroupReference chains into full own+inherited+standard attribute lists
        ▼
Inheritance resolution
        │  cdm_rag/graph.py - build_graph(): seed the banking entities, follow their FKs
        │  and ancestor chains, resolve bare target names to entity ids
        ▼
Entity/relationship graph  (Graph: EntityNode, Edge - audit edges flagged, not dropped)
        │  cdm_rag/chunks.py - three chunk types built from the graph
        ▼
Entity chunks · Relationship chunks · Attribute chunks   (176 total)
        │  cdm_rag/embeddings.py - BAAI/bge-small-en-v1.5 (local, offline)
        ▼
Chroma vector store (cdm_rag/store.py - local, file-based, cosine similarity)
        │
        ▼
Router (cdm_rag/router.py) - tries the strongest available answer first, in this order:
  1. two known entities named   → Graph.relations_between(): direct edge, else ranked
                                   near-misses, else (only if both are empty) a multi-hop
                                   path via Graph.find_path()
  2. exactly one known entity   → Graph.entity_detail()   (full authoritative record)
  3. one known FK attribute     → Graph.attribute_detail() (independent of its declaring entity)
  4. none of the above          → vector search only
     (a structural match from 1-3 is always paired with top-k vector search as secondary context)
        │
        ▼
Generation (cdm_rag/generate.py + cdm_rag/llm_client.py — Groq, openai/gpt-oss-120b)
```


## 3. Design decisions

### Why a Graph

The schema is parsed into a real entity/relationship graph (`cdm_rag/graph.py`), not chunked as text and left to embedding similarity alone. Concretely: entities become `EntityNode` objects, and every field that points at another entity becomes an `Edge` object recording who points to whom, via which attribute. Once that graph exists in memory, questions about relationships can be answered by filtering a list of edges.

The concrete reason this matters: "How does a Contact relate to an Organization?" is not answerable by finding the most similar chunk. There is no direct edge between Contact and Organization at all, and the correct answer requires knowing that *(a)* no such edge exists — provably, by checking every edge, not by failing to find a similar-sounding one, and *(b)* Contact does connect to other entities (Account, via `employer` and `parentCustomer`) that a similarity search might not rank highly for a query that never mentions Account. `Graph.relations_between(name_a, name_b)` answers this exhaustively by construction. Every direct edge between the two named entities (both directions, including audit), and when there are none, every one of each entity's own near-miss edges, ranked. This was added after a manual smoke test showed the model correctly reporting the `employer` near-miss but silently missing `parentCustomer`. This turned out to be a context defect.

### The audit-edge finding

CDM entities carry a set of boilerplate FK attributes present on nearly every entity (`createdBy`, `modifiedBy`, `createdOnBehalfBy`, `modifiedOnBehalfBy`, `ownerId`, `owningUser`, `owningTeam`, `owningBusinessUnit`, `transactionCurrencyId`, `organizationId`, `AUDIT_ATTRIBUTES` in `cdm_rag/relationships.py`). These carry no domain meaning: every entity in the schema has the same fields, so their presence says nothing distinctive about what any one entity actually is. Measured directly across the whole `applicationCommon` manifest (effectively the full CDM, not just banking), audit edges dominate, **5,625 of 5,977 relationships (94.1%)**. Scoped down to this project's actual in-scope graph (the banking seed entities plus everything reachable from them), the picture is very different. **41 of 133 edges (30.8%) are audit**, the rest business relationships, most of the manifest-wide audit boilerplate belongs to entities the banking model does not reach.

Audit edges are excluded from `relations_between()`'s default edge-following (`outgoing()`/`incoming()` default to `include_audit=False`) so they don't crowd out real relationships, but they are never dropped from the graph itself, instead flagged (`Edge.is_audit`) and `relations_between()` always checks them first when asking "is there any link at all," because sometimes an audit-only relationship is the honest answer. Organization is the concrete case: it has 11 incoming audit `organizationId` edges and nothing else, so its `relations_between()` note states plainly that it has no business relationships to explain, rather than the API staying silent about why none was found.

### Seed-layer entity resolution

The same entity name is defined multiple times across the CDM, at increasing levels of specificity. First, a generic base definition, then a general-purpose CRM layer built on top of it, and finally this project's actual banking layer, built on top of that. CDM's own import-order resolution turned out to disagree with what a banking document actually means. A bare reference to `"Account"` written in a banking entity file resolves, under strict CDM import order, to the generic *Core*-layer Account, never the banking one, because of how imports are declared. Before this was caught, that meant **banking Account and banking Contact had zero incoming edges**: every other banking entity's reference to "Account" or "Contact" was silently landing on the wrong, generic layer, even though the actual connection is the same one, just resolved to the wrong target.

The fix (`cdm_rag/graph.py`, `9224dab`). When a bare target name matches exactly one seed entity, the edge resolves to that seed entity instead of following import order. This is a build-time decision, made once when the graph is constructed. Each `EdgeTarget` still records both `resolved_by` (`"seed_layer"` or `"import_order"`) and `import_order_id` (what strict CDM resolution would have picked), purely so the override stays inspectable and reviewable later, not because anything at runtime chooses between the two. Scoped narrowly: applied uniformly, it changed **45 targets, none of them audit edges** (verified, not assumed); a name written with an explicit moniker is never overridden; and if two or more seed entities ever shared the same bare name, import order still decides (this doesn't occur in the current corpus scope, so it's undefended, see Known limitations).

### Near-miss ranking

The first version of `relations_between()`'s near-miss fallback returned *every* non-audit edge an entity had. For Contact asked about Organization, that meant all 14 of Contact's non-audit edges went into the prompt as "near misses". This buried the two actually relevant ones (`employer`, `parentCustomer`) in noise, and the model missed both.

An embedding-similarity ranking of each candidate edge's text against the query was tried first and rejected on measurement, it was too unreliable on short attribute-name text. `employer` shares no words and no obvious surface-level meaning with "Organization" to a general-purpose embedding model, and it never ranked first across three different query phrasings tested. What shipped instead is a structural signal: each candidate is scored on lexical overlap (does the missing entity's name appear in the edge's own attribute or target name) *and* whether its target is one of the graph's recurring polymorphic "party" types — Account or Contact, the closest structural analog this schema has to a general counterparty concept. This is why `employer` is correctly kept despite zero textual overlap with "Organization". Zero-scoring candidates are dropped outright, survivors capped at `NEAR_MISS_LIMIT = 5`. This cut Contact's near-miss set from 14 to 3 for the Organization case.

### Three chunk types, and why attribute chunks were added

Entity chunks (51) and relationship chunks (92) cover the obvious cases: "what is X" and "how does X relate to Y." A third, attribute chunk type (33, after excluding audit and standard fields - 176 total) was added after a specific hole surfaced in evaluation: *"What can the `regardingObject` attribute point to?"* returned "not in context," even though the answer existed in the graph. The entities `regardingObject` can point to were present and fine; the problem was that `regardingObject` is *declared by* `CampaignResponse`, and `CampaignResponse` was never one of this project's seed entities - only referenced as a side character by other entities. Only seed entities get their fields turned into real graph edges (a documented structural limit of `build_graph()`), so `regardingObject` never became an edge at all, and since the question never names `CampaignResponse` itself, only the attribute, vector search over entity/relationship chunks had nothing to find.

Attribute chunks fix this by being built from *every* entity's resolved fields directly and keyed by the attribute's own name, so a question naming only the attribute, with no entity name at all, can still find it (`Graph.attribute_detail()`, `router.detect_attribute()`).

### The single-entity direct-lookup split, and the "core attributes of Account" ambiguity

This is the most interesting failure the project surfaced, and it took three iterations to get right.

**The initial problem.** "What are the core attributes of the Account entity?" and "What does the banking Account inherit from?" both answered wrong or incompletely, even though vector search *did* retrieve an Account chunk, because an `EntityChunk`'s text is deliberately summarized for embedding similarity (own attributes in full, inherited attributes truncated to "+N more," only the immediate parent named). That summary is the right shape for *finding* the entity via search; it's the wrong shape for answering about it once found, because the details a detailed question needs were never in the chunk to begin with. The first fix was when a question names exactly one known entity, `router.py` builds context from `Graph.entity_detail()` instead. The full, untruncated own/inherited/standard attribute lists (type and description each), the complete ancestor chain, and every non-audit relationship in both directions. This is the same data `GET /entities/{name}` serves, and the same underlying principle as the graph-not-chunks decision above.

**The ambiguity this exposed.** Account (banking)'s real ancestor chain genuinely includes a layer literally named **Core** (`Account (Core)`). So "core attributes" has two legitimate readings that are both textually present in the real data: the everyday English sense ("the main, essential attributes") and the literal technical sense (the attributes inherited from the ancestor named Core).

Two fixes were tried and rejected, with evidence, before the one that worked:

1. **Strengthen the section header's wording** to explicitly warn against exactly this kind of collision. This did not work. the model still occasionally resolved the question toward the Core-layer reading, even with the correct data and a direct, explicit warning both present and first in context (`0641529`).
   
2. **Filter the other layers' duplicate Account chunks out of secondary vector search**, so the wrong `Account (Core)` chunk couldn't compete at all, plus enrich `entity_detail`'s inherited-attribute rendering with full type and description. This removed the *external* wrong chunk, but the model then found the same ambiguity *inside* the single correct chunk it was already reading, since "Core" is a real ancestor name sitting right there in its own parent chain. (`6caff10`)

**What actually worked**: detect the lexical collision itself, a whole-word, case-insensitive match between a word in the question and a real ancestor-layer label in the resolved entity's own
parent chain, and, only when a genuine collision is found, instruct the model to disclose the ambiguity explicitly and answer **both** readings, rather than silently picking one
(`router._ancestor_layer_collision`, `da23982`).

### Multi-hop traversal — a deliberately last-resort fallback

`relations_between()` (direct edge, then near-miss) only looks one hop out from each named entity. A genuine multi-hop question, "what's the path from Collateral to Bank?" — has a real 3-hop answer (`Collateral → FinancialProduct → Branch → Bank`) that neither a direct-edge check nor a near-miss search would ever find.

`Graph.find_path()` does breadth-first search over the graph's non-audit edges.

The more important decision is *where* this fits in the router, and it's deliberately last, not first, for a specific reason: a direct edge, a near-miss, and a multi-hop path are three progressively weaker kinds of claim, not three attempts at the same question. A direct edge is the strongest possible answer. A near-miss isn't claiming a relationship exists between the two named things at all, it's a redirect to something more useful. A multi-hop path is a real, provable connection, but an indirect one. If multi-hop were tried first, or without the direct/near-miss checks ahead of it, it would very likely find *some* technically-real but meaningless path for almost any pair of entities, especially through audit-edge chains, and present it as if it were a meaningful answer. `find_path()` only runs when `relations_between()` has already confirmed *both* zero direct edges and zero near-misses on both sides, so it only ever surfaces when it's genuinely the best available answer, never as a shortcut that could paper over the honest "no relationship" case.

### Embedding model choice

`BAAI/bge-small-en-v1.5` via `sentence-transformers`, run locally rather than through a hosted embeddings API. It's offline (no per-call cost, no network dependency at query time, can't be deprecated or rate-limited out from under the project the way a hosted endpoint could, this happened to the LLM choice below, mid-project), 384-dim output, 512-token max sequence length.
Across all 176 indexed chunks, mean length is **~103 tokens**, the longest is **340 tokens** (an entity chunk), comfortably under 512, verified with the model's own tokenizer, not a word-count heuristic. Any future chunk that *did* exceed the limit is logged loudly before encoding rather than silently truncated (`embeddings.embed()`).

One BGE-specific detail from its model card, easy to miss: retrieval quality improves when a fixed instruction string (`"Represent this sentence for searching relevant passages: "`) is prepended to *queries* only, never to the indexed passages being searched through. This build's model ships empty default prompts, so `sentence-transformers` won't add it automatically — missing this would have silently degraded search quality with no error to point at the cause, so `embed(..., is_query=True)` adds it explicitly on the query side only.

### LLM choice

Groq, running `openai/gpt-oss-120b`. All Groq SDK usage is isolated behind `cdm_rag/llm_client.py`, the only module in the codebase that imports the `groq` package at all (enforced by a test).

## 4. Known limitations

- **Multi-hop is last-resort and single-shortest-path.** `find_path()` (BFS, `max_hops=4`) only
  runs after `relations_between()` finds zero direct edges and zero near-misses — a direct edge,
  a near-miss, and a multi-hop path are progressively weaker claims. Running multi-hop first would likely surface meaningless paths (often through
  audit edges) for almost any pair. Verified directly: Contact/Organization never calls
  `find_path()` at all, and a test guards this. Doesn't enumerate every path or exceed 4 hops.
- **Near-miss enumeration is sometimes summarized, not itemized.** Context can correctly contain
  several ranked near-misses (e.g. Q2: `employer`, `master`, `parentCustomer`) while the answer
  names only one, in prose. Two prompt fixes (an enumeration rule, then a few-shot example) didn't
  reliably help, the few-shot example sometimes made it worse. Answers stay accurate and
  grounded, just not always itemized.
- **A fixed token cap once caused silent truncation** Q11
  hit `max_tokens` and was cut off mid-sentence, yet its check still passed (it verified a fact
  appeared, not completeness). Fix: `llm_client.chat()` checks the API's own `finish_reason`, and
  on `"length"` raises `ResponseTruncatedError` instead of returning partial text;
  `generate.answer()` retries once with doubled `max_tokens`, else propagates a clear error. The
  eval script also flags an abrupt-looking ending (`looks_truncated()`).
- **Attribute-name collisions are undefended, not just untested.** Zero real collisions across 45
  candidate FK names in scope, but a genuine collision would silently merge into one misleading
  chunk. A synthetic test pins this behavior; it isn't guarded against, since it doesn't occur.
- **Groq free-tier limits were hit** — 8,000 TPM and ~200,000 TPD, both observed
  directly (one run hit 429 at 197,911/200,000). A single large question can use a meaningful
  share of the per-minute budget alone. Mitigated with a timeout + one retry, the truncation-retry
  above, and optional `DEMO_ACCESS_KEY` / `DEMO_RATE_LIMIT_PER_HOUR` on the public deployment.
- **Corpus scope is banking + referenced common objects, not the full CDM.** Healthcare, Retail,
  etc. are out of scope, and the system says so plainly rather than guessing — verified in
  adversarial testing.

## 5. Evaluation

`scripts/eval.py` runs 20 questions against the real pipeline (real graph, real Chroma, real Groq)
— kept out of `pytest` since it checks a live, non-deterministic API. Each question gets an
automated PASS/FAIL checked against the graph's own ground truth, or **NEEDS HUMAN REVIEW** when
the judgment isn't mechanical.

Grouped into nine categories by capability, not left as a flat list:

| Category | Tests | Result |
|---|---|---|
| Entity attributes | Own vs. inherited, full ancestor chain | 4 / 4 |
| Direct relationships | Real edge, correctly cited | 2 / 2 |
| No-relationship handling | Proving a negative, near-miss redirect | 1 / 1 |
| Reverse / multi-hop traversal | Incoming edges, path-finding | 2 / 3 |
| Attribute-level lookup | Polymorphic FK, any declaring entity | 1 / 1 |
| Hallucination resistance | False premises, fabricated entities | 3 / 4 |
| Scope boundaries | Out-of-corpus, says so plainly | 2 / 2 |
| Comparative reasoning | Open-ended | needs review |
| System self-awareness | Confidence, provenance | needs review |

**Overall: 15/20 auto-verified (15 passed, 0 failed), 5 needing manual review.** Two categories
rest on a single question each (no-relationship handling, attribute-level lookup) a known
coverage gap.

`eval_results.md` is gitignored (regenerated by the script, non-deterministic). Three excerpts:

**Q1** — opens by disclosing the ambiguity rather than resolving it silently:

> **Ambiguity note:** "core attributes" could mean the Core-layer inherited attributes, or the
> entity's full attribute set. Below are both interpretations...

**Q2** — a genuine no-relationship answer:

> There is **no direct relationship** between Contact (banking) and Organization... Organization
> has no non-audit (business) relationships anywhere in this graph — it functions here as
> infrastructure/tenant metadata.

**Q16** — multi-hop traversal:

> There is no direct relationship between Collateral and Bank... Collateral → FinancialProduct
> (via `financialProduct`) → Branch (via `branch`) → Bank (via `bank`).

## 6. Running it locally

Requires Python 3.10+ and a local checkout of the CDM schema repository as a sibling directory
(`../CDM/schemaDocuments`), or `CDM_CORPUS_PATH` pointed elsewhere.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY; LLM_MODEL already defaults to openai/gpt-oss-120b
```

Environment variables (`cdm_rag/config.py`, `cdm_rag/llm_client.py`):

| Variable | Required | Default |
|---|---|---|
| `CDM_CORPUS_PATH` | no | `../CDM/schemaDocuments` (sibling of this repo) |
| `EMBEDDING_MODEL` | no | `BAAI/bge-small-en-v1.5` |
| `GROQ_API_KEY` | for `/ask` and eval | — |
| `LLM_MODEL` | for `/ask` and eval | — (no hardcoded fallback; errors immediately if unset) |

Run the API:

```bash
uvicorn cdm_rag.api:app --host 0.0.0.0 --port 8000
```

Run the tests (176 tests; anything needing the real corpus skips cleanly with a clear message if
`CDM_CORPUS_PATH` isn't set to a real checkout):

```bash
pytest
```

Run the evaluation script (real API calls, real time — see Evaluation above):

```bash
python scripts/eval.py          # all questions
python scripts/eval.py 11 20    # resume just Q11-Q20, e.g. after a rate-limit interruption
```

## 7. Running with Docker

The image is self-contained at runtime: it ships a pre-built Chroma index and the embedding
model's weights, so a container starts instantly with no first-request cold-start. The build is a
single, standard build context (`docker build .`).

### Build

```bash
docker build -t cdm-rag:latest .
```

Run from inside this directory (`cdm-rag/`). The build downloads the embedding model and builds
the vector index.

### Run

`GROQ_API_KEY` (and optionally `LLM_MODEL`) must be passed at **run** time — they are never
baked into the image. Everything except `POST /ask` works without them. `DEMO_ACCESS_KEY` /
`DEMO_RATE_LIMIT_PER_HOUR` are optional (see Section 9, below) and off by default.

```bash
docker run --rm -p 8000:8000 --env-file .env cdm-rag:latest
# or explicitly:
docker run --rm -p 8000:8000 -e GROQ_API_KEY=... -e LLM_MODEL=openai/gpt-oss-120b cdm-rag:latest
```

Or with Compose (reads `.env` automatically for both build and run):

```bash
docker compose build
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

## 8. Deploying to Render

`render.yaml` at the repo root is a Render Blueprint: from the
[Render dashboard](https://dashboard.render.com/blueprints) No manual service configuration is needed beyond what it declares, except:

- **`GROQ_API_KEY`** marked `sync: false` in `render.yaml`, so Render prompts for it in the
  dashboard.
- **`DEMO_ACCESS_KEY`** (optional) — also `sync: false`; set it in the dashboard to require an
  `X-Demo-Key` header on `POST /ask` (see Section 9). Leave it unset to leave `/ask` open.
- **`DEMO_RATE_LIMIT_PER_HOUR`** — defaults to `15` in `render.yaml`; edit or remove that entry
  to change or disable it.

Free-tier notes: a free Render web service spins down after a period of inactivity and takes some time to cold-start on the next request, expect the first request after idle to be sloweven though the Chroma index itself is pre-built.

## 9. Protecting a public demo deployment

`POST /ask` is the only endpoint that costs a real Groq API call, so it's the only one with any protection, and both mechanisms below are optional, independently toggled by environment variable, and true no-ops when unset, local development and any deployment that doesn't opt in are completely unaffected. See `cdm_rag/demo_guard.py`.

- **`DEMO_ACCESS_KEY`** — when set, `POST /ask` requires a matching `X-Demo-Key: <value>` header,
  else `401`. `GET /health` and `GET /entities/{name}` are never affected.
- **`DEMO_RATE_LIMIT_PER_HOUR`** - when set to a positive integer, `POST /ask` is capped to that
  many requests per rolling hour, globally across all callers, `429`ing once exceeded. A single in-memory counter, reset whenever the process restarts. A request rejected by `DEMO_ACCESS_KEY` never consumes a slot of this budget (checked first).

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" -H "X-Demo-Key: your-secret-here" \
  -d '{"question": "How does Contact relate to Organization?"}'
```

## 10. How this was built

This was developed with Claude Code assisting on implementation under my direction. I made the architecture decisions and verified findings against the real corpus at each step and designed the evaluation approach.
