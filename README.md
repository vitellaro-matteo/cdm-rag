# cdm-rag

A FastAPI service that answers questions about the Microsoft Common Data Model (CDM) banking
accelerator schema — its entities, attributes, and relationships — grounded only in the schema
itself.

## 1. Overview

This is a RAG (retrieval-augmented generation) API over the Microsoft CDM banking model plus the
common/foundation objects it references. It was built as a technical assignment: given the CDM
schema documents, expose a way to ask natural-language questions about entities, attributes, and
relationships, with answers that don't hallucinate — an entity, attribute, or relationship the
model names must actually exist in the schema, and a question with no real answer ("how does
Contact relate to Organization?") should get an honest "no direct relationship" rather than an
invented one.

It's worth being precise about what "RAG" means here, because this project uses two different
retrieval strategies feeding the same generation step, not just one. Classic RAG — embed a
question, find the most similar stored text, hand it to an LLM — is one half of this system
(`cdm_rag/store.py`, vector search over chunks). The other half is retrieval by direct, structured
lookup against a real in-memory graph (`cdm_rag/graph.py`) whenever the question can be pinned to
specific, known entities or attributes. Both are retrieval; the graph half exists because
similarity search alone cannot prove a negative ("there is no connection between X and Y") or
answer precisely from data that was deliberately compressed for embedding. Section 3 below is
about why that split exists and what it took to get it right.

It's a FastAPI app with three endpoints:

- `GET /health` — liveness/readiness.
- `GET /entities/{name}` — the full resolved record for one entity (own/inherited/standard
  attributes, ancestor chain, relationships both directions) — no LLM call, pure graph lookup.
- `POST /ask` — the RAG endpoint: routes a free-text question, retrieves grounding context, and
  generates an answer via an LLM.

## 2. Architecture

```
CDM schema documents (.cdm.json)
        │  cdm_rag/inheritance.py — parse entity docs, resolve extendsEntity /
        │  attributeGroupReference chains into full own+inherited+standard attribute lists
        ▼
Inheritance resolution
        │  cdm_rag/graph.py — build_graph(): seed the banking entities, follow their FKs
        │  and ancestor chains, resolve bare target names to real entity ids
        ▼
Entity/relationship graph  (Graph: EntityNode, Edge — audit edges flagged, not dropped)
        │  cdm_rag/chunks.py — three chunk types built from the graph
        ▼
Entity chunks · Relationship chunks · Attribute chunks   (176 total)
        │  cdm_rag/embeddings.py — BAAI/bge-small-en-v1.5 (local, offline)
        ▼
Chroma vector store (cdm_rag/store.py — local, file-based, cosine similarity)
        │
        ▼
Router (cdm_rag/router.py) — tries the strongest available answer first, in this order:
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

A reader should be able to stop here with the shape of the system; the rest of this document is
about *why* each stage looks the way it does, not what it does.

## 3. Design decisions

### Graph, not flat documents

The schema is parsed into a real entity/relationship graph (`cdm_rag/graph.py`), not chunked as
flat text and left to embedding similarity alone. Concretely: entities become `EntityNode`
objects, and every field that points at another entity becomes an `Edge` object recording who
points to whom, via which attribute. Once that graph exists in memory, questions about
relationships can be answered by *filtering a list of edges* — an exhaustive, checkable operation
— rather than by hoping the right sentence happens to embed close enough to the question.

The concrete reason this matters: "How does a Contact relate to an Organization?" is not
answerable by finding the most similar chunk. There is no direct edge between Contact and
Organization at all, and the correct answer requires knowing that *(a)* no such edge exists —
provably, by checking every edge, not by failing to find a similar-sounding one — and *(b)*
Contact does connect to other entities (Account, via `employer` and `parentCustomer`) that a
similarity search might not rank highly for a query that never mentions Account.
`Graph.relations_between(name_a, name_b)` answers this exhaustively by construction: every direct
edge between the two named entities (both directions, including audit), and when there are none,
every one of each entity's own near-miss edges, ranked. This was added after a manual smoke test
showed the model correctly reporting the `employer` near-miss but silently missing
`parentCustomer` — not a prompt defect, a context defect: vector top-k only returns what embeds
closest to the query, and a real near-miss edge can simply lose that race.

### The audit-edge finding

CDM entities carry a set of boilerplate FK attributes present on nearly every entity
(`createdBy`, `modifiedBy`, `createdOnBehalfBy`, `modifiedOnBehalfBy`, `ownerId`, `owningUser`,
`owningTeam`, `owningBusinessUnit`, `transactionCurrencyId`, `organizationId` —
`AUDIT_ATTRIBUTES` in `cdm_rag/relationships.py`). These carry no domain meaning: every entity in
the schema has the same fields, so their presence says nothing distinctive about what any one
entity actually is. Measured directly against the real corpus: across the whole
`applicationCommon` manifest (effectively the full CDM, not just banking), audit edges dominate —
**5,625 of 5,977 relationships (94.1%)**. Scoped down to this project's actual in-scope graph (the
banking seed entities plus everything reachable from them), the picture is very different:
**41 of 133 edges (30.8%) are audit**, the rest genuine business relationships — most of the
manifest-wide audit boilerplate belongs to entities the banking model never reaches.

Audit edges are excluded from `relations_between()`'s default edge-following
(`outgoing()`/`incoming()` default to `include_audit=False`) so they don't crowd out real
relationships, but they are never dropped from the graph itself — only flagged (`Edge.is_audit`)
— and `relations_between()` always checks them first when asking "is there any link at all,"
because sometimes an audit-only relationship *is* the honest answer. Organization is the concrete
case: it has 11 incoming audit `organizationId` edges and nothing else, so its
`relations_between()` note states plainly that it has no business relationships to explain,
rather than the API staying silent about why none was found.

### Seed-layer entity resolution

The same entity name is defined multiple times across the CDM, at increasing levels of
specificity — a generic base definition, a general-purpose CRM layer built on top of it, and
finally this project's actual banking layer, built on top of that. CDM's own import-order
resolution turned out to disagree with what a banking document actually means: a bare reference
to `"Account"` written in a banking entity file resolves, under strict CDM import order, to the
generic *Core*-layer Account — never the banking one — because of how imports are declared.
Before this was caught, that meant **banking Account and banking Contact had zero incoming
edges**: every other banking entity's reference to "Account" or "Contact" was silently landing on
the wrong, generic layer, even though the actual connection is the same one, just resolved to the
wrong target.

The fix (`cdm_rag/graph.py`, `9224dab`): when a bare target name matches exactly one seed entity,
the edge resolves to that seed entity instead of following import order. This is a build-time
decision, made once when the graph is constructed — not something re-evaluated per question. Each
`EdgeTarget` still records both `resolved_by` (`"seed_layer"` or `"import_order"`) and
`import_order_id` (what strict CDM resolution would have picked), purely so the override stays
inspectable and reviewable later, not because anything at runtime chooses between the two. Scoped
narrowly: applied uniformly, it changed **45 targets, none of them audit edges** (verified, not
assumed); a name written with an explicit moniker is never overridden; and if two or more seed
entities ever shared the same bare name, import order still decides (this doesn't occur in the
current corpus scope, so it's undefended — see Known limitations).

### Near-miss ranking

The first version of `relations_between()`'s near-miss fallback returned *every* non-audit edge
an entity had. For Contact asked about Organization, that meant all 14 of Contact's non-audit
edges went into the prompt as "near misses" — burying the two actually relevant ones (`employer`,
`parentCustomer`) in noise, and the model missed both.

An embedding-similarity ranking of each candidate edge's text against the query was tried first
and rejected on measurement, not assumption: it was too unreliable on short attribute-name text.
`employer` shares no words and no obvious surface-level meaning with "Organization" to a
general-purpose embedding model, and it never ranked first across three different query phrasings
tested. What shipped instead is a structural signal: each candidate is scored on lexical overlap
(does the missing entity's name appear in the edge's own attribute or target name) *and* whether
its target is one of the graph's recurring polymorphic "party" types — Account or Contact, the
closest structural analog this schema has to a general counterparty concept. This is why
`employer` is correctly kept despite zero textual overlap with "Organization": the match isn't on
wording, it's on structural role. Zero-scoring candidates are dropped outright, survivors capped
at `NEAR_MISS_LIMIT = 5`. This cut Contact's near-miss set from 14 to 3 for the Organization case.

### Three chunk types, and why attribute chunks were added

Entity chunks (51) and relationship chunks (92) cover the obvious cases: "what is X" and "how does
X relate to Y." A third, attribute chunk type (33, after excluding audit and standard fields — 176
total) was added after a specific hole surfaced in evaluation: *"What can the `regardingObject`
attribute point to?"* returned "not in context," even though the answer existed in the graph. The
entities `regardingObject` can point to were present and fine; the problem was that
`regardingObject` is *declared by* `CampaignResponse`, and `CampaignResponse` was never one of
this project's seed entities — only referenced as a side character by other entities. Only seed
entities get their fields turned into real graph edges (a documented structural limit of
`build_graph()`), so `regardingObject` never became an edge at all, and since the question never
names `CampaignResponse` itself, only the attribute, vector search over entity/relationship chunks
had nothing to find.

Attribute chunks fix this by being built from *every* entity's resolved fields directly — seed or
not — and keyed by the attribute's own name, so a question naming only the attribute, with no
entity name at all, can still find it (`Graph.attribute_detail()`, `router.detect_attribute()`).
Before shipping "one chunk per attribute name" as safe, every one of the corpus's 45 candidate FK
names was checked for a real collision — the same name mapping to different target-entity sets on
two different entities — and none was found; a synthetic test pins the (currently unreached) merge
behavior for a future maintainer if that ever changes.

### The single-entity direct-lookup split, and the "core attributes of Account" ambiguity

This is the most interesting failure the project surfaced, and it took three iterations to get
right.

**The initial problem.** "What are the core attributes of the Account entity?" and "What does the
banking Account inherit from?" both answered wrong or incompletely, even though vector search
*did* retrieve an Account chunk — because an `EntityChunk`'s text is deliberately summarized for
embedding similarity (own attributes in full, inherited attributes truncated to "+N more," only
the immediate parent named). That summary is the right shape for *finding* the entity via search;
it's the wrong shape for *answering* about it once found, because the details a detailed question
needs were never in the chunk to begin with. The fix: when a question names exactly one known
entity, `router.py` builds context from `Graph.entity_detail()` instead — the full, untruncated
own/inherited/standard attribute lists (type and description each), the complete ancestor chain,
and every non-audit relationship in both directions. This is the same data `GET /entities/{name}`
serves, and the same underlying principle as the graph-not-chunks decision above, applied a second
time: once you know exactly which entity a question is about, stop trusting the compressed,
search-optimized version of it and go get the real thing.

**The ambiguity this exposed.** Account (banking)'s real ancestor chain genuinely includes a layer
literally named **Core** (`Account (Core)`). So "core attributes" has two legitimate readings that
are both textually present in the real data: the everyday English sense ("the main, essential
attributes") and the literal technical sense (the attributes inherited from the ancestor named
Core). This is a genuine, unavoidable lexical collision, not a retrieval bug — the correct,
complete, first-priority context was already being retrieved when this surfaced.

Two fixes were tried and rejected, with evidence, before the one that worked:

1. **Strengthen the section header's wording** to explicitly warn against exactly this kind of
   collision. Did not work: the model still occasionally resolved the question toward the
   Core-layer reading, even with the correct data and a direct, explicit warning both present and
   first in context (`0641529`). The lesson here isn't that the instruction was too weak —
   asking a model to notice a plausible alternative reading and then silently suppress it in
   favor of another is an inherently unreliable kind of instruction to give, regardless of how
   strongly it's worded.
2. **Filter the other layers' duplicate Account chunks out of secondary vector search**, so the
   wrong `Account (Core)` chunk couldn't compete at all, plus enrich `entity_detail`'s
   inherited-attribute rendering with full type and description. This removed the *external*
   wrong chunk, but the model then found the same ambiguity *inside* the single correct chunk it
   was already reading, since "Core" is a real ancestor name sitting right there in its own parent
   chain. (A query-augmentation alternative — appending the resolved entity name to the search
   query — was measured via real embedding distances first and found to only narrow, never flip,
   the ranking against the same-named duplicate, so it wasn't pursued as the primary fix.)
   (`6caff10`)

**What actually worked**: detect the lexical collision itself — a whole-word, case-insensitive
match between a word in the question and a real ancestor-layer label in the resolved entity's own
parent chain — and, only when a genuine collision is found, instruct the model to disclose the
ambiguity explicitly and answer **both** readings, rather than silently picking one
(`router._ancestor_layer_collision`, `da23982`). This is not a stronger version of attempt 1's
instruction; it's a different kind of ask entirely. Attempts 1 and 2 both required the model to
notice an alternative reading and then suppress it while still holding onto it enough to
correctly avoid it — a genuinely hard thing to do reliably regardless of wording. Asking the model
to notice the ambiguity and be transparent about both readings, instead of resolving it
unilaterally, turned out to be something it could do reliably. Scoped to fire only on a genuine
whole-word collision against the entity's real ancestor names, not a hardcoded word list; verified
the system prompt is byte-identical on non-colliding questions, so no other question's behavior
changed. Verified against a real API call: the model now opens with an explicit ambiguity note,
then answers both the Core-layer reading and the full-attribute-set reading in full (see
Evaluation, Q1).

### Multi-hop traversal — a deliberately last-resort fallback

`relations_between()` (direct edge, then near-miss) only looks one hop out from each named
entity. A genuine multi-hop question — "what's the path from Collateral to Bank?" — has a real
3-hop answer (`Collateral → FinancialProduct → Branch → Bank`) that neither a direct-edge check
nor a near-miss search would ever find. This was initially scoped out and documented as a known
limitation rather than built under time pressure; it was added afterward, once there was time to
build and verify it properly without risking anything already working.

`Graph.find_path()` does breadth-first search over the graph's non-audit edges. This is a
deliberate choice, not a simplification: the graph is unweighted — every edge is equally
meaningful, there's no real notion of one connection being "closer" or "cheaper" than another —
so a shortest-*weighted*-path algorithm like Dijkstra's would degenerate to exactly the same
result as BFS while adding complexity the data doesn't need.

The more important decision is *where* this fits in the router, and it's deliberately last, not
first, for a specific reason: a direct edge, a near-miss, and a multi-hop path are three
progressively weaker kinds of claim, not three attempts at the same question. A direct edge is the
strongest possible answer. A near-miss isn't claiming a relationship exists between the two named
things at all — it's a redirect to something more useful. A multi-hop path is a real, provable
connection, but an indirect one. If multi-hop were tried first, or without the direct/near-miss
checks ahead of it, it would very likely find *some* technically-real but meaningless path for
almost any pair of entities — especially through audit-edge chains — and present it as if it were
a meaningful answer. `find_path()` only runs when `relations_between()` has already confirmed
*both* zero direct edges and zero near-misses on both sides, so it only ever surfaces when it's
genuinely the best available answer, never as a shortcut that could paper over the honest "no
relationship" case. This is verified directly, not just asserted: Contact/Organization — the
project's headline no-relationship case — is confirmed to never even call `find_path()` (near-miss
edges exist for Contact, so the fallback condition is never met), and a test fails loudly if that
ever changes.

### Embedding model choice

`BAAI/bge-small-en-v1.5` via `sentence-transformers`, run locally rather than through a hosted
embeddings API: it's offline (no per-call cost, no network dependency at query time, can't be
deprecated or rate-limited out from under the project the way a hosted endpoint could — this
happened to the LLM choice below, mid-project), 384-dim output, 512-token max sequence length.
Real chunk sizes were measured against that limit rather than assumed safe: across all 176 indexed
chunks, mean length is **~103 tokens**, the longest is **340 tokens** (an entity chunk) —
comfortably under 512, verified with the model's own tokenizer, not a word-count heuristic. Any
future chunk that *did* exceed the limit is logged loudly before encoding rather than silently
truncated (`embeddings.embed()`).

One BGE-specific detail from its model card, easy to miss: retrieval quality improves when a fixed
instruction string (`"Represent this sentence for searching relevant passages: "`) is prepended to
*queries* only, never to the indexed passages being searched through. This build's model ships
empty default prompts, so `sentence-transformers` won't add it automatically — missing this would
have silently degraded search quality with no error to point at the cause, so `embed(...,
is_query=True)` adds it explicitly on the query side only.

### LLM choice

Groq, running `openai/gpt-oss-120b` — chosen for a genuinely free tier (no card required) and
inference speed (server-side completion time was consistently well under 2s even for large
answers in evaluation; see below). The original model choice (`llama-3.3-70b-versatile`) turned
out to already be deprecated on the free tier by the time it was actually tried; the working model
name was found by querying Groq's own API for the account's currently available models directly,
rather than trusting documentation that could already be stale.

All Groq SDK usage is isolated behind `cdm_rag/llm_client.py` — the only module in the codebase
that imports the `groq` package at all (enforced by a test) — so swapping providers later (OpenAI,
Anthropic, a local server) means changing one file; nothing else ever sees a provider-specific
type. This isolation exists specifically because of the model-deprecation experience above.

## 4. Known limitations

- **Near-miss enumeration wording gap.** The model sometimes summarizes near-misses in prose
  rather than mechanically naming every one, especially when a strong explanatory note is also
  present. A real example from evaluation (Q2, "How does a Contact relate to an Organization?"):
  the context correctly contained three ranked near-miss edges (`employer`, `master`,
  `parentCustomer`), but the answer stated "there is no indirect near-miss path either" instead of
  naming them. Two prompt-engineering attempts (an explicit enumeration rule, then a few-shot
  example) did not fix this — the few-shot example made it *worse*, causing the model to lean
  entirely into the explanatory note and stop naming near-misses at all. Documented as a known
  limitation rather than pursued further (`d3c9b72`): the answers remain accurate and grounded,
  just not always itemized in the requested format.
- **Attribute-name collisions are undefended, not just untested.** Verified zero real collisions
  across all 45 candidate FK names in the current corpus scope, but if two entities ever declared
  the same attribute name against genuinely different target types, the current design would
  merge them into one (potentially misleading) attribute chunk. A synthetic test pins this
  behavior for a future maintainer; it isn't guarded against, because it doesn't occur here.
- **Multi-hop traversal is single-shortest-path, not exhaustive.** `find_path()` (BFS) returns
  one shortest chain of edges, capped at `max_hops=4` by default, and only ever runs as a
  last-resort fallback after `relations_between()` finds nothing (see Design decisions above). It
  does not enumerate every possible path, and a real connection that requires more than 4 hops
  would not be found — this cap is a deliberate boundary against unbounded search, not a
  discovered limit of the data.
- **Groq free-tier rate limits are real and were hit during testing** — both a per-minute (TPM)
  and a separate, harder daily (TPD) ceiling. Observed directly: **8,000 TPM**, and
  **~200,000 TPD** (one test run hit a 429 at 197,911/200,000 tokens used, quoting a 33m26s wait
  for the next request). A single large-context question can consume a meaningful fraction of the
  per-minute budget on its own — the Q1 ambiguity answer alone used ~5,265 prompt tokens plus up
  to ~2,300 completion tokens in one call, roughly 90% of the entire per-minute budget by itself —
  so heavy consecutive use (a full evaluation run) can and did exhaust the daily quota mid-run.
  Mitigations in `llm_client.py`: a 15-second client-side timeout with one retry (the exact
  backoff comes from Groq's own quoted retry-after time, not a blind guess), and a
  `max_tokens=3000` completion cap sized with real margin above the largest completion actually
  observed (~2,298 tokens — an earlier, tighter 2,200 cap was measured to truncate a real answer
  mid-sentence before being corrected). These bound worst-case behavior — fail fast and
  predictably rather than hang — but do not remove the underlying account-level limit. A live demo
  session asking many questions in a short window should expect this.
- **Corpus scope is banking + referenced common objects, not the full CDM.** Healthcare, Retail,
  and other CDM accelerator domains are out of scope. The system is expected — and was verified in
  adversarial testing — to say so plainly if asked, rather than guessing as if it had that data.

## 5. Evaluation

`scripts/eval.py` runs a fixed set of questions against the real running pipeline (real graph,
real Chroma index, real Groq API) — deliberately not part of the `pytest` suite, since it's
checking a live, non-deterministic API rather than fixed-input behavior. Each question gets either
an automated PASS/FAIL from a check tied to the real graph's own ground truth (e.g. "does the
context cite the real Branch→Bank edge as a direct edge, not a near-miss," computed from the graph
itself, never hand-typed), or is marked **NEEDS HUMAN REVIEW** when the judgment genuinely isn't
mechanical (e.g. "is this free-form entity comparison actually useful?"). A question is never
given a faked verdict just to report a cleaner-looking pass rate — a check that can't honestly
verify correctness is more useful flagged than pretending to be automated.

`eval_results.md` is gitignored and regenerated by running the script; it depends on a live API
key and isn't deterministic, so it isn't committed as source. The most recent complete run of the
original 10-question set:

**9/10 automatically verified (9 passed, 0 failed), 1 needing manual review.**

| Q | Question | Result | Total (client) | Prompt tok | Completion tok |
|---|---|---|---|---|---|
| 1 | Core attributes of the Account entity | PASS | 7.9s | 5265 | 2298 |
| 2 | How does Contact relate to Organization | PASS | 1.1s | 1197 | 348 |
| 3 | How does Branch relate to Bank | PASS | 1.1s | 953 | 360 |
| 4 | What entities reference Bank | PASS | 0.8s | 1776 | 274 |
| 5 | What can regardingObject point to | PASS | 0.7s | 904 | 226 |
| 6 | What does banking Account inherit from | PASS | 1.8s | 4763 | 347 |
| 7 | What is a CryptoWallet entity | PASS | 0.9s | 1493 | 157 |
| 8 | Difference between Account and Contact | NEEDS REVIEW | 2.0s | 1003 | 804 |
| 9 | What entities exist in the banking model | PASS | 1.6s | 1458 | 647 |
| 10 | How does FinancialProduct relate to a customer | PASS | 1.3s | 3184 | 346 |

Two real excerpts, so this section has evidence a reader can check without an API key:

**Q1** (the ambiguity fix in action, see Design decisions above) opens:

> **Ambiguity note:**
> The phrase *"core attributes of the Account entity"* could be read in two ways:
> 1. **Core‑layer attributes** – the attributes that Account (banking) inherits from its
>    **Account (Core)** ancestor.
> 2. **All attributes of the Account entity** – the complete set of attributes (both its own
>    banking‑layer attributes and everything it inherits).
>
> Below are both interpretations, each tied to the schema information provided.

**Q2** ("How does a Contact relate to an Organization?") — a genuine no-relationship answer,
and also a live example of the near-miss enumeration gap noted above:

> There is **no direct relationship** between **Contact (banking)** and **Organization** in the
> provided schema.
>
> The only information about **Organization** is the note in chunk [5], which states that
> "Organization has no non-audit (business) relationships anywhere in this graph… it functions
> here as infrastructure/tenant metadata, not a business entity with its own relationships."

A second, harder eval set (Q11–Q20 — false premises, out-of-corpus scope questions, meta-questions
about the system's own confidence and provenance, and Q16, a genuine multi-hop question) was added
to `scripts/eval.py`. Q16 ("What's the path from Collateral to Bank?") was run live against the
real API once `find_path()` existed and passed: the model opened with "no direct relationship,"
then correctly named the full real 3-hop chain (`financialProduct → branch → bank`) with every
attribute name right. The full Q11–Q20 run has not completed end to end — it is blocked mid-run on
the Groq daily quota described above.

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

Run the tests (161 tests; anything needing the real corpus skips cleanly with a clear message if
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

## 8. How this was built

This was developed with Claude Code assisting on implementation under my direction: I made the
architecture decisions, verified findings against the real corpus at each step rather than
trusting a plausible-sounding explanation, and designed the evaluation approach. Several sections
above describe a case where an initial hypothesis was tested, measured, and found wrong before the
real fix was identified — the audit-edge scope numbers, the near-miss ranking approach, and the
Account/Core ambiguity fix all went through this process. That's a genuine record of how this was
built, not retrofitted narrative, and it's documented here on the same terms as everything that
worked the first time.
