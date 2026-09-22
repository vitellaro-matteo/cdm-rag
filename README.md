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
invented one. It's a FastAPI app with three endpoints:

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
Router (cdm_rag/router.py)
  ├─ two known entities named  → Graph.relations_between()  (direct edges + ranked near-misses)
  ├─ exactly one known entity  → Graph.entity_detail()       (full authoritative record)
  ├─ one known FK attribute    → Graph.attribute_detail()    (independent of its declaring entity)
  └─ none of the above         → vector search only
     (a structural match is always followed by top-k vector search as secondary context)
        │
        ▼
Generation (cdm_rag/generate.py + cdm_rag/llm_client.py — Groq, openai/gpt-oss-120b)
```

A reader should be able to stop here with the shape of the system; the rest of this document is
about *why* each stage looks the way it does, not what it does.

## 3. Design decisions

### Graph, not flat documents

The schema is parsed into a real entity/relationship graph (`cdm_rag/graph.py`), not chunked as
flat text and left to embedding similarity alone. The concrete reason: "How does a Contact relate
to an Organization?" is not answerable by finding the most similar chunk — there is no direct
edge between Contact and Organization at all, and the correct answer requires knowing that
*(a)* no such edge exists, and *(b)* Contact does connect to other entities (Account, via
`employer` and `parentCustomer`) that a similarity search might not rank highly for a query that
never mentions Account. `Graph.relations_between(name_a, name_b)` answers this exhaustively by
construction: every direct edge between the two named entities (both directions, including
audit), and when there are none, every one of each entity's own near-miss edges, ranked. This
was added after a manual smoke test showed the model correctly reporting the `employer` near-miss
but silently missing `parentCustomer` — not a prompt defect, a context defect: vector top-k only
returns what embeds closest to the query, and a real near-miss edge can simply lose that race.

### The audit-edge finding

CDM entities carry a set of boilerplate FK attributes present on nearly every entity
(`createdBy`, `modifiedBy`, `createdOnBehalfBy`, `modifiedOnBehalfBy`, `ownerId`, `owningUser`,
`owningTeam`, `owningBusinessUnit`, `transactionCurrencyId`, `organizationId` —
`AUDIT_ATTRIBUTES` in `cdm_rag/relationships.py`), which carry no domain meaning. Measured
directly against the real corpus: across the whole `applicationCommon` manifest (effectively the
full CDM, not just banking), audit edges dominate — **5,625 of 5,977 relationships (94.1%)**.
Scoped down to this project's actual in-scope graph (the banking seed entities plus everything
reachable from them), the picture is very different: **41 of 133 edges (30.8%) are audit**, the
rest genuine business relationships — most of the manifest-wide audit boilerplate belongs to
entities the banking model never reaches. Audit edges are excluded from `relations_between()`'s
default edge-following (`outgoing()`/`incoming()` default to `include_audit=False`) so they don't
crowd out real relationships, but they are never dropped from the graph itself — only flagged
(`Edge.is_audit`) — and `relations_between()` always checks them first when asking "is there any
link at all," because sometimes an audit-only relationship *is* the honest answer. Organization is
the concrete case: it has 11 incoming audit `organizationId` edges and nothing else, so its
`relations_between()` note states plainly that it has no business relationships to explain,
rather than the API staying silent about why none was found.

### Seed-layer entity resolution

CDM's own import-order resolution turned out to disagree with what a banking document actually
means. A bare reference to `"Account"` written in a banking entity file resolves, under strict
CDM import order, to the *Core*-layer Account — not the banking one — because of how imports are
declared. Before this was caught, that meant **banking Account and banking Contact had zero
incoming edges**: every other banking entity's reference to "Account" or "Contact" was silently
landing on the wrong, generic layer. The fix (`cdm_rag/graph.py`, `9224dab`): when a bare target
name matches exactly one seed entity, the edge resolves to that seed entity instead of following
import order; each `EdgeTarget` records both `resolved_by` (`"seed_layer"` or `"import_order"`)
and `import_order_id` (what strict CDM resolution would have picked), so the deviation stays
inspectable rather than silently applied. Scoped narrowly: applied uniformly, it changed
**45 targets, none of them audit edges** (verified, not assumed); a name written with an explicit
moniker is never overridden; and if two or more seed entities ever shared the same bare name,
import order still decides (this doesn't occur in the current corpus scope, so it's undefended,
matching how the rest of this project treats an edge case that isn't real yet — see Known
limitations).

### Near-miss ranking

The first version of `relations_between()`'s near-miss fallback returned *every* non-audit edge
an entity had. For Contact asked about Organization, that meant all 14 of Contact's non-audit
edges went into the prompt as "near misses" — burying the two actually relevant ones (`employer`,
`parentCustomer`) in noise, and the model missed both. An embedding-similarity ranking of each
candidate edge's text against the query was tried first and rejected on measurement: it was too
unreliable on short attribute-name text — `employer` never ranked first across three different
query phrasings tested. What shipped instead is a structural signal: each candidate is scored on
lexical overlap (does the missing entity's name appear in the edge's own attribute or target
name) *and* whether its target is one of the graph's recurring polymorphic "party" types (i.e.
Account or Contact — the closest structural analog this schema has to a general counterparty
concept). Zero-scoring candidates are dropped outright, survivors capped at `NEAR_MISS_LIMIT = 5`.
This cut Contact's near-miss set from 14 to 3 for the Organization case, and is what catches a
near-miss with *no* textual overlap at all — `Contact.employer → Account` shares no words with
"Organization," but is structurally a party-type edge.

### Three chunk types, and why attribute chunks were added

Entity chunks (51) and relationship chunks (92) cover the obvious cases: "what is X" and "how does
X relate to Y." A third, attribute chunk type (33, after excluding audit and standard fields — 176
total) was added after a specific hole surfaced in evaluation: *"What can the `regardingObject`
attribute point to?"* returned "not in context," even though the answer existed in the graph.
`regardingObject` is declared on `CampaignResponse`, which is never a seed entity — and only seed
entities get their FKs expanded into edges (a documented structural limit of `build_graph()`) — so
vector search never surfaced it, because the question never names `CampaignResponse` itself, only
the attribute. Attribute chunks are keyed by FK attribute name and looked up independently of
which entity happens to declare or inherit them (`Graph.attribute_detail()`,
`router.detect_attribute()`). Before shipping "one chunk per attribute name" as safe, every one of
the corpus's 45 candidate FK names was checked for a real collision — the same name mapping to
different target-entity sets on two different entities — and none was found; a synthetic test
pins the (currently unreached) merge behavior for a future maintainer if that ever changes.

### The single-entity direct-lookup split, and the "core attributes of Account" ambiguity

This is the most interesting failure the project surfaced, and it took three iterations to get
right.

**The initial problem.** "What are the core attributes of the Account entity?" and "What does the
banking Account inherit from?" both answered wrong or incompletely, even though vector search
*did* retrieve an Account chunk — because an `EntityChunk`'s text is deliberately summarized for
embedding similarity (own attributes in full, inherited attributes truncated to "+N more," only
the immediate parent named). That's the right shape for *finding* the entity; it's the wrong shape
for *answering* about it once found. The fix: when a question names exactly one known entity,
`router.py` builds context from `Graph.entity_detail()` instead — the full, untruncated
own/inherited/standard attribute lists (type and description each), the complete ancestor chain,
and every non-audit relationship in both directions. This is the same data `GET /entities/{name}`
serves.

**The ambiguity this exposed.** Account (banking)'s real ancestor chain genuinely includes a layer
literally named **Core** (`Account (Core)`). So "core attributes" is a real, unavoidable lexical
collision in this schema, not a retrieval bug: given the correct, complete, first-priority
`entity_detail` chunk, the model still read "core" as "the Core-layer inherited attributes" rather
than "the entity's own essential attributes" — because that reading is factually true of the real
data.

Two fixes were tried and rejected, with evidence:

1. **Strengthen the section header's wording** to explicitly warn against exactly this kind of
   collision ("...even if a word in the question happens to coincidentally match a layer name...").
   Did not work: the model still occasionally prioritized its own reading of the question's
   wording over an explicit instruction, even with the correct data and a direct warning both
   present and first in context (`0641529`).
2. **Filter the other layers' duplicate Account chunks out of secondary vector search**, so the
   wrong `Account (Core)` chunk couldn't compete at all, plus enrich `entity_detail`'s
   inherited-attribute rendering with full type/description. This removed the *external* wrong
   chunk, but the model then found the same ambiguity *inside* the single correct chunk it was
   already reading, since "Core" is a real ancestor name sitting right there in the parent chain.
   (A query-augmentation alternative — appending the resolved entity name to the search query —
   was measured via real embedding distances first and found to only narrow, never flip, the
   ranking against the same-named duplicate, so it wasn't pursued as the primary fix; the
   structural filter was the one of the two that actually removed the confusable input, for its
   own narrower purpose.) (`6caff10`)

**What actually worked**: detect the lexical collision itself — a whole-word, case-insensitive
match between a word in the question and a real ancestor-layer label in the resolved entity's own
parent chain — and, only when a genuine collision is found, instruct the model to disclose the
ambiguity explicitly and answer **both** readings, rather than silently picking one
(`router._ancestor_layer_collision`, `da23982`). This isn't a retrieval fix — by this point the
context was already provably correct and complete — it accepts that the ambiguity is real and asks
the model to be transparent about it instead of resolving it unilaterally. Scoped to fire only on
a genuine whole-word collision against the entity's real ancestor names (not a hardcoded word
list); verified the system prompt is byte-identical on non-colliding questions, so no other
question's behavior changed. Verified against a real API call: the model now opens with an
explicit "Ambiguity note," then answers both the Core-layer reading and the full-attribute-set
reading in full (see Evaluation, Q1).

### Embedding model choice

`BAAI/bge-small-en-v1.5` via `sentence-transformers`, run locally rather than through a hosted
embeddings API: it's offline (no per-call cost, no network dependency at query time, can't be
deprecated or rate-limited out from under the project the way a hosted endpoint could), 384-dim
output, 512-token max sequence length. Real chunk sizes were measured against that limit rather
than assumed safe: across all 176 indexed chunks, mean length is **~103 tokens**, the longest is
**340 tokens** (an entity chunk) — comfortably under 512, verified with the model's own tokenizer,
not a word-count heuristic. Any future chunk that *did* exceed the limit is logged loudly before
encoding rather than silently truncated (`embeddings.embed()`). One BGE-specific detail from its
model card, easy to miss: retrieval quality improves when a fixed instruction string
(`"Represent this sentence for searching relevant passages: "`) is prepended to *queries* only,
never to the indexed passages — this build's model ships empty default prompts, so
`sentence-transformers` won't add it automatically, so `embed(..., is_query=True)` does it
explicitly.

### LLM choice

Groq, running `openai/gpt-oss-120b` — chosen for a genuinely free tier (no card required) and
inference speed (server-side completion time was consistently well under 2s even for large
answers in evaluation; see below). All Groq SDK usage is isolated behind
`cdm_rag/llm_client.py` — the only module in the codebase that imports the `groq` package at all
(enforced by a test) — so swapping providers later (OpenAI, Anthropic, a local server) means
changing one file; nothing else ever sees a provider-specific type.

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
- **Reverse-relationship and multi-hop support is basic.** `relations_between()` looks one hop of
  near-miss out from each named entity; it does not chain hops. A genuine multi-hop question
  (e.g. "what's the path from Collateral to Bank?" — a real 3-hop path exists via `FinancialProduct`
  and `Branch`, with no direct edge) relies entirely on whatever the model can piece together from
  retrieved chunks, not on graph traversal built for the purpose. `entity_detail()` does expose an
  entity's own full incoming/outgoing edge list, so one-hop reverse lookups ("what references
  Bank?") work well; anything beyond that is not purpose-built.
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
mechanical (e.g. "is this free-form entity comparison actually useful?") — those never get a faked
verdict.

`eval_results.md` is gitignored and regenerated by running the script (`python scripts/eval.py`);
it depends on a live API key and isn't deterministic, so it isn't committed as source. The most
recent complete run of the original 10-question set:

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

A second, harder eval set (Q11–Q20 — false premises, out-of-corpus scope questions, a genuine
multi-hop path question, and meta-questions about the system's own confidence/provenance) has
also been added to `scripts/eval.py`, with ground-truth checks derived the same way (against the
real graph, verified before the check was written, not assumed from the question's own phrasing).
As of this writing that run has not completed — it is blocked mid-run on the Groq daily quota
described above.

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

Run the tests (152 tests; anything needing the real corpus skips cleanly with a clear message if
`CDM_CORPUS_PATH` isn't set to a real checkout):

```bash
pytest
```

Run the evaluation script (real API calls, real time — see Evaluation above):

```bash
python scripts/eval.py          # all 20 questions
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
architecture decisions, verified findings against the real corpus at each step (not just
trusting a plausible-sounding explanation), and designed the evaluation approach. Several sections
above describe a case where an initial hypothesis was tested, measured, and found wrong before the
real fix was identified — that's a genuine record of how this was built, not retrofitted
narrative, and it's documented here on the same terms as everything that worked the first time.
