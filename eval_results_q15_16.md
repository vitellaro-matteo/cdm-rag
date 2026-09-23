# cdm-rag evaluation report

**Summary: 1/2 automatically verified (0 passed, 1 failed), 1 needing manual review.**

## Performance profile

All times in seconds. The embedding model is warmed once before question 1 (see `main()`), so "Vector query" reflects steady-state per-query cost, not the one-time ~8-10s model-load cost a fresh process pays on its first ever query.

| Q | Detection | Graph lookup | Vector query | Context asm. | Prompt build | LLM call (client) | Total (client) | Prompt tok | Completion tok | Queue+Prompt (server) | Completion (server) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 0.002 | 0.000 | 2.413 | 0.000 | 0.000 | 1.968 | 4.383 | 1342 | 640 | 0.294 | 1.346 |
| 16 | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 1.412 | 1.438 | 1103 | 542 | 0.259 | 1.123 |

## Q15: Which entities does Bank connect to, directly or indirectly, going up to two steps away?

**Status:** NEEDS HUMAN REVIEW  (**4.4s**)

**Answer:**

> **Direct (one‑step) connections**
> 
> - **Branch (banking)** – Branch refers to Bank via the attribute **bank** (foreign‑key bankId).  
>   *Source:* Bank’s relationship list – “is referred to by Branch … via attribute bank (foreign key bankId)”【2】; attribute definition confirms the reference【5】.
> 
> - **Syndicates (banking)** – Syndicates refers to Bank via the attribute **bank** (foreign‑key bankId).  
>   *Source:* Bank’s relationship list – “is referred to by Syndicates … via attribute bank (foreign key bankId)”【2】; attribute definition confirms the reference【5】.
> 
> **Indirect (two‑step) connections**
> 
> The only entities that are one step away from Bank are **Branch** and **Syndicates**.  
> The provided context does **not** contain any further relationships (incoming or outgoing) for either Branch or Syndicates, so no additional entities are reachable in a second step.
> 
> **Summary**
> 
> - Directly connected entities: **Branch**, **Syndicates**.  
> - No indirect (two‑step) connections are described in the given context.

**Sources / context** (8 items):

- *(section header)* Full schema lookup for Bank -- this is the authoritative, complete record for the specific entity named in this question. Prefer it over anything below, even if a word in the question happens to coincidentally match a layer name, attribute name, or other label appearing elsewhere in this context (a coincidental wording match does not mean a different entity or layer was meant):
- chunk_type=entity_detail, source=entity_lookup, entity=Bank
- *(section header)* Additional context from search:
- chunk_type=relationship, source=vector_search, attribute=qualifyingOpportunity
- chunk_type=attribute, source=vector_search, attribute=bank
- chunk_type=relationship, source=vector_search, attribute=originatingLead
- chunk_type=entity, source=vector_search, entity=LeadToOpportunitySalesProcess
- chunk_type=relationship, source=vector_search, attribute=Contact

**Automated checks:** none -- read the answer above and judge for yourself.

## Q16: What's the path from Collateral to Bank?

**Status:** FAIL  (**1.4s**)

**Answer:**

> There is **no direct relationship** between **Collateral** and **Bank** in the CDM schema.
> 
> **Indirect (multi‑hop) path**
> 
> The schema does provide a three‑hop connection:
> 
> 1. **Collateral → FinancialProduct** – via the attribute **`financialProduct`** (foreign key `financialProductId`).  
>    *Context:* “Collateral has a many‑to‑one relationship to FinancialProduct… via attribute `financialProduct`.”
> 
> 2. **FinancialProduct → Branch** – via the attribute **`branch`** (foreign key `branchId`).  
>    *Context:* “The attribute `branch` is a reference (declared on FinancialProduct (banking)) that points to Branch.”
> 
> 3. **Branch → Bank** – via the attribute **`bank`** (foreign key `bankId`).  
>    *Context:* “The attribute `bank` is a reference (declared on Branch (banking) …) that points to Bank.”
> 
> So the full indirect route is:
> 
> **Collateral** —(`financialProduct`)-> **FinancialProduct** —(`branch`)-> **Branch** —(`bank`)-> **Bank**.
> 
> This matches the multi‑hop path described in the context (3 hops: Collateral → FinancialProduct → Branch → Bank).

**Sources / context** (8 items):

- *(section header)* Direct schema lookup for Collateral and Bank (the highest-confidence facts for this question):
- chunk_type=multi_hop_path, source=multi_hop_path
- *(section header)* Additional context from search:
- chunk_type=entity, source=vector_search, entity=Collateral
- chunk_type=relationship, source=vector_search, attribute=financialProduct
- chunk_type=attribute, source=vector_search, attribute=financialProduct
- chunk_type=attribute, source=vector_search, attribute=bank
- chunk_type=attribute, source=vector_search, attribute=branch

**Automated checks:**
- [ ] answer does not falsely deny any path exists (real path: Collateral -> FinancialProduct -> Branch -> Bank) -- no direct relationship
- [x] answer's path includes FinancialProduct (the real first hop from Collateral)
- [x] answer's path includes Branch (the real connector to Bank -- without it the path doesn't reach Bank)
