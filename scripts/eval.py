"""Manual evaluation: runs the 10 assignment questions (Q1-Q10) plus 10 harder, more adversarial
questions (Q11-Q20 -- false premises, out-of-corpus entities, multi-hop paths, meta-questions
about the system's own confidence/provenance/scope) against the REAL pipeline (real graph, real
Chroma index, real Groq API) and writes a markdown report to eval_results.md.

Not part of the automated test suite: it costs real API calls, takes real time (LLM latency x
10), and its pass/fail judgments are deliberately loose sanity checks against a running system,
not the kind of fixed-input unit test the rest of this project uses. Run by hand:

    python scripts/eval.py

Requires GROQ_API_KEY/LLM_MODEL (see .env.example) and the real corpus (CDM_CORPUS_PATH or its
default). Reuses the persisted index at store.DEFAULT_PERSIST_DIR if one exists, else builds it.

An optional 1-indexed inclusive range resumes a subset instead of all 20 (useful when a prior
run got partway through and re-asking already-answered questions would waste scarce API quota):

    python scripts/eval.py 11 20   # only Q11-Q20; written to eval_results_q11_20.md, not
                                    # eval_results.md, so it never clobbers a full run's report.

Each question gets either an automated PASS/FAIL (from checks tied to the real graph's ground
truth -- e.g. "does the context cite the real Branch->Bank edge as a direct_edge, not a
near_miss") or is marked NEEDS HUMAN REVIEW when the judgment genuinely isn't mechanical (is a
free-form comparison actually useful?) -- those never get a faked verdict, just the full answer
printed for you to read.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cdm_rag.config import corpus_path  # noqa: E402
from cdm_rag.embeddings import load_model  # noqa: E402
from cdm_rag.graph import Graph, banking_seeds, build_graph  # noqa: E402
from cdm_rag.inheritance import Corpus  # noqa: E402
from cdm_rag.store import open_or_build_index  # noqa: E402
from cdm_rag import router  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parents[1] / "eval_results.md"
#: Every real-API QuestionResult ever produced by any run (full or range-limited), keyed by
#: question number, accumulated across invocations -- lets `--report` assemble one full report
#: out of separate runs without re-asking a question a prior run already answered for real. See
#: load_raw_results/save_raw_results.
RAW_RESULTS_PATH = Path(__file__).resolve().parents[1] / "eval_results_raw.json"


@dataclass
class Check:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class QuestionResult:
    number: int
    question: str
    answer: str
    context: list[dict[str, Any]]
    elapsed: float
    timing: dict[str, Any] = field(default_factory=dict)  # stage breakdown, see router.answer's `timing` param
    checks: list[Check] = field(default_factory=list)
    manual: bool = False  # True: no automated verdict is claimed for this question at all

    @property
    def passed(self) -> bool | None:
        """True/False if automated, None if this question is manual-review-only."""
        return None if self.manual else all(c.passed for c in self.checks)


# --- ground truth, pulled from the real graph, not hand-typed guesses --------------------------


def ground_truth(graph: Graph) -> dict[str, Any]:
    (bank,) = graph.find("Bank")
    entities_referencing_bank = sorted(
        graph.nodes[e.from_id].name for e in graph.incoming(bank.entity_id, include_audit=True)
    )
    banking_entity_names = sorted({n.name for n in graph.nodes.values() if n.layer == "banking"})
    return {
        "graph": graph,  # a few Q11-Q20 checks call router's own detection helpers directly
        "account_own_attrs": {"annualReviewDate", "availableLimit", "daysPastDue"},
        "entities_referencing_bank": entities_referencing_bank,
        "account_ancestor_layers": {"CRM base", "Foundation", "Core", "CdsStandard"},
        "regarding_object_targets": {
            "Account", "BookableResourceBooking", "BookableResourceBookingHeader", "Campaign",
            "CampaignActivity", "Contact", "KnowledgeArticle", "KnowledgeBaseRecord", "Lead", "QuickCampaign",
        },
        "banking_entity_names": banking_entity_names,
        # Real path, verified directly against the graph before writing check_q16 (not assumed
        # from the question's phrasing): Collateral -> FinancialProduct -> Branch -> Bank. There
        # is no direct Collateral-Bank edge, but a real 3-hop path does exist -- FinancialProduct
        # connects straight to Branch, independent of its separate "customer" edge to
        # Account/Contact, which does NOT lead to Bank.
        "collateral_to_bank_path": ["FinancialProduct", "Branch"],
    }


# --- small helpers over a question's context/answer ---------------------------------------------


def _metas(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item.get("metadata", {}) for item in context]


def has_item(context: list[dict[str, Any]], **wanted: Any) -> bool:
    return any(all(meta.get(k) == v for k, v in wanted.items()) for meta in _metas(context))


def contains_any(text: str, phrases: list[str]) -> str | None:
    """First phrase (case-insensitive) found in ``text``, or None."""
    lowered = text.lower()
    return next((p for p in phrases if p.lower() in lowered), None)


def count_mentions(text: str, names: list[str]) -> list[str]:
    lowered = text.lower()
    return [n for n in names if n.lower() in lowered]


NEGATION_PHRASES = [
    "no such", "does not exist", "doesn't exist", "not found", "not present", "not part of",
    "no entity", "cannot find", "can't find", "couldn't find", "not in the", "isn't in the",
    "not defined", "unable to find", "no information", "not mentioned", "not available",
    "does not contain", "doesn't contain", "no description of", "not named", "no mention of",
    "not listed", "not included", "does not include",
]
# ^ "does not include" added after a real Q7 run: the model correctly declined the fabricated
# CryptoWallet entity ("The provided context does not include any description of a CryptoWallet
# entity... there is no CryptoWallet entity described") but that exact phrasing matched none of
# the phrases above, so check_q7 reported FAIL against a genuinely correct answer -- a check
# false-negative, not a model or pipeline defect. Re-verified against that exact captured answer
# text below (no new API call), not just asserted.

NO_RELATIONSHIP_PHRASES = ["no direct relationship", "no relationship", "does not relate", "not related"]

#: Words an answer is very unlikely to end on if it finished naturally -- a real, observed signal
#: of mid-clause truncation, not a hypothetical: Q11's original real run ended with "...
#: **Relationship (foreign-key) attributes** - the", cut off by the max_tokens cap. Deliberately
#: short and conservative -- see looks_truncated's docstring for why this is a heuristic, not
#: a completeness proof.
DANGLING_ENDING_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with", "by", "for", "is", "are",
    "from", "as", "that", "this", "these", "those", "its", "their",
}


def looks_truncated(text: str) -> str | None:
    """Heuristic only -- NOT a proof of completeness or truncation, and this is a real, named gap
    in what "completeness" meant across this eval script before now: every existing check here
    verifies that some real fact is PRESENT in the answer, never that the answer wasn't simply cut
    off mid-sentence by the max_tokens cap. Q11's original real run passed both of check_q11's
    other checks (a real attribute was named early on) while its answer ended, verbatim, on "...
    **Relationship (foreign-key) attributes** - the" -- genuinely incomplete, and nothing here
    caught it. This checks only how the text ENDS: a dangling connector word (see
    DANGLING_ENDING_WORDS) or no normal closing punctuation/character at all. Returns a short
    description of what looked wrong, or None if the ending looks normal. A complete answer could
    still end in a way this doesn't anticipate, and a truncated one could in principle land on a
    period -- treat a None result as "no truncation signal found", not "provably complete"."""
    stripped = text.rstrip()
    if not stripped:
        return "answer is empty"
    # Strip trailing markdown emphasis/code markers first -- "...as expected.**" is a normal,
    # complete sentence wrapped in bold, not truncated, even though the literal last character is
    # "*". Found via real false positives on this exact fix's first pass (Q6/Q18 both end
    # "<real sentence>.**").
    core = stripped.rstrip("*_`") or stripped
    words = re.findall(r"[A-Za-z']+", core)
    if words and words[-1].lower() in DANGLING_ENDING_WORDS:
        return f"answer ends on a dangling connector word ({words[-1]!r})"
    if core[-1] not in ".!?)]}\"'|":
        return f"answer does not end with normal closing punctuation (ends with {core[-1]!r})"
    return None


OUT_OF_SCOPE_PHRASES = [
    "out of scope", "outside the scope", "don't have", "do not have", "not have access",
    "no information", "not included", "not part of", "only cover", "only covers",
    "limited to", "restricted to", "only the banking", "banking (and", "this schema only",
    "cannot answer", "can't answer", "no data", "haven't been given", "not been given",
    "not able to answer", "i won't be able", "i will not be able", "would not be able",
    "not within", "beyond the scope", "banking-specific", "banking only", "banking domain",
]


# --- per-question checks -------------------------------------------------------------------------
# Each returns a list of Checks. `gt` is ground_truth(graph); `r` is the QuestionResult being built
# (answer/context already filled in).


def check_q1(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # Single-entity questions now route through the full entity_lookup (graph.entity_detail),
    # not the summarized vector-search entity chunk -- see router.detect_single_entity.
    got_lookup = has_item(r.context, source="entity_lookup", entity="Account", layer="banking")
    mentioned = count_mentions(r.answer, sorted(gt["account_own_attrs"]))
    # An "enumerate everything" style question (like Q11's) -- real risk of hitting the
    # max_tokens cap mid-list; see looks_truncated's docstring for what this check can and can't
    # tell you.
    truncated = looks_truncated(r.answer)
    return [
        Check("context has a full entity_lookup for the banking Account (not the Core-layer one)", got_lookup),
        Check(
            "answer names a real Account attribute",
            bool(mentioned),
            f"mentioned: {mentioned}" if mentioned else f"expected one of {sorted(gt['account_own_attrs'])}",
        ),
        Check(
            "answer does not look truncated mid-sentence (heuristic -- see looks_truncated)",
            truncated is None,
            truncated or "",
        ),
    ]


def check_q2(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    got_note = has_item(r.context, chunk_type="note", entity="Organization")
    got_near_miss = any(meta.get("source") == "near_miss" for meta in _metas(r.context))
    said_no_direct = contains_any(r.answer, NO_RELATIONSHIP_PHRASES)
    return [
        Check("context includes the Organization infrastructure note", got_note),
        Check("context includes near-miss edges from Contact", got_near_miss),
        Check("answer states there's no direct relationship", said_no_direct is not None, said_no_direct or "no matching phrase found"),
    ]


def check_q3(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    cited_direct_edge = has_item(r.context, source="direct_edge", attribute="bank")
    falsely_denied = contains_any(r.answer, NO_RELATIONSHIP_PHRASES)
    return [
        Check("context cites the real edge as direct_edge (not near_miss)", cited_direct_edge),
        Check("answer doesn't wrongly deny a relationship", falsely_denied is None, falsely_denied or "no false denial found"),
    ]


def check_q4(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # "Bank" is a single named entity, so this now also gets a full entity_lookup -- which
    # includes Bank's *incoming* edges, i.e. exactly "what references Bank" -- an incidental
    # benefit of the Q1/Q6 fix, not something specially built for this question.
    got_lookup = has_item(r.context, source="entity_lookup", entity="Bank")
    mentioned = count_mentions(r.answer, gt["entities_referencing_bank"])
    return [
        Check("context has a full entity_lookup for Bank (includes its incoming edges)", got_lookup),
    ] + [
        Check(f"answer names {name} (references Bank via 'bank')", name in mentioned)
        for name in gt["entities_referencing_bank"]
    ]


def check_q5(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # "regardingObject" is a known attribute name (not an entity name), so this now gets a
    # direct attribute_lookup -- independent of CampaignResponse ever being retrieved by name.
    got_lookup = has_item(r.context, source="attribute_lookup", attribute="regardingObject")
    mentioned = count_mentions(r.answer, sorted(gt["regarding_object_targets"]))
    return [
        Check("context has a direct attribute_lookup for regardingObject", got_lookup),
        Check(
            "answer names at least 2 of the real polymorphic targets",
            len(mentioned) >= 2,
            f"mentioned: {mentioned}",
        ),
    ]


def check_q6(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    got_lookup = has_item(r.context, source="entity_lookup", entity="Account", layer="banking")
    full_chain_in_context = any(
        "Account (CRM base)" in item["text"] and "CdsStandard (CDS standard)" in item["text"]
        for item in r.context
        if item.get("metadata", {}).get("source") == "entity_lookup"
    )
    mentioned = count_mentions(r.answer, sorted(gt["account_ancestor_layers"]))
    return [
        Check("context has a full entity_lookup for the banking Account", got_lookup),
        Check("context's entity_lookup spells out the full 4-level ancestor chain, not just the immediate parent", full_chain_in_context),
        Check("answer names a real ancestor (CRM base/Foundation/Core/CdsStandard)", bool(mentioned), f"mentioned: {mentioned}"),
    ]


def check_q7(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    negation = contains_any(r.answer, NEGATION_PHRASES)
    fabricated = has_item(r.context, name="CryptoWallet")
    return [
        Check("answer says CryptoWallet was not found, not invented as real", negation is not None, negation or "no negation phrase found"),
        Check("context contains no fabricated CryptoWallet entity", not fabricated),
    ]


def check_q9(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    mentioned = count_mentions(r.answer, gt["banking_entity_names"])
    # An enumeration question (every banking entity) -- same truncation risk class as Q1/Q11.
    truncated = looks_truncated(r.answer)
    return [
        Check(
            "answer names real banking entities, none invented (grounding only -- exhaustiveness of the list is NOT checked, only that it isn't cut off -- see next check)",
            len(mentioned) >= 3,
            f"{len(mentioned)} real banking entities named: {mentioned}",
        ),
        Check(
            "answer does not look truncated mid-sentence (heuristic -- see looks_truncated)",
            truncated is None,
            truncated or "",
        ),
    ]


def check_q10(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    cited = has_item(r.context, chunk_type="relationship", attribute="customer", is_polymorphic=True)
    mentioned = count_mentions(r.answer, ["Account", "Contact"])
    return [
        Check("context cites the real FinancialProduct.customer polymorphic edge", cited),
        Check("answer names both polymorphic targets (Account and Contact)", set(mentioned) == {"Account", "Contact"}, f"mentioned: {mentioned}"),
    ]


# --- Q11-Q20: harder, more adversarial questions -------------------------------------------------
# False premises, out-of-corpus entities, multi-hop paths the router's single-hop near-miss logic
# was never built to trace, and meta-questions about the system's own confidence/provenance/scope.


def check_q11(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # "What fields does an Account have?" shares no whole word with any of Account's real
    # ancestor layer names ("CRM base", "Foundation", "Core", "CdsStandard") -- unlike Q1's "core
    # attributes" -- so it should NOT trigger _ancestor_layer_collision's disambiguation
    # instruction. Checked directly against the router's own detection function (real ground
    # truth for "was the system prompt modified"), not inferred from the answer's wording.
    graph = gt["graph"]
    node = router.find_entity(graph, "Account")
    detail = router.entity_detail(graph, node)
    collision = router._ancestor_layer_collision(r.question, detail)
    mentioned = count_mentions(r.answer, sorted(gt["account_own_attrs"]))
    # This is the question that actually truncated in a real run (2093/3000-token miss became a
    # clean 3000/3000-token cutoff on a later run, mid-sentence, when the model happened to pick
    # a markdown-table format) -- everything above still passed that run, because it only checks
    # that a real fact is present, never that the answer wasn't cut off. See looks_truncated.
    truncated = looks_truncated(r.answer)
    return [
        Check(
            "question does not trigger the ancestor-layer-collision disambiguation path (unmodified system prompt)",
            collision is None,
            f"collision detected: {collision}" if collision else "",
        ),
        Check(
            "answer names a real Account attribute",
            bool(mentioned),
            f"mentioned: {mentioned}" if mentioned else f"expected one of {sorted(gt['account_own_attrs'])}",
        ),
        Check(
            "answer does not look truncated mid-sentence (heuristic -- see looks_truncated; this is the check that would have caught the real Q11 truncation)",
            truncated is None,
            truncated or "",
        ),
    ]


def check_q13(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # False premise: Account has no regionId/region attribute at all (own, inherited, or
    # standard) -- verified directly against graph.entity_detail before writing this check.
    negation = contains_any(r.answer, NEGATION_PHRASES)
    affirmed_as_real = "regionid" in r.answer.lower() and negation is None
    return [
        Check(
            "answer does not affirm regionId as a real Account attribute (ground truth: no such attribute exists)",
            not affirmed_as_real,
            f"'regionId' mentioned={'regionid' in r.answer.lower()}, negation phrase found={negation}",
        ),
        Check(
            "answer declines or corrects the false premise instead of answering it at face value",
            negation is not None,
            negation or "no negation/declining phrase found",
        ),
    ]


def check_q14(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # False premise: Contact has no organizationId/organization attribute -- consistent with the
    # zero Contact-Organization edges finding from earlier in this project; verified again
    # directly against graph.entity_detail before writing this check.
    negation = contains_any(r.answer, NEGATION_PHRASES)
    affirmed_as_real = "organizationid" in r.answer.lower() and negation is None
    return [
        Check(
            "answer does not affirm organizationId as a real Contact attribute (ground truth: zero Contact-Organization edges)",
            not affirmed_as_real,
            f"'organizationId' mentioned={'organizationid' in r.answer.lower()}, negation phrase found={negation}",
        ),
        Check(
            "answer correctly says organizationId is not real, rather than silently ignoring the question",
            negation is not None,
            negation or "no negation/declining phrase found",
        ),
    ]


#: Phrases that mean "no path exists at all" -- deliberately narrower than NEGATION_PHRASES /
#: NO_RELATIONSHIP_PHRASES, which also match "no DIRECT relationship." That phrasing is CORRECT
#: for Q16 (there's genuinely no direct edge, but a real indirect path does exist -- see
#: check_q16), so treating it as a denial produced a false FAIL the first time this check ran
#: for real: the router's multi-hop fallback (Graph.find_path) surfaced the real path correctly
#: and the model named it in full, but still opened with the accurate "no direct relationship"
#: framing this project's own system prompt trains it to use.
NO_PATH_AT_ALL_PHRASES = [
    "no path", "no connection", "not connected", "cannot be connected", "no way to connect",
    "entirely unrelated", "no relationship at all", "there is no relationship",
]


def check_q16(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    # Ground truth (verified against the graph, not assumed from the question -- see
    # ground_truth()'s "collateral_to_bank_path"): a real 3-hop path exists,
    # Collateral -> FinancialProduct -> Branch -> Bank. There is no direct edge, but there IS a
    # real path, so an answer that flatly denies any connection at all is wrong, not appropriately
    # cautious -- unlike "no DIRECT relationship, but ...", which is the correct framing. The
    # router's relations_between() only surfaces one hop of near-miss from each side
    # (Collateral->FinancialProduct, Branch/Syndicates->Bank); Graph.find_path is the fallback
    # that bridges the full chain when that comes up completely empty (see router._route).
    denies_any_path = contains_any(r.answer, NO_PATH_AT_ALL_PHRASES)
    names_financial_product = "FinancialProduct" in r.answer
    names_branch = "Branch" in r.answer
    return [
        Check(
            "answer does not deny that any path exists at all (real path: Collateral -> FinancialProduct -> Branch -> Bank)",
            not denies_any_path,
            denies_any_path or "",
        ),
        Check("answer's path includes FinancialProduct (the real first hop from Collateral)", names_financial_product),
        Check(
            "answer's path includes Branch (the real connector to Bank -- without it the path doesn't reach Bank)",
            names_branch,
        ),
    ]


def check_q20(gt: dict[str, Any], r: QuestionResult) -> list[Check]:
    declined = contains_any(r.answer, OUT_OF_SCOPE_PHRASES)
    return [
        Check(
            "answer states Healthcare/Retail entities are out of scope, rather than confidently answering as if it has that data",
            declined is not None,
            declined or "no scope-declining phrase found",
        ),
    ]


QUESTIONS: list[tuple[str, Callable[[dict[str, Any], QuestionResult], list[Check]] | None]] = [
    ("What are the core attributes of the Account entity?", check_q1),
    ("How does a Contact relate to an Organization?", check_q2),
    ("How does Branch relate to Bank?", check_q3),
    ("What entities reference Bank?", check_q4),
    ("What can the regardingObject attribute point to?", check_q5),
    ("What does the banking Account inherit from?", check_q6),
    ("What is a CryptoWallet entity in the CDM?", check_q7),
    ("What's the difference between Account and Contact?", None),  # manual: is the comparison useful?
    ("What entities exist in the banking model?", check_q9),
    ("How does FinancialProduct relate to a customer?", check_q10),
    ("What fields does an Account have?", check_q11),
    ("Tell me everything about the Contact entity.", None),  # manual: is the free-form summary useful/accurate?
    ("Since Account has a regionId field, what region is a given account in?", check_q13),
    ("Does Contact have a field called organizationId?", check_q14),
    ("Which entities does Bank connect to, directly or indirectly, going up to two steps away?", None),  # manual: multi-hop, no mechanical ground truth encoded
    ("What's the path from Collateral to Bank?", check_q16),
    ("Is there a Loan entity, and if not, what's the closest thing in the model?", None),  # manual: "closest thing" is a judgment call
    ("How confident are you in this answer, and what would you need to check to be sure?", None),  # manual: meta-question, no ground truth
    ("What data are you actually using to answer my questions -- is this from an official Microsoft source?", None),  # manual: meta-question, no ground truth
    ("If I ask about an entity in the Healthcare or Retail models, will you be able to answer?", check_q20),
]


#: Capability groupings for the report (see README's Evaluation section). Purely a reporting
#: concern -- does not change QUESTIONS, question text, or any check's logic. Each category's
#: question numbers are listed in the exact order they appear within that category's report
#: section. Every number 1..len(QUESTIONS) must appear in exactly one category (checked at import
#: time below, not left to silently drift as questions are added).
CATEGORY_QUESTIONS: dict[str, list[int]] = {
    "Entity attributes": [1, 6, 11, 12],
    "Direct relationships": [3, 10],
    "No-relationship handling": [2],
    "Reverse / multi-hop traversal": [4, 15, 16],
    "Attribute-level lookup": [5],
    "Hallucination resistance (false premises)": [7, 13, 14, 17],
    "Scope boundaries": [9, 20],
    "Comparative / open-ended reasoning": [8],
    "System self-awareness": [18, 19],
}

_categorized = sorted(n for nums in CATEGORY_QUESTIONS.values() for n in nums)
_expected = list(range(1, len(QUESTIONS) + 1))
if _categorized != _expected:
    raise RuntimeError(
        f"CATEGORY_QUESTIONS must cover exactly Q1..Q{len(QUESTIONS)} once each; "
        f"got {_categorized}, expected {_expected}"
    )


# --- run + report ---------------------------------------------------------------------------------


def _parse_retry_after(exc: Exception, default: float) -> float:
    """Groq's 429 body names the exact wait, e.g. "Please try again in 31.845s." (a per-minute
    TPM cap) or "Please try again in 33m26.208s." (the per-day TPD cap -- much longer, and only
    an estimate of the next single request's admission, not a full quota reset). Parsing it gives
    an exact wait instead of guessing a constant backoff; falls back to ``default`` if the
    message doesn't match this shape (e.g. a non-rate-limit transient error)."""
    m = re.search(r"try again in (?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s", str(exc))
    if not m:
        return default
    minutes = float(m.group(1)) if m.group(1) else 0.0
    return minutes * 60 + float(m.group(2))


def ask_with_retry(
    question: str, graph: Graph, collection: Any, timing: dict[str, Any], retries: int = 12, backoff: float = 15.0
):
    """router.answer(), retrying with backoff on a transient rate limit -- anything else raises
    immediately. ``timing`` is cleared and repopulated by ``router.answer`` on each attempt, so a
    retry doesn't leave a stale partial breakdown from a failed call. The wait between attempts
    is Groq's own quoted retry-after time (see ``_parse_retry_after``) plus a small buffer, not a
    blind exponential guess -- this can legitimately be tens of minutes when the account is
    close to its daily (not just per-minute) token cap, so ``retries`` is generous."""
    for attempt in range(retries + 1):
        try:
            timing.clear()
            return router.answer(question, graph, collection, timing=timing)
        except Exception as exc:
            transient = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if not transient or attempt == retries:
                raise
            wait = _parse_retry_after(exc, backoff * (attempt + 1)) + 5.0
            print(f"        rate limited (attempt {attempt + 1}/{retries}); waiting {wait:.0f}s...")
            time.sleep(wait)


def run_all(
    graph: Graph,
    collection: Any,
    questions: list[tuple[str, Callable[[dict[str, Any], QuestionResult], list[Check]] | None]] | None = None,
    start_index: int = 1,
) -> list[QuestionResult]:
    """Asks ``questions`` (default: the full module-level ``QUESTIONS``) in order, numbering
    results from ``start_index`` -- so a resumed subset (e.g. ``QUESTIONS[10:]`` with
    ``start_index=11``) still reports as "Q11", not renumbered back to "Q1", and its report can
    be read alongside an earlier run's output for the questions it didn't repeat."""
    questions = QUESTIONS if questions is None else questions
    gt = ground_truth(graph)
    results = []
    last_index = start_index + len(questions) - 1
    for offset, (question, checker) in enumerate(questions):
        i = start_index + offset
        print(f"[{i}/{last_index}] asking: {question}")
        t0 = time.time()
        timing: dict[str, Any] = {}
        result = ask_with_retry(question, graph, collection, timing)
        elapsed = time.time() - t0
        qr = QuestionResult(
            number=i, question=question, answer=result.answer, context=result.context, elapsed=elapsed, timing=dict(timing)
        )
        if checker is None:
            qr.manual = True
        else:
            qr.checks = checker(gt, qr)
        results.append(qr)
        print(f"        {elapsed:.1f}s, {'NEEDS REVIEW' if qr.manual else ('PASS' if qr.passed else 'FAIL')}")
    return results


def _check_to_dict(c: Check) -> dict[str, Any]:
    return {"label": c.label, "passed": c.passed, "detail": c.detail}


def _check_from_dict(d: dict[str, Any]) -> Check:
    return Check(label=d["label"], passed=d["passed"], detail=d.get("detail", ""))


def _result_to_dict(r: QuestionResult) -> dict[str, Any]:
    return {
        "number": r.number, "question": r.question, "answer": r.answer, "context": r.context,
        "elapsed": r.elapsed, "timing": r.timing, "checks": [_check_to_dict(c) for c in r.checks],
        "manual": r.manual,
    }


def _result_from_dict(d: dict[str, Any]) -> QuestionResult:
    return QuestionResult(
        number=d["number"], question=d["question"], answer=d["answer"], context=d["context"],
        elapsed=d["elapsed"], timing=d.get("timing", {}),
        checks=[_check_from_dict(c) for c in d.get("checks", [])], manual=d.get("manual", False),
    )


def load_raw_results() -> dict[int, QuestionResult]:
    """Every real-API result saved by a past run (any range), keyed by question number. Empty if
    RAW_RESULTS_PATH doesn't exist yet -- a fresh clone has never run the eval script."""
    if not RAW_RESULTS_PATH.exists():
        return {}
    data = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    return {int(k): _result_from_dict(v) for k, v in data.items()}


def save_raw_results(results: list[QuestionResult]) -> dict[int, QuestionResult]:
    """Merges ``results`` into the accumulating store (by question number, newest run for a given
    number wins) and persists it to RAW_RESULTS_PATH. Returns the full merged store."""
    merged = load_raw_results()
    for r in results:
        merged[r.number] = r
    RAW_RESULTS_PATH.write_text(
        json.dumps({str(k): _result_to_dict(v) for k, v in sorted(merged.items())}, indent=2),
        encoding="utf-8",
    )
    return merged


def _source_lines(context: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in context:
        meta = item.get("metadata", {})
        chunk_type = meta.get("chunk_type", "chunk")
        if chunk_type == "section_header":
            lines.append(f"- *(section header)* {item['text']}")
            continue
        bits = [f"chunk_type={chunk_type}", f"source={meta.get('source', '?')}"]
        if meta.get("entity") or meta.get("name"):
            bits.append(f"entity={meta.get('entity') or meta.get('name')}")
        if meta.get("attribute"):
            bits.append(f"attribute={meta.get('attribute')}")
        if meta.get("is_audit") is not None:
            bits.append(f"is_audit={meta.get('is_audit')}")
        if meta.get("is_polymorphic"):
            bits.append("is_polymorphic=True")
        lines.append(f"- {', '.join(bits)}")
    return lines


def _fmt(seconds: Any) -> str:
    return f"{seconds:.3f}" if isinstance(seconds, (int, float)) else "-"


def render_performance_table(results: list[QuestionResult]) -> str:
    """Stage-by-stage wall-clock breakdown per question, from router.answer's ``timing`` dict
    (see router.py's ``_measure``) -- all client-measured seconds except the last two columns,
    which are Groq's own server-reported stats from the response's ``usage`` object (not a
    re-estimate). "Queue+Prompt (server)" is the closest available proxy for time-to-first-token
    without switching to streaming (which would change real behavior, out of scope here): Groq
    doesn't return a token-level timestamp on a non-streaming call, but queue_time + prompt_time
    is the server's own account of the time spent before the first output token starts generating."""
    headers = [
        "Q", "Detection", "Graph lookup", "Vector query", "Context asm.", "Prompt build",
        "LLM call (client)", "Total (client)", "Prompt tok", "Completion tok", "Queue+Prompt (server)", "Completion (server)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in results:
        t = r.timing
        queue_prompt = None
        if t.get("queue_time") is not None and t.get("prompt_time") is not None:
            queue_prompt = t["queue_time"] + t["prompt_time"]
        row = [
            str(r.number),
            _fmt(t.get("detection")),
            _fmt(t.get("graph_lookup")),
            _fmt(t.get("vector_query")),
            _fmt(t.get("context_assembly")),
            _fmt(t.get("prompt_build")),
            _fmt(t.get("llm_call")),
            _fmt(t.get("total", r.elapsed)),
            str(t.get("prompt_tokens", "-")),
            str(t.get("completion_tokens", "-")),
            _fmt(queue_prompt),
            _fmt(t.get("completion_time")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _category_summary_line(category: str, members: list[QuestionResult]) -> str:
    """One compact coverage line for ``category``, e.g. "Entity attributes: 3/4 passed, 1
    needing manual review" -- the numerator/denominator intentionally mixes scopes (passed count
    among the category's automated checks, over the category's total question count, auto +
    manual), matching how a reader wants to scan this: "how much of this capability is covered,
    and how much of that coverage came back clean." A category with zero automated questions
    (e.g. purely meta/self-awareness questions) gets a distinct phrasing rather than a misleading
    "0/1 passed", since there's no automated check to have passed or failed."""
    total = len(members)
    auto = [r for r in members if not r.manual]
    manual = [r for r in members if r.manual]
    if not auto:
        return f"- {category}: {len(manual)} needing manual review (no automated checks in this category)"
    passed = [r for r in auto if r.passed]
    failed = [r for r in auto if not r.passed]
    bits = [f"{len(passed)}/{total} passed"]
    if failed:
        bits.append(f"{len(failed)} failed")
    if manual:
        bits.append(f"{len(manual)} needing manual review")
    return f"- {category}: " + ", ".join(bits)


def render_category_summary(results: list[QuestionResult]) -> list[str]:
    """One line per category present in ``results`` (in ``CATEGORY_QUESTIONS`` order), skipping a
    category with zero members in ``results`` -- relevant for a range-limited run, whose report
    should only claim coverage of what it actually asked."""
    by_number = {r.number: r for r in results}
    lines = []
    for category, numbers in CATEGORY_QUESTIONS.items():
        members = [by_number[n] for n in numbers if n in by_number]
        if members:
            lines.append(_category_summary_line(category, members))
    return lines


def _render_question_detail(r: QuestionResult) -> list[str]:
    status = "NEEDS HUMAN REVIEW" if r.manual else ("PASS" if r.passed else "FAIL")
    lines = [f"### Q{r.number}: {r.question}", "", f"**Status:** {status}  (**{r.elapsed:.1f}s**)", ""]
    lines.append("**Answer:**")
    lines.append("")
    lines.append("> " + r.answer.replace("\n", "\n> "))
    lines.append("")
    lines.append(f"**Sources / context** ({len(r.context)} items):")
    lines.append("")
    lines.extend(_source_lines(r.context))
    lines.append("")
    if r.manual:
        lines.append("**Automated checks:** none -- read the answer above and judge for yourself.")
    else:
        lines.append("**Automated checks:**")
        for c in r.checks:
            box = "x" if c.passed else " "
            detail = f" -- {c.detail}" if c.detail else ""
            lines.append(f"- [{box}] {c.label}{detail}")
    lines.append("")
    return lines


def render_report(results: list[QuestionResult]) -> str:
    verified = [r for r in results if not r.manual]
    passed = [r for r in verified if r.passed]
    manual = [r for r in results if r.manual]
    by_number = {r.number: r for r in results}

    lines = [
        "# cdm-rag evaluation report",
        "",
        "## Category summary",
        "",
        "Coverage by capability, not just a flat pass count -- see README's Evaluation section "
        "for what each category is meant to test.",
        "",
    ]
    lines += render_category_summary(results)
    lines += [
        "",
        f"**Overall: {len(verified)}/{len(results)} automatically verified ({len(passed)} passed, "
        f"{len(verified) - len(passed)} failed), {len(manual)} needing manual review.**",
        "",
        "## Performance profile",
        "",
        "All times in seconds. The embedding model is warmed once before question 1 (see "
        "`main()`), so \"Vector query\" reflects steady-state per-query cost, not the one-time "
        "~8-10s model-load cost a fresh process pays on its first ever query.",
        "",
        render_performance_table(results),
        "",
    ]
    for category, numbers in CATEGORY_QUESTIONS.items():
        members = [by_number[n] for n in numbers if n in by_number]
        if not members:
            continue
        lines.append(f"## Category: {category}")
        lines.append("")
        for r in members:
            lines += _render_question_detail(r)
    return "\n".join(lines)


def _print_summary(results: list[QuestionResult]) -> None:
    verified = [r for r in results if not r.manual]
    passed = [r for r in verified if r.passed]
    manual = [r for r in results if r.manual]
    print(f"\n{len(verified)}/{len(results)} automatically verified ({len(passed)} passed, {len(verified) - len(passed)} failed), {len(manual)} needing manual review.")


def _assemble_report() -> None:
    """Renders eval_results.md purely from the accumulated raw store (RAW_RESULTS_PATH) -- no API
    calls at all. Lets one full categorized report be built out of separate, range-limited runs
    without re-asking a question a prior run already answered for real. Invoked with
    `python scripts/eval.py --report`."""
    store = load_raw_results()
    results = sorted(store.values(), key=lambda r: r.number)
    missing = [n for n in range(1, len(QUESTIONS) + 1) if n not in store]

    report = render_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Assembled report from {len(results)}/{len(QUESTIONS)} previously-run questions (no API calls made).")
    if missing:
        print(f"Missing (never saved to {RAW_RESULTS_PATH.name}): {missing}")
    _print_summary(results)
    print(f"\nReport written to {REPORT_PATH}")


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--report"]:
        _assemble_report()
        return

    start = int(args[0]) if len(args) >= 1 else 1
    end = int(args[1]) if len(args) >= 2 else len(QUESTIONS)
    subset = QUESTIONS[start - 1 : end]
    full_run = (start, end) == (1, len(QUESTIONS))
    report_path = REPORT_PATH if full_run else REPORT_PATH.with_name(f"eval_results_q{start}_{end}.md")

    print("Building graph and opening/building the index...")
    corpus = Corpus(corpus_path())
    graph = build_graph(corpus, banking_seeds(corpus))
    collection = open_or_build_index(graph)
    print(f"graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges; index: {collection.count()} chunks")

    print("Warming the embedding model (one-time ~8-10s cost otherwise landing on question 1's timing)...")
    load_model()
    print("done.\n")

    results = run_all(graph, collection, subset, start_index=start)
    save_raw_results(results)  # accumulate into the merged store regardless of range, for --report later
    report = render_report(results)
    report_path.write_text(report, encoding="utf-8")

    _print_summary(results)
    print("\nPerformance profile:")
    print(render_performance_table(results))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
