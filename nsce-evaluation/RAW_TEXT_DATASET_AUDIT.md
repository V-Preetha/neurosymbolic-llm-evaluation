# Raw-Text Dataset Audit — Neural-to-Symbolic Verification Bridge (Experiment 2)

Status: **design/audit only.** No dataset downloaded, no code written, the 360-example ProofWriter held-out test untouched. This document supports the decision of what feeds Experiment 2 (end-to-end raw-text verification) — it is not itself the implementation.

## 1. What's already in the repo

| Component | Reusable for the raw-text bridge? |
|---|---|
| `src/nsce/parser.py`, `symbolic_reasoner.py` | **Yes, as the reasoning core.** Horn-clause forward-chaining engine, `Atom`/`Rule` IR — see below (§6). It currently only accepts ProofWriter's own controlled sentence grammar as input text, not free prose; that's the actual gap, not the reasoner itself. |
| `src/nsce/verifier.py`, `pipeline.py`, `feedback.py` | Yes — RETAIN/CORRECT/ABSTAIN control flow generalizes unchanged; it just needs to receive a `theory` built by a new extraction step instead of `parse_theory(record["input"])`. |
| `nsce_dataset_package/normalized/proofwriter_{dev,test}.jsonl` | Reused as **Experiment 1 (symbolic-engine validation)** only, per the pivot — not the new headline set. 360-example held-out test remains frozen and unrun. |
| `src/datasets/felm.py`, `src/evaluation/felm_audit.py` | FELM claims are free prose with retrieved Wikipedia evidence — i.e., inherently a retrieval setup. That conflicts with the "no RAG" constraint for Experiment 2's primary architecture (the corpus must be bounded and supplied, not retrieved). FELM's evidence-sufficiency scoring logic is reusable as a *claim-extraction quality* auxiliary, but FELM itself is not a candidate primary dataset here.
| `src/io_guard.py` field whitelisting | Directly reusable pattern — extend `AllowedInput` (or a new equivalent) to whitelist `corpus`/`question` only, keeping gold logic/evidence/labels out of inference-time code paths for the new dataset too. |

**Conclusion: no natural-language, bounded-corpus, gold-verdict dataset with rule structure currently exists in this project.** ProofWriter is facts/rules, not prose; FELM is prose but retrieval-based, not bounded-corpus. A new dataset is needed for Experiment 2.

## 2. Candidate datasets

### 2.1 FOLIO — *recommended primary*
Expert-written NL reasoning dataset with parallel FOL annotations. 1,204 train + 203 validation examples (public split; test held by authors), MIT license, GitHub `Yale-LILY/FOLIO`, arXiv 2209.00840 / ACL 2024. Each example: a set of premises (natural prose, not templated), a conclusion/question, and a **True / False / Unknown** label — exactly the trichotomy the existing reasoner already outputs. Premises are open-domain (real-world entities/relations), written by people, not generated from a fixed grammar, so genuinely tests language understanding rather than template parsing. Gold FOL formulas exist per example but are **not required at inference** — they are exactly the kind of gold logical form the fairness rule permits for oracle/translator-eval use only.

### 2.2 ParaRules (RuleTaker/ProofWriter paraphrase split) — *recommended for the naturalized-ProofWriter robustness arm*
Crowdworkers rephrased RuleTaker's templated facts/rules into free-form English while preserving the underlying theory (~40k questions over ~2k theories). E.g. "Bob is cold." → "In the snow sits Bob, crying from being cold." This directly answers Task-list item 5 ("can ProofWriter become a more natural-language robustness dataset without contaminating the held-out set?") — **yes, via ParaRules, which is a distinct, separately-released split, not a transformation of the frozen 360-example test set.** It shares ProofWriter's D0–D5 depth stratification and label set, so it plugs into the existing `Atom/Rule` symbolic core once the free-text extraction step exists. This is the correct instrument for Experiment 3 (robustness to paraphrase) rather than the Experiment 2 headline number, since its logical skeleton is still template-generated even though the surface text is not.

### 2.3 ContractNLI — *secondary/domain-generalization candidate, not primary*
607 NDAs, 17 hypotheses applied per document, entailment/contradiction/not-mentioned labels, span-level evidence annotations, Stanford NLP release. Bounded per-document corpus (no retrieval), real prose, some negation and multi-hop. Weaknesses: legal-domain vocabulary and rule structure are harder to symbolize with a general-purpose predicate schema than FOLIO's open-domain sentences; would need substantial domain-specific predicate design. Keep as a possible Experiment-3 generalization check, not the primary dataset.

### 2.4 ConTRoL (long-context contextual reasoning) — audited, not recommended
Long premises (paragraph-length), entail/contradict/neutral labels, genuine multi-hop. Rejected as primary: premises draw heavily on background/world knowledge beyond what's stated in the passage, which conflicts with "avoid a dataset requiring uncontrolled external world knowledge for the primary experiment" and makes clean symbolization unreliable.

### 2.5 CLUTRR (kinship reasoning) — audited, not recommended
Natural-language short stories requiring inferring family relations by chaining implicit relational rules; genuinely multi-hop and rule-based. Rejected as primary: labels are relation-classification (not support/contradict/insufficient), and rules are never explicit in the text — they must be induced across many training stories rather than read off one bounded document, which does not fit "explicit or implicit rules stated within the bounded corpus."

### 2.6 FEVER / SciFact-style claim verification — audited, not recommended as primary
Real prose, SUPPORTS/REFUTES/NOT ENOUGH INFO labels, some multi-hop. Rejected as primary because the standard task setup retrieves evidence from a large external corpus (Wikipedia / a paper corpus) — using it without retrieval means hand-restricting to gold evidence sentences only, which quietly changes the task and risks reviewer objections that this is "FEVER with retrieval removed," not a fair NL-reasoning benchmark. Usable only if evidence is bounded to the gold sentence set *and* this restriction is explicitly reported as a deviation from the standard task — lower priority than FOLIO/ParaRules.

## 3. Comparison table

| Criterion | FOLIO | ParaRules | ContractNLI | ConTRoL | CLUTRR | FEVER/SciFact |
|---|---|---|---|---|---|---|
| Raw prose (not templated) | Yes | Yes (surface only; logic skeleton templated) | Yes | Yes | Yes | Yes |
| Bounded evidence (no retrieval needed) | Yes | Yes | Yes | Yes | Yes | No (standard setup retrieves) |
| 3-way label (support/contradict/insufficient) | Yes (True/False/Unknown) | Yes | Yes (entail/contra/not-mentioned) | Yes | No (relation classes) | Yes (nominally) |
| Multi-hop | Yes | Yes (by depth) | Some | Yes | Yes | Some |
| Explicit/implicit negation | Yes | Yes | Some | Some | Rare | Some |
| Conditional/rule structure present | Yes | Yes (explicit "if...then") | Weak | Weak | Implicit only, not stated | Weak |
| Reasonably symbolizable | Yes (open-domain but bounded predicates per example) | Yes (already symbolizable — same grammar as ProofWriter under paraphrase) | Partial (legal-domain schema needed) | Low (world-knowledge dependent) | Low (rules never stated) | Low (mostly single-fact entailment) |
| Gold evidence for eval | Partial | Yes (theory) | Yes (spans) | Partial | N/A | Yes (evidence sentences) |
| Gold logical form for eval | Yes (FOL) | Yes (original template facts/rules) | No | No | No | No |
| Leakage risk | Low (real-world content varies per item) | Low | Low | Medium (background knowledge shortcuts) | Medium (relation shortcuts) | Medium (claim wording artifacts, well documented in literature) |
| Size | 1,204 train + 203 val (public) | ~40k Qs / ~2k theories | 607 docs × 17 hypotheses | ~8k | ~10k+ | Large (FEVER: 185k) |
| License | MIT | Same license as RuleTaker/ProofWriter (AI2, research use) | Documented per Stanford release, research use | Research use | MIT-ish/research use | CC-BY-SA (FEVER) |

## 4. Recommendation

- **Primary dataset for Experiment 2 (headline end-to-end raw-text verification): FOLIO.** It is the only candidate that is simultaneously genuine open-domain prose (not disguised templates), bounded per-example, natively 3-way (True/False/Unknown — matches the existing reasoner's supported/contradicted/unknown output with no relabeling), has explicit conditional and negation structure suitable for the existing Horn-clause fragment, and ships gold FOL for oracle-only use (never at inference), consistent with the fairness rule.
- **Robustness/paraphrase arm (Experiment 3) and the "naturalize ProofWriter" instrument (Task-list item 5): ParaRules.** It is a separate, existing release built exactly for this purpose (testing paraphrase robustness on RuleTaker/ProofWriter-style theories) and does not require touching or regenerating the frozen 360-example ProofWriter held-out test.
- **Do not adopt ContractNLI, ConTRoL, CLUTRR, or FEVER/SciFact as the primary dataset.** Each fails at least one hard requirement (implicit-only rules, retrieval dependency, non-trichotomous labels, or heavy world-knowledge dependence). ContractNLI is the best-positioned secondary/generalization candidate if a second domain is wanted later.
- **FELM stays out of Experiment 2 entirely** — it is retrieval-based by construction, which conflicts with the explicit "not a RAG study" constraint for the primary architecture.

## 5. Intermediate symbolic representation (proposed, extends the existing IR)

Keep `Atom(predicate, arguments, negated)` and `Rule(premises, conclusion, source)` as-is — do not replace them. Extend, minimally:

1. **Binary/relational atoms beyond the fixed 6-verb table.** FOLIO/ParaRules need an open predicate vocabulary (`is_a`, `contains`, `coated_with`, ...) rather than `VERB_FORMS`'s closed set. Predicate names become extraction output (from the new extraction step, §6), not a hardcoded lexicon — but the *shape* (`predicate(arg1[, arg2])`, optional `negated` flag) stays identical.
2. **Entity aliasing table**, separate from `Atom` itself: a per-document `{alias_string: canonical_entity}` map produced by the new normalization step (Task 5), consulted before atoms are constructed. This keeps `Atom` itself alias-free and keeps the reasoner's unification logic untouched.
3. Everything else stays constrained on purpose, matching the existing "unsupported operators fail loudly" design: still single implicit variable per rule, still conjunctive-only premises, still no disjunction/existentials/function symbols. Malformed or unsupported extractions raise the same `ParserError` path (→ `ABSTAIN`), not a best-effort partial parse.

This is additive to `SYMBOLIC_BRIDGE_SPEC.md` §1–§7, not a rewrite of the reasoner (§6 of that spec, `forward_chain`/`classify`, is reused unchanged).

## 6. Proposed pipeline (raw text → symbolic, new components only)

```
raw corpus + question
        │
        ▼
Frozen Qwen3-8B (existing QwenRunner, unchanged)  →  initial answer
        │
        ▼
Claim extraction (NEW, small)         — Qwen's answer → target Atom(s) via schema-constrained
                                          generation, same frozen model, separate prompt/call
        │
        ▼
Corpus information extraction (NEW)   — raw corpus → candidate entities/relations/rules via
                                          schema-constrained generation (JSON-schema-validated
                                          output; malformed items dropped, logged, never passed
                                          to the reasoner)
        │
        ▼
Entity normalization (NEW, small)     — per-document alias map, rule-based first (string/edit-
                                          distance match within document), LLM-assisted only for
                                          residual ambiguous cases, logged either way
        │
        ▼
Language → symbolic form (NEW)        — validated JSON propositions → Atom/Rule objects,
                                          reusing Atom/Rule dataclasses unchanged
        │
        ▼
Deterministic reasoner (EXISTING, unchanged) — forward_chain / classify
        │
        ▼
Verifier / RETAIN-CORRECT-ABSTAIN (EXISTING, unchanged control flow, new theory source)
```

The only genuinely new code is: claim extraction, corpus information extraction, entity normalization, and the JSON-proposition → `Atom`/`Rule` mapper. The reasoner, verifier state machine, and I/O whitelisting pattern are reused.

## 7. Per-stage evaluation plan

| Stage | How evaluated independently |
|---|---|
| Corpus information extraction | Precision/recall of extracted atoms/rules against a small hand-annotated gold-proposition subset (order of ~50–100 FOLIO examples), not the full set — this is an eval annotation, never fed to inference. |
| Entity normalization | Cluster-purity / alias-accuracy against the same hand-annotated subset. |
| Claim extraction | Exact-match / logical-equivalence of extracted claim atom against a hand-labeled target for a sample of Qwen answers. |
| Symbolization (proposition → Atom/Rule) | Parse-success rate + spot-check of correctness on the validated-JSON subset; failures must raise `ParserError`, never silently drop a premise. |
| Symbolic reasoning | Already validated (Experiment 1, 72/72 pilot; 56/56 unit tests) — reused, not re-evaluated here except for regression tests on any new predicate shapes. |
| Correction/abstention policy | Beneficial-vs-harmful correction rate, same metric already defined in `NEW_STUDY_DESIGN.md`. |
| End-to-end | Accuracy/macro-F1 vs Conditions A/B, plus the error-decomposition breakdown (Task 7's `*_ERROR` taxonomy) attributing each failure to exactly one stage above. |

## 8. Compute note (RTX 5050, 8GB, nf4)

FOLIO's public set (1,204 train + 203 val) is small enough that a hand-annotated gold-proposition subset (~50–100 items) for stage-wise evaluation, plus a held-out slice of the remainder for the end-to-end run, is feasible within a few Qwen3-8B nf4 inference passes per item (initial answer + claim extraction + corpus extraction — 2–3 extra forward passes per example versus Condition A, all still the same frozen 8B checkpoint already smoke-tested in Phase B). No new model needs downloading. ParaRules is much larger (~40k questions) — a stratified sample by depth, sized similarly to the existing 360-example ProofWriter test convention, is the right scope rather than running it in full.

## 9. Minimal prototype plan (before scaling)

1. Pull ~15–20 FOLIO validation examples by hand, no download automation yet — read them directly to sanity-check that the existing `Atom/Rule` shape (extended per §5) can represent typical premises without hitting the "unsupported operator" wall constantly.
2. Hand-write (not LLM-generate) gold propositions for those ~15–20 examples — this becomes the first slice of the stage-wise gold-annotation set in §7, and doubles as a manual check on whether FOLIO is as symbolizable in practice as it looks on paper.
3. Only after that manual check, prototype the corpus-information-extraction prompt (schema-constrained) against those same items and compare to the hand-written gold — this is the actual bottleneck the central research question is about, so it should be the first thing built, before claim extraction or entity normalization.
4. Wire the existing reasoner against the prototype's output on just those ~15–20 items end-to-end, by hand-inspecting each `bridge_action`, before writing any batch-run harness.
5. Only after that walkthrough looks sound, formalize into `run_*` scripts mirroring the existing `experiments/run_nsce.py` pattern, and decide sample sizes for the real dev/test split.

Do not download ParaRules or automate FOLIO ingestion until step 1–4 confirm the representation and extraction approach are viable on a small hand-checked sample.
