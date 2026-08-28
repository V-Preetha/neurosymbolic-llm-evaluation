# FOLIO Neural-to-Symbolic Translation Prototype — Results

Status: **frozen, one-shot run. Not re-run, not prompt-tuned.** 30-item sample, Qwen3-8B (`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, nf4, seed 42, temperature 0, `max_new_tokens=400` per sentence). Raw artifacts: [`translations.jsonl`](translations.jsonl), [`metrics.json`](metrics.json), [`error_analysis.json`](error_analysis.json), [`gold_oracle.json`](gold_oracle.json), [`sample_manifest.json`](sample_manifest.json). This phase stops at translation + verification — no correction/RETAIN-CORRECT-ABSTAIN layer attached yet, per the phase's explicit stop condition.

## Central question

> Can a frozen Qwen3-8B reliably construct executable first-order symbolic representations from raw natural-language evidence?

**No, not yet, and not by a small margin.** The numbers below are unambiguous on this specific prompting/architecture combination.

## 1. IR design

Frozen, unchanged from `FOLIO_SYMBOLIC_IR.md` — 8 node types, documented separately. Not revisited in this phase.

## 2. Compiler test results

`tests/test_folio_compiler.py`: 13/13 passing (unary/binary predicates, negation, conjunction, disjunction, implication, universal/existential quantification incl. nested and multi-variable, iff, shared symbol table across premises+hypothesis).

## 3. Solver test results

`tests/test_folio_solver.py`: 9/9 passing (True/False/Uncertain classification, inconsistent-premises handling, translation-failure propagation, unsupported-node handling, unbound-variable handling, existential+conjunctive query).

## 4. Total project test-suite result

**116/116 passing** (56 pre-existing ProofWriter/FELM tests + 60 new FOLIO tests, incl. `tests/test_folio_ir.py`, `test_folio_normalization.py`, `test_folio_field_whitelisting.py`, `test_folio_translation.py`). ProofWriter path unmodified and unaffected — the only change touching shared code was an additive `QwenRunner.generate_text()` method (`src/models/qwen.py`); `generate_choice`/`generate_label` (used by every existing ProofWriter/FELM branch) are untouched.

## 5. Prototype sample composition

30 items from `Yale-LILY/FOLIO`'s public **validation** split (not train — train has no gold `conclusion-FOL`, needed for the oracle-agreement metric below; not the frozen 360-example ProofWriter test, which is a different dataset entirely). Deterministic hash-based selection, 10/10/10 label balance, **zero overlap** with the 18-item feasibility-audit sample (explicitly excluded from the selection pool). Manifest + checksum: `sample_manifest.json`. Construct coverage: negation 22/30, universal 28/30, disjunction 16/30, conjunction 25/30, existential 6/30, multi-variable 6/30, binary relations 12/30, multi-step (≥5 premises) 28/30, xor 5/30.

## 6. Schema-valid translation rate

- **Per sentence: 78.9%** (intuitive-sounding, but misleading in isolation — see below)
- **Per item (every premise + hypothesis in the theory schema-valid): 30.0%** (9/30)

A theory needs *every* sentence to parse cleanly to be usable at all — one bad sentence out of six kills the whole item. This is why the per-item rate is so much lower than the per-sentence rate, and it's the number that actually matters for the pipeline.

## 7. Z3-compilable translation rate

**26.7%** (8/30) reached the compiler successfully — a proper subset of the 9/30 (30%) that were schema-valid per sentence. The one item in between (idx 60) is exact and diagnosed: every individual sentence parsed cleanly in isolation, but `schema.validate_theory`'s whole-theory arity-consistency check (§4 of `FOLIO_SYMBOLIC_IR.md`) caught a predicate used with two different argument counts across different sentences of the same theory — invisible to per-sentence validation by construction, since arity consistency is only checkable once every sentence in a theory is known. This is exactly why per-sentence validation (§6) and whole-theory validation (this section) are reported as two different numbers rather than collapsed into one.

## 8. Generated-logic → Z3 official-label accuracy

- **Over all 30 items: 13.3%** (4/30 — the honest headline number: most items never produced any verdict at all)
- **Over the 8 items that did produce a verdict: 50.0%** (4/8)

## 9. Generated-logic verdict vs. gold-FOL oracle agreement

**50.0%** (4/8 comparable items). Gold-FOL oracle itself: **29/30 (96.7%)** official-label accuracy on this same 30-item set (`gold_oracle.json`) — one unresolved case (idx 87) due to a gold-FOL annotation using a bare comma where a conjunction symbol was clearly intended, a genuine FOLIO annotation artifact, not a bug in this project's parser (confirmed by re-running the same 18-item feasibility-sample parser unchanged and reproducing its exact prior 16/18 result). This gap — **96.7% oracle ceiling vs. 13.3% end-to-end** — is the core empirical finding of this phase.

## 10. Accuracy by logical construct

(Official-label accuracy, all 30 items, denominator = items containing that construct per the gold-FOL-derived tags)

| Construct | n | accuracy |
|---|---|---|
| existential | 6 | 50.0% |
| binary_relation | 12 | 16.7% |
| conjunction | 25 | 16.0% |
| disjunction | 16 | 12.5% |
| multi_step | 28 | 14.3% |
| universal | 28 | 7.1% |
| negation | 22 | 4.5% |
| xor | 5 | 0.0% |
| multi_variable | 6 | 0.0% |

Every rate here is low, but the *relative* ordering is informative: universal-quantifier and negation items (the two most common constructs in this sample) score worst, and xor/multi-variable score at 0% outright — consistent with the manual audit below, where the single most common individual failure was Qwen dropping the `forall` wrapper on "If X then Y" conditionals.

## 11. Translation-error breakdown

Structural (auto-tagged from `error_reason`, `error_analysis.json`):

| Category | Count |
|---|---|
| SCHEMA_ERROR | 32 |
| QUANTIFIER_SCOPE_ERROR (unbound variable) | 10 |
| PREDICATE_ERROR (arity conflict) | 1 |

Semantic (manual audit, `error_analysis.json` → `manual_semantic_audit`):

- **QUANTIFIER_ERROR (missing `forall` wrapper on universal conditionals)** — the single most common individual failure mode, appearing directly in **7 of 30** items (idx 51, 153, 155, 170, 171, 172, 180/181 share the same theory) despite an explicit matching few-shot example in the prompt.
- **MISSING_PROPOSITION** — key binary relations dropped during translation (e.g. `plays_for(ailton, nautico)` vanished entirely, idx 199-203), leaving a compiled theory too weak to derive anything even when every sentence was individually schema-valid.
- **COREFERENCE_ERROR / entity inconsistency** — the same real-world entity rendered as 3 different IR constants across sentences in one theory (`aitlon_silva` / `aitlon` / `aitlon silva`, idx 199-203) — confirms the risk flagged in `normalization.py`'s design notes as a real, observed failure, not a hypothetical one.
- **HALLUCINATED_PROPOSITION / QUANTIFIER_ERROR combined** — a ground fact about one named individual ("Ailton Silva does not play for a football club") was translated into a fabricated universal claim over an invented `person(x)` predicate (idx 200).
- **NEGATION_ERROR (wrong node shape)** — `{"type": "predicate", "name": "not", "args": [...]}` used instead of the documented `{"type": "not", "body": ...}` node (idx 9) — negation is conceptually applied to the right content but not encoded in the frozen schema's required shape.
- **SCHEMA_ERROR via truncation** — long enumerative sentences (e.g. listing six wild-turkey subspecies) produced JSON cut off mid-object at the 400-token budget — a prompting/budget issue, not a semantic-understanding failure; worth widening the token budget in a future iteration rather than a finding about Qwen's reasoning per se.

## 12. Examples of successful translation

**idx 6** (`monkeypox`, gold True): a genuine 2-hop chain translated correctly —
```
∀x (human(x) → mammal(x))
∀x (mammal(x) → animal(x))
```
This is a correctly-formed multi-hop universal rule chain; it produced `Uncertain` not because the logic was malformed, but because no premise sentence asserted a ground fact instantiating `human(_)` for any concrete individual (see §11 audit note) — a genuinely interesting boundary case, not a translation defect in the narrow sense.

**idx 202 / 203** (`Ailton`): existential and simple-atom hypotheses translated cleanly and compiled; the theory correctly returned `Uncertain` and matched gold exactly, since the underlying premises genuinely don't settle Fluminense-related claims.

## 13. Examples of failed translation

See §11 for idx 9 (negation node-shape), idx 51 (missing forall), idx 199/200 (missing proposition, hallucinated quantifier, entity drift) with exact raw model output quoted in `error_analysis.json`.

## 14. Is Qwen3-8B adequate as the semantic parser (as prompted here)?

**Not at this prototype's prompting strategy.** 30% per-item schema validity and 13.3% end-to-end official-label accuracy (50% among the minority that even produced a verdict) is far below what would justify moving to the correction/verification layer. The failure modes are concentrated and structural (a specific quantifier-wrapper omission, node-shape confusion for negation, entity-name drift, dropped relations) rather than diffuse or exotic — which is actually the encouraging part: these look like fixable prompting/schema-communication problems (more few-shot coverage of the "If X then Y" → `forall` pattern specifically, an explicit reminder to reuse the exact same entity string across sentences, a stricter reminder about the `not` node shape, more generation budget for long enumerations) rather than evidence that an 8B model fundamentally cannot do this task. That distinction matters for what to try next, but it is not evidence this prototype's configuration is "good enough" — it is not.

## 15. Is predicate/entity normalization a major bottleneck?

**Yes, confirmed directly, not hypothetically.** idx 199-203 is direct, observed evidence: one real-world entity ("Ailton Silva") was rendered as three distinct, unmerged IR constants across sentences within a single theory, which alone is sufficient to break the entailment the theory needed. This validates the concern raised in `normalization.py`'s own design notes before any data was collected. It sits alongside — not instead of — the missing-forall and dropped-relation problems as a top contributor; a full accounting of "how much of the 87% end-to-end failure is normalization vs. dropped propositions vs. quantifier-shape" needs a larger, purpose-built ablation (e.g. running the normalization Tier-2 clustering from `normalization.py` against these translations and re-measuring), which is future work, not something this 30-item prototype can cleanly decompose on its own.

## 16. Ready to connect to the correction/verification layer?

**No.** Per the phase's own stop condition, this was never attempted, and the numbers above are the reason not to attempt it yet: with only 8/30 items even producing a symbolic verdict, a RETAIN/CORRECT/ABSTAIN layer built on top of this translator would spend the overwhelming majority of its time in ABSTAIN (translation failure) rather than doing any actual verification — which would not be a meaningful test of the neuro-symbolic bridge hypothesis. The translator needs iteration first (see §14's specific, targeted suggestions), evaluated the same way (frozen prototype sample, no peeking at official labels during iteration), before the correction layer is worth building.

## Infrastructure note (not a translation-quality finding)

One item (idx 172's hypothesis sentence) took **4,797 seconds (≈80 minutes)** for a single generation call that produced only ~280 tokens — 100-1000x slower than every other call in the same run under identical settings. Solver latency across all items was 0.3-163ms, confirming reasoning is not the bottleneck anywhere in this pipeline. The translation-latency outlier looks like a transient system-level stall (thermal/driver/contention on the laptop GPU) rather than a property of that specific input, and is reported here for transparency rather than folded into any accuracy number.

## What this phase did NOT do (by design)

- Did not re-run or prompt-tune after seeing the 13.3%/50% numbers — this is the frozen, one-shot result, exactly as instructed.
- Did not touch the 360-example ProofWriter held-out test.
- Did not attach a correction/verification (RETAIN/CORRECT/ABSTAIN) layer.
- Did not give Qwen gold FOL, gold labels, or Z3 results at any point (`tests/test_folio_field_whitelisting.py` proves this structurally, not just by convention).
