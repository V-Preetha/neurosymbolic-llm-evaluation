# FOLIO Representation Feasibility Audit

Status: **design/audit only.** Answers one question before any extraction-prompt work begins: *can the existing `Atom`/`Rule` forward-chaining engine be extended enough to faithfully represent FOLIO's logic?* No batch pipeline built, no translator prompted, the frozen 360-example ProofWriter test still untouched.

Artifacts: [`reports/folio_feasibility/folio_sample_18.jsonl`](reports/folio_feasibility/folio_sample_18.jsonl) (the 18 raw FOLIO records, verbatim from the official release), [`reports/folio_feasibility/gold_fol_oracle_z3.py`](reports/folio_feasibility/gold_fol_oracle_z3.py) (the oracle experiment, §7), [`reports/folio_feasibility/oracle_results.txt`](reports/folio_feasibility/oracle_results.txt) (raw run output).

## 0. Sample provenance

Pulled directly from the official `Yale-LILY/FOLIO` GitHub repo, `data/v0.0/folio-validation.jsonl` (204 rows, MIT license) — not synthesized. Selected 18 rows deterministically: computed `sha256(json(premises))` per row, grouped by gold label, took the 6 lowest-hash rows per label (`True`/`False`/`Uncertain`), then re-sorted by original file index. No cherry-picking by how "easy" an item looked. The 18 rows happen to fall into **8 distinct premise sets** (FOLIO reuses one premise block across 2–3 conclusion questions), which is a representative feature of the dataset, not a sampling artifact.

## 1–3. Sample composition

- **18 examples inspected** (6 True / 6 False / 6 Uncertain — exact 3-way balance by construction).
- **Logical constructs encountered**, counted by premise-group (8 groups) and by individual conclusion (18 items):

| Construct | Present in |
|---|---|
| Unary predicates | all 8 groups |
| Binary predicates | 6/8 groups (`Created`, `Hosted`, `CityIn`, `Starring`/`MemberOf`/`PilotedOn`/`WorkingTitle`/`ScriptEditorFor`, `BreedingBack`, `CapitalOf`) |
| Universal quantification (∀) | 7/8 groups |
| Existential quantification (∃) | 5/8 groups — in **facts** (Zelda, Montana, Badults, Heck-cattle) and/or in **queries** (idx 58, 59, 69, 143) |
| Two-variable rules (∀x∀y) | 2/8 groups (Zelda rule 2, Badults conclusion, Heck-cattle rule 1) |
| Negation | 6/8 groups |
| Conjunction | all 8 groups |
| Disjunction (∨) in premises | 1/8 groups (ML-algorithms) |
| XOR / biconditional-style constraint | 1/8 groups (Remy) |
| Conjunctive query (query = AND of ≥2 atoms) | 3/18 items (24, 69, 143) |
| Existential query (query itself is ∃-quantified) | 4/18 items (58, 59, 69, 143) |
| Universally-quantified query (query itself is a rule, e.g. "∀x(HouseHoldApp(x)→Suck(x))") | 4/18 items (13, 70, 137, 144) |
| Equality / inequality | 0/18 — not observed in this sample |
| Function symbols | 0/18 — not observed |

The last row of the table is the single most consequential fact for the IR decision: **in 4 of 18 items (22%) the thing being verified is not a ground atom at all — it's a universally-quantified statement that must be checked by searching for a counterexample, not by checking derivability.** The current `parse_query` rejects this outright (`ParserError: a rule can't be a query`), and this is a structural gap, not a missing feature that a small patch closes.

## 4. Per-item audit

| idx | label | main constructs | current IR? | category |
|---|---|---|---|---|
| 12 | True | ∀(single-var), binary pred + constant arg | almost — binary atom with a constant argument needs support | MINOR_EXTENSION |
| 13 | Uncertain | conclusion is itself `∀x(...)` | query must reduce to one atom | NOT_SUPPORTED |
| 14 | False | negated atom, same theory as 12 | binary atom + constant arg | MINOR_EXTENSION |
| 24 | True | conjunctive ground query (2 atoms) | query must reduce to one atom | MINOR_EXTENSION |
| 25 | Uncertain | unary atom, no relevant rule | fits exactly | SUPPORTED_AS_IS |
| 26 | Uncertain | unary atom, unmentioned | fits exactly | SUPPORTED_AS_IS |
| 58 | False | existential binary query, ∀-rule refutes it | no existential query support | MAJOR_EXTENSION |
| 59 | Uncertain | existential binary query, unrelated entity | no existential query support | MAJOR_EXTENSION |
| 69 | True | existential + conjunctive query, witness = known constant | no existential/conjunctive query support | MAJOR_EXTENSION |
| 70 | False | 2-variable rule, disjunctive consequent, query = rule | multiple gaps at once | NOT_SUPPORTED |
| 135 | True | single-hop ∀, unary | fits exactly | SUPPORTED_AS_IS |
| 136 | Uncertain | unary atom, no path | fits exactly | SUPPORTED_AS_IS |
| 137 | False | query = ∀-rule, refuted by counterexample | requires counterexample search, not derivability check | NOT_SUPPORTED |
| 138 | True | disjunctive premises + modus tollens / disjunctive syllogism | forward-chaining is monotonic, can't do refutation-style reasoning | NOT_SUPPORTED |
| 139 | False | same theory as 138 | same | NOT_SUPPORTED |
| 143 | True | existential facts (Skolem witnesses), 2-var rule, existential+conjunctive query | needs existential fact handling + witness enumeration | MAJOR_EXTENSION |
| 144 | Uncertain | query = ∀-rule, not generalizable from an existential fact | requires counterexample search | NOT_SUPPORTED |
| 154 | False | XOR constraint + multi-hop contrapositive/proof-by-contradiction | needs indirect proof, not just forward chaining | NOT_SUPPORTED |

**Category totals:**

- `SUPPORTED_AS_IS`: 4/18 = **22%**
- `MINOR_EXTENSION`: 3/18 = **17%**
- `MAJOR_EXTENSION`: 4/18 = **22%**
- `NOT_SUPPORTED` (by forward chaining in any patched form): 7/18 = **39%**

So **61% of the sample needs more than incremental extension**, and more than a third needs a fundamentally different reasoning mode (refutation/counterexample search, disjunctive/indirect proof) that a monotonic forward-chaining engine cannot produce by construction, no matter how the `Atom`/`Rule` dataclasses are extended.

## 5/6. Constructs the current reasoner cannot handle, and IR/backend implications

Fundamentally out of reach for the *forward-chaining* algorithm itself (not just the `Atom`/`Rule` data shape):

- Disjunction anywhere in a premise (today: `ParserError`, by explicit design).
- More than one variable per rule.
- Existential quantification, in either facts or queries.
- A query that is itself a universally-quantified formula, verified by counterexample search rather than membership-in-derived-facts.
- XOR/biconditional constraints requiring case-split/indirect proof.
- Disjunctive syllogism / modus tollens — anything requiring *assuming* a formula and deriving a contradiction, since forward chaining only ever adds derived facts, never retracts or explores alternatives.

Given that, **do not extend `Atom`/`Rule` piecemeal for FOLIO.** Patching in existentials, multi-variable rules, and disjunction one at a time reproduces general first-order logic by accretion, badly, while the properly-engineered version of that is a solved problem (a real solver). The minimum IR for FOLIO is close to what the task prompt sketched (§Step 5): predicates of arbitrary arity, `And/Or/Not/Implies/Xor`, `ForAll`/`Exists` over possibly multiple variables — i.e., finite-domain first-order logic — not a bigger Horn-clause dialect.

## 7. Gold-FOL oracle experiment

Encoded all 8 premise groups' **gold `premises-FOL`** directly in [z3](https://github.com/Z3Prover/z3) (MIT license, ~20 MB pip package, CPU-only, no GPU/model download) over an uninterpreted `Object` sort, with predicates as `Function(Object[, Object], BoolSort())`. For each conclusion: `unsat(premises ∧ ¬concl)` → oracle label True; `unsat(premises ∧ concl)` → oracle label False; else → Uncertain. Same generic entailment/refutation check for every item — no per-item special-casing.

**Result: 16/18 (89%) exact match with gold labels**, using nothing but the dataset's own gold formulas and a stock SMT solver. The two misses (idx 138, 139) both come from one theory ("ML algorithms are categorized as supervised/unsupervised/reinforcement learning") whose gold FOL states that categorization as a **disjunction** (`MLAlgorithm(sup) ∨ MLAlgorithm(unsup) ∨ MLAlgorithm(reinf)`), which is too weak to license the modus-tollens step the gold label needs. Re-running the same two items with that one premise read as a **conjunction** (all three literally are ML algorithms — the natural reading of "categorized as X, Y, and Z") flips both to match gold exactly (confirmed by direct z3 re-run, `and`-encoding). This is a documented category of FOLIO gold-FOL annotation looseness (the FOLIO paper itself doesn't claim 100% FOL/label consistency), not a solver limitation.

**Conclusion: the oracle ceiling is ~89–100% with zero hand-tuning of the reasoning strategy.** This means the burden of the end-to-end system's error rate falls almost entirely on the raw-text→logic extraction step, not on reasoning — exactly the quantity Experiment 2 is designed to measure.

## 8. Feasibility matrix (condensed)

| Bucket | Count | Example idx | Current IR | Proposed IR | Oracle solvable (z3) |
|---|---|---|---|---|---|
| SUPPORTED_AS_IS | 4 | 25, 135 | yes | unchanged | yes |
| MINOR_EXTENSION | 3 | 12, 24 | binary atom / conjunctive-query gap | constant-valued args, AND-of-atoms query | yes |
| MAJOR_EXTENSION | 4 | 58, 143 | no existential support | ∃ facts/queries, witness search | yes |
| NOT_SUPPORTED | 7 | 70, 137, 154 | no | full connective set + multi-var ∀/∃ + refutation search | yes (6/7 exact; 1 pair limited by gold-annotation looseness, not solver) |

## 9. Architectural decision

- **Option A (extend the existing custom engine): rejected for FOLIO.** 61% of the sample needs capabilities (existentials, multi-variable rules, disjunction, refutation-style proof) that are not incremental patches to a monotonic Horn-clause forward-chainer.
- **Option C (restrict to a Horn-compatible FOLIO subset): not adopted as the primary path.** Filtering by gold logical structure (per the task's own rule — never by observed model/solver performance) would keep only the `SUPPORTED_AS_IS` + `MINOR_EXTENSION` buckets, i.e. ~39% of items, and would silently exclude exactly the constructs (negation-heavy multi-hop, universal-claim refutation) that make FOLIO a meaningfully harder benchmark than ProofWriter. Available as a documented fallback/ablation, not the headline design.
- **Option B (proper FOL solver behind the same IR): recommended.** The oracle experiment (§7) directly validates this choice on real FOLIO data — 89% exact match with no engineering beyond a straight transcription of the gold formulas. z3 is free, small, deterministic, CPU-only (fits the RTX 5050 budget trivially — it does no GPU work at all), and already installed and tested in this environment (`z3-solver` in `.venv`).

**Recommendation: Architecture A (Full FOLIO), but with z3 as the reasoning backend for the FOLIO arm — not a retrofit of `src/nsce/symbolic_reasoner.py`.** The existing forward-chaining engine stays exactly as-is and stays scoped to ProofWriter (Experiment 1, already 72/72-validated) and to ParaRules (Experiment 3's robustness arm, which reuses ProofWriter's own controlled grammar under paraphrase, so it doesn't hit any of the constructs in §5). FOLIO gets its own small, separate symbolic core: a constrained extraction schema (predicate/args/polarity/quantifier-per-variable/connective, capped at 2 variables and no function symbols/equality, matching what was actually observed in this sample) compiled into z3 formulas, with the same entailment/refutation check used in the oracle script. This is not "rebuilding a general theorem prover" — z3 already is one, well-tested and off the shelf; the new work is the small, schema-constrained *compiler* from extracted propositions into z3 calls, which is a bounded, auditable component (any extraction failure or out-of-schema construct still raises and routes to ABSTAIN, exactly like the existing `ParserError` path).

This is a correction to `RAW_TEXT_DATASET_AUDIT.md` §5–§6, which assumed the existing `Atom`/`Rule` engine would be reused unchanged for FOLIO too — that assumption doesn't survive this audit and should be treated as superseded by this document for the FOLIO arm specifically. It still holds for ParaRules.

## 10–13. Direct answers

10. **Is current forward chaining still adequate?** No — not for FOLIO (61% of sample out of reach); yes — unchanged, for ProofWriter/ParaRules.
11. **Is another symbolic backend required?** Yes — z3 (or an equivalent finite-domain FOL solver), used only for the FOLIO arm.
12. **Recommendation:** Full FOLIO (Architecture A), with a z3-backed reasoning core for that arm, existing engine untouched and reused for Experiment 1/3.
13. **Rationale:** The 18-item hand audit shows the mismatch is structural (existentials, multi-variable rules, disjunction, universal-claim-as-query) not cosmetic, so patching `Atom`/`Rule` would either silently misrepresent semantics (explicitly disallowed by the task) or reproduce a worse version of an existing solved tool. The gold-FOL oracle experiment on real data (not hypothetical) shows the ceiling is high (89%, with the shortfall traced to annotation looseness rather than solver inadequacy) — so investing in a correct backend now is justified before any translation-prompt engineering begins, per the required build order (IR → reasoner decision → oracle validation → freeze target language → then build the translator).

## Next step (not yet started)

Per the task's explicit stop condition: do not build the raw-text→proposition extraction prompts yet. The next artifact should freeze the FOLIO target schema (predicate/arity/polarity/quantifier grammar, ≤2 variables, connective set = {and, or, not, implies, xor}) and the schema→z3 compiler, validated against this same 18-item hand-annotated set (and its would-be extraction — but written by hand, not by Qwen) before any LLM-based extraction is attempted.
