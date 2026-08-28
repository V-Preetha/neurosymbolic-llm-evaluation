# Translation-v2 — Frozen-30 Results (V1 vs. V2-C)

Status: **V2-C (full AST-guided pipeline, theory-shared vocabulary, no repair)
frozen and evaluated once** on the exact same 30-item sample used for
Translation-v1 (`reports/folio_translation_prototype/sample_manifest.json`).
**V2-D (repair) was not run to completion on the frozen 30** — see §7; a
small dev-set preview is reported instead, by explicit decision, given the
wall-clock cost documented in §6. Raw artifacts:
[`frozen30_variant_C.jsonl`](frozen30_variant_C.jsonl),
[`metrics_variant_C.json`](metrics_variant_C.json),
[`error_analysis_variant_C.json`](error_analysis_variant_C.json),
[`dev_run_variant_D.jsonl`](dev_run_variant_D.jsonl). Design rationale:
[`TRANSLATION_V2_DESIGN.md`](../../TRANSLATION_V2_DESIGN.md).

## Central question

> Can an AST-guided, decomposed neural-to-symbolic translator substantially
> improve the reliability of raw-English → formal-logic conversion compared
> with the original one-shot Qwen3-8B translator?

**Not yet, on end-to-end accuracy — the headline numbers are statistically
tied.** But the architecture measurably changed *which* failure mode
dominates, in exactly the direction the design predicted (gross JSON/tree
malformation almost disappeared; a specific, describable quantifier-scope
bug became the dominant failure instead). That's a real, actionable result,
just not yet a win on the metric that matters most for NSCE.

## 1. Full test-suite result

**162/162 passing** (existing ProofWriter/FELM/v1-FOLIO suite + 46 new
Translation-v2 unit tests across `test_folio_v2_vocabulary.py`,
`test_folio_v2_ast_builder.py`, `test_folio_v2_validator.py`,
`test_folio_v2_decomposer.py`, `test_folio_v2_repair.py`,
`test_folio_v2_pipeline.py`). Translation-v1 (`src/folio/translator.py`) and
the shared backend (`ir.py`, `schema.py`, `compiler.py`, `solver.py`,
`io_guard.py`, `normalization.py`) are untouched. ProofWriter path
unaffected.

## 2. Development process summary

- Dev subset: 18 items from FOLIO's validation split, disjoint from both
  the frozen 30 and the 18 feasibility-audit items (`select_dev_sample.py`,
  `dev_manifest.json`).
- Smoke-tested the pipeline against the real model before any frozen run;
  immediately observed the predicted dominant failure mode (unbound
  variable from a missing/misplaced `forall`) and, separately, confirmed
  the decomposition JSON schema itself is easy for the model to follow
  (clean atoms/nodes/root output on the first try in most cases).
- Found and fixed a real bug during dev debugging: the first repair prompt
  caused Qwen to either echo the broken JSON back unchanged, or "fix" it by
  moving `root` to a new node that silently orphaned part of the original
  structure. Rewrote the repair prompt with an explicit two-option
  procedure and a worked example. This raised repair success on the same
  failing dev items from 0/4 to 1/4 — a real but partial improvement (see
  §7).
- Never touched the frozen 30 or gold FOL during any of this.

## 3. V1 vs. V2-C metric table

| Metric | V1 | V2-C |
|---|---:|---:|
| Schema/AST-valid rate, per sentence | 78.9% | 74.4%¹ |
| Schema/AST-valid rate, per item (whole theory) | **30.0%** (9/30) | **26.7%** (8/30) |
| Z3-compilable rate, per item | 26.7% (8/30) | 26.7% (8/30) |
| Gold-FOL → Z3 oracle agreement | 50.0% (4/8 comparable) | 50.0% (4/8 comparable) |
| Official-label accuracy, over all 30 | **13.3%** (4/30) | **13.3%** (4/30) |
| Official-label accuracy, over items with a verdict | 50.0% (4/8) | 50.0% (4/8) |
| Mean wall-clock per item | 323s | 1275s² |

¹ Not perfectly apples-to-apples with V1 — see the measurement caveat in §5.
² Confounded by hardware, not an architectural cost — see §6.

**On this n=30 sample, V2-C is statistically indistinguishable from V1 on
every accuracy metric that matters.** Both produced a usable verdict for
exactly 8/30 items; both got the official label right on exactly 4/30; the
oracle-agreement rate is identical. This is the honest headline result and
it does not favor V2.

## 4. What actually changed: the error-mode shift

| Error category | V1 count | V2-C count |
|---|---:|---:|
| SCHEMA_ERROR (malformed JSON / broken node shape) | **32** | **2** |
| QUANTIFIER_SCOPE_ERROR (unbound variable / shadowing) | 10 | **32** |
| PREDICATE_ERROR / PREDICATE_ARITY_ERROR | 1 | 6 |
| ENTITY_ERROR (new category — v1 had no entity table) | n/a | 6 |

This is the real finding. Translation-v1's dominant failure was the model
simply not producing well-formed nested JSON in one shot (32 occurrences).
Translation-v2's decomposition schema (flat atoms + id-referenced nodes)
essentially solved that — malformed-shape errors collapsed to 2. But the
same underlying weakness now shows up almost undisguised as a **specific,
describable quantifier-scope bug**: the model consistently wraps `forall`
around only the *antecedent* of an implication, leaving the *consequent*
unbound, instead of wrapping the whole implication. Concrete example, idx 6
("Humans are mammals."):

```json
{"atoms": [{"id": "A1", "predicate": "human", "args": ["x"]},
           {"id": "A2", "predicate": "mammal", "args": ["x"]}],
 "nodes": [{"id": "N1", "op": "forall", "variables": ["x"], "body": "A1"},
           {"id": "N2", "op": "implies", "antecedent": "N1", "consequent": "A2"}],
 "root": "N2"}
```

i.e. it built `(∀x human(x)) → mammal(x)` instead of `∀x(human(x) →
mammal(x))` — `x` in `mammal(x)` is outside the quantifier. This exact
pattern recurred across most of the 32 `unbound_variable`/
`variable_shadowing` cases. It's the same pattern the repair prompt's
worked example targets (§7), and the same one the two failed dev-set
repairs also got wrong in a related way (moving the quantifier to wrap the
wrong sub-tree). This reads as a genuine Qwen3-8B capability limitation in
this specific spot — assigning quantifier scope over an antecedent →
consequent boundary — not a defect in the decomposition architecture,
which is otherwise doing its job (see §9's positive findings below).

`PREDICATE_ARITY_ERROR` (6) and `ENTITY_ERROR` (6) are new categories v1
could not surface distinctly: the theory-shared vocabulary layer caught
genuine cross-sentence inconsistencies (e.g. `help` used with 2 args in one
sentence of a theory and 3 in another — idx 76/77/78) and one repeated
decomposition-hygiene bug where the model listed its own atom ids (`"A2"`,
`"A3"`, `"A4"`) as if they were entity surface forms, which happens to
canonicalize to variable-shaped strings and is now caught explicitly
(`entity_variable_collision`, idx 173/174) rather than silently producing
wrong Z3 semantics.

## 5. Measurement caveat (own this one honestly)

For 6/30 items, the theory-wide vocabulary build itself failed (a genuine
cross-sentence `arity_conflict`, caught by `PredicateTable.build()` before
any per-sentence attempt is recorded). The pipeline's early-return path for
this case does not preserve the individual per-sentence decomposition
results in the output record, even though those LLM calls did happen. This
drops ~37 sentences from V2's per-sentence denominator that are present in
V1's, so the **per-sentence** valid-rate comparison (78.9% vs. 74.4%) is
not perfectly apples-to-apples. The **per-item** (headline) rate is
unaffected — those 6 items are correctly counted as invalid either way.

## 6. Latency — confounded by hardware, not a clean comparison

V2-C's frozen-30 run took **~10.6 wall-clock hours** for 30 items, driven
by a handful of catastrophic per-item slowdowns (idx 199: 2.3h; idx 201:
2.4h; idx 173: 1.3h for one item) inside an otherwise-normal 60-900s/item
distribution. `nvidia-smi` during the run showed normal temperatures and
clock speeds (no thermal throttling) but VRAM at 7.3/8.1 GB on an 8GB
laptop GPU — consistent with Windows' CUDA driver falling back to
much-slower shared system memory when a particular item's KV-cache pushed
past the VRAM ceiling. V2-C actually issued *fewer* total LLM calls than V1
(162 vs. 199, because the 6 vocabulary-failed items' sentences weren't all
attempted-and-recorded per §5) yet took ~4x longer wall-clock — a hardware
artifact of this specific run, not a property of the architecture. Any
future timing comparison should run on a GPU with real memory headroom, or
add an explicit per-call generation timeout.

## 7. Repair (V2-D) — dev-set preview only, not run on the frozen 30

Given §6's run took 10.6 hours for V2-C alone, and V2-D would add a second
LLM call for every failing sentence (roughly 32+ more calls, each subject
to the same VRAM-overflow risk), running V2-D to completion on the frozen
30 was not attempted this session. Instead, before/after repair is reported
on the 4 unbound-variable sentences hit during dev debugging (§2):

| | count |
|---|---:|
| Repair attempted | 4 |
| Repair succeeded | 1 (25%) |
| Repair failed (`repair_failed`) | 3 |

The one success matched the repair prompt's own worked example closely
(wrap the `and` of antecedent+negated-consequent in a `forall`, converting
to `implies`). The three failures show the model either reproducing the
broken JSON unchanged, or applying the fix to the wrong node (adding a
variable that doesn't even appear in any atom; moving the quantifier to
wrap a different sub-tree than the one that needed it) — the same
scope-reasoning weakness identified in §4, now visible in the repair
attempt itself. This is a small sample (n=4) and should not be read as a
precise repair success rate — it's a preview, not a result.

**A concrete, cheap next step**: the §4 bug pattern (`forall` wrapping only
an implication's antecedent) is specific and mechanical enough that a
**deterministic auto-fix** (detect `implies` whose antecedent is a
lone-atom `forall` and whose consequent uses the same unbound variable;
rewrite to `forall` wrapping the whole `implies`) could plausibly repair a
large fraction of these **without any further LLM call at all** — worth
trying before investing in more repair-prompt LLM calls.

## 8. Predicate/entity vocabulary — not fully exercised

Tier-2 (embedding-based) predicate clustering was not wired into this run
(`embed_fn=None` throughout) — only tier-1 surface canonicalization ran.
The literal `researcher`/`is_researcher`/`works_as_researcher` case the
phase spec names would only be caught by tier-1 for the `is_researcher`
variant (its `is_` prefix is stripped); `works_as_researcher` would need
tier-2 embedding similarity, which wasn't exercised here. Wiring
`src/retrieval/embeddings.EmbeddingModel` in as `embed_fn` is a small,
already-supported change (`PredicateTable.build(..., embed_fn=...)`) not
done this session for time reasons.

## 9. Examples

**V2 fixed what V1 failed** (idx 10, gold label False): the sentence "Tom
is also not a Gould's wild turkey, or a Merriam's wild turkey, or a Rio
Grande wild turkey" — a 6-way disjunction-with-negation — broke V1's
one-shot nested-JSON generation (`malformed_node`). V2's flat
atoms-then-references decomposition handled it cleanly; the whole item
validated, compiled, and produced the correct verdict (`contradicted` →
`False`, matching gold).

**V2 still fails** (idx 6, "Humans are mammals."): see §4 — the
antecedent-only quantifier-scope bug. Structurally simple, single-variable,
single-implication sentence; still fails.

## 10. Is the bridge reliable enough for the NSCE correction loop?

**No — do not connect it yet**, per the phase's own stop condition. Both
V1 and V2-C fail to produce any verdict on 22/30 and produce the *wrong*
verdict or no verdict on 26/30 items. V2-C's architectural changes are
real and traceable (§4, §9) but have not yet moved the headline accuracy
number on this frozen sample. The dominant remaining failure looks
capability-limited (a specific, nameable Qwen3-8B quantifier-scope
weakness — §4, §7) rather than a gap in the decomposition/AST-builder/
validator architecture itself, which is why §7 suggests trying a
deterministic fix for that one pattern before spending more LLM calls on
it. A full, properly-resourced V2-D run (with a VRAM-safe environment or a
generation timeout) is the natural next measurement before revisiting this
question.
