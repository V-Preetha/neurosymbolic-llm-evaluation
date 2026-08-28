# Translation-v2 — Structured, AST-Guided NL→FOL Design

Status: implemented, unit-tested, dev-debugged; frozen and evaluated once on
the original frozen-30 Translation-v1 set for a clean paired comparison
(see `reports/folio_translation_v2/`).

## 0. Why this exists

Translation-v1 (`src/folio/translator.py`) asks Qwen3-8B to produce one
*complete, correctly-nested* JSON logic tree per sentence in a single
generation. On the frozen 30-item evaluation set this produced:

| Metric | V1 |
|---|---:|
| Whole-theory schema valid | 9/30 (30.0%) |
| Z3 compilable | 8/30 (26.7%) |
| Official-label accuracy | 4/30 (13.3%) |
| Gold-FOL → Z3 oracle agreement | 29/30 |

The last row shows the *backend* (IR, schema validation, compiler, solver)
is not the bottleneck — it reproduces the official label correctly from
gold FOL 29/30 times. The bottleneck is exactly what the task states:
**raw natural language → correct symbolic logic**, specifically a one-shot
8B model getting brace-nesting, quantifier placement, and cross-sentence
predicate naming right in a single unconstrained generation.

## 1. Prior-method inspiration vs. our implementation vs. our contribution

**Prior-method inspiration.** A body of NL→FOL / compositional semantic
parsing work (broadly: "NL2LOGIC"-style pipelines and compositional
neural-symbolic parsers) has shown that one-shot end-to-end generation of a
complete logical form is fragile, and that decomposing translation into
(a) identifying semantic components (predicates, entities, quantifiers)
before (b) assembling them into a formal structure improves syntactic
executability and semantic fidelity, versus asking a model to emit the
whole nested structure at once. We did not invent this idea; we adapt its
*architectural* shape (decompose → normalize → assemble → validate → repair)
to this project's frozen 8-node FOLIO IR and existing Z3 backend.

**Our implementation.** Everything in `src/folio/v2/` is new code written
against this project's own IR (`src/folio/ir.py`), written specifically so
the LLM never has to get bracket-nesting right and never has to invent
predicate names independently per sentence. Concretely:

- A single structured-decomposition prompt per sentence (not a whole
  theory, not a nested AST) — see §2.
- A component-reference graph (atoms + nodes-by-id) instead of an inline
  nested tree — the model writes `{"op": "implies", "antecedent": "N1",
  "consequent": "A4"}`, never `{"antecedent": {"type": "and", "args": [...]}}`.
- Deterministic, theory-shared entity/predicate vocabulary tables
  (`vocabulary.py`) built on top of this project's *existing*,
  already-tested `normalization.py` (previously implemented but never
  wired into a live pipeline — v1's translator never called it).
- A deterministic graph-to-AST resolver (`ast_builder.py`) that recursively
  walks `root` through `atoms`/`nodes` into the exact same `ir.Expr`
  dataclasses `schema.py`/`compiler.py`/`solver.py` already consume.
- A validator (`validator.py`) that composes `ast_builder` + the
  project's existing `schema.py` rather than reimplementing scope/arity
  checking.
- A one-shot, reason-gated repair pass (`repair.py`).

**Our NSCE integration.** None yet, by explicit instruction (§20 of the
phase spec). Translation-v2's output format (`premise_dicts`/
`hypothesis_dict`, identical in shape to Translation-v1's) is compatible
with the existing `solver.classify_theory` call used everywhere else in
this codebase, so wiring it into the NSCE correction loop later is a
call-site change, not a redesign — but that wiring is explicitly out of
scope for this phase.

**Our experimental contribution.** The ablation design in §6 (which of
{decomposition, global vocabulary, repair} account for how much of the
improvement), and the paired V1-vs-V2 comparison on the same frozen 30
items, gold-FOL oracle, and Z3 backend.

## 2. Pipeline

```
RAW ENGLISH (one sentence: one premise, or the hypothesis)
    |
    v   decomposer.py -- ONE Qwen3-8B call per sentence
semantic decomposition JSON:
    entities: [{surface, canonical}]        -- proper-name constants (Step B input)
    atoms:    [{id, predicate, args}]       -- flat predicate applications (Step D "clauses")
    nodes:    [{id, op, ...}]               -- boolean/quantifier structure, by id-reference (Step E/F)
    root:     "<atom or node id>"
    |
    v   vocabulary.py -- deterministic, theory-shared
EntityTable  (surface -> canonical constant, tier-1 normalization.py reuse)
PredicateTable (raw name -> canonical name+arity, tier-1 + tier-2 normalization.py reuse)
    |
    v   ast_builder.py -- deterministic, Step F
    recursively resolves `root` through atoms/nodes into ir.Expr
    (ir.Predicate / Not / And / Or / Implies / Iff / ForAll / Exists --
     the exact same 8-node grammar as Translation-v1, unchanged)
    |
    v   validator.py -- deterministic, Step G (reuses schema.py entirely)
    1. ast_builder.build_expr (dangling/duplicate/cyclic ids, unknown ops)
    2. schema.validate_scope, per sentence (unbound vars, shadowing)
    3. schema.validate_theory, once over premises+hypothesis (cross-sentence arity)
    |
    v   [repair.py -- at most one Qwen3-8B call, only for repairable reasons]
    |
    v   ir.to_dict(expr) -> src/folio/solver.classify_theory (UNCHANGED from v1)
    |
    v   formal verdict (True / False / Uncertain / inconsistent_premises / ...)
```

The LLM is called at most twice per sentence (decompose, optional repair)
and never sees Z3, never sees the whole theory in one generation, and
never sees a gold label or gold FOL at any stage (`io_guard.FolioAllowedInput`
still gates every entry point exactly as in v1; `decompose_sentence`'s only
inputs are `runner` and one raw sentence string).

## 3. Decomposition schema (Step A / C / D / E)

See `src/folio/v2/decomposer.py`'s `DECOMPOSE_PROMPT` for the exact prompt
and worked examples (including the "Some scientist admires every
researcher" nested-scope example that distinguishes ∃y(Scientist(y) ∧
∀x(...)) from a flattened, non-equivalent reading — the case §7 of the
phase spec calls out explicitly).

Node `op` values are exactly the 7 non-`predicate` node types of the
existing frozen IR (`not`, `and`, `or`, `implies`, `iff`, `forall`,
`exists`) — the decomposition graph is structurally the same grammar as
`ir.py`, just with children referenced by id instead of nested inline, so
`ast_builder.py` needs no translation logic beyond graph resolution and
vocabulary substitution.

## 4. Global entity/predicate tables (Step B / C)

`vocabulary.EntityTable` and `vocabulary.PredicateTable` are built once per
theory (across every premise and the hypothesis) and reused as each
sentence's atoms are resolved into `ir.Predicate` nodes:

- Entities: tier-1 surface canonicalization only (`normalization.
  canonicalize_entity_surface`), per that module's own documented
  scope — cross-sentence entity coreference beyond exact-string-after-
  canonicalization is a translator-prompt responsibility (the model
  is asked to emit a consistent canonical form), not attempted here.
  Guards against a canonicalized entity accidentally colliding with the
  variable namespace (`entity_variable_collision`, new in v2 — v1 had no
  entity table at all).
- Predicates: tier-1 + tier-2 (`normalization.normalize_predicates`),
  clustering same-theory predicate names by embedding cosine similarity
  when an `embed_fn` is supplied. This is the direct fix for the v1
  failure mode named in the phase spec: `researcher` / `is_researcher` /
  `works_as_researcher` becoming three unrelated Z3 predicates. Arity
  conflicts for the same canonical name are raised as `TranslationFailure
  ("arity_conflict", ...)` at vocabulary-build time, before any AST is
  built.

## 5. Static validation before Z3 (Step G)

`validator.py` never calls Z3. It composes three already-narrow checks
rather than reimplementing them: `ast_builder.build_expr` (graph-shaped
errors), `schema.validate_scope` (per-sentence, so a failure is
attributable to exactly one sentence — required for repair eligibility),
and `schema.validate_theory` (cross-sentence arity, run once at the end of
`pipeline.translate_theory_v2` regardless of ablation setting, so the
"no compilation on a structurally invalid theory" guarantee holds even in
the no-global-vocabulary ablation).

Error taxonomy (raised as `ir.TranslationFailure(reason, detail)`, same
exception type v1 already used, extended with new reasons this
architecture can now distinguish):

```
invalid_json                 -- decomposer output isn't parseable JSON
missing_field                -- required key absent (id, predicate, root, ...)
malformed_node                -- wrong type/shape (e.g. and/or with <2 args)
unknown_operator              -- node.op not one of the 7 structural ops
dangling_reference            -- id referenced but not defined
duplicate_id                  -- same id reused across atoms/nodes
cyclic_reference               -- id graph contains a cycle
unbound_variable              -- ir.py/schema.py, reused unchanged
variable_shadowing            -- ir.py/schema.py, reused unchanged
entity_shaped_variable        -- ir.py/schema.py, reused unchanged
entity_variable_collision     -- new: canonicalized entity is variable-shaped
arity_conflict                -- ir.py/schema.py, reused unchanged
unknown_predicate_reference   -- atom names a predicate vocabulary never saw
repair_failed                 -- the one repair attempt still didn't validate
```

## 6. Repair (Step 10)

`repair.py`'s `REPAIRABLE_REASONS` whitelist is exactly the sentence-local,
structural reasons above — never `arity_conflict` (spans two sentences;
fixing it would mean guessing which sentence is "wrong", which the phase
spec forbids), never `unknown_predicate_reference` (would require renaming
a predicate, which the repair prompt explicitly forbids the model from
doing). Enforced by `pipeline._validate_and_maybe_repair`: at most one
`repair.attempt_repair` call per sentence; if the repaired decomposition
still fails validation, the sentence's final reason becomes
`repair_failed` and no further attempt is made. `pre_repair_valid` is
recorded on every `SentencePipelineResult` so before/after-repair metrics
can be computed without a second run.

## 7. Ablation surface — what's actually toggled, and why not a literal A/B/C/D switch

`pipeline.translate_theory_v2` exposes two flags:

- `use_global_vocabulary`: `False` gives every sentence its own throwaway
  `EntityTable`/`PredicateTable` (no cross-sentence identity sharing);
  `True` shares one pair across the whole theory.
- `enable_repair`: gates the one-shot repair pass.

The AST-guided, id-graph decomposition + deterministic assembly
(`decomposer.py` + `ast_builder.py`) is **not** an ablatable toggle in this
implementation — it is the single load-bearing mechanism the whole design
rests on (Step F), used identically whether or not global vocabulary or
repair is enabled. A literal V2-A ("decomposition only, no AST-guided
assembly") would require a second, materially different assembly code path
(e.g. a flat conditions/conclusion reader that cannot represent nested
quantifier scope), which duplicates real backend logic for a config this
phase's own priority note says can be skipped under time pressure ("If
runtime is prohibitive, prioritize V1, V2-C, V2-D"). We therefore map:

| Ablation label | `use_global_vocabulary` | `enable_repair` | What it measures |
|---|---|---|---|
| V2-A | `False` | `False` | decomposition's structural win alone, without cross-sentence naming consistency |
| V2-B / V2-C | `True` | `False` | + theory-shared entity/predicate vocabulary (this is "the full AST-guided pipeline" in our implementation, since AST-guided assembly is constant across all rows) |
| V2-D | `True` | `True` | + constrained repair |

This is a deliberate, documented simplification of the phase spec's literal
4-way ablation grid, not an oversight — see `reports/folio_translation_v2/`
for which rows were actually run and why, given the same runtime
constraints the phase spec itself anticipates.

## 8. What is reused unchanged from Translation-v1's backend

Per the phase spec's explicit instruction not to replace working code:
`src/folio/ir.py`, `src/folio/schema.py`, `src/folio/compiler.py`,
`src/folio/solver.py`, `src/folio/io_guard.py`, and
`src/folio/normalization.py` (previously implemented and tested, now
wired into a live pipeline for the first time) are all untouched.
`src/folio/translator.py` (Translation-v1) is untouched and remains the
frozen baseline; nothing in `src/folio/v2/` imports from it except the
shared IR/schema/solver/io_guard/normalization modules every FOLIO
component already depends on.

## 9. NSCE correction loop — explicitly not connected

Per §20 of the phase spec, this phase stops at "does Translation-v2
substantially improve the reliability of raw-English → formal-logic
conversion" and does not proceed to `Qwen answer -> NSCE correction`. See
the accompanying results report for whether the bridge looks reliable
enough to justify that next step.
