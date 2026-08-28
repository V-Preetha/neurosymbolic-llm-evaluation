# NL2Logic + Z3 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a faithful recursive NL2Logic semantic parser, typed FOL AST, two-pass Z3 compiler, separate solver, complete parser traces, component-level bridge training records, and bridge metrics without changing experiment sampling.

**Architecture:** Qwen3-8B is injected into a recursive selector/subparser pipeline that returns only typed AST nodes. A declaration pass and expression pass compile validated ASTs to Z3, after which a separate solver returns True/False/Unknown. Evaluation persists every recursive trace step and uses the same pipeline for Conditions 2 and 4.

**Tech Stack:** Python 3.12, Qwen3-8B/Transformers, PEFT, Z3, `unittest`, JSONL.

**Spec:** `docs/superpowers/specs/2026-08-25-nl2logic-z3-bridge-design.md`

## Global Constraints

- Preserve Conditions 1-4 and all existing condition identifiers.
- Preserve frozen manifest loading and ProofWriter `main` training size 10,000.
- Condition 2 uses the base Qwen3-8B parser; Condition 4 uses its bridge adapter.
- Fine-tuning supervises semantic-parser component responses only, never compiler or Z3 behavior.
- Save successful, partial, and failed recursive parser events to condition-run `parser_traces.jsonl`.
- Run only unit, mock, and no-model smoke tests. Do not load Qwen weights, train an adapter, or run `main`/`full` evaluation.
- This workspace is not a Git repository, so commit steps are inapplicable; preserve unrelated files in place.

---

### Task 1: Typed AST validation

**Files:**
- Create: `final_experiment/fol_ast.py`
- Create: `final_experiment/tests/test_fol_ast.py`
- Modify: `final_experiment/bridge.py`

**Interfaces:**
- Produces: `BridgeError(status: str, detail: str)`, `extract_json(text)`, `validate_expr(node, arities=None, bound=None)`, `validate_theory(value)`, and `normalized_logic(value)`.
- Consumes: existing dictionary ASTs returned by `gold_targets.py`.

- [ ] **Step 1: Write failing AST tests** covering a valid quantified theory, free-variable rejection, arity-conflict rejection, and malformed JSON.
- [ ] **Step 2: Run** `python -m unittest final_experiment.tests.test_fol_ast -v` and verify missing-module failure.
- [ ] **Step 3: Implement** strict AST validation and canonical normalization in `fol_ast.py`; make `bridge.py` re-export the new functions.
- [ ] **Step 4: Re-run the targeted test** and then the existing unit suite.

### Task 2: Explicit two-pass compiler and separate solver

**Files:**
- Create: `final_experiment/z3_compiler.py`
- Create: `final_experiment/z3_solver.py`
- Create: `final_experiment/tests/test_z3_pipeline.py`
- Modify: `final_experiment/bridge.py`

**Interfaces:**
- Produces: `DeclarationTable`, `CompiledTheory`, `collect_declarations(theory)`, `compile_theory(theory)`, `solve_compiled(compiled, timeout_ms)`, `solve(theory, timeout_ms)`, and `semantic_agreement(predicted, gold, timeout_ms)`.
- Consumes: validated dictionary theories from Task 1.

- [ ] **Step 1: Write failing tests** that hand-check declaration contents, arity conflicts, True/False/Unknown results, inconsistent-premise failure, and equivalent/different semantic agreement.
- [ ] **Step 2: Run** the targeted test and verify missing APIs fail.
- [ ] **Step 3: Implement pass 1** as a pure traversal that collects constants and predicate arities and verifies bound-variable scope.
- [ ] **Step 4: Implement pass 2** using only the declaration table to construct Z3 expressions.
- [ ] **Step 5: Implement solver checks and semantic agreement** with explicit unknown/timeout status mapping.
- [ ] **Step 6: Re-run targeted and full unit tests.**

### Task 3: Recursive NL2Logic parser and exhaustive traces

**Files:**
- Create: `final_experiment/nl2logic_parser.py`
- Create: `final_experiment/tests/test_nl2logic_parser.py`
- Modify: `final_experiment/prompts.py`

**Interfaces:**
- Produces: `NL2LogicParser(generate, dataset, condition, run_id, trace_sink, max_depth=12)`, `parse_sentence(...)`, and `parse_theory(row)`.
- Generator signature: `(prompt: str, bridge: bool) -> {raw, latency_ms, input_tokens, output_tokens}`.
- Trace sink signature: `(event: dict) -> None`.

- [ ] **Step 1: Write a failing scripted-generator test** for selector → binary parser → two child selector/atomic paths and assert exact depth-first trace fields.
- [ ] **Step 2: Write failing failure-path tests** for malformed selector JSON and cyclic/unchanged child clauses; assert partial and failed trace events include failure reasons.
- [ ] **Step 3: Run** the targeted tests and verify the missing parser fails.
- [ ] **Step 4: Implement component prompts and JSON decoders** for parser selector, relation selector, four atomic parsers, quantified parser, binary parser, and unary parser.
- [ ] **Step 5: Implement recursive construction** with deterministic step IDs, parent/child paths, cycle detection, maximum depth, and per-node validation.
- [ ] **Step 6: Finalize every trace event in `finally`** so raw decisions, chosen subparser, child clauses, AST node, validation, and failure reason survive exceptions.
- [ ] **Step 7: Re-run targeted and full unit tests.**

### Task 4: Component-level bridge fine-tuning records

**Files:**
- Create: `final_experiment/nl2logic_training.py`
- Create: `final_experiment/tests/test_nl2logic_training.py`
- Modify: `final_experiment/training.py`
- Modify: `final_experiment/prompts.py`

**Interfaces:**
- Produces: `canonical_clause(node)`, `component_records(row)`, and `training_examples(rows, task)` returning `{prompt, target, text}` records.
- Consumes: gold theories from `proofwriter_target` and `folio_target` after manifest selection.

- [ ] **Step 1: Write failing tests** asserting that an implication yields selector, binary-parser, relation-selector, and atomic-parser records with JSON targets and that direct training remains one row per example.
- [ ] **Step 2: Run** the targeted tests and verify missing APIs fail.
- [ ] **Step 3: Implement deterministic canonical clause rendering and AST traversal** that creates component records for every gold node.
- [ ] **Step 4: Refactor training tokenization** to expand only already-selected bridge rows and mask every component prompt from loss.
- [ ] **Step 5: Re-run targeted and full unit tests without invoking `train()`.**

### Task 5: Evaluation integration and condition-specific JSONL traces

**Files:**
- Create: `final_experiment/tests/test_evaluation_bridge.py`
- Modify: `final_experiment/evaluation.py`
- Modify: `final_experiment/core.py`

**Interfaces:**
- Evaluation writes `predictions.jsonl` and `parser_traces.jsonl` under the condition-specific run directory.
- Bridge prediction translation fields include `ast_valid`, `compile_success`, `executable`, normalized agreement, and semantic agreement.

- [ ] **Step 1: Write a failing isolated bridge-evaluation test** using a scripted model runner and temporary output path; assert one prediction, complete trace JSONL, and matching held-out example ID.
- [ ] **Step 2: Run** the test and verify current one-shot bridge behavior fails it.
- [ ] **Step 3: Refactor Conditions 2/4** to instantiate the same `NL2LogicParser` with the condition-specific runner, parse the row, compile, solve, and accumulate latency/tokens.
- [ ] **Step 4: Persist traces after every example**, including parser exceptions, and populate parser/compiler/solver/error fields without converting system failures to Unknown.
- [ ] **Step 5: Compute exact normalized and Z3 semantic agreement** against dataset gold targets when available.
- [ ] **Step 6: Re-run targeted and full unit tests.**

### Task 6: Aggregate metrics, smoke workflow, and documentation

**Files:**
- Create: `final_experiment/tests/test_bridge_metrics.py`
- Modify: `final_experiment/analysis.py`
- Modify: `final_experiment/smoke.py`
- Modify: `final_experiment/README.md`

**Interfaces:**
- `analysis.translation(rows)` returns counts/rates for AST validity, compile success, executable examples, semantic-agreement categories, and final solver accuracy.
- No-model smoke exercises a scripted recursive parse, two-pass compilation, trace creation, and three-way solver.

- [ ] **Step 1: Write failing metric tests** with literal successful, partial, compile-failed, and semantically-equivalent fixtures.
- [ ] **Step 2: Run** the targeted test and verify missing aggregate fields fail.
- [ ] **Step 3: Implement aggregates** with explicit numerators and rates.
- [ ] **Step 4: Replace the one-shot bridge smoke check** with scripted recursive NL2Logic parsing and assert traces contain required fields.
- [ ] **Step 5: Update README** with architecture, trace artifact, metrics, unchanged sampling, and exact training/evaluation commands; state that smoke does not train.
- [ ] **Step 6: Re-run targeted and full unit tests.**

### Task 7: Final constrained verification

**Files:**
- Verify only; no production changes unless a failing test first reproduces the issue.

**Interfaces:**
- Consumes all prior tasks.
- Produces fresh test evidence and a requirements checklist.

- [ ] **Step 1: Run** `python -m unittest discover -s final_experiment/tests -v`.
- [ ] **Step 2: Run** `python -m final_experiment.scripts.smoke` without `--with-model`.
- [ ] **Step 3: Inspect output directories** and confirm no checkpoint, adapter, `main`, or `full` run was created by verification.
- [ ] **Step 4: Re-read the spec** and map every requirement to implementation and test evidence.
- [ ] **Step 5: Report exact smoke-only commands and stop before any model loading or training command is executed.**
