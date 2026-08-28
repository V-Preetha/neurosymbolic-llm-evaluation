# Frozen Bridge Single-Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make C3 and C4 reuse one dataset-specific direct LoRA while C2 and C4 reuse one frozen NL2Logic/AST/compiler/Z3 bridge, with no active bridge-training expansion.

**Architecture:** Central condition routing resolves model adapter and bridge use independently. Training and tokenizer auditing operate only on one direct record per selected row. Evaluation keeps the existing bridge execution and metric pipeline but records a stable bridge identity for C2/C4.

**Tech Stack:** Python 3.12, unittest, Transformers/PEFT, typed JSON FOL AST, Z3, YAML.

**Spec:** `final_experiment/docs/superpowers/specs/2026-08-26-frozen-bridge-single-adapter-design.md`

## Global Constraints

- Base model is `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Only `proofwriter_direct_qwen3_1_7b_lora` and `folio_direct_qwen3_1_7b_lora` are active adapters.
- C1/C2 load no adapter; C3/C4 load the same dataset direct adapter.
- C2/C4 use the same frozen parser, AST, two-pass compiler, and solver.
- No model weights are loaded and no training is started during implementation verification.

---

### Task 1: Condition routing and frozen bridge identity

**Files:**
- Create: `final_experiment/tests/test_experiment_design.py`
- Modify: `final_experiment/core.py`
- Modify: `final_experiment/evaluation.py`

**Interfaces:**
- Produces: `adapter_name_for_condition(dataset: str, condition: int) -> str | None`
- Produces: `uses_frozen_bridge(condition: int) -> bool`
- Produces: `FROZEN_BRIDGE_ID: str`

- [ ] Write tests asserting literal routing: C1/C2 `None`, C3/C4 the same direct adapter, and only C2/C4 use the bridge.
- [ ] Run the focused tests and observe failures caused by the missing routing API.
- [ ] Add the routing functions, remove active bridge adapter names, and make evaluation use the routing API.
- [ ] Add the same frozen bridge ID to C2/C4 run metadata and prediction translation metadata.
- [ ] Run the focused tests and observe them pass.

### Task 2: Direct-only training contract

**Files:**
- Modify: `final_experiment/tests/test_experiment_design.py`
- Modify: `final_experiment/tests/test_sampling_manifests.py`
- Modify: `final_experiment/training.py`

**Interfaces:**
- `build_parser()` accepts only the literal task `direct`.
- `train(dataset, task, mode, ...)` rejects every non-direct task before importing or loading model weights.

- [ ] Write tests that `--task bridge` is rejected and programmatic bridge training raises before any model import.
- [ ] Run the focused tests and observe the old bridge choice fail the contract.
- [ ] Remove component-expansion imports and bridge branches from active training.
- [ ] Materialize one direct record per row and keep direct answer-token masking, dynamic padding, epoch evaluation/save, and best-checkpoint behavior.
- [ ] Run the focused tests and observe them pass.

### Task 3: Direct-only tokenizer audit and lossless limit

**Files:**
- Modify: `final_experiment/tests/test_experiment_design.py`
- Modify: `final_experiment/token_audit.py`
- Modify: `final_experiment/training_data.py`
- Modify: `final_experiment/configs/model.yaml`
- Modify: `final_experiment/environment.py`

**Interfaces:**
- Produces: `smallest_lossless_standard_limit(observed_max: int) -> int`.
- Tokenizer audit groups are exactly `direct_train` and `direct_validation` for each dataset.

- [ ] Write tests using hand-derived maxima that 292 maps to 292, 293 maps to 320, and audit group construction never calls bridge expansion.
- [ ] Run focused tests and observe failures.
- [ ] Change the audit to direct records only and report active training scope and zero truncations.
- [ ] Implement standard-limit selection and validate the configured limit against the observed maximum.
- [ ] Set the configured limit to 292 because the existing frozen-manifest audit recorded a direct maximum of 292 and zero direct-record truncation.
- [ ] Run focused tests and observe them pass without loading a tokenizer or model.

### Task 4: Documentation and inactive archival code

**Files:**
- Modify: `final_experiment/README.md`
- Modify: `final_experiment/nl2logic_training.py`
- Modify: `final_experiment/tests/test_nl2logic_training.py`

**Interfaces:**
- Component-supervision utilities remain importable only as explicitly archived research utilities and are not consumed by training or audit code.

- [ ] Add an archival module docstring and make tests describe it as inactive legacy coverage rather than an active training path.
- [ ] Update README condition definitions, commands, adapter count, token limit, and expected training times.
- [ ] Remove every active README/config instruction to train a bridge adapter.

### Task 5: Required verification

**Files:**
- Verify all modified files and preserve completed adapter directories.

- [ ] Run `python -m unittest discover -s final_experiment/tests -v` and require zero failures.
- [ ] Run `python -m compileall -q final_experiment` and require exit code 0.
- [ ] Run `python -m final_experiment.scripts.environment` and confirm model identity, 292 limit, and no model load.
- [ ] Run `python -m final_experiment.scripts.smoke` without `--with-model` and require all non-model checks pass.
- [ ] Confirm both completed direct `final_adapter` directories still exist and no bridge adapter is referenced by active routing.
