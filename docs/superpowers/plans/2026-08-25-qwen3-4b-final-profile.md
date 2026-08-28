# Qwen3-4B Final Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze Qwen3-4B and a 5k-by-3-epoch ProofWriter profile while preventing incompatible 8B adapter reuse.

**Architecture:** Pin model identity in configuration, isolate checkpoint names, validate artifact metadata at resume/evaluation boundaries, and migrate the frozen main manifest recoverably.

**Tech Stack:** Python 3.12, Transformers, PEFT, YAML, unittest.

**Spec:** `docs/superpowers/specs/2026-08-25-qwen3-4b-final-profile-design.md`

### Task 1: Frozen configuration and namespaces

- [ ] Add failing tests for model ID/revision, ProofWriter 5k main size, three epochs, and 4B checkpoint names.
- [ ] Update configuration and adapter namespaces.
- [ ] Run targeted tests.

### Task 2: Adapter compatibility gate

- [ ] Add failing tests for matching, missing, and Qwen3-8B metadata.
- [ ] Implement metadata writing and validation before resume/evaluation adapter loading.
- [ ] Run targeted tests.

### Task 3: Recoverable manifest migration

- [ ] Verify the exact current manifest path and its 10,000-example metadata.
- [ ] Move it to `train_main_10000_seed42.json` without overwriting an existing archive.
- [ ] Ensure preparation will create a new frozen 5,000-example `train_main.json`.

### Task 4: Documentation and verification

- [ ] Update commands and experiment description.
- [ ] Run the full unit suite, compilation, and no-model smoke only.
