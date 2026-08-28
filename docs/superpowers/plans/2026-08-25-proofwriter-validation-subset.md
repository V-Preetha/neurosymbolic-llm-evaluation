# ProofWriter Validation Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use a frozen stratified 5,000-example ProofWriter main validation subset with epoch-level evaluation/checkpointing and auditable bridge-record counts.

**Architecture:** Preserve the full validation manifest and add a mode-specific training-validation manifest. Training consumes the subset by default and optionally evaluates the full split once after fitting.

**Tech Stack:** Python 3.12, Hugging Face Transformers/Datasets, YAML, unittest.

**Spec:** `docs/superpowers/specs/2026-08-25-proofwriter-validation-subset-design.md`

## Global Constraints

- ProofWriter main training rows remain 10,000.
- Full validation remains 85,468 and is not repeatedly evaluated.
- Training evaluation and checkpoint saving occur once per epoch.
- No training or live dataset preparation is run during verification.

### Task 1: Frozen training-validation manifest

- [ ] Add failing tests for deterministic 5,000-style stratification, manifest seed/count metadata, full-manifest preservation, and subset loading.
- [ ] Implement `training_validation` configuration, creation, persistence, and loading without changing `validation.json`.
- [ ] Run targeted and full tests.

### Task 2: Epoch schedule and optional full validation

- [ ] Add failing tests for epoch strategies, best-model loading, and the `--final-full-validation` CLI flag.
- [ ] Update training to use the subset and optionally evaluate the full manifest once after fitting.
- [ ] Run targeted and full tests.

### Task 3: Bridge expansion reporting

- [ ] Add failing tests for selector/atomic/quantified/logical counts and total partitioning.
- [ ] Print counts before any model loading.
- [ ] Update documentation and run the no-model smoke suite.
