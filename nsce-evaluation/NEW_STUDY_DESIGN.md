# Revised NSCE Study Design — External Neuro-Symbolic Bridge vs Frozen Qwen3-8B

Status: **Phase A (Audit), Phase B (Qwen3-8B migration), Phase C (dataset preparation), and Phase D (symbolic bridge validation) complete.** Held-out test predictions have NOT been run. Next: Phase E (translation audit — see §12 of `SYMBOLIC_BRIDGE_SPEC.md`, already answered) and the frozen dev-pilot run of all three conditions on Qwen3-8B.

## Frozen runtime (Phase B)

- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Config: `configs/experiment_qwen8b.json` (versioned; `configs/experiment.json` for the historical Qwen3-4B pilot is untouched).
- NF4 4-bit, double quantization, FP16 compute, single GPU (`cuda:0`), no CPU offload, peak VRAM ≈ 6984 MB.
- Smoke test: `reports/revised_study/qwen8_smoke_test.json` — passed.

## Dataset artifacts (Phase C)

- `reports/revised_study/dataset_audit.md` / `.json` — dev split audit (360 examples, balanced 3×6 grid, 0 exact duplicates, 0 world-level dev/test overlap; 105 shared query surface strings across distinct worlds, which is benign — see audit for detail).
- `data/revised_study/proofwriter_dev_pilot.jsonl` + `_manifest.json` — 72-example deterministic stratified dev/debugging pilot (seed 42, 4 per label×depth cell), **not** the held-out set.
- `data/revised_study/proofwriter_test_manifest.json` — frozen metadata for the untouched 360-example held-out test split (checksum, label counts, sample IDs). No predictions have been run on it.
- `src/io_guard.py` + `tests/test_field_whitelisting.py` — field whitelisting (`AllowedInput`: facts/rules/query only) enforced in `src/prompts.py`, `src/nsce/verifier.py`, `src/nsce/feedback.py`, with executable proof that `ground_truth`/`evidence`/`metadata`/`candidate_response` are never read by prompt or verifier code, covering LLM-only, self-check, and NS bridge paths.

## Symbolic bridge validation (Phase D)

- `SYMBOLIC_BRIDGE_SPEC.md` — full representation spec (facts, negation, rules, query, polarity, supported/unsupported operators, UNKNOWN derivation, proof-trace provenance, UNKNOWN-vs-ABSTAIN, and the Phase E translation-source clarification).
- `tests/test_symbolic_bridge_validation.py` — 13 deterministic, Qwen-free tests: direct support/contradiction, unknown (unrelated fact + missing premise), one-hop, two-hop, negated query, contradictory facts, malformed/unsupported input (unsupported verb, disjunction, malformed conditional), and translation-failure-vs-reasoning-outcome separation via `verify_record`. All pass; full suite is 56/56 passing.
- `src/nsce/pipeline.py` now logs an explicit `bridge_action` (`RETAIN`/`CORRECT`/`ABSTAIN`) distinguishing the task label `UNKNOWN` from the system action `ABSTAIN` (translation failure).

## Documentation drift (recorded, not fixed yet — per explicit instruction)

1. `README.md` still says "implements Phases 1 through 8" — Phase 9 exists and has executed results.
2. `MAJOR PROJECT REPORT.pdf` Appendix A ("Coding") contains unrelated FastAPI/Telegram-bot/MNIST-digit-addition code — none of it is part of this codebase.
3. The PDF's Chapter 4 ("Results and Discussions") figures have no traceable methodology, dataset, or sample size tying them to this pipeline.

**The revised experimental artifacts generated from this study (this file, `SYMBOLIC_BRIDGE_SPEC.md`, `reports/revised_study/*`, `data/revised_study/*`) are the source of truth going forward — the PDF's existing Results chapter must not be cited as evidence for the revised paper.**

## Research question

> Does an external neural-to-symbolic verification bridge improve the factual and logical reliability of a frozen Qwen3-8B, relative to the same frozen model with no verification and with pure neural self-verification?

## Hypotheses

- **H1**: Qwen3-8B + external neural-to-symbolic bridge achieves significantly greater logical/factual reliability than the same frozen Qwen3-8B without external symbolic verification.
- **H0**: The bridge produces no statistically meaningful reliability improvement over the unverified baseline.

## Independent variable

Verification condition: {LLM-only, Self-verification, NS Bridge} — three levels, same frozen Qwen3-8B, same prompts/decoding config held constant across conditions except for the verification step itself.

## Dependent variables

Accuracy, macro-F1, per-class P/R/F1 (SUPPORTED/CONTRADICTED/UNKNOWN), false-support rate, contradiction-detection rate, appropriate-abstention rate, changed-answer rate, beneficial vs harmful correction rate, translation-validity rate, symbolic-verdict accuracy, latency, model calls/example, peak VRAM.

## Controlled variables

Model ID, quantization, compute dtype, seed, temperature, max_new_tokens, prompt templates, dataset split, hardware.

---

## 1. Existing architecture — what the project actually does now

The repo currently implements **Phases 1–9** on `Qwen/Qwen3-4B-Instruct-2507` (the README claims only "Phases 1 through 8" — **stale, needs updating**; Phase 9 is fully implemented and has executed results in `results/phase9/`).

Config is centralized in `configs/experiment.json` and loaded identically by every `run_*.py`: `nf4` 4-bit quant (double-quant, fp16 compute), `temperature=0.0`, `seed=42`, `max_new_tokens=128` (Phase 9 overrides to `8`, since it only needs a single label token, and hard-gates on `nf4` + GPU name `"RTX 5050"`).

Branches that already exist:

| Phase | Branch | What it does |
|---|---|---|
| 2 | `run_llm.py` | Qwen zero-shot on ProofWriter. Executed, n=20, accuracy 0.45. |
| 3 | `run_self_check.py` | Pure neural self-verification (re-prompt, no symbolic access). Executed, n=20, final accuracy 0.45 (4 beneficial + 4 harmful corrections). |
| 4 | `run_symbolic.py` | **Deterministic symbolic solver only, no LLM.** Executed on full 720 ProofWriter dev+test: 100% coverage, 100% accuracy. |
| 5 | `run_nsce.py` | LLM + symbolic verifier + feedback/correction loop (ProofWriter). Executed, n=20: initial 0.45 → symbolic 1.0 → **final 0.80**, correction success rate 0.636. |
| 6 | `run_rag.py` | RAG over a synthetic ProofWriter-distractor corpus (6,609 docs, 40 distractors/query, MiniLM + FAISS). Executed, accuracy 0.45, recall@5 0.193. |
| 7 | `run_felm.py` | FELM evidence-grounded claim verification, explicit abstention. **5-sample smoke test only — both the script and README explicitly disclaim it as non-representative.** |
| 8 | `run_felm_audit.py` | Label-blind evidence-sufficiency audit over all 4,426 FELM claim segments; only 130 (2.9%) have sufficient evidence to reach the verifier. Executed at full scale. |
| 9 | `run_phase9.py` + `build_phase9_pilot.py` | Frozen, hash-verified **50-claim FELM pilot**, four conditions (llm_only/self_check/rag/nsce) sharing one retrieval pass over a frozen 162,696-chunk Wikipedia corpus. Executed: llm_only 0.46, self_check 0.42, rag 0.50, **nsce 0.52** (macro-F1 0.455/0.424/0.503/0.519), correction success rate 0.16. |

## 2. Reusable components

- **The ProofWriter symbolic bridge is genuine and reusable as-is for the logic-reasoning arm of Condition C.** `src/nsce/parser.py` + `symbolic_reasoner.py` + `verifier.py` implement a real deterministic Horn-clause forward-chaining engine (`Atom`/`Rule` typed intermediate representation, fixed-point derivation, proof trace built from actual execution state, not LLM narration). No gold-label leakage found in this path — verified by direct code citation and by an existing unit test (`tests/test_phase1.py`) that asserts `"ground_truth" not in prompt`.
- `src/baselines/self_check.py` — confirmed pure neural re-prompting, zero symbolic imports. Usable unmodified as Condition B.
- `src/models/qwen.py` (`QwenRunner`) — model-agnostic; `model_id` is a parameter, never hardcoded. Swapping to Qwen3-8B needs **only a config edit**, not a code change (see §7).
- ProofWriter dataset pipeline (`src/datasets/proofwriter.py`, `nsce_dataset_package/normalized/proofwriter_{dev,test}.jsonl`) — 720 examples, balanced by label × reasoning-depth, each carrying explicit `facts`/`rules`, i.e. already symbolization-ready.
- FELM evidence-sufficiency audit machinery (`src/evaluation/felm_audit.py`) — label-blind, semantic+lexical+numeric scoring of evidence sufficiency per claim. This is architecture-agnostic and directly reusable as a "claim extraction / evidence gating" front-end for any future factual-claim bridge.
- Dataset package provenance: `nsce_dataset_package/SHA256SUMS.txt`, `DATASHEET.md`, `validation_report.json` — solid, already covers leakage-control guidance explicitly.
- The Phase 9A frozen Wikipedia corpus + embedding infra (`build_factual_corpus.py`, `embed_factual_corpus_gpu.py`, `data/factual_corpus/*`) is **not** RAG-baseline-specific — Phase 9's `rag` and `nsce` conditions both depend on it. Don't delete it when deprecating RAG as a standalone architecture branch.

## 3. Methodological problems that must be flagged before scaling up

1. **RAG (Phase 6)** is a standalone architecture branch per the master plan and must not be run again as a headline comparison. Its code/reports/config keys should be marked historical (see §"Obsolete portions" below) — but its *infrastructure* (embeddings, FAISS index wrapper, frozen corpus) is shared with Phase 9 and must be kept.
2. **The FELM/factual-claim "symbolic" verifier (`factual_structured.py`/`factual_verifier.py`) is NOT a genuine external bridge.** It only performs deterministic reasoning for a narrow arithmetic/exact-numeric-frame subset; everything else falls through to an LLM NLI call (`runner.generate_choice`). If FELM claims are meant to exercise Condition C, most of that "verification" is actually neural — this violates the master plan's explicit requirement (§7) that the bridge not be "Qwen → another prompt asking Qwen to reason logically." **Recommendation: scope the primary Condition-C benchmark to the ProofWriter/logical-reasoning domain only**, where a genuine bridge already exists, and treat FELM/factual claims as an optional secondary benchmark with an honestly-labeled, mostly-neural verifier (or invest separately in extending the deterministic coverage before including it in headline numbers).
3. **Gold-label leakage risk in field naming** (not currently exploited by any prompt-builder, but structurally present): `build_phase9_pilot.py`'s flat pilot record puts `gold_label` as a sibling field next to `claim_text`/`corpus_query`; the canonical schema's `symbolic.expected_status` directly duplicates `ground_truth.label`. Any new "common experiment interface" (master plan §16) must explicitly whitelist input fields passed into prompts/bridge components rather than serializing whole records.
4. **The frozen 50-claim pilot (`reports/phase9_manual_audit_50.jsonl`) is FELM-only** — no ProofWriter claims are in it (see §4). It also reuses several source responses across 2-3 claims each (`felm_test_000496` ×3, `felm_test_000601` ×2, etc.), so the 50 "examples" are not 50 fully independent source texts — a caveat for effective-n in any statistical test on this specific set.
5. Phase 1's recorded smoke-test artifact (`results/raw/phase1_smoke.json`) shows `quantization: "bitsandbytes_int8"`, inconsistent with the `nf4` baseline used everywhere else — treat as a stale historical artifact, not a valid baseline reference.
6. Phase 8's headline verifier-quality numbers (selective_accuracy 0.791) characterize only the 2.9% of FELM claims with sufficient evidence — not FELM performance broadly.
7. **The `MAJOR PROJECT REPORT.pdf` does not describe this codebase.** Its Chapter 3 (Sprint methodology, generic SPL/SRL bidirectional-feedback architecture) and Chapter 4 (comparative radar/bar charts of "LLMs" vs "This AI") have no stated methodology, dataset, or sample size tying them to the actual Qwen/ProofWriter/FELM pipeline — they read as illustrative, not measured. **Appendix A's "Coding" listing is entirely unrelated code** (a generic FastAPI dashboard with Telegram-bot polling, and a toy MNIST-digit-addition neuro-symbolic demo) — none of it appears anywhere in `nsce-evaluation/`. This is a significant discrepancy between the write-up and the implementation; flagging it explicitly since the master plan requires comparing the report against actual files rather than assuming it's accurate.

## 4. Dataset assessment

- **ProofWriter (720 examples, `proofwriter_dev.jsonl` + `proofwriter_test.jsonl`)** is the right primary benchmark for the bridge: explicit `facts`/`rules` per record, already balanced by label × reasoning depth (6 depths × 3 labels × 20 per split per cell), and **the full 360-record test split has never been spent in any pilot** — it can become the primary held-out set directly, no new sampling needed, far exceeding n=50.
- **FELM (847 responses / 4,426 claim segments)**: only 50 claims have ever been drawn (the Phase 9 pilot); 3,548 were deemed eligible after exclusion filters, leaving **~3,498 unused, already-vetted claims** available via the same deterministic seeded-hash sampling method (`build_phase9_pilot.py`) for a larger secondary/generalization benchmark. However, FELM claims carry no explicit premises — only free text + retrieved Wikipedia evidence chunks — so they are **not directly symbolizable** without a new extraction step, and per the master plan (§13/§14) must be reported as a separate secondary benchmark, not combined into one headline accuracy with the ProofWriter numbers.
- `DATASHEET.md` documents that FELM evaluates factuality of **pre-existing fixed ChatGPT responses**, not our own model's outputs — an important scope caveat if FELM is used at all.
- Checksums (`SHA256SUMS.txt`) exist for the dataset package but not for the specific 50-claim report files — those are reproducible via `build_phase9_pilot.py` + seed + the recorded claim-ID list (`felm_phase9_pilot_ids.json`), just not via a hash manifest.

## 5. Bridge assessment

**The ProofWriter branch already satisfies the master plan's "real symbolic bridge" bar**: explicit typed intermediate representation (`Atom`/`Rule`), deterministic forward-chaining reasoner, proof trace generated from actual derivation state (not fabricated by the LLM), and a correction policy (RETAIN when LLM agrees with symbolic verdict, CORRECT-attempt via feedback prompt when it disagrees). One gap: there is **no explicit ABSTAIN branch** currently — an "unknown" symbolic verdict is treated as an ordinary label value rather than a distinct decline-to-answer action, and a parser failure silently falls back to keeping the initial LLM answer rather than abstaining. This is a small, well-scoped code change, not a rebuild.

The FELM/factual branch does **not** meet the bar (see §3.2) and should not be presented as Condition C without further work.

## 6. Proposed minimal modifications

1. Update `README.md` to describe Phase 9 and correct the "Phases 1–8" claim.
2. Point `configs/experiment.json`'s `model_id` at `Qwen/Qwen3-8B` once the checkpoint is downloaded and the smoke test (master plan §4) passes. No other source changes required for the ProofWriter/symbolic path.
3. Add an explicit ABSTAIN branch to `src/nsce/pipeline.py` (currently `unknown` is handled as an ordinary label; parser failures should abstain rather than silently defaulting to the initial LLM answer).
4. Scope the primary Condition-C benchmark to ProofWriter (logic reasoning) only, given the FELM verifier's current mostly-neural character. Treat FELM as an optional, separately-reported secondary/generalization benchmark, and treat the existing FELM 50-claim pilot as *development* data (already used for pipeline debugging), not final held-out data.
5. Freeze the unused ProofWriter test split (360, or a documented stratified subset) as the primary held-out evaluation set; draw a larger stratified sample of the ~3,498 unused eligible FELM claims (same deterministic hash method, disjoint claim IDs from the existing 50) only if the secondary factual benchmark is pursued.
6. When implementing the common experiment interface (`run_llm_only`/`run_self_verification`/`run_ns_bridge`), explicitly whitelist which record fields are visible to prompt-builders and bridge components, given the leakage risks identified in §3.3.
7. Mark RAG-specific code/reports/config keys historical without touching the shared retrieval/embedding/frozen-corpus infrastructure that Phase 9's other conditions depend on.
8. Decide separately (with the user) what to do about the mismatch between `MAJOR PROJECT REPORT.pdf`'s narrative/results and the actual codebase before that document is used as a source of truth for the revised paper.

## 7. Qwen3-8B readiness

- **Not cached.** Confirmed absent from `C:\Users\Preetha\.cache\huggingface\hub` (only up to Qwen3-4B/4B-VL present).
- Download: `huggingface-cli download Qwen/Qwen3-8B` (or `hf download ...`), ~16.4 GB (bf16 weights, 5 safetensors shards). **Not downloaded — requires explicit user authorization before proceeding to Phase B**, per the file-download policy for this session.
- No code changes needed to point the existing `QwenRunner` at Qwen3-8B — `model_id` is already a config-driven parameter.
- Environment already verified capable of 4-bit (nf4) inference on the RTX 5050 8GB: PyTorch 2.11.0+cu128 with `sm_120` in its arch list, bitsandbytes 0.50.1, transformers 4.51.0, CUDA available. Estimated weight footprint in nf4 ≈ 4.1–4.5 GB, leaving reduced but workable headroom versus the 4B model for KV cache/activations — the master plan's own smoke-test step (§4) is the right way to confirm this empirically rather than estimate further.

---

## Obsolete / historical (do not delete, do not use as primary baseline)

- `configs/experiment.json` pointing at `Qwen/Qwen3-4B-Instruct-2507` and all of its results under `results/raw/*` — retained as the historical pilot model, never compared head-to-head against Qwen3-8B NS results per the non-negotiable rules.
- Phase 6 RAG code/reports (`experiments/run_rag.py`, `src/baselines/rag.py`, `src/retrieval/corpus.py`, `data/rag/*`, `results/raw/rag_phase6*`) — RAG is dropped as a standalone architecture branch in the revised study.
- Phase 7's 5-sample FELM smoke test results — never representative, don't cite as a benchmark number.
- Phase 1's `bitsandbytes_int8`-quantized smoke result — inconsistent with the nf4 baseline used everywhere else.

## Next step

Awaiting user authorization to download `Qwen/Qwen3-8B` (~16.4 GB) before proceeding to Phase B (model swap, smoke test, config freeze).
