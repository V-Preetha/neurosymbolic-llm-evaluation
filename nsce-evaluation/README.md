# NSCE Comparative Evaluation

Research prototype comparing four architectures around the same frozen base model, `Qwen/Qwen3-4B-Instruct-2507`. This repository currently implements **Phases 1 through 8**.

## Phase 1 scope

- Fixed experiment configuration and prompt template.
- Read-only loading from the sibling `nsce_dataset_package`.
- Canonical normalization of 20 ProofWriter development samples.
- Strict logical-label extraction.
- Local-cache-only Qwen loading and one deterministic smoke inference.
- Unit tests for schema validity and output parsing.

Phase 2 adds the LLM-only branch. Phase 3 adds same-model self-verification. Phase 4 validates the deterministic solver. Phase 5 adds external symbolic verification and correction. Phase 6 adds pinned SentenceTransformers embeddings, exact FAISS retrieval, deterministic distractors, and RAG failure separation. Phase 7 adds strict FELM normalization and evidence-grounded claim verification with explicit abstention. Phase 8 separates evidence availability from verifier quality using label-blind evidence audits, FELM-FULL/FELM-EVIDENCE exports, and a deterministic manual-review subset.

## Environment setup (PowerShell)

```powershell
$python = 'C:\Users\Preetha\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
```

The model loader sets `local_files_only=True`, preventing an accidental checkpoint change or network download. The exact model must already be present in the Hugging Face cache.

`requirements-phase1.txt` pins PyTorch's official CUDA 12.8 build. Verify GPU visibility after installation:

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Verify normalization without loading the model

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe experiments\run_phase1.py --normalize-only
```

## Run the Phase 1 inference smoke test

```powershell
& .\.venv\Scripts\python.exe experiments\run_phase1.py
```

## Run the Phase 2 LLM-only smoke evaluation

```powershell
& .\.venv\Scripts\python.exe experiments\run_llm.py --limit 20
```

## Run the Phase 3 self-verification smoke evaluation

```powershell
& .\.venv\Scripts\python.exe experiments\run_self_check.py --limit 20
```

## Run the Phase 4 independent symbolic validation

```powershell
& .\.venv\Scripts\python.exe experiments\run_symbolic.py --split all
```

## Run the Phase 5 NSCE correction loop

```powershell
& .\.venv\Scripts\python.exe experiments\run_nsce.py --limit 20
```

## Run the Phase 6 RAG evaluation

```powershell
& .\.venv\Scripts\python.exe experiments\run_rag.py --limit 20
```

## Run the Phase 7 FELM integration smoke evaluation

```powershell
& .\.venv\Scripts\python.exe experiments\run_felm.py --limit 5 --selection evidence
```

This Phase 7 command intentionally selects evidence-bearing records for integration testing.
It is not a representative FELM benchmark score. Missing or insufficient evidence produces
`Unknown`; the benchmark's `atomic_facts.support_status` and claim labels are used only after
verification for scoring and are never included in verifier prompts.

## Run the Phase 8 controlled FELM evidence audit

```powershell
& .\.venv\Scripts\python.exe experiments\run_felm_audit.py
```

Phase 8 audits all claims but runs Qwen only on the label-blind, sufficient-evidence subset.
It does not retrieve web evidence and must not be reported as a full FELM benchmark result.

## Build the Phase 9A frozen factual corpus

```powershell
& .\.venv\Scripts\python.exe experiments\build_factual_corpus.py --download
```

Phase 9A uses the first 10,000 eligible main-namespace, non-redirect articles in
source order from the official `enwiki-20260701` shard `p1p41242`. This fixed
selection is independent of FELM. The resulting corpus checksum in
`data/factual_corpus/manifest.json` is authoritative for the later factual experiment.

The document embedding/index stage is CUDA-only and resumable in 128-row checkpoints:

```powershell
& .\.venv\Scripts\python.exe experiments\embed_factual_corpus_gpu.py
```

For long runs, launch this script as a hidden background process and redirect stdout to
`data/factual_corpus/embedding_gpu.log`. It validates existing checkpoint shapes and
continues from the first missing batch without falling back to CPU.

Outputs:

- `data/processed/proofwriter.jsonl`: exactly 20 canonical development records.
- `results/raw/phase1_smoke.json`: smoke-test status, prediction, latency, and token counts.

## Reproducibility controls

The settings in `configs/experiment.json` are authoritative: temperature `0.0`, maximum new tokens `128`, seed `42`, NF4 4-bit GPU loading, RAG top-k `5`, and labels `True / False / Unknown`. The Qwen wrapper uses greedy decoding (`do_sample=False`) with decoding constrained to exactly one declared benchmark label followed by EOS; temperature is therefore recorded but intentionally not passed to generation. The same fixed model loading and label constraint must be used by every architecture.

Raw data is never modified. Override its location with `NSCE_DATASET_PACKAGE` if needed.
