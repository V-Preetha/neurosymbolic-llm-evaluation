# Neurosymbolic LLM Evaluation

Implementation and experimental results for evaluating logical reasoning in large language
models through LoRA fine-tuning and a neuro-symbolic (NL2Logic → Z3) bridge.

## Layout

| Path | Contents |
| --- | --- |
| `final_experiment/` | Main pipeline — LoRA fine-tuning, NL2Logic parser, FOL AST, Z3 compiler/solver, evaluation, statistics, and analysis. |
| `final_experiment/results/` | Run outputs: predictions, parser traces, bridge diagnostics, and comparison summaries. |
| `nsce-evaluation/` | NSCE evaluation harness — experiment drivers, retrieval/RAG tooling, prompts, baselines, and tests. |
| `nsce-evaluation/results/`, `nsce-evaluation/reports/` | Phase results, figures, tables, and audit reports. |
| `nsce_dataset_package/` | Normalized dataset release (ProofWriter, FOLIO, FELM) with schema, datasheet, and checksums. |

## Setup

Each experiment directory is a self-contained Python project:

```bash
cd final_experiment
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

## Not tracked here

Virtual environments, model checkpoints and caches, and the large derived corpora
(`final_experiment/data/processed/`, `nsce-evaluation/data/factual_corpus/`) are excluded —
regenerate them with the build scripts in each project.

The benchmark harness lives in its own repository:
[nsce-llm-reliability-benchmark](https://github.com/V-Preetha/nsce-llm-reliability-benchmark).
