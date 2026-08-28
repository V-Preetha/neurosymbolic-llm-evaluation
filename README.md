# Neurosymbolic LLM Evaluation

Research code, datasets, and paper sources for **"Evaluating Logical Reasoning in Large Language Models Through Fine-Tuning and Neuro-Symbolic Bridging."**

## Layout

| Path | Contents |
| --- | --- |
| `final_experiment/` | Main experiment pipeline: LoRA fine-tuning, NL2Logic parsing, Z3 compilation/solving, evaluation, statistics, and paper artifact generation. |
| `nsce-evaluation/` | NSCE evaluation harness: experiment drivers, RAG/factual-corpus tooling, prompts, reports, and tests. |
| `nsce_dataset_package/` | Normalized dataset release (ProofWriter, FOLIO, FELM) with schema, datasheet, citation, and checksums. |
| `Evaluating_Logical_Reasoning_.../` | Final paper: LaTeX source, generated tables, and figures. |
| `conference-latex-template/` | IEEE conference template and the `nsce_final` paper build. |
| `docs/` | Design specs and implementation plans (NL2Logic→Z3 bridge, ProofWriter validation subset, Qwen3-4B profile). |
| root PDFs / DOCX | Paper drafts, project report, and reference material. |

## Setup

Each experiment directory is a self-contained Python project:

```bash
cd final_experiment
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

## Not tracked here

Virtual environments, model checkpoints and caches, and the large derived corpora
(`final_experiment/data/processed/`, `nsce-evaluation/data/factual_corpus/`) are excluded
via `.gitignore` — regenerate them with the build scripts in each project.

The benchmark harness lives in its own repository:
[nsce-llm-reliability-benchmark](https://github.com/V-Preetha/nsce-llm-reliability-benchmark).
