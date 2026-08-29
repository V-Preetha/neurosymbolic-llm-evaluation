# Evaluating Logical Reasoning in LLMs Through Fine-Tuning and Neuro-Symbolic Bridging

Code and canonical evaluation artifacts for a matched 2×2 study on ProofWriter, comparing
direct LoRA fine-tuning of Qwen3-1.7B against routing that same fine-tuned checkpoint through
a frozen NL2Logic → FOL AST → Z3 symbolic bridge.

The headline result is negative and is the point of the paper: fine-tuning helps substantially,
and adding the symbolic bridge on top of it **hurts**, because natural-language-to-logic
translation fidelity — not solver capability — is the bottleneck.

| Condition | Model | Answer path | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| **C1** | Qwen3-1.7B (base) | direct | 48.5% | 0.251 |
| **C3** | Qwen3-1.7B + LoRA (ckpt-250) | direct | 82.5% | 0.826 |
| **C4** | Qwen3-1.7B + same LoRA | frozen NL2Logic → Z3 | 49.5% | 0.459 |

C4 corrects 12 of C3's errors but overturns 78 of its correct answers. The −33.0 pp change is
significant by exact two-sided McNemar testing (*p* = 5.20 × 10⁻¹³). 89.5% of bridge outputs
execute, yet 80 executable verdicts are wrong.

> **C2** (base model + bridge) is implemented and runnable, but the paper reports the C1/C3/C4
> contrast; see `final_experiment/README.md` for the full condition matrix.

## Repository layout

| Path | Contents |
| --- | --- |
| `final_experiment/` | The complete pipeline: data preparation, LoRA training, NL2Logic parser, typed FOL AST, two-pass Z3 compiler, solver, evaluation, and statistics. |
| `final_experiment/results/final200_proofwriter/` | **The frozen canonical run.** Every number in the paper is regenerated from these files. |
| `final_experiment/analysis/` | Metrics, bootstrap intervals, McNemar testing, bridge diagnostics, tables, and figures. |
| `final_experiment/analysis/generated/` | Committed analysis outputs: `results.json`, per-example CSV, depth and bridge-failure breakdowns. |
| `final_experiment/tests/`, `final_experiment/analysis/tests/` | 131 tests covering the parser, AST, compiler, solver, metrics, and artifact provenance. |

## Canonical evidence and provenance

The sole experimental source is `final_experiment/results/final200_proofwriter`: a
label-and-QDep-stratified sample of 200 held-out ProofWriter examples (seed 42; 53 True,
53 False, 94 Unknown). Each condition holds 200 predictions with identical ordered
identifiers, identical gold labels, and a matching frozen subset hash.

Before computing anything, the analysis validates row count, identifier order, gold-label
identity, subset hash, and checkpoint identity, and requires C3 and C4 to reference the same
checkpoint-250 adapter. Bootstrap intervals use seed 42 and 10,000 replicates. **No model
inference is rerun to produce the paper.**

Exploratory prototype, profiling, and earlier frozen runs are deliberately not versioned here —
the paper's provenance checks exclude them by construction. They remain available on the
[`archive/nsce-exploration`](https://github.com/V-Preetha/neurosymbolic-llm-evaluation/tree/archive/nsce-exploration)
branch, along with an earlier, separate NSCE retrieval/reliability line of work that this paper
does not draw on.

## Reproducing the paper's numbers

No GPU and no model weights are needed for this step — it reads the committed canonical run.

```bash
python3.12 -m venv final_experiment/.venv
source final_experiment/.venv/bin/activate    # Windows: final_experiment\.venv\Scripts\Activate.ps1
python -m pip install -r final_experiment/requirements.txt
python -m final_experiment.analysis.generate_paper_artifacts
```

Run every command from the repository root so `final_experiment` resolves as an importable
package. To verify the artifacts against the canonical source:

```bash
python -m unittest discover -s final_experiment/tests
python -m pytest final_experiment/analysis/tests
```

Reproducing the run itself — data preparation, LoRA training, and the four evaluation
conditions — requires a GPU and is documented in
[`final_experiment/README.md`](final_experiment/README.md), including the frozen manifest
hashes and the base-model revision pin.

## Not versioned here

Virtual environments, LoRA checkpoints, model caches, and the ~1.3 GB processed corpora are
excluded; rebuild them with `python -m final_experiment.scripts.prepare_data`. The two small
`data/processed/*/audit.json` manifests **are** committed, because they are the only files the
analysis reads from that tree — so the paper's tables and figures regenerate from a fresh clone
without rebuilding the corpora.

The LaTeX manuscript sources are not published in this repository.

## Third-party code

The bridge implements the AST-guided NL-to-FOL translation method of NL2Logic
(Putra et al.; reference implementation: [`peng-gao-lab/nl2logic`](https://github.com/peng-gao-lab/nl2logic), MIT).
The parser, typed AST, two-pass compiler, and Z3 solver in this repository are a
self-contained implementation with no runtime dependency on that codebase; the bridge is
frozen at identity `nl2logic_ast_two_pass_z3_v1` and is never fine-tuned.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
