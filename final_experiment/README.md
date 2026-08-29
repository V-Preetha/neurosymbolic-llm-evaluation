# Qwen3-1.7B Direct LoRA × Frozen Neuro-Symbolic Bridge

This package implements a 2×2 experiment on ProofWriter (`tasksource/proofwriter`) and FOLIO (`yale-nlp/FOLIO`). The factors are direct LoRA fine-tuning of the neural model and addition of one fixed neuro-symbolic bridge:

- **C1:** frozen Qwen3-1.7B → direct True/False/Unknown.
- **C2:** frozen Qwen3-1.7B → frozen NL2Logic → FOL AST → two-pass compiler → Z3.
- **C3:** Qwen3-1.7B + dataset direct LoRA → direct True/False/Unknown.
- **C4:** Qwen3-1.7B + the same dataset direct LoRA as C3 → the same frozen bridge as C2.

The base model is frozen at `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. There is exactly one active adapter per dataset: `proofwriter_direct_qwen3_1_7b_lora` and `folio_direct_qwen3_1_7b_lora`. C3 and C4 resolve to the same adapter directory. C1 and C2 load no adapter. Missing or mismatched model identity metadata is rejected before adapter loading.

## Frozen bridge

The bridge identity is `nl2logic_ast_two_pass_z3_v1`. Its parser, typed AST, declaration pass, expression pass, and Z3 solver are identical in C2 and C4 and are never fine-tuned. `--task bridge` is rejected by the training CLI. Legacy component-supervision utilities remain only as inactive archival research code and are not consumed by active training or tokenizer auditing.

Every recursive parser step is written to the condition run's `parser_traces.jsonl`, including successful, partial, and failed steps. Events retain selector decisions, chosen subparsers, child clauses, returned AST nodes, validation status, failure reasons, ancestry, latency, and tokens. Evaluation retains AST validity, compile success, executable rate, normalized/gold logic agreement, semantic agreement, solver accuracy, system failures, corrections/regressions, gains, and interaction effects.

FOLIO gold logic supports equality and inequality, zero-arity predicates, numeric and hyphenated constants, Unicode identifiers, quantifier variants, free official constants, and predicate arity overloading. Gold parse, validation, and compilation cover 896/896 training rows and 105/105 validation rows without filtering or synthesized replacement ASTs.

## Frozen samples and tokenization

ProofWriter `main` uses seed 42 and frozen 2,000-row training, 2,000-row training-validation, and 2,000-row held-out evaluation manifests. Training/development rows come from the controlled-language pool and are jointly stratified by label and QDep. The unchanged held-out sample retains NatLang rows.

- train IDs SHA-256: `514fbeedf81ad1f3e11b46f09698c57a9a39ff819d67aa24f61e180c7b2ec45c`
- training-validation IDs SHA-256: `2536e485a99dceba6e61f685d9d3a42e4e8e0c42394e92b43504f0da99789847`
- evaluation IDs SHA-256: `1d032b6c1172de008b3fd9ec5b081c1da20308f399c83c67b7ceeee283cb4edc`

The active tokenizer audit measures only direct train/development records—one record per selected row. The existing frozen-manifest audit found maxima of 287 (ProofWriter train), 274 (ProofWriter validation), 292 (FOLIO train), and 227 (FOLIO validation). Therefore `max_sequence_length` is 292 with zero truncation. Training uses dynamic per-batch padding and masks prompt tokens so direct loss is computed only on answer tokens plus EOS.

ProofWriter trains for at most three epochs. FOLIO inherits the five-epoch default. Both evaluate and save once per epoch and reload the best checkpoint. The complete 85,468-row ProofWriter validation split remains preserved for an optional one-time post-training `--final-full-validation` run.

## Setup and permitted checks

All commands are run from the **repository root** (the directory that contains
`final_experiment/`), so that `final_experiment` resolves as an importable package.
Python 3.12 is required.

Linux / macOS:

```bash
python3.12 -m venv final_experiment/.venv
source final_experiment/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r final_experiment/requirements.txt
python -m unittest discover -s final_experiment/tests -v
python -m compileall -q final_experiment
python -m final_experiment.scripts.environment
python -m final_experiment.scripts.smoke
```

Windows PowerShell:

```powershell
py -3.12 -m venv final_experiment\.venv
final_experiment\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r final_experiment\requirements.txt
python -m unittest discover -s final_experiment\tests -v
python -m compileall -q final_experiment
python -m final_experiment.scripts.environment
python -m final_experiment.scripts.smoke
```

The no-model smoke command exercises schemas, sampling, parser traces, AST/compiler/Z3 behavior, metrics, and result saving. It does not load Qwen weights. Do not add `--with-model` unless model inference is explicitly intended.

## Data preparation and the only two training commands

```bash
python -m final_experiment.scripts.prepare_data --dataset proofwriter --mode main
python -m final_experiment.scripts.prepare_data --dataset folio --mode main

python -m final_experiment.scripts.train --dataset proofwriter --task direct --mode main
python -m final_experiment.scripts.train --dataset folio --task direct --mode main
```

There are no bridge-training commands. The completed direct adapters are reused by both C3 and C4.

## Evaluation commands

Run all four conditions against each dataset's same frozen evaluation manifest:

```bash
python -m final_experiment.scripts.evaluate --dataset proofwriter --condition 1 --mode main
python -m final_experiment.scripts.evaluate --dataset proofwriter --condition 2 --mode main
python -m final_experiment.scripts.evaluate --dataset proofwriter --condition 3 --mode main
python -m final_experiment.scripts.evaluate --dataset proofwriter --condition 4 --mode main

python -m final_experiment.scripts.evaluate --dataset folio --condition 1 --mode main
python -m final_experiment.scripts.evaluate --dataset folio --condition 2 --mode main
python -m final_experiment.scripts.evaluate --dataset folio --condition 3 --mode main
python -m final_experiment.scripts.evaluate --dataset folio --condition 4 --mode main
```

Parser/compiler/solver/model failures remain system failures and are never converted to logical `Unknown`. No command launches training automatically.
