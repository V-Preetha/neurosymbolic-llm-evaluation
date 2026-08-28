# Canonical Evidence IEEE Paper Design

## Purpose

Create a reproducible, evidence-conservative IEEE conference paper from the repository's latest complete matched experiment. The paper will answer when symbolic verification helps or harms a fine-tuned language model and how strongly the result depends on natural-language-to-logic translation fidelity.

The manuscript will not preserve the project's earlier intended positive claim. It will report the observed negative/mixed result: task-specific fine-tuning improves ProofWriter classification substantially, while the current NL2Logic-to-Z3 bridge reverses much of that gain when applied to the same fine-tuned checkpoint.

## Title and framing

Approved title:

> Evaluating Logical Reasoning in Large Language Models Through Fine-Tuning and Neuro-Symbolic Bridging

The central research question is:

> When does symbolic verification help or harm a fine-tuned language model, and how sensitive is a neuro-symbolic pipeline to translation fidelity?

The paper will describe the system as a controlled comparison among direct base inference, direct fine-tuned inference, and fine-tuned inference through a symbolic bridge. It will not claim that the bridge improves the language model, implement a reconciliation policy that does not exist, or describe self-check and symbolic-only baselines that have no matched prediction artifacts.

## Canonical evidence boundary

The sole primary experiment is:

`final_experiment/results/final200_proofwriter`

The three canonical condition directories are:

- `c1_base_direct`
- `c3_finetuned_direct`
- `c4_finetuned_bridge`

All three contain 200 predictions on the same ordered example IDs. Their metadata records the same frozen subset hash:

`1b4665a0472a3aa15541be13d538336a00bd8ed1d104cd76bf80b8b360bd9092`

The primary bridge ablation is C3 versus C4 because both use the same checkpoint-250 adapter. C1 versus C3 measures the effect of fine-tuning. C1 versus C4 is descriptive only because two factors differ.

The analysis pipeline will reject the run if:

- any condition has a row count other than 200;
- example IDs differ in membership or order;
- gold labels differ across conditions;
- the frozen-subset hashes differ;
- C3 and C4 do not reference the identical checkpoint path;
- any gold label lies outside True, False, and Unknown.

Older 2,000-example runs will not be combined with this experiment. They used earlier generation caps or pre-boundary-fix behavior and lack a complete matched C4. Prototype, profile, frozen-5, and frozen-20 runs are development history and will not supply headline metrics.

## Implemented methods

### C1: Base direct

Qwen/Qwen3-1.7B at pinned revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, loaded with NF4 quantization, receives the answer-only direct prompt and returns True, False, or Unknown.

### C3: Fine-tuned direct

The same base model loads the ProofWriter LoRA adapter from `checkpoint-250` and uses the same direct inference path and prompt as C1.

### C4: Fine-tuned bridge

The same base model and exact checkpoint-250 adapter drive the frozen descriptive NL2Logic component prompts. Recursive parser outputs form a typed first-order-logic AST, the two-pass compiler declares symbols and compiles expressions, and Z3 maps entailment, contradiction, or neither to True, False, or Unknown.

The C4 symbolic verdict is the final prediction. There is no separate retain/correct/abstain reconciliation module in the canonical implementation. “Intervention” therefore means the observed prediction transition from C3 to C4 on the paired examples, not a downstream policy action.

## Verified configuration sources

Configuration values will be read from executable code and frozen metadata rather than copied from README prose. Sources include:

- `configs/model.yaml`
- `configs/proofwriter.yaml`
- `training.py`
- checkpoint `adapter_config.json`
- checkpoint `trainer_state.json`
- training `metadata.json`
- canonical run `config.json` and condition metadata

The paper will distinguish configured metadata from executed behavior where they disagree. In particular, executable training code selects the best checkpoint using validation loss with `greater_is_better=False`, even though a YAML field names macro F1. Checkpoint-250 is verified as the selected ProofWriter checkpoint, and the final adapter weight hash matches checkpoint-250.

## Dataset accounting

ProofWriter statistics will come from prepared-data audit and frozen manifests:

- source train: 585,552;
- source validation: 85,468;
- source test: 174,476;
- selected controlled-language training subset: 2,000;
- selected training-validation subset: 2,000;
- held-out main manifest: 2,000;
- canonical evaluation subset: 200 stratified by label and QDep from the held-out main manifest, seed 42.

The paper will state the canonical label distribution: 53 True, 53 False, and 94 Unknown.

FOLIO will appear only as a prepared extension and limitation:

- train: 896;
- development: 105;
- held-out: 203;
- adapter checkpoints and gold-logic tests exist;
- no prediction-level FOLIO evaluation artifact exists.

No FOLIO model accuracy, F1, significance, or comparative result will be reported.

## Reproducible analysis architecture

Create a focused `analysis/` package with these responsibilities:

- `analysis/generate_paper_artifacts.py`: entry point that loads canonical predictions, validates provenance, calculates results, and orchestrates outputs.
- `analysis/metrics.py`: strict multiclass metrics, confidence intervals, paired tests, intervention counts, depth analysis, and bridge reliability.
- `analysis/figures.py`: deterministic IEEE-sized Matplotlib figures in PDF and 300-DPI PNG.
- `analysis/tables.py`: machine-generated LaTeX table fragments and CSV audit tables.
- `analysis/tests/test_paper_analysis.py`: tests for strict invalid handling, paired alignment, transition counts, and consistency checks.
- `analysis/generated/`: JSON and CSV outputs containing every value consumed by plots and tables.

The paper and figures will consume one generated results object rather than recalculating metrics independently. This avoids divergent values across text, tables, and plots.

## Metric definitions

All headline classification metrics are strict over all 200 examples. A missing or invalid prediction is mapped to an `INVALID` outcome that cannot match any gold class.

Report for each condition:

- total correct and incorrect;
- strict accuracy;
- valid output rate;
- accuracy among valid outputs, labeled clearly as conditional;
- macro precision, recall, and F1 over the three gold classes;
- weighted F1;
- per-class precision, recall, F1, and support;
- a confusion matrix with True, False, Unknown, and Invalid prediction columns.

Absolute accuracy change is C4 accuracy minus C3 accuracy. Relative error reduction uses the requested formula and may be negative. If negative, the manuscript will describe it as increased error rather than rhetorically presenting a “reduction.”

## Statistical analysis

Use deterministic paired bootstrap resampling with seed 42 and 10,000 replicates.

Calculate percentile 95% confidence intervals for:

- each condition's accuracy;
- each condition's macro F1;
- paired C4-minus-C3 accuracy;
- paired C4-minus-C3 macro F1.

Use the exact two-sided McNemar/binomial test on discordant C3/C4 correctness outcomes. Report the discordant counts, exact p-value, and sample size. Do not infer that translation failures alone caused every regression; the significance test concerns final paired correctness only.

## Intervention analysis

For each paired C3/C4 example, assign exactly one transition:

- wrong to correct;
- correct to correct;
- correct to wrong;
- wrong to wrong.

Calculate correction rate, harm rate, and net corrections from those counts. Examples with invalid C4 output remain incorrect under strict scoring and therefore contribute to the appropriate transition category.

The intervention figure and table will be prominent because they directly show the asymmetry between helpful and harmful changes.

## Reasoning-depth analysis

The canonical rows contain `qdep`. Report accuracy and valid-output coverage by QDep for C1, C3, and C4. Depths 4 and 5 contain only two examples each and will be combined into a `4+` group. Every depth table and figure will display group sample sizes.

The manuscript will avoid causal claims about depth because the groups are observational and highly imbalanced.

## Bridge reliability and error evidence

From C4 predictions and parser traces, report:

- schema-valid, parser-valid, AST-valid, compile-success, and executable rates;
- parser, compiler, and solver status counts;
- malformed-output and schema-invalid counts;
- inconsistent-premise count;
- semantic-equivalence status counts;
- exact and normalized logic match rates where recorded;
- latency and recursive model-call summaries;
- fallback counts only when objectively recorded in trace fields.

Error categories will follow trace-supported stage boundaries:

- parser failure;
- compilation failure;
- solver failure;
- executable but incorrect symbolic verdict;
- invalid output.

The analysis will not label every executable wrong answer as a translation error unless the available semantic-agreement or gold-logic fields support that statement. “Symbolic verdict overriding a correct model answer” is an observable paired transition, not a claim about the solver's internal fault.

## Figures

Generate each figure in vector PDF and 300-DPI PNG under `paper/figures/`:

1. `overall_comparison`: grouped accuracy and macro-F1 for C1, C3, and C4.
2. `per_class_f1`: True, False, and Unknown F1 for C1, C3, and C4.
3. `intervention_outcomes`: the four C3-to-C4 correctness transitions.
4. `confusion_matrices`: side-by-side strict C3 and C4 confusion matrices, including Invalid for C4.
5. `reasoning_depth`: accuracy versus grouped QDep with sample sizes.
6. `bridge_reliability`: successful and failed parser/compiler/solver stages or a trace-supported failure breakdown.

Styling will use readable IEEE column widths, color-blind-safe colors, restrained grids, embedded labels, and no decorative effects.

## Tables

Generate IEEE-compatible fragments under `paper/generated_tables/`:

1. Dataset statistics.
2. Model and fine-tuning configuration.
3. Canonical ProofWriter results.
4. Per-class precision, recall, and F1.
5. C3-to-C4 intervention analysis.
6. Bridge reliability and failure counts.
7. Statistical comparison and confidence intervals.

The manuscript will use `booktabs`, compact column spacing, and `table*` only when a table cannot remain legible in one column. Best values will be bolded only for descriptive comparison, not to imply that conditions differ significantly without a paired test.

## Manuscript structure

Create `paper/main.tex` using `\documentclass[conference]{IEEEtran}` with:

1. Introduction
2. Related Work
3. System and Implemented Conditions
4. Experimental Setup
5. Results
6. Intervention and Error Analysis
7. Discussion
8. Limitations and Threats to Validity
9. Conclusion

The abstract will be 150–220 words and will state the negative/mixed outcome numerically. The discussion will distinguish solver determinism from translation fidelity and will not claim that symbolic reasoning is generally ineffective.

## Bibliography policy

Create `paper/references.bib` using only independently verified entries. Existing BibTeX may be reused only after checking primary publisher, DOI, or original-paper metadata. Candidate topics include ProofWriter, FOLIO, Logic-LM, LINC, NL2Logic, Z3, Qwen3, chain-of-thought, and neuro-symbolic computing.

Unverified or obsolete references will be omitted rather than invented. The paper will not cite the old manuscript as experimental evidence.

## LaTeX and PDF workflow

Copy the verified `IEEEtran.cls` from the local conference template into `paper/` for a self-contained build. Compile with the locally installed MiKTeX tools using:

1. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
2. `bibtex main`
3. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
4. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`

Before the first PDF-authoring command, invoke the required artifact-operation marker exactly once. Render the compiled PDF with Poppler and inspect every page. Fix compilation errors, missing references, overfull tables, clipped figures, unreadable labels, and visual defects before delivery.

## Numerical consistency gates

The final verification script will assert that:

- canonical row counts and IDs match;
- generated JSON, CSV, LaTeX tables, and figure source data agree;
- every headline number in the manuscript appears in generated analysis output;
- C3 and C4 use checkpoint-250;
- strict invalid handling is applied consistently;
- FOLIO performance metrics are absent;
- no self-check or symbolic-only experimental row is present;
- all figure files exist in both formats;
- `main.pdf` exists after a clean compile;
- bibliography and cross-references resolve;
- rendered pages have been visually inspected.

## Deliverables

- `paper/main.tex`
- `paper/references.bib`
- `paper/main.pdf`
- `paper/IEEEtran.cls`
- `paper/figures/*.pdf`
- `paper/figures/*.png`
- `paper/generated_tables/*.tex`
- `analysis/*.py`
- `analysis/tests/*.py`
- `analysis/generated/*`

No separate long-form report will be created.
