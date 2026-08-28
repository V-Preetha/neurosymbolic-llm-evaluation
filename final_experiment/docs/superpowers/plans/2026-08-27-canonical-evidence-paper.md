# Canonical Evidence IEEE Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible IEEE conference paper and compiled PDF from the single matched 200-example ProofWriter experiment in `results/final200_proofwriter`.

**Architecture:** A small Python analysis package validates the matched evidence, derives every metric from raw JSONL predictions, and writes one canonical JSON result plus CSV and LaTeX fragments. Deterministic plotting code consumes that result, while `paper/main.tex` consumes only generated tables and figures. Tests lock down strict invalid-as-error scoring, paired transitions, provenance, and headline values before manuscript authoring.

**Tech Stack:** Python 3.12, pytest, NumPy, pandas, scikit-learn, SciPy, Matplotlib, IEEEtran LaTeX, BibTeX, MiKTeX, Poppler.

**Spec:** `docs/superpowers/specs/2026-08-27-canonical-evidence-paper-design.md`

## Global Constraints

- Use only `results/final200_proofwriter` for experimental results.
- Conditions are C1 base direct, C3 checkpoint-250 direct, and C4 checkpoint-250 descriptive NL2Logic-to-Z3 bridge.
- Count invalid predictions as wrong in all headline metrics.
- Treat C1-to-C3 as the fine-tuning effect and C3-to-C4 as the bridge effect; C1-to-C4 is descriptive only.
- Do not merge older 2,000-example runs or report FOLIO performance.
- Do not add self-check, symbolic-only, reconciliation, or verification-policy baselines.
- The symbolic verdict is C4's final prediction.
- Use seed 42 and 10,000 paired bootstrap replicates.
- Generate every figure as vector PDF and 300-DPI PNG.
- The directory is not a Git repository, so commit steps are replaced with explicit verification checkpoints.

---

### Task 1: Canonical Loader and Provenance Validation

**Files:**
- Create: `analysis/__init__.py`
- Create: `analysis/canonical.py`
- Create: `analysis/tests/__init__.py`
- Create: `analysis/tests/test_canonical.py`

**Interfaces:**
- Consumes: canonical directory containing `metadata.json` and three condition directories.
- Produces: `load_canonical_run(root: Path) -> CanonicalRun`, where `CanonicalRun` contains ordered `conditions: dict[str, list[dict]]`, `metadata: dict`, and `subset_hash: str`.

- [ ] **Step 1: Write failing provenance tests**

Create tests that load the real canonical directory and assert 200 rows per condition, identical ordered IDs and gold labels, the expected subset hash, and identical checkpoint references for C3/C4. Add synthetic tests proving that row-count, ID-order, gold-label, hash, checkpoint, and label-vocabulary mismatches raise `ValueError`.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_canonical.py -v`

Expected: collection or import failure because `analysis.canonical` does not exist.

- [ ] **Step 3: Implement the loader**

Implement immutable `CanonicalRun`, JSON/JSONL readers, condition constants, and explicit validators. The label vocabulary is exactly `{"True", "False", "Unknown"}` and ordered example IDs must match, not merely set membership.

- [ ] **Step 4: Run the focused tests**

Run the Task 1 command and require all tests to pass.

---

### Task 2: Strict Metrics and Paired Statistical Analysis

**Files:**
- Create: `analysis/metrics.py`
- Create: `analysis/tests/test_metrics.py`

**Interfaces:**
- Consumes: aligned prediction-row dictionaries from `CanonicalRun`.
- Produces: `condition_metrics(rows: list[dict]) -> dict`, `paired_analysis(c3: list[dict], c4: list[dict]) -> dict`, `depth_analysis(conditions: dict[str, list[dict]]) -> list[dict]`, and `bootstrap_intervals(conditions: dict[str, list[dict]], seed: int = 42, replicates: int = 10000) -> dict`.

- [ ] **Step 1: Write failing strict-scoring tests**

Use tiny hand-calculated arrays to assert that invalid/None predictions never match, confusion matrices include an Invalid prediction column, macro metrics use only the three gold classes, and valid-only accuracy has an explicit denominator. Add paired-transition tests for W-to-C, C-to-C, C-to-W, W-to-W and exact McNemar counts.

- [ ] **Step 2: Run tests and confirm failure**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_metrics.py -v`

- [ ] **Step 3: Implement metric functions**

Compute precision, recall, F1, support, accuracy, valid rate, valid-only accuracy, weighted F1, strict confusion matrices, transitions, correction rate `W_to_C / C3_wrong`, harm rate `C_to_W / C3_correct`, net corrections, exact two-sided binomial McNemar p-value, grouped QDep (`0`, `1`, `2`, `3`, `4+`), and deterministic percentile bootstrap intervals.

- [ ] **Step 4: Lock down canonical headline values**

Add real-data assertions: C1 accuracy 0.485 and macro F1 approximately 0.251064; C3 0.825 and 0.826268; C4 0.495 and 0.458925; C4 valid rate 0.895; transitions 12/87/78/23; net -66; exact McNemar p approximately `5.20243e-13`.

- [ ] **Step 5: Run Task 2 tests and Task 1 regression tests**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_metrics.py final_experiment\analysis\tests\test_canonical.py -v`

---

### Task 3: Bridge Diagnostics and Trace-Supported Failure Taxonomy

**Files:**
- Create: `analysis/bridge_diagnostics.py`
- Create: `analysis/tests/test_bridge_diagnostics.py`

**Interfaces:**
- Consumes: C4 `predictions.jsonl` and `parser_traces.jsonl`.
- Produces: `bridge_diagnostics(predictions: list[dict], traces: list[dict]) -> dict` with stage counts, executability, error taxonomy, semantic-status counts, latency summaries, model-call summaries, and trace-supported fallback counts.

- [ ] **Step 1: Inspect representative raw C4 records and encode field contracts in failing tests**

Assert parser/AST/compile success of 180/200, solver success and executable count 179/200, and one inconsistent-premises outcome. Assert error/status counts are derived from row fields rather than hard-coded.

- [ ] **Step 2: Run focused tests and confirm failure**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_bridge_diagnostics.py -v`

- [ ] **Step 3: Implement defensive diagnostics extraction**

Join predictions and traces by example ID, validate one-to-one coverage, count schema/parser/AST/compiler/solver states, classify only trace-supported failures, summarize semantic-equivalence fields, and compute latency quantiles and recursive-call totals. Report fallback only if explicit trace fields exist; otherwise record `available: false`.

- [ ] **Step 4: Run bridge and full analysis tests**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests -v`

---

### Task 4: Generated Results, Audit CSVs, and IEEE Tables

**Files:**
- Create: `analysis/tables.py`
- Create: `analysis/generate_paper_artifacts.py`
- Create: `analysis/tests/test_generated_artifacts.py`
- Generate: `analysis/generated/results.json`
- Generate: `analysis/generated/per_example.csv`
- Generate: `analysis/generated/depth.csv`
- Generate: `analysis/generated/bridge_failures.csv`
- Generate: `paper/generated_tables/*.tex`

**Interfaces:**
- Consumes: Tasks 1-3 functions and canonical source files.
- Produces: `build_results(run: CanonicalRun) -> dict`, `write_generated_artifacts(results: dict, analysis_dir: Path, paper_dir: Path) -> None`, and seven standalone LaTeX table fragments.

- [ ] **Step 1: Write failing artifact-consistency tests**

Assert generated JSON includes provenance, metrics, bootstrap, paired, depth, diagnostics, and dataset/configuration sections. Assert the per-example CSV has one row per canonical ID and C1/C3/C4 prediction, validity, correctness, QDep, and transition columns. Assert table fragments contain no unresolved placeholders and agree numerically with JSON.

- [ ] **Step 2: Run focused tests and confirm failure**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_generated_artifacts.py -v`

- [ ] **Step 3: Implement the generator and table escaping**

Write deterministic JSON/CSV outputs and IEEE-compatible `booktabs` table fragments for dataset statistics, model/training configuration, overall results, per-class scores, interventions, bridge reliability, and significance/confidence intervals. Escape LaTeX special characters centrally.

- [ ] **Step 4: Generate artifacts**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m final_experiment.analysis.generate_paper_artifacts`. The script must print the canonical source path and generated file list.

- [ ] **Step 5: Run all analysis tests**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests -v`

---

### Task 5: Publication-Quality Figures

**Files:**
- Create: `analysis/figures.py`
- Create: `analysis/tests/test_figures.py`
- Generate: `paper/figures/overall_comparison.{pdf,png}`
- Generate: `paper/figures/per_class_f1.{pdf,png}`
- Generate: `paper/figures/intervention_outcomes.{pdf,png}`
- Generate: `paper/figures/confusion_matrices.{pdf,png}`
- Generate: `paper/figures/reasoning_depth.{pdf,png}`
- Generate: `paper/figures/bridge_reliability.{pdf,png}`

**Interfaces:**
- Consumes: `analysis/generated/results.json` only.
- Produces: `generate_all_figures(results: dict, output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing figure tests**

Assert six exact figure stems, both formats per stem, non-empty files, PNG dimensions suitable for print, and PDF page count one. Assert plotted source arrays equal values in generated results.

- [ ] **Step 2: Run tests and confirm failure**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests\test_figures.py -v`

- [ ] **Step 3: Implement deterministic plotting**

Use a color-blind-safe palette, IEEE single/double-column dimensions, embedded value labels, restrained grids, 300-DPI PNG export, vector PDF export, and closed figures. Include Invalid as the fourth prediction column in C4's confusion matrix and label every QDep group with sample size.

- [ ] **Step 4: Generate and inspect figures**

Run the generator, render the six PDFs to PNG previews if necessary, and inspect all plots for clipping, overlap, small type, misleading scales, and consistency with generated JSON.

- [ ] **Step 5: Run all analysis tests**

Run from the parent workspace: `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests -v`

---

### Task 6: Verified Bibliography and IEEE Manuscript

**Files:**
- Create: `paper/main.tex`
- Create: `paper/references.bib`
- Copy: `paper/IEEEtran.cls`

**Interfaces:**
- Consumes: generated tables/figures and verified primary-source bibliographic metadata.
- Produces: a self-contained IEEEtran manuscript with no manually divergent result calculations.

- [ ] **Step 1: Verify references from primary sources**

Check publisher, DOI, arXiv, or original project metadata for ProofWriter, FOLIO, NL2Logic, Logic-LM, LINC, Z3, Qwen3, chain-of-thought, LoRA, QLoRA, and neuro-symbolic background. Omit any entry that cannot be verified.

- [ ] **Step 2: Create the bibliography**

Write complete BibTeX entries with stable keys, author lists, titles, venue or archive, year, DOI/URL where applicable, and no invented page ranges.

- [ ] **Step 3: Author the IEEE paper**

Use title `Evaluating Logical Reasoning in Large Language Models Through Fine-Tuning and Neuro-Symbolic Bridging`. Write a 150-220 word abstract; Introduction; Related Work; System and Implemented Conditions; Experimental Setup; Results; Intervention and Error Analysis; Discussion; Limitations and Threats to Validity; Conclusion. State the negative/mixed outcome and distinguish deterministic symbolic execution from fallible translation.

- [ ] **Step 4: Enforce claim boundaries**

Search the manuscript to ensure it does not call C4 a reconciliation or verification policy, claim NSCE improves performance, report FOLIO performance, combine legacy runs, or mention unmatched self-check/symbolic-only baselines as results.

---

### Task 7: Compile, Render, and Verify the PDF

**Files:**
- Generate: `paper/main.pdf`
- Generate temporarily: `paper/main.aux`, `paper/main.bbl`, `paper/main.blg`, `paper/main.log`
- Generate temporarily: rendered page PNGs under a dedicated verification directory.

**Interfaces:**
- Consumes: complete manuscript, bibliography, tables, figures, and IEEE class.
- Produces: compiled and visually verified `paper/main.pdf`.

- [ ] **Step 1: Mark the PDF artifact operation exactly once**

Before the first PDF-authoring command run:

`node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pdf`

- [ ] **Step 2: Compile the bibliography cycle**

From `paper/`, run `pdflatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, then `pdflatex` twice more. Require exit code zero for every command.

- [ ] **Step 3: Check logs and document invariants**

Fail on undefined citations/references, missing files, fatal errors, clipped or materially overfull boxes, absent figures, or a missing/empty `main.pdf`. Extract PDF text and verify all canonical headline values and the absence of forbidden claims/results.

- [ ] **Step 4: Render and inspect every page**

Use Poppler `pdftoppm -png -r 150 main.pdf <verification-prefix>`, enumerate every rendered page, and visually inspect each for column overflow, table clipping, figure legibility, blank pages, font substitution, broken references, and poor float placement. Patch and recompile until clean.

- [ ] **Step 5: Run final verification**

Run `final_experiment\.venv\Scripts\python.exe -m pytest final_experiment\analysis\tests -v`, regenerate artifacts, recompile cleanly, compare generated-table values to `analysis/generated/results.json`, and confirm PDF/PNG figure pairs and `paper/main.pdf` exist and are non-empty.
