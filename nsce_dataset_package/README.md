# NSCE research dataset package

This package provides analysis-ready data for a controlled comparison of LLM-only, self-verification, RAG+LLM, and an external Neuro-Symbolic Cognition Engine (NSCE). It contains **1,567 normalized records**: 720 balanced ProofWriter records and all 847 FELM records. The downloaded FELM file contributes 4,426 segment-level factuality annotations.

## Recommended experimental use

- Use `normalized/proofwriter_test.jsonl` (360 rows) for the final logical-reasoning comparison. It has 20 rows in every label×depth cell: three labels across depths 0–5 (60 samples per depth; 120 per label).
- Use `normalized/proofwriter_dev.jsonl` (360 rows) only for prompt, parser, retriever, and hyperparameter development. Do not report it as held-out performance.
- Use `normalized/felm_test.jsonl` (847 responses, 4,426 segments in the downloaded version) for factuality-verifier evaluation. Report segment-level balanced accuracy, precision, recall, and F1 because the labels are imbalanced.
- Use exactly the same sample IDs, base LLM, decoding settings, answer parser, and knowledge access across all four architectures.
- Run each stochastic condition with at least three seeds and report mean, standard deviation, paired bootstrap confidence intervals, and paired significance tests.

The package provides 1,207 held-out top-level units (360 ProofWriter test + 847 FELM test). Statistical power depends on the observed paired effect, so report confidence intervals rather than claiming adequacy from row count alone.

## Files

- `normalized/master.jsonl`: all normalized records.
- `normalized/proofwriter_dev.jsonl`: development-only logic set.
- `normalized/proofwriter_test.jsonl`: held-out logic set.
- `normalized/felm_test.jsonl`: held-out factuality set.
- `rag/proofwriter_corpus.jsonl`: statement-level ProofWriter retrieval corpus.
- `rag/proofwriter_queries.jsonl`: queries, labels, candidate pools, and in-instance relevance IDs.
- `rag/retrieval_results_template.jsonl`: fill with each retriever's top-5 output.
- `sample_index.csv`: compact analysis index.
- `schema.json`: machine-readable schema for the normalized master records.
- `validation_report.json`: counts, distributions, and integrity checks.
- `DATASHEET.md`: provenance, limitations, and leakage policy.
- `CITATION.bib`: benchmark citations.
- `build_dataset.py`: deterministic rebuild script (seed 20260814).
- `SHA256SUMS.txt`: integrity hashes.

## Label policy

ProofWriter labels map `True`→`true`/`entailed`, `False`→`false`/`contradicted`, and `Uncertain`→`unknown`.

FELM is a binary factuality benchmark. Its `true` segments map to `supported`; `false` segments map to `factual_error`. A false FELM label is **not** automatically relabeled `contradicted`, because the original annotation does not consistently distinguish contradiction from insufficient support. Store an NSCE's predicted `SUPPORTED`, `CONTRADICTED`, or `UNKNOWN` separately during evaluation.

## RAG caveat

ProofWriter `relevant_doc_ids` identify every statement from the originating theory, not the minimal proof chain. This supports instance-level retrieval recall but should not be described as gold proof-level relevance. The candidate pool contains the originating theory plus up to 40 deterministic cross-instance distractors. FELM references are retained under `provenance.reference_urls`; pages should be snapshotted with access dates if used as a live RAG corpus.

## Rebuild

Place the two balanced ProofWriter JSONL source files and FELM `all.jsonl` in a raw directory using the filenames expected by the script, then run:

```powershell
python build_dataset.py --raw PATH_TO_RAW --out PATH_TO_OUTPUT
```

No model-generated content is added by normalization.
