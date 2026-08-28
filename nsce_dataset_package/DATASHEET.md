# Dataset datasheet

## Motivation and composition

The package supports two distinct evaluations: formal logical inference (ProofWriter) and factuality classification of LLM response segments (FELM). They share a superset record schema but are not forced into identical reasoning mechanics.

The downloaded ProofWriter derivative contains 360 development and 360 test records, with 20 records in every label×depth cell over three labels and depths 0–5. The downloaded FELM file contains 847 test responses and 4,426 annotated segments across math, reasoning, science/technology, world knowledge, and writing/recommendation. These observed counts differ slightly from some upstream summary text and are recorded in `validation_report.json`.

## Provenance

- ProofWriter: OWA balanced derivative by Theo X. Olausson, based on Tafjord, Dalvi, and Clark (2021). Each normalized row retains its source ID and split. License reported by the derivative dataset card: CC BY 4.0.
- FELM: Chen et al. (2023). Each normalized row retains its source index, source category, domain, annotation comments, error types, and reference URLs. Dataset license: CC BY-NC-SA 4.0.

Raw benchmark files are intentionally not bundled. The normalized files are transformations of downloaded benchmark records; users remain responsible for complying with upstream terms and attribution.

## Transformations

ProofWriter theory sentences are split into facts and rules using the explicit initial token `If`. Labels and splits are normalized. Natural-language statements are not automatically converted into executable logic because the balanced derivative does not expose the original formal triples/rules.

FELM response segments become claim objects with one aligned claim-label object each. Original binary labels are preserved. Reference URLs and any bundled reference content are retained.

## Leakage controls

- Tune only on `proofwriter_dev`; evaluate once on `proofwriter_test`.
- FELM has no separate development split in this package. Freeze prompts and thresholds using ProofWriter development or a separately documented calibration set before scoring FELM.
- Never place target labels, annotator comments, or error types in generation or verification prompts.
- For RAG, retrieve only from the declared corpus/candidate pool and log retrieved IDs.
- Record model/version/date because benchmark contamination cannot be ruled out for pretrained LLMs.

## Known limitations

- ProofWriter is synthetic and does not represent all real-world reasoning.
- The balanced ProofWriter derivative omits original proof traces and formal logical forms; proof-quality evaluation requires the full upstream release.
- FELM evaluates factuality classification of fixed ChatGPT responses, not factual correctness of new responses from the experimental base model.
- FELM's class imbalance makes raw accuracy insufficient.
- Reference pages can change or disappear. Snapshot evidence with URL, access date, and content hash.
- `factual_error` does not encode whether the error is contradicted or merely unsupported.

## Ethics and intended use

This package is intended for research evaluation, not training or fine-tuning the compared base LLM. Do not use benchmark labels to tune on the held-out sets. FELM's non-commercial ShareAlike license may restrict downstream use.
