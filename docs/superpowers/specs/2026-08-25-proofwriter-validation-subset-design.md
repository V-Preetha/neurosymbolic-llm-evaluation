# ProofWriter Training Validation Subset Design

ProofWriter `main` training uses a frozen 5,000-example subset of the complete 85,468-example official validation split. Selection is deterministic with the experiment seed and stratified jointly by normalized label and `QDep`. The full validation data and its existing `validation.json` manifest remain unchanged.

Preparation writes `validation_train_main.json` with dataset, source split, mode, seed, size, `stratify_by: [label, QDep]`, per-stratum counts, ID digest, and ordered example IDs. Existing valid manifests are reused.

Training loads this subset for checkpoint selection, evaluates once per epoch, saves once per epoch, and retains `load_best_model_at_end=true`. `--final-full-validation` optionally tokenizes and evaluates the complete validation manifest once after training; it is off by default.

Before bridge training starts or model weights load, output reports original selected rows, total expanded NL2Logic records, and a complete partition: selector (parser and relation selectors), atomic (four relation parsers), quantified, and logical (unary/binary).

Frozen train/test manifests, the 10,000-example ProofWriter main sample, FOLIO behavior, and four experimental conditions are unchanged. Verification is limited to unit/mock/no-model smoke tests.
