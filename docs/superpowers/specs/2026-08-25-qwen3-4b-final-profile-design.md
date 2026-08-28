# Qwen3-4B Final Experiment Profile

The final experiment uses `Qwen/Qwen3-4B` pinned to immutable Hugging Face revision `1cfa9a7208912126459214e8b04321603b3df60c` for all four conditions. ProofWriter main training selects 5,000 seed-42 examples stratified by label and QDep and trains for three epochs. Its 5,000-example training-validation and 2,000-example held-out evaluation selections remain unchanged.

The previous 10,000-example `train_main.json` is preserved as `train_main_10000_seed42.json`; a newly prepared deterministic 5,000-example manifest becomes `train_main.json`. No scale-up profile is activated automatically.

All adapter checkpoint namespaces include `qwen3_4b`. Metadata stores base model ID and revision. Training resume and evaluation adapter loading reject missing or mismatched model identity before loading weights, preventing Qwen3-8B artifacts from entering the 4B experiment.

Verification is limited to unit tests, compilation, and no-model smoke checks. No preparation or training command is run.
