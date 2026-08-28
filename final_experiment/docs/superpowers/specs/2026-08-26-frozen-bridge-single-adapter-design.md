# Frozen Bridge Single-Adapter Experimental Design

## Objective

The final experiment measures the interaction between two binary factors: direct LoRA fine-tuning of Qwen3-1.7B and addition of a fixed neuro-symbolic bridge. The bridge itself is never trained.

## Conditions

- C1 loads the frozen Qwen3-1.7B base model and predicts True/False/Unknown directly.
- C2 loads the same frozen base model and uses the frozen NL2Logic parser, typed FOL AST, two-pass compiler, and Z3 solver.
- C3 loads the dataset-specific direct LoRA and predicts True/False/Unknown directly.
- C4 loads the exact same dataset-specific direct LoRA as C3 and uses the exact same frozen bridge implementation as C2.

Each dataset has exactly one active adapter name and checkpoint lineage: `proofwriter_direct_qwen3_1_7b_lora` or `folio_direct_qwen3_1_7b_lora`. Conditions 3 and 4 must resolve to the same adapter directory. Conditions 1 and 2 must not load an adapter.

## Frozen bridge contract

The active bridge is `NL2LogicParser -> typed FOL AST -> declaration pass -> expression pass -> Z3`. Parser, compiler, and solver code and configuration are identical between C2 and C4. Evaluation metadata records a stable frozen-bridge identity. Fine-tuning changes only the neural model factor: C4 runs the direct LoRA under the same parser prompts used by C2.

Parser traces for successful, failed, and partial parses remain condition-specific. AST validity, compile success, executable rate, semantic/gold-FOL agreement, final solver accuracy, paired corrections/regressions, gains, and interaction effects remain active.

## Training and CLI

Only direct records are valid training inputs. The training CLI accepts `--task direct` for compatibility with existing commands and rejects `--task bridge`. The training module must never call component-record expansion in an active training path. It produces one direct LoRA per dataset and retains exact base-model compatibility checks, frozen manifests, dynamic padding, target-token-only loss, epoch evaluation/checkpointing, and best-checkpoint selection.

Old bridge checkpoint directories and component-supervision utilities may remain as inactive archival artifacts, but no active configuration, adapter registry, training CLI, tokenizer audit, or C1-C4 evaluation route may reference a bridge adapter.

## Token length

The tokenizer audit processes only active direct train and training-validation records. It must report zero truncations at its selected limit. Configure `max_sequence_length: 292` only if the audit confirms the observed maximum across every active direct train/validation group is at most 292. If the maximum is greater, select the smallest lossless standard limit from 320, 384, 512, 640, 768, 1024, 1536, 2048, or a higher explicitly supported standard limit. No active record may be truncated.

The existing tokenizer report already records direct-group maxima of 287 for ProofWriter training, 274 for ProofWriter validation, 292 for FOLIO training, and 227 for FOLIO validation. Implementation tests must independently enforce that the audit excludes bridge expansion and that configured length is not below the audited direct maximum.

## Verification

Tests are written and observed failing before production changes. Required verification after implementation is limited to:

- the complete unit-test suite;
- Python compilation;
- environment checks;
- no-model smoke tests.

No tokenizer/model weights may be downloaded or loaded during final verification. No training may be started or resumed.
