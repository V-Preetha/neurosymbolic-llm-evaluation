# Frozen 72-Example ProofWriter Development Pilot -- Qwen3-8B

Model: `Qwen/Qwen3-8B` (revision `b968826d9c46dd6066d109eabc6255188de91218`), quantization `nf4`, GPU `NVIDIA GeForce RTX 5050 Laptop GPU`.

Pilot size: 72. Total run time: 141.111 s (model load 90.498 s). Peak VRAM: 6984.2 MB.

Qwen generation calls: shared initial=72, self-check verification=72, NS-bridge correction=41 (only on conflict), total=185.

## Primary metrics

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| llm_only | 0.4306 | 0.3529 | 0.3529 |
| self_check | 0.4444 | 0.3661 | 0.3661 |
| ns_bridge | 1.0 | 1.0 | 1.0 |

## Per-class metrics

### llm_only

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| TRUE/Supported | 0.6667 | 0.5 | 0.5714 | 24 |
| FALSE/Contradicted | 0.0 | 0.0 | 0.0 | 24 |
| UNKNOWN | 0.3519 | 0.7917 | 0.4872 | 24 |

### self_check

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| TRUE/Supported | 0.6842 | 0.5417 | 0.6047 | 24 |
| FALSE/Contradicted | 0.0 | 0.0 | 0.0 | 24 |
| UNKNOWN | 0.3585 | 0.7917 | 0.4935 | 24 |

### ns_bridge

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| TRUE/Supported | 1.0 | 1.0 | 1.0 | 24 |
| FALSE/Contradicted | 1.0 | 1.0 | 1.0 | 24 |
| UNKNOWN | 1.0 | 1.0 | 1.0 | 24 |

## Confusion matrices

### llm_only (rows=gold, cols=predicted)

| gold\pred | true | false | unknown | none |
|---|---|---|---|---|
| true | 12 | 0 | 12 | 0 |
| false | 1 | 0 | 23 | 0 |
| unknown | 5 | 0 | 19 | 0 |

### self_check (rows=gold, cols=predicted)

| gold\pred | true | false | unknown | none |
|---|---|---|---|---|
| true | 13 | 0 | 11 | 0 |
| false | 1 | 0 | 23 | 0 |
| unknown | 5 | 0 | 19 | 0 |

### ns_bridge (rows=gold, cols=predicted)

| gold\pred | true | false | unknown | none |
|---|---|---|---|---|
| true | 24 | 0 | 0 | 0 |
| false | 0 | 24 | 0 | 0 |
| unknown | 0 | 0 | 24 | 0 |

## Reasoning-depth breakdown (accuracy)

| Depth | LLM-only | Self-check | NS Bridge |
|---|---:|---:|---:|
| 0 | 0.6667 | 0.6667 | 1.0 |
| 1 | 0.4167 | 0.5 | 1.0 |
| 2 | 0.3333 | 0.3333 | 1.0 |
| 3 | 0.5 | 0.5 | 1.0 |
| 4 | 0.25 | 0.25 | 1.0 |
| 5 | 0.4167 | 0.4167 | 1.0 |

### Reasoning-depth breakdown (macro F1)

| Depth | LLM-only | Self-check | NS Bridge |
|---|---:|---:|---:|
| 0 | 0.5556 | 0.5556 | 1.0 |
| 1 | 0.3443 | 0.4167 | 1.0 |
| 2 | 0.254 | 0.254 | 1.0 |
| 3 | 0.4127 | 0.4127 | 1.0 |
| 4 | 0.1978 | 0.1978 | 1.0 |
| 5 | 0.3333 | 0.3333 | 1.0 |

## Self-check change analysis

- Changed: 1 (1.39%)
- Wrong -> correct: 1
- Correct -> wrong: 0
- Wrong -> different wrong: 0
- Unchanged wrong: 40

## Symbolic bridge action analysis

- RETAIN: 31, CORRECT: 41, ABSTAIN: 0
- Among CORRECT -- wrong Qwen -> correct bridge: 41, correct Qwen -> wrong bridge: 0, wrong Qwen -> different wrong bridge: 0
- Among RETAIN -- correctly retained: 31, incorrectly retained: 0

## Case A-E counts

- Case A (Qwen wrong -> bridge correct): 41
- Case B (Qwen correct -> bridge wrong): 0
- Case C (Qwen wrong -> self-check correct): 1
- Case D (Qwen wrong, self-check still wrong -> bridge correct): 40
- Case E (Qwen correct -> self-check wrong): 0

Exact example IDs are in `metrics.json` -> `case_analysis`.

## Symbolic verdict validity check

- Evaluated: 72/72, accuracy: 1.0
- Parser failures (ABSTAIN): 0
- Mismatching example IDs: []
- Parser failure example IDs: []

## Latency

- Mean LLM-only latency/example: 766.237 ms
- Mean self-check latency/example: 1511.613 ms
- Mean NS-bridge latency/example: 1201.992 ms
- Mean symbolic-reasoning latency/example: 4.33 ms
- Peak VRAM: 6984.2 MB

## Preliminary paired statistics (pilot, n=72 -- not confirmatory)

### LLM-only vs Self-check

- Contingency: {'both_correct': 31, 'both_wrong': 40, 'only_first_correct': 0, 'only_second_correct': 1}
- McNemar chi2 (continuity-corrected): 0.0, exact two-sided p=1.0
- Bootstrap accuracy diff (95% CI): 0.0139 [0.0, 0.0417] (10000 resamples, seed=42)

### LLM-only vs NS Bridge

- Contingency: {'both_correct': 31, 'both_wrong': 0, 'only_first_correct': 0, 'only_second_correct': 41}
- McNemar chi2 (continuity-corrected): 39.0244, exact two-sided p=0.0
- Bootstrap accuracy diff (95% CI): 0.5694 [0.4583, 0.6806] (10000 resamples, seed=42)
