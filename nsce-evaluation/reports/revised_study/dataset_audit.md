# ProofWriter Dataset Audit (Phase C1)

Source dev split: `E:\SRM\Projects and Internships\Univeristy of Hull\nsce_dataset_package\normalized\proofwriter_dev.jsonl`

Source test split (structure-only, not touched for predictions): `E:\SRM\Projects and Internships\Univeristy of Hull\nsce_dataset_package\normalized\proofwriter_test.jsonl`

## Development split

- Total examples: **360**
- Label distribution: {'unknown': 120, 'false': 120, 'true': 120}
- Reasoning-depth distribution: {'0': 60, '1': 60, '2': 60, '3': 60, '4': 60, '5': 60}
- Structure tag counts (a record may carry multiple tags): {'conditional_rule': 357, 'conjunction': 341, 'has_rules': 357, 'negation': 251, 'facts_only': 3}
- Exact duplicate (question+facts+rules) records: 0
- Duplicate questions (same question text, different worlds): 66
- Distinct underlying worlds (unique facts+rules sets): 360
- Worlds queried by more than one question: 0 (sample: {})
- High fact-overlap pairs among first 50 records (Jaccard>0.5): 0 found

### Label x depth grid (development)

| Label | Depth | Count |
|---|---|---|
| false | 0 | 20 |
| false | 1 | 20 |
| false | 2 | 20 |
| false | 3 | 20 |
| false | 4 | 20 |
| false | 5 | 20 |
| true | 0 | 20 |
| true | 1 | 20 |
| true | 2 | 20 |
| true | 3 | 20 |
| true | 4 | 20 |
| true | 5 | 20 |
| unknown | 0 | 20 |
| unknown | 1 | 20 |
| unknown | 2 | 20 |
| unknown | 3 | 20 |
| unknown | 4 | 20 |
| unknown | 5 | 20 |

## Test split (structure-only audit)

- Total examples: **360**
- Label distribution: {'unknown': 120, 'false': 120, 'true': 120}
- Reasoning-depth distribution: {'0': 60, '1': 60, '2': 60, '3': 60, '4': 60, '5': 60}
- **No predictions were run on this split.** This audit only reads facts/rules/question/label/depth metadata to check for leakage against the development split.

## Dev/test leakage check

- Distinct worlds in dev: 360
- Distinct worlds in test: 360
- Worlds shared between dev and test: 0
- Questions shared between dev and test: 105
- **Verdict: NO WORLD-LEVEL LEAKAGE. Query surface text is reused across distinct worlds (expected in ProofWriter's small closed entity/verb vocabulary) -- this is NOT sample-level leakage since the underlying facts/rules differ, but is recorded for transparency.**

## Notes

- "Structure tags" are a coarse heuristic (substring/keyword based), not a formal classification; see `dataset_audit.json` for per-tag counts. They approximate: negation, conjunction, presence of conditional (`If ... then`) rules, universal-quantifier rules (`All ... are ...`), and facts-only records.
- Full per-record data (including world-key hashes) is in `dataset_audit.json`'s intermediate computation but is stripped from the final file to keep it compact; rerun this script for full detail.