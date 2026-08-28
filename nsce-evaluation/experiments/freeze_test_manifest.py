"""Phase C4: freeze held-out test split metadata WITHOUT running any predictions.

Records count, label distribution, depth distribution, a checksum of the raw source
file, and the exact sample IDs so the final held-out evaluation is reproducible.
This script never renders a prompt, never calls a model, and never inspects
`candidate_response` -- it only reads the counts needed to describe the split.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl

TEST_PATH = dataset_package_path() / "normalized" / "proofwriter_test.jsonl"
OUT_MANIFEST = PROJECT_ROOT / "data" / "revised_study" / "proofwriter_test_manifest.json"


def main() -> None:
    raw_bytes = TEST_PATH.read_bytes()
    checksum = hashlib.sha256(raw_bytes).hexdigest()

    records = list(iter_jsonl(TEST_PATH))
    label_counts = Counter(record["ground_truth"]["label"] for record in records)
    depth_counts = Counter(record["metadata"]["reasoning_depth"] for record in records)
    sample_ids = [record["sample_id"] for record in records]

    manifest = {
        "purpose": "FINAL HELD-OUT EVALUATION SPLIT. Do not run predictions until the bridge, "
                   "prompts, and dataset are frozen. Do not inspect predictions on this split "
                   "during development.",
        "source_file": str(TEST_PATH),
        "source_sha256": checksum,
        "count": len(records),
        "label_counts": dict(label_counts),
        "depth_counts": {str(k): v for k, v in sorted(depth_counts.items())},
        "sample_ids": sample_ids,
        "predictions_run": False,
        "frozen_at_phase": "C4",
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "count": len(records),
        "label_counts": dict(label_counts),
        "source_sha256": checksum,
        "predictions_run": False,
    }, indent=2))
    print(f"\nWrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
