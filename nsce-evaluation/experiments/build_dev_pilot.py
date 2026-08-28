"""Phase C2: build a fixed, deterministic ProofWriter development/pilot subset.

Sampling is stratified by (label, reasoning_depth) -- the dev split's own native
balance grid (3 labels x 6 depths x 20 examples/cell, see dataset_audit.md) -- using
a seeded hash sort per cell, matching the method already used for the historical
FELM Phase 9 pilot (build_phase9_pilot.py). This subset is for bridge development
and debugging only; it is drawn entirely from the untouched development split and
never touches proofwriter_test.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl

SEED = 42
PER_CELL = 4  # 3 labels x 6 depths x 4 = 72 examples

DEV_PATH = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
OUT_JSONL = PROJECT_ROOT / "data" / "revised_study" / "proofwriter_dev_pilot.jsonl"
OUT_MANIFEST = PROJECT_ROOT / "data" / "revised_study" / "proofwriter_dev_pilot_manifest.json"


def sort_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def structure_tags(record: dict) -> list[str]:
    facts = record["input"]["facts"]
    rules = record["input"]["rules"]
    question = record["input"]["question"]
    all_text = " ".join(facts + rules + [question]).lower()
    tags = []
    if "not " in all_text:
        tags.append("negation")
    if any(" and " in rule.lower() for rule in rules):
        tags.append("conjunction")
    if not rules:
        tags.append("facts_only")
    if any(rule.lower().startswith("if ") for rule in rules):
        tags.append("conditional_rule")
    if any("people" in rule.lower() or "things" in rule.lower() for rule in rules):
        tags.append("universal_quantifier")
    return tags


def main() -> None:
    records = list(iter_jsonl(DEV_PATH))
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        key = (record["ground_truth"]["label"], record["metadata"]["reasoning_depth"])
        cells[key].append(record)

    selected: list[dict] = []
    for key in sorted(cells):
        bucket = sorted(cells[key], key=lambda r: sort_key(SEED, r["sample_id"]))
        selected.extend(bucket[:PER_CELL])

    selected.sort(key=lambda r: r["sample_id"])

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for record in selected
    ]
    content = "\n".join(lines) + "\n"
    OUT_JSONL.write_text(content, encoding="utf-8", newline="\n")
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

    label_counts: dict[str, int] = defaultdict(int)
    depth_counts: dict[str, int] = defaultdict(int)
    structure_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        label_counts[record["ground_truth"]["label"]] += 1
        depth_counts[str(record["metadata"]["reasoning_depth"])] += 1
        for tag in structure_tags(record):
            structure_counts[tag] += 1

    manifest = {
        "purpose": "Development/debugging pilot only -- not a held-out evaluation set.",
        "source_split": str(DEV_PATH),
        "source_split_size": len(records),
        "selection_method": "stratified by (label, reasoning_depth); per-cell sort key sha256(f'{seed}|{sample_id}'); first N per cell",
        "seed": SEED,
        "per_cell_target": PER_CELL,
        "pilot_size": len(selected),
        "label_counts": dict(label_counts),
        "depth_counts": dict(sorted(depth_counts.items())),
        "structure_tag_counts": dict(structure_counts),
        "sample_ids": [record["sample_id"] for record in selected],
        "output_path": str(OUT_JSONL),
        "output_sha256": checksum,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "pilot_size": len(selected),
        "label_counts": dict(label_counts),
        "depth_counts": dict(sorted(depth_counts.items())),
        "checksum": checksum,
    }, indent=2))
    print(f"\nWrote {OUT_JSONL}")
    print(f"Wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
