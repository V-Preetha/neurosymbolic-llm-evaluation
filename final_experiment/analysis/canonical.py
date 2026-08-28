"""Load and validate the sole matched experiment used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


LABELS = ("True", "False", "Unknown")
EXPECTED_SUBSET_HASH = "1b4665a0472a3aa15541be13d538336a00bd8ed1d104cd76bf80b8b360bd9092"
CONDITION_DIRS = {
    "C1": "c1_base_direct",
    "C3": "c3_finetuned_direct",
    "C4": "c4_finetuned_bridge",
}


@dataclass(frozen=True)
class CanonicalRun:
    root: Path
    metadata: dict[str, Any]
    condition_metadata: dict[str, dict[str, Any]]
    conditions: dict[str, list[dict[str, Any]]]
    subset_hash: str
    example_ids: tuple[str, ...]
    checkpoint: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing canonical file: {path}")


def load_canonical_run(root: Path) -> CanonicalRun:
    root = Path(root).resolve()
    _require(root / "metadata.json")
    metadata = read_json(root / "metadata.json")
    condition_metadata: dict[str, dict[str, Any]] = {}
    conditions: dict[str, list[dict[str, Any]]] = {}

    for name, directory in CONDITION_DIRS.items():
        condition_root = root / directory
        _require(condition_root / "metadata.json")
        _require(condition_root / "predictions.jsonl")
        condition_metadata[name] = read_json(condition_root / "metadata.json")
        conditions[name] = read_jsonl(condition_root / "predictions.jsonl")

    if any(len(rows) != 200 for rows in conditions.values()):
        raise ValueError("canonical conditions must each contain exactly 200 predictions")

    reference_ids = tuple(row.get("example_id") for row in conditions["C1"])
    reference_gold = tuple(row.get("gold_label") for row in conditions["C1"])
    for name, rows in conditions.items():
        if tuple(row.get("example_id") for row in rows) != reference_ids:
            raise ValueError(f"ordered example IDs differ for {name}")
        if tuple(row.get("gold_label") for row in rows) != reference_gold:
            raise ValueError(f"gold labels differ for {name}")
        invalid_gold = sorted({row.get("gold_label") for row in rows} - set(LABELS))
        if invalid_gold:
            raise ValueError(f"invalid gold labels for {name}: {invalid_gold}")
        meta_ids = tuple(condition_metadata[name].get("test_ids", ()))
        if meta_ids != reference_ids:
            raise ValueError(f"metadata IDs differ for {name}")

    hashes = {meta.get("frozen_subset_sha256") for meta in condition_metadata.values()}
    if hashes != {EXPECTED_SUBSET_HASH}:
        raise ValueError(f"unexpected or mismatched frozen subset hashes: {sorted(str(x) for x in hashes)}")
    if metadata.get("frozen_subset_sha256") != EXPECTED_SUBSET_HASH:
        raise ValueError("top-level frozen subset hash does not match canonical evidence")

    c3_checkpoint = condition_metadata["C3"].get("adapter_checkpoint")
    c4_checkpoint = condition_metadata["C4"].get("adapter_checkpoint")
    if not c3_checkpoint or c3_checkpoint != c4_checkpoint or not c3_checkpoint.endswith("checkpoint-250"):
        raise ValueError("C3 and C4 must use the identical checkpoint-250 adapter")

    return CanonicalRun(
        root=root,
        metadata=metadata,
        condition_metadata=condition_metadata,
        conditions=conditions,
        subset_hash=EXPECTED_SUBSET_HASH,
        example_ids=reference_ids,
        checkpoint=c3_checkpoint,
    )
