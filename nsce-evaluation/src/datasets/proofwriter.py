"""ProofWriter adapter for the immutable NSCE dataset package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.schema import canonical_record, validate_record


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def normalize_proofwriter(source: dict[str, Any]) -> dict[str, Any]:
    """Project an existing ProofWriter source row into the requested canonical schema."""
    if source.get("dataset") != "proofwriter":
        raise ValueError(f"Expected ProofWriter row, got {source.get('dataset')!r}")
    record = canonical_record()
    for key in record:
        if key in source:
            record[key] = source[key]
    errors = validate_record(record)
    if errors:
        raise ValueError(f"Invalid normalized record {record.get('sample_id')}: {'; '.join(errors)}")
    return record


def write_phase1_sample(source_path: Path, output_path: Path, limit: int = 20) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for source in iter_jsonl(source_path):
            if count >= limit:
                break
            record = normalize_proofwriter(source)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
    if count != limit:
        raise ValueError(f"Requested {limit} samples but found {count}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize the Phase 1 ProofWriter sample")
    parser.add_argument(
        "--source", type=Path,
        default=dataset_package_path() / "normalized" / "proofwriter_dev.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    count = write_phase1_sample(args.source, args.output, args.limit)
    print(json.dumps({"status": "ok", "records": count, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()

