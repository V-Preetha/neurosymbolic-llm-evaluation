"""Strict FELM adapter for claim-level factuality evaluation."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from src.datasets.schema import TOP_LEVEL_KEYS, validate_record


def _finite_json(value: Any) -> Any:
    """Replace non-finite source values so outputs remain strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def iter_felm(path: Path) -> Iterator[dict[str, Any]]:
    """Read source JSONL permissively; normalization removes bare NaN values."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid FELM JSON on line {line_number}: {exc}") from exc


def normalize_felm_record(source: dict[str, Any]) -> dict[str, Any]:
    """Return canonical fields and repair two non-finite response texts."""
    record = _finite_json({key: deepcopy(source.get(key)) for key in TOP_LEVEL_KEYS})
    response = record["candidate_response"]
    if not isinstance(response.get("text"), str):
        response["text"] = " ".join(
            claim["text"].strip()
            for claim in response.get("claims", [])
            if isinstance(claim, dict) and isinstance(claim.get("text"), str)
        )
    errors = validate_record(record)
    if errors:
        raise ValueError(f"Invalid FELM record {record.get('sample_id')}: {'; '.join(errors)}")
    return record


def write_felm_dataset(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for source_record in iter_felm(source):
            record = normalize_felm_record(source_record)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count
