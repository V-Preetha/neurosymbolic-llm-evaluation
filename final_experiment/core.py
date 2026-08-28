from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
LABELS = ("True", "False", "Unknown")
CONDITIONS = {
    1: "base_direct",
    2: "base_bridge",
    3: "finetuned_direct",
    4: "finetuned_bridge",
}
ADAPTER_NAMES = {
    "proofwriter": "proofwriter_direct_qwen3_1_7b_lora",
    "folio": "folio_direct_qwen3_1_7b_lora",
}
FROZEN_BRIDGE_ID = "nl2logic_ast_two_pass_z3_v1"


def uses_frozen_bridge(condition: int) -> bool:
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid experimental condition: {condition}")
    return condition in (2, 4)


def adapter_name_for_condition(dataset: str, condition: int) -> str | None:
    if dataset not in ADAPTER_NAMES:
        raise ValueError(f"Invalid experiment dataset: {dataset}")
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid experimental condition: {condition}")
    return ADAPTER_NAMES[dataset] if condition in (3, 4) else None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_id(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {"true": "True", "false": "False", "unknown": "Unknown", "uncertain": "Unknown"}
    if text not in mapping:
        raise ValueError(f"Invalid logical label: {value!r}")
    return mapping[text]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install final_experiment/requirements.txt") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        out[key] = deep_update(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def resolved_config(dataset: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deep_update(load_yaml(ROOT / "configs" / "model.yaml"), load_yaml(ROOT / "configs" / f"{dataset}.yaml"))
    return deep_update(cfg, overrides or {})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def new_run_dir(dataset: str, stage: str, run_id: str | None = None) -> Path:
    rid = run_id or f"{utc_stamp()}_{dataset}_{stage}_{os.getpid()}"
    path = ROOT / "results" / rid
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def runtime_metadata(config: dict[str, Any]) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    gpu = cuda = None
    try:
        import torch
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "python": sys.version, "os": platform.platform(), "git_commit": commit,
        "versions": {x: package_version(x) for x in ("torch", "transformers", "peft", "datasets", "z3-solver", "scikit-learn")},
        "cuda": cuda, "gpu": gpu,
    }


@dataclass
class PredictionRow:
    dataset: str
    run_id: str
    example_id: str
    condition: int
    gold_label: str
    predicted_label: str | None = None
    raw_model_response: str | None = None
    generated_logic: str | None = None
    parser_status: str | None = None
    compiler_status: str | None = None
    solver_status: str | None = None
    solver_result: str | None = None
    qdep: int | None = None
    maxD: int | None = None
    NFact: int | None = None
    NRule: int | None = None
    latency_ms: float | None = None
    token_count: int | None = None
    confidence: float | None = None
    correctness: bool = False
    error_category: str | None = None
    timing: dict[str, float] = field(default_factory=dict)
    translation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
