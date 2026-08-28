"""Fixed, shared experiment configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
DEFAULT_DATASET_PACKAGE = PROJECT_ROOT.parent / "nsce_dataset_package"


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    temperature: float
    max_new_tokens: int
    seed: int
    quantization: str
    logical_labels: tuple[str, ...]
    rag_top_k: int
    embedding_model_id: str
    embedding_model_revision: str
    rag_distractors: int
    proofwriter_phase1_samples: int
    model_revision: str | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "ExperimentConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["logical_labels"] = tuple(payload["logical_labels"])
        return cls(**payload)


def dataset_package_path() -> Path:
    """Return the immutable source dataset package path."""
    configured = os.environ.get("NSCE_DATASET_PACKAGE")
    return Path(configured).resolve() if configured else DEFAULT_DATASET_PACKAGE
