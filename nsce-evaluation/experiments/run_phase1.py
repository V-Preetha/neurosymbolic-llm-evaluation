"""Normalize 20 ProofWriter samples and run one deterministic Qwen smoke inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import ExperimentConfig, PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl, write_phase1_sample
from src.models.qwen import QwenRunner


def render_prompt(record: dict, template: str) -> str:
    facts = "\n".join(f"- {item}" for item in record["input"]["facts"])
    rules = "\n".join(f"- {item}" for item in record["input"]["rules"])
    return template.format(facts=facts, rules=rules, question=record["input"]["question"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "raw" / "phase1_smoke.json",
    )
    args = parser.parse_args()
    config = ExperimentConfig.load()
    processed = PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl"
    source = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
    normalized_count = write_phase1_sample(source, processed, config.proofwriter_phase1_samples)
    record = next(iter(iter_jsonl(processed)))
    template = (PROJECT_ROOT / "prompts" / "proofwriter_llm_only.txt").read_text(encoding="utf-8")
    prompt = render_prompt(record, template)
    payload: dict = {
        "status": "normalization_verified",
        "model_id": config.model_id,
        "settings": {
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "quantization": config.quantization,
        },
        "normalized_records": normalized_count,
        "sample_id": record["sample_id"],
        "ground_truth": record["ground_truth"]["label"],
        "prompt_chars": len(prompt),
    }
    if not args.normalize_only:
        runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
        generation = runner.generate_label(prompt)
        payload.update({
            "status": "inference_complete",
            "model_runtime": runner.runtime_info,
            "raw_output": generation.raw_text,
            "prediction": generation.prediction,
            "format_valid": generation.prediction is not None,
            "latency_ms": generation.latency_ms,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
