"""Run the Phase 2 LLM-only ProofWriter smoke evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.baselines.llm_only import evaluate_proofwriter_record
from src.config import ExperimentConfig, PROJECT_ROOT
from src.datasets.proofwriter import iter_jsonl
from src.models.qwen import QwenRunner
from src.prompts import PROOFWRITER_LLM_ONLY_TEMPLATE, load_template


def summarize(results: list[dict], runtime: dict, template: str, config: ExperimentConfig) -> dict:
    valid_latencies = [row["latency_ms"] for row in results if row["latency_ms"] is not None]
    by_depth: dict[str, dict] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["reasoning_depth"]].append(row)
    for depth, rows in sorted(grouped.items()):
        by_depth[str(depth)] = {
            "samples": len(rows),
            "correct": sum(row["correct_finally"] for row in rows),
            "accuracy": sum(row["correct_finally"] for row in rows) / len(rows),
        }
    correct = sum(row["correct_finally"] for row in results)
    return {
        "phase": 2,
        "method": "llm_only",
        "model_id": config.model_id,
        "settings": {
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "seed": config.seed,
            "quantization": config.quantization,
        },
        "model_runtime": runtime,
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "samples": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else None,
        "format_valid": sum(row["final_prediction"] is not None for row in results),
        "error_categories": dict(Counter(row["error_category"] or "none" for row in results)),
        "average_latency_ms": statistics.fmean(valid_latencies) if valid_latencies else None,
        "total_input_tokens": sum(row["input_tokens"] or 0 for row in results),
        "total_output_tokens": sum(row["output_tokens"] or 0 for row in results),
        "by_reasoning_depth": by_depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl")
    parser.add_argument("--start", type=int, default=0, help="Zero-based input offset")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "llm_only_phase2.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "llm_only_phase2_summary.json")
    args = parser.parse_args()

    config = ExperimentConfig.load()
    template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
    records = list(iter_jsonl(args.input))[args.start : args.start + args.limit]
    if len(records) != args.limit:
        raise ValueError(f"Requested {args.limit} records, found {len(records)}")
    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            result = evaluate_proofwriter_record(runner, record, template)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({"completed": index, "sample_id": result["sample_id"], "prediction": result["final_prediction"], "correct": result["correct_finally"]}))

    summary = summarize(results, runner.runtime_info, template, config)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
