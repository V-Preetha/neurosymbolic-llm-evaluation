"""Run the Phase 3 same-model self-verification evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.baselines.self_check import evaluate_proofwriter_record
from src.config import ExperimentConfig, PROJECT_ROOT
from src.datasets.proofwriter import iter_jsonl
from src.models.qwen import QwenRunner
from src.prompts import (
    PROOFWRITER_LLM_ONLY_TEMPLATE,
    PROOFWRITER_SELF_CHECK_TEMPLATE,
    load_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "results" / "raw" / "llm_only_phase2.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "self_check_phase3.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "self_check_phase3_summary.json")
    args = parser.parse_args()

    config = ExperimentConfig.load()
    initial_template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
    verification_template = load_template(PROOFWRITER_SELF_CHECK_TEMPLATE)
    records = list(iter_jsonl(args.input))[: args.limit]
    baselines = {row["sample_id"]: row for row in iter_jsonl(args.baseline)}
    if len(records) != args.limit:
        raise ValueError(f"Requested {args.limit} records, found {len(records)}")
    missing = [record["sample_id"] for record in records if record["sample_id"] not in baselines]
    if missing:
        raise ValueError(f"Missing Phase 2 baseline records: {missing}")

    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
    results: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            result = evaluate_proofwriter_record(runner, record, initial_template, verification_template)
            baseline_prediction = baselines[result["sample_id"]]["final_prediction"]
            result["phase2_initial_prediction"] = baseline_prediction
            result["initial_matches_phase2"] = result["initial_prediction"] == baseline_prediction
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({
                "completed": index, "sample_id": result["sample_id"],
                "initial": result["initial_prediction"], "final": result["final_prediction"],
                "changed": result["answer_changed"], "correct": result["correct_finally"],
            }))

    initially_wrong = [row for row in results if not row["correct_initially"]]
    initially_correct = [row for row in results if row["correct_initially"]]
    corrected = sum(row["correction_successful"] for row in results)
    harmful = sum(row["correct_initially"] and not row["correct_finally"] for row in results)
    summary = {
        "phase": 3, "method": "self_check", "model_id": config.model_id,
        "settings": {
            "temperature": config.temperature, "max_new_tokens": config.max_new_tokens,
            "seed": config.seed, "quantization": config.quantization,
        },
        "model_runtime": runner.runtime_info,
        "initial_prompt_sha256": hashlib.sha256(initial_template.encode("utf-8")).hexdigest(),
        "verification_prompt_sha256": hashlib.sha256(verification_template.encode("utf-8")).hexdigest(),
        "samples": len(results),
        "initial_correct": sum(row["correct_initially"] for row in results),
        "initial_accuracy": sum(row["correct_initially"] for row in results) / len(results),
        "final_correct": sum(row["correct_finally"] for row in results),
        "final_accuracy": sum(row["correct_finally"] for row in results) / len(results),
        "answers_changed": sum(row["answer_changed"] for row in results),
        "initially_incorrect": len(initially_wrong), "successful_corrections": corrected,
        "correction_success_rate": corrected / len(initially_wrong) if initially_wrong else None,
        "initially_correct": len(initially_correct), "harmful_revisions": harmful,
        "harmful_revision_rate": harmful / len(initially_correct) if initially_correct else None,
        "phase2_initial_mismatches": sum(not row["initial_matches_phase2"] for row in results),
        "fairness_check_passed": all(row["initial_matches_phase2"] for row in results),
        "format_valid": sum(row["final_prediction"] is not None for row in results),
        "error_categories": dict(Counter(row["error_category"] or "none" for row in results)),
        "average_total_latency_ms": statistics.fmean(row["latency_ms"] for row in results),
        "total_input_tokens": sum(row["input_tokens"] for row in results),
        "total_output_tokens": sum(row["output_tokens"] for row in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

