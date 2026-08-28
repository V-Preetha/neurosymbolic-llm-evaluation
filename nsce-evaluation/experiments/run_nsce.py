"""Run the Phase 5 ProofWriter NSCE correction-loop evaluation."""

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

from src.config import ExperimentConfig, PROJECT_ROOT
from src.datasets.proofwriter import iter_jsonl
from src.models.qwen import QwenRunner
from src.nsce.pipeline import evaluate_proofwriter_record
from src.prompts import PROOFWRITER_LLM_ONLY_TEMPLATE, load_template


FEEDBACK_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "proofwriter_nsce_feedback.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "results" / "raw" / "llm_only_phase2.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "nsce_phase5.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "nsce_phase5_summary.json")
    args = parser.parse_args()

    config = ExperimentConfig.load()
    initial_template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
    feedback_template = load_template(FEEDBACK_TEMPLATE_PATH)
    records = list(iter_jsonl(args.input))[: args.limit]
    baselines = {row["sample_id"]: row for row in iter_jsonl(args.baseline)}
    if len(records) != args.limit:
        raise ValueError(f"Requested {args.limit} records, found {len(records)}")
    if any(record["sample_id"] not in baselines for record in records):
        raise ValueError("Phase 2 baseline is missing one or more requested records")

    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
    results: list[dict] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            result = evaluate_proofwriter_record(runner, record, initial_template, feedback_template)
            phase2_prediction = baselines[result["sample_id"]]["final_prediction"]
            result["phase2_initial_prediction"] = phase2_prediction
            result["initial_matches_phase2"] = result["initial_prediction"] == phase2_prediction
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({
                "completed": index, "sample_id": result["sample_id"],
                "initial": result["initial_prediction"],
                "symbolic": result["symbolic_prediction"],
                "final": result["final_prediction"],
                "correct": result["correct_finally"],
            }))

    initially_wrong = [row for row in results if not row["correct_initially"]]
    initially_correct = [row for row in results if row["correct_initially"]]
    corrected = sum(row["correction_successful"] for row in results)
    false_rejections = sum(
        row["correct_initially"] and row["verification_status"] == "conflict"
        for row in results
    )
    ignored = sum(
        row["correction_attempted"] and not row["feedback_accepted"] for row in results
    )
    summary = {
        "phase": 5, "method": "nsce", "model_id": config.model_id,
        "settings": {
            "temperature": config.temperature, "max_new_tokens": config.max_new_tokens,
            "seed": config.seed, "quantization": config.quantization,
        },
        "model_runtime": runner.runtime_info,
        "initial_prompt_sha256": hashlib.sha256(initial_template.encode("utf-8")).hexdigest(),
        "feedback_prompt_sha256": hashlib.sha256(feedback_template.encode("utf-8")).hexdigest(),
        "samples": len(results),
        "initial_correct": sum(row["correct_initially"] for row in results),
        "initial_accuracy": sum(row["correct_initially"] for row in results) / len(results),
        "symbolic_correct": sum(row["symbolic_correct"] for row in results),
        "symbolic_accuracy": sum(row["symbolic_correct"] for row in results) / len(results),
        "final_correct": sum(row["correct_finally"] for row in results),
        "final_accuracy": sum(row["correct_finally"] for row in results) / len(results),
        "conflicts": sum(row["verification_status"] == "conflict" for row in results),
        "correction_attempts": sum(row["correction_attempted"] for row in results),
        "initially_incorrect": len(initially_wrong), "successful_corrections": corrected,
        "correction_success_rate": corrected / len(initially_wrong) if initially_wrong else None,
        "initially_correct": len(initially_correct), "false_rejections": false_rejections,
        "false_rejection_rate": false_rejections / len(initially_correct) if initially_correct else None,
        "feedback_ignored": ignored,
        "feedback_acceptance_rate": (
            1 - ignored / sum(row["correction_attempted"] for row in results)
            if any(row["correction_attempted"] for row in results) else None
        ),
        "parser_failures": sum(row["verification_status"] == "parser_failure" for row in results),
        "phase2_initial_mismatches": sum(not row["initial_matches_phase2"] for row in results),
        "fairness_check_passed": all(row["initial_matches_phase2"] for row in results),
        "format_valid": sum(row["final_prediction"] is not None for row in results),
        "error_categories": dict(Counter(row["error_category"] or "none" for row in results)),
        "average_total_latency_ms": statistics.fmean(row["latency_ms"] for row in results),
        "average_symbolic_latency_ms": statistics.fmean(row["symbolic_latency_ms"] for row in results),
        "total_input_tokens": sum(row["input_tokens"] for row in results),
        "total_output_tokens": sum(row["output_tokens"] for row in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

