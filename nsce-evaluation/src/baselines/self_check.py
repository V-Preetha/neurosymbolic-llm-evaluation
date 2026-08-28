"""Architecture 2: initial inference followed by same-model self-verification."""

from __future__ import annotations

from typing import Any

from src.models.qwen import QwenRunner
from src.prompts import render_proofwriter_prompt, render_self_check_prompt


def evaluate_proofwriter_record(
    runner: QwenRunner,
    record: dict[str, Any],
    initial_template: str,
    verification_template: str,
) -> dict[str, Any]:
    ground_truth = record["ground_truth"]["label"]
    initial = runner.generate_label(render_proofwriter_prompt(record, initial_template))
    if initial.prediction is None:
        return {
            "sample_id": record["sample_id"], "dataset": record["dataset"],
            "method": "self_check", "initial_prediction": None, "final_prediction": None,
            "ground_truth": ground_truth, "correct_initially": False, "correct_finally": False,
            "verification_status": None, "symbolic_prediction": None,
            "verification_attempted": False, "answer_changed": False,
            "correction_attempted": False, "correction_successful": False,
            "retrieved_docs": [], "proof_trace": [], "error_category": "output_format_error",
            "initial_raw_output": initial.raw_text, "verification_raw_output": None,
            "initial_latency_ms": initial.latency_ms, "verification_latency_ms": 0.0,
            "latency_ms": initial.latency_ms, "initial_input_tokens": initial.input_tokens,
            "initial_output_tokens": initial.output_tokens, "verification_input_tokens": 0,
            "verification_output_tokens": 0, "input_tokens": initial.input_tokens,
            "output_tokens": initial.output_tokens,
            "reasoning_depth": record["metadata"]["reasoning_depth"],
        }

    verification = runner.generate_label(
        render_self_check_prompt(record, initial.prediction, verification_template)
    )
    final_prediction = verification.prediction
    correct_initially = initial.prediction == ground_truth
    correct_finally = final_prediction == ground_truth
    answer_changed = final_prediction is not None and final_prediction != initial.prediction
    correction_successful = (not correct_initially) and correct_finally
    if final_prediction is None:
        error_category = "output_format_error"
    elif correct_finally:
        error_category = None
    elif correct_initially and answer_changed:
        error_category = "self_check_changed_correct_to_incorrect"
    else:
        error_category = "self_check_failed_to_correct"

    return {
        "sample_id": record["sample_id"], "dataset": record["dataset"],
        "method": "self_check", "initial_prediction": initial.prediction,
        "final_prediction": final_prediction, "ground_truth": ground_truth,
        "correct_initially": correct_initially, "correct_finally": correct_finally,
        "verification_status": "completed" if final_prediction is not None else "format_error",
        "symbolic_prediction": None, "verification_attempted": True,
        "answer_changed": answer_changed, "correction_attempted": True,
        "correction_successful": correction_successful, "retrieved_docs": [],
        "proof_trace": [], "error_category": error_category,
        "initial_raw_output": initial.raw_text,
        "verification_raw_output": verification.raw_text,
        "initial_latency_ms": initial.latency_ms,
        "verification_latency_ms": verification.latency_ms,
        "latency_ms": round(initial.latency_ms + verification.latency_ms, 3),
        "initial_input_tokens": initial.input_tokens,
        "initial_output_tokens": initial.output_tokens,
        "verification_input_tokens": verification.input_tokens,
        "verification_output_tokens": verification.output_tokens,
        "input_tokens": initial.input_tokens + verification.input_tokens,
        "output_tokens": initial.output_tokens + verification.output_tokens,
        "reasoning_depth": record["metadata"]["reasoning_depth"],
    }

