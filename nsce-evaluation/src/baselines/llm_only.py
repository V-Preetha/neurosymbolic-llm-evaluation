"""Architecture 1: one unverified response from the shared base LLM."""

from __future__ import annotations

from typing import Any

from src.models.qwen import QwenRunner
from src.prompts import render_proofwriter_prompt


def evaluate_proofwriter_record(
    runner: QwenRunner,
    record: dict[str, Any],
    template: str,
) -> dict[str, Any]:
    prompt = render_proofwriter_prompt(record, template)
    ground_truth = record["ground_truth"]["label"]
    try:
        generation = runner.generate_label(prompt)
        prediction = generation.prediction
        correct = prediction == ground_truth
        error_category = None if correct else (
            "output_format_error" if prediction is None else "llm_reasoning_error"
        )
        return {
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "method": "llm_only",
            "initial_prediction": prediction,
            "final_prediction": prediction,
            "ground_truth": ground_truth,
            "correct_initially": correct,
            "correct_finally": correct,
            "verification_status": None,
            "symbolic_prediction": None,
            "correction_attempted": False,
            "correction_successful": False,
            "retrieved_docs": [],
            "proof_trace": [],
            "error_category": error_category,
            "raw_output": generation.raw_text,
            "latency_ms": generation.latency_ms,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "reasoning_depth": record["metadata"]["reasoning_depth"],
        }
    except Exception as exc:
        return {
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "method": "llm_only",
            "initial_prediction": None,
            "final_prediction": None,
            "ground_truth": ground_truth,
            "correct_initially": False,
            "correct_finally": False,
            "verification_status": None,
            "symbolic_prediction": None,
            "correction_attempted": False,
            "correction_successful": False,
            "retrieved_docs": [],
            "proof_trace": [],
            "error_category": "inference_error",
            "raw_output": None,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_depth": record["metadata"]["reasoning_depth"],
            "error_message": f"{type(exc).__name__}: {exc}",
        }

