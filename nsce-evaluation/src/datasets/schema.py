"""Canonical dataset schema shared by all experiment branches."""

from __future__ import annotations

from typing import Any


TOP_LEVEL_KEYS = {
    "sample_id", "dataset", "task_type", "input", "candidate_response",
    "ground_truth", "evidence", "symbolic", "metadata",
}


def canonical_record() -> dict[str, Any]:
    return {
        "sample_id": "",
        "dataset": "",
        "task_type": "",
        "input": {"question": "", "context": None, "facts": [], "rules": []},
        "candidate_response": {"text": None, "claims": []},
        "ground_truth": {
            "label": None, "reference_answer": None, "acceptable_answers": [],
            "incorrect_answers": [], "claim_labels": [],
        },
        "evidence": {"documents": [], "atomic_facts": [], "source_available": False},
        "symbolic": {
            "entities": [], "predicates": [], "rules": [], "query": None,
            "expected_status": None,
        },
        "metadata": {
            "domain": None, "reasoning_depth": None, "hallucination_type": None,
            "source_split": None,
        },
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the record is Phase-1 valid."""
    errors: list[str] = []
    if set(record) != TOP_LEVEL_KEYS:
        errors.append(f"top-level keys differ: {sorted(set(record) ^ TOP_LEVEL_KEYS)}")
    if not isinstance(record.get("sample_id"), str) or not record.get("sample_id"):
        errors.append("sample_id must be a non-empty string")
    if record.get("dataset") not in {"proofwriter", "felm"}:
        errors.append("dataset must be proofwriter or felm")
    if record.get("task_type") not in {"logical_reasoning", "claim_factuality"}:
        errors.append("invalid task_type")
    input_value = record.get("input")
    if not isinstance(input_value, dict):
        errors.append("input must be an object")
    else:
        if not isinstance(input_value.get("question"), str) or not input_value["question"].strip():
            errors.append("input.question must be a non-empty string")
        for name in ("facts", "rules"):
            values = input_value.get(name)
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                errors.append(f"input.{name} must be a string array")
    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, dict):
        errors.append("ground_truth must be an object")
    elif record.get("dataset") == "proofwriter" and ground_truth.get("label") not in {
        "true", "false", "unknown",
    }:
        errors.append("ProofWriter ground_truth.label must be true, false, or unknown")
    return errors
