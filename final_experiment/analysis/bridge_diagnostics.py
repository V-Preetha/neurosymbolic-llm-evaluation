"""Trace-supported diagnostics for the descriptive NL2Logic-to-Z3 bridge."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def bridge_diagnostics(predictions: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_ids = {row["example_id"] for row in predictions}
    unknown_trace_ids = {row.get("example_id") for row in traces} - prediction_ids
    if unknown_trace_ids:
        raise ValueError(f"parser traces contain unknown example IDs: {sorted(unknown_trace_ids)}")

    def translation_count(field: str) -> int:
        return sum(bool(row.get("translation", {}).get(field)) for row in predictions)

    error_categories = Counter(row.get("error_category") or "correct" for row in predictions)
    semantic = Counter(row.get("translation", {}).get("semantic_equivalence") or "not_available" for row in predictions)
    fallback_rows = [
        row
        for row in traces
        if row.get("fallback_from")
        or row.get("fallback_reason")
        or row.get("relation_fallback_from")
        or row.get("relation_fallback_reason")
    ]
    totals = [float(row.get("timing", {}).get("total_ms", row.get("latency_ms", 0.0))) for row in predictions]
    recursive_calls = [int(row.get("translation", {}).get("recursive_llm_call_count", 0)) for row in predictions]
    model_calls = [int(row.get("translation", {}).get("model_invocation_count", 0)) for row in predictions]

    return {
        "total": len(predictions),
        "schema_valid": translation_count("schema_valid"),
        "parser_valid": translation_count("parser_valid"),
        "ast_valid": translation_count("ast_valid"),
        "compile_success": translation_count("compile_success"),
        "executable": translation_count("executable"),
        "solver_success": sum(row.get("solver_status") == "success" for row in predictions),
        "parser_status": dict(sorted(Counter(str(row.get("parser_status")) for row in predictions).items())),
        "compiler_status": dict(sorted(Counter(str(row.get("compiler_status")) for row in predictions).items())),
        "solver_status": dict(sorted(Counter(str(row.get("solver_status")) for row in predictions).items())),
        "error_categories": dict(error_categories),
        "semantic_equivalence": dict(semantic),
        "exact_match": sum(bool(row.get("translation", {}).get("exact_match")) for row in predictions),
        "normalized_logic_match": sum(
            bool(row.get("translation", {}).get("normalized_logic_match")) for row in predictions
        ),
        "fallback": {
            "available": any(
                field in row
                for row in traces
                for field in ("fallback_from", "fallback_reason", "relation_fallback_from", "relation_fallback_reason")
            ),
            "trace_step_count": len(fallback_rows),
            "example_count": len({row.get("example_id") for row in fallback_rows}),
        },
        "trace_steps": len(traces),
        "latency_ms": {
            "mean": float(np.mean(totals)),
            "median": float(np.median(totals)),
            "p95": float(np.percentile(totals, 95)),
            "sum": float(np.sum(totals)),
        },
        "recursive_llm_calls": {"total": sum(recursive_calls), "mean": float(np.mean(recursive_calls))},
        "model_invocations": {"total": sum(model_calls), "mean": float(np.mean(model_calls))},
    }
