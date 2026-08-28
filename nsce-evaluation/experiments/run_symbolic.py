"""Phase 4: independently validate the deterministic ProofWriter solver."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl
from src.nsce.parser import ParserError, parse_query, parse_theory
from src.nsce.symbolic_reasoner import classify


def evaluate(record: dict) -> dict:
    started = time.perf_counter()
    try:
        theory = parse_theory(record["input"]["facts"] + record["input"]["rules"])
        query = parse_query(record["input"]["question"])
        result = classify(query, theory.facts, theory.rules)
        correct = result.prediction == record["ground_truth"]["label"]
        return {
            "sample_id": record["sample_id"], "dataset": "proofwriter",
            "method": "deterministic_symbolic", "ground_truth": record["ground_truth"]["label"],
            "symbolic_status": result.status, "symbolic_prediction": result.prediction,
            "correct": correct, "parser_status": "supported", "parser_errors": [],
            "proof_trace": list(result.proof_trace), "inconsistent": result.inconsistent,
            "parsed_facts": len(theory.facts), "parsed_rules": len(theory.rules),
            "derived_atoms": len(result.derivations),
            "reasoning_depth": record["metadata"]["reasoning_depth"],
            "source_split": record["metadata"]["source_split"],
            "error_category": None if correct else "symbolic_reasoning_failure",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except ParserError as exc:
        return {
            "sample_id": record["sample_id"], "dataset": "proofwriter",
            "method": "deterministic_symbolic", "ground_truth": record["ground_truth"]["label"],
            "symbolic_status": None, "symbolic_prediction": None, "correct": False,
            "parser_status": "failed", "parser_errors": [str(exc)], "proof_trace": [],
            "inconsistent": False, "parsed_facts": None, "parsed_rules": None,
            "derived_atoms": None, "reasoning_depth": record["metadata"]["reasoning_depth"],
            "source_split": record["metadata"]["source_split"],
            "error_category": "symbolic_parser_failure",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def grouped_metrics(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        str(group): {
            "samples": len(items),
            "parser_supported": sum(item["parser_status"] == "supported" for item in items),
            "correct": sum(item["correct"] for item in items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
        }
        for group, items in sorted(groups.items(), key=lambda pair: str(pair[0]))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "symbolic_phase4.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "symbolic_phase4_summary.json")
    args = parser.parse_args()

    source_root = dataset_package_path() / "normalized"
    files = {
        "dev": [source_root / "proofwriter_dev.jsonl"],
        "test": [source_root / "proofwriter_test.jsonl"],
        "all": [source_root / "proofwriter_dev.jsonl", source_root / "proofwriter_test.jsonl"],
    }[args.split]
    records = [record for path in files for record in iter_jsonl(path)]
    results = [evaluate(record) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")

    supported = [row for row in results if row["parser_status"] == "supported"]
    correct = sum(row["correct"] for row in results)
    summary = {
        "phase": 4, "method": "deterministic_symbolic", "samples": len(results),
        "parser_supported": len(supported),
        "parser_coverage": len(supported) / len(results) if results else None,
        "correct": correct, "overall_accuracy": correct / len(results) if results else None,
        "supported_syntax_accuracy": (
            sum(row["correct"] for row in supported) / len(supported) if supported else None
        ),
        "parser_failures": len(results) - len(supported),
        "reasoning_failures": sum(row["error_category"] == "symbolic_reasoning_failure" for row in results),
        "inconsistent_theories": sum(row["inconsistent"] for row in results),
        "predictions": dict(Counter(row["symbolic_prediction"] for row in results)),
        "average_latency_ms": statistics.fmean(row["latency_ms"] for row in results),
        "by_split": grouped_metrics(results, "source_split"),
        "by_reasoning_depth": grouped_metrics(results, "reasoning_depth"),
        "error_categories": dict(Counter(row["error_category"] or "none" for row in results)),
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

