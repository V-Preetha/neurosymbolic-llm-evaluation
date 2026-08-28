"""Phase 7: normalize FELM and run evidence-grounded claim verification."""

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

from src.config import ExperimentConfig, PROJECT_ROOT, dataset_package_path
from src.datasets.felm import iter_felm, normalize_felm_record, write_felm_dataset
from src.models.qwen import QwenRunner
from src.nsce.factual_verifier import verify_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--selection", choices=("evidence", "all"), default="evidence")
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "felm_phase7.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "felm_phase7_summary.json")
    args = parser.parse_args()

    source = dataset_package_path() / "normalized" / "felm_test.jsonl"
    processed = PROJECT_ROOT / "data" / "processed" / "felm.jsonl"
    normalized_count = write_felm_dataset(source, processed)
    if args.normalize_only:
        print(json.dumps({"normalized": normalized_count, "output": str(processed)}))
        return

    records = [normalize_felm_record(row) for row in iter_felm(source)]
    if args.selection == "evidence":
        records = [row for row in records if row["evidence"]["documents"]]
    selected = records[: args.limit]
    if not selected:
        raise ValueError("No FELM records selected")

    config = ExperimentConfig.load()
    prompt = (PROJECT_ROOT / "prompts" / "felm_verify.txt").read_text(encoding="utf-8")
    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(selected, start=1):
            result = verify_record(runner, record, prompt)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({"completed": index, "sample_id": result["sample_id"],
                              "claims": len(result["claims"])}))

    claims = [claim for result in results for claim in result["claims"]]
    decided = [claim for claim in claims if claim["verification_status"] != "unknown"]
    predicted_errors = [claim for claim in claims if claim["verification_status"] == "contradicted"]
    true_errors = [claim for claim in claims if claim["ground_truth"] == "factual_error"]
    true_positive = sum(claim["ground_truth"] == "factual_error" for claim in predicted_errors)
    unique_evidence_ids = {
        evidence.get("doc_id", evidence.get("evidence_id"))
        for claim in claims for evidence in claim["evidence"]
    }
    summary = {
        "phase": 7, "method": "evidence_claim_verification", "model_id": config.model_id,
        "selection": args.selection, "selection_note": (
            "Smoke subset intentionally starts with evidence-bearing records; not a representative score."
            if args.selection == "evidence" else "First records in source order."
        ),
        "normalized_records": normalized_count, "samples": len(results), "claims": len(claims),
        "unique_evidence_items": len(unique_evidence_ids),
        "claim_evidence_attachments": sum(len(claim["evidence"]) for claim in claims),
        "ground_truth_labels": dict(Counter(claim["ground_truth"] for claim in claims)),
        "statuses": dict(Counter(claim["verification_status"] for claim in claims)),
        "methods": dict(Counter(claim["verification_method"] for claim in claims)),
        "coverage": len(decided) / len(claims),
        "exact_claim_accuracy": sum(claim["correct"] for claim in claims) / len(claims),
        "selective_accuracy": (
            sum(claim["correct"] for claim in decided) / len(decided) if decided else None
        ),
        "factual_error_precision": true_positive / len(predicted_errors) if predicted_errors else None,
        "factual_error_recall": true_positive / len(true_errors) if true_errors else None,
        "unsupported_claim_rate": sum(
            claim["verification_status"] != "supported" for claim in claims
        ) / len(claims),
        "unknown_rate": sum(claim["verification_status"] == "unknown" for claim in claims) / len(claims),
        "average_verifier_latency_ms": statistics.fmean(claim["latency_ms"] for claim in claims),
        "total_input_tokens": sum(claim["input_tokens"] for claim in claims),
        "total_output_tokens": sum(claim["output_tokens"] for claim in claims),
        "model_runtime": runner.runtime_info,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "integrity": {
            "ground_truth_used_only_after_verification": True,
            "atomic_fact_support_status_excluded_from_verifier": True,
            "missing_evidence_maps_to_unknown": True,
            "binary_factual_error_scored_as_contradicted": True,
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
