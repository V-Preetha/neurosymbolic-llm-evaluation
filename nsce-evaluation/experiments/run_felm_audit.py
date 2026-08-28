"""Phase 8: audit bundled FELM evidence, then evaluate only sufficient-evidence claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import ExperimentConfig, PROJECT_ROOT, dataset_package_path
from src.datasets.felm import iter_felm, normalize_felm_record
from src.evaluation.felm_audit import (
    audit_claim_extraction, audit_evidence, audit_normalization, reconcile_evidence_ids,
)
from src.evaluation.felm_metrics import LABEL_MAP, verification_metrics
from src.models.qwen import QwenRunner
from src.nsce.factual_verifier import verify_claim
from src.retrieval.embeddings import EmbeddingModel


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_audits(source_rows: list[dict], records: list[dict]) -> list[dict]:
    embedding_model = EmbeddingModel(PROJECT_ROOT / ".model_cache" / "all-MiniLM-L6-v2")
    unique_documents: list[str] = []
    document_position: dict[str, int] = {}
    claims_with_evidence: list[str] = []
    claim_positions: dict[str, int] = {}
    for record in records:
        usable = [
            document["text"] for document in record["evidence"]["documents"]
            if isinstance(document, dict) and isinstance(document.get("text"), str)
            and document["text"].strip()
        ]
        for text in usable:
            if text not in document_position:
                document_position[text] = len(unique_documents)
                unique_documents.append(text)
        if usable:
            for claim in record["candidate_response"]["claims"]:
                claim_positions[claim["claim_id"]] = len(claims_with_evidence)
                claims_with_evidence.append(claim["text"])

    document_vectors = embedding_model.encode_documents(unique_documents)
    claim_vectors = embedding_model.encode_queries(claims_with_evidence)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    audits: list[dict] = []
    for record in records:
        source = source_by_id[record["sample_id"]]
        extraction = audit_claim_extraction(source, record)
        normalization = audit_normalization(source, record)
        documents = [
            document for document in record["evidence"]["documents"]
            if isinstance(document, dict) and isinstance(document.get("text"), str)
            and document["text"].strip()
        ]
        gold_by_id = {
            item["claim_id"]: LABEL_MAP[item["label"]]
            for item in record["ground_truth"]["claim_labels"]
        }
        for claim in record["candidate_response"]["claims"]:
            similarity = None
            if documents:
                vector = claim_vectors[claim_positions[claim["claim_id"]]]
                similarity = max(
                    float(vector @ document_vectors[document_position[document["text"]]])
                    for document in documents
                )
            evidence_audit = audit_evidence(claim["text"], documents, similarity)
            audits.append({
                "felm_full_id": claim["claim_id"], "sample_id": record["sample_id"],
                "claim_id": claim["claim_id"], "claim_text": claim["text"],
                "domain": record["metadata"]["domain"], "gold_label": gold_by_id[claim["claim_id"]],
                "evidence": documents, **evidence_audit, **extraction, **normalization,
            })
    return audits


def evaluate_evidence_claims(audits: list[dict], records: list[dict], prompt: str, runner: QwenRunner) -> list[dict]:
    records_by_id = {record["sample_id"]: record for record in records}
    rows: list[dict] = []
    for index, audit in enumerate(audits, start=1):
        record = records_by_id[audit["sample_id"]]
        claim = next(
            item for item in record["candidate_response"]["claims"]
            if item["claim_id"] == audit["claim_id"]
        )
        verification = verify_claim(runner, record, claim, prompt)
        prediction = verification["verification_status"].upper()
        correct = prediction == audit["gold_label"]
        if prediction == "UNKNOWN":
            failure = "VERIFIER_UNNECESSARY_UNKNOWN"
        elif prediction == "SUPPORTED" and audit["gold_label"] == "CONTRADICTED":
            failure = "VERIFIER_FALSE_SUPPORT"
        elif prediction == "CONTRADICTED" and audit["gold_label"] == "SUPPORTED":
            failure = "VERIFIER_FALSE_CONTRADICTION"
        else:
            failure = None
        rows.append({
            "sample_id": audit["sample_id"], "claim_id": audit["claim_id"],
            "claim": audit["claim_text"], "gold_label": audit["gold_label"],
            "prediction": prediction, "evidence_status": "SUFFICIENT_EVIDENCE",
            "correct": correct, "verification_status": verification["verification_method"],
            "abstained": prediction == "UNKNOWN", "failure_category": failure,
            "latency_ms": verification["latency_ms"],
            "input_tokens": verification["input_tokens"],
            "output_tokens": verification["output_tokens"],
        })
        if index % 25 == 0 or index == len(audits):
            print(json.dumps({"verified": index, "total": len(audits)}), flush=True)
    return rows


def deterministic_audit_sample(full_rows: list[dict], result_by_id: dict[str, dict], seed: int) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in full_rows:
        result = result_by_id.get(row["claim_id"])
        prediction = result["prediction"] if result else "UNKNOWN"
        groups[(prediction, row["domain"], row["audit_status"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda row: row["claim_id"])
        rng.shuffle(values)
    selected: list[dict] = []
    keys = sorted(groups)
    while len(selected) < 50 and keys:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < 50:
                row = groups[key].pop()
                result = result_by_id.get(row["claim_id"])
                selected.append({
                    "sample_id": row["sample_id"], "claim_id": row["claim_id"],
                    "domain": row["domain"], "claim_text": row["claim_text"],
                    "gold_label": row["gold_label"],
                    "prediction": result["prediction"] if result else "UNKNOWN",
                    "evidence_status": row["audit_status"],
                    "evidence_present": row["evidence_present"],
                    "evidence_relevant": row["evidence_relevant"],
                    "evidence_sufficient": row["evidence_sufficient"],
                    "semantic_similarity": row["semantic_similarity"],
                    "lexical_coverage": row["lexical_coverage"],
                    "numeric_coverage": row["numeric_coverage"],
                    "claim_extraction_valid": row["claim_extraction_valid"],
                    "claim_normalization_valid": row["claim_normalization_valid"],
                    "evidence_excerpt": " | ".join(
                        document["text"][:500].replace("\n", " ") for document in row["evidence"]
                    ),
                    "manual_evidence_status": "", "manual_prediction": "",
                    "manual_extraction_valid": "", "reviewer_notes": "",
                })
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def write_manual_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig.load()
    source_path = dataset_package_path() / "normalized" / "felm_test.jsonl"
    source_rows = list(iter_felm(source_path))
    records = [normalize_felm_record(row) for row in source_rows]
    audits = build_audits(source_rows, records)
    full_path = PROJECT_ROOT / "data" / "processed" / "felm_full_claims.jsonl"
    evidence_path = PROJECT_ROOT / "data" / "processed" / "felm_evidence_claims.jsonl"
    write_jsonl(full_path, audits)
    evidence_rows = [
        row for row in audits
        if row["evidence_sufficient"] and row["claim_extraction_valid"]
        and row["claim_normalization_valid"] and row["claim_text"].strip()
    ]
    write_jsonl(evidence_path, evidence_rows)
    if args.audit_only:
        print(json.dumps({
            "claims": len(audits), "statuses": dict(Counter(row["audit_status"] for row in audits)),
            "felm_evidence_claims": len(evidence_rows),
        }, indent=2))
        return

    prompt = (PROJECT_ROOT / "prompts" / "felm_verify.txt").read_text(encoding="utf-8")
    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
    results = evaluate_evidence_claims(evidence_rows, records, prompt, runner)
    result_path = PROJECT_ROOT / "results" / "raw" / "felm_evidence_phase8.jsonl"
    write_jsonl(result_path, results)
    result_by_id = {row["claim_id"]: row for row in results}
    manual_rows = deterministic_audit_sample(audits, result_by_id, config.seed)
    audit_jsonl = PROJECT_ROOT / "reports" / "felm_phase8_manual_audit_50.jsonl"
    audit_csv = PROJECT_ROOT / "reports" / "felm_phase8_manual_audit_50.csv"
    write_jsonl(audit_jsonl, manual_rows)
    write_manual_csv(audit_csv, manual_rows)

    metrics = verification_metrics(results)
    status_counts = Counter(row["audit_status"] for row in audits)
    failure_counts = {
        "NO_EVIDENCE": status_counts["NO_EVIDENCE"],
        "IRRELEVANT_EVIDENCE": status_counts["IRRELEVANT_EVIDENCE"],
        "PARTIAL_EVIDENCE": status_counts["PARTIAL_EVIDENCE"],
        "CLAIM_EXTRACTION_FAILURE": sum(not row["claim_extraction_valid"] for row in audits),
        "CLAIM_NORMALIZATION_FAILURE": sum(not row["claim_normalization_valid"] for row in audits),
        "VERIFIER_FALSE_SUPPORT": sum(row["failure_category"] == "VERIFIER_FALSE_SUPPORT" for row in results),
        "VERIFIER_FALSE_CONTRADICTION": sum(row["failure_category"] == "VERIFIER_FALSE_CONTRADICTION" for row in results),
        "VERIFIER_UNNECESSARY_UNKNOWN": sum(row["failure_category"] == "VERIFIER_UNNECESSARY_UNKNOWN" for row in results),
        "CORRECT_ABSTENTION": sum(row["audit_status"] != "SUFFICIENT_EVIDENCE" for row in audits),
        "CORRECT_SUPPORTED": sum(row["correct"] and row["prediction"] == "SUPPORTED" for row in results),
        "CORRECT_CONTRADICTED": sum(row["correct"] and row["prediction"] == "CONTRADICTED" for row in results),
    }
    pipeline_unknown = len(audits) - len(evidence_rows) + sum(row["abstained"] for row in results)
    summary = {
        "phase": 8, "scope": "FELM evidence audit and conditional verifier evaluation",
        "warning": "Phase 7 smoke and Phase 8 conditional results are not full FELM benchmark results.",
        "model_id": config.model_id, "model_runtime": runner.runtime_info,
        "audit_method": {
            "label_blind": True, "verifier_predictions_excluded": True, "web_retrieval": False,
            "semantic_relevance_threshold": 0.25, "lexical_relevance_threshold": 0.15,
            "semantic_sufficiency_threshold": 0.50, "lexical_sufficiency_threshold": 0.50,
            "all_claim_numbers_must_be_covered": True,
        },
        "felm_full": {
            "claims": len(audits), "evidence_statuses": dict(status_counts),
            "total_evidence_coverage": len(evidence_rows) / len(audits),
            "pipeline_unknown_rate": pipeline_unknown / len(audits),
            "missing_evidence_rate": status_counts["NO_EVIDENCE"] / len(audits),
            "irrelevant_evidence_rate": status_counts["IRRELEVANT_EVIDENCE"] / len(audits),
        },
        "felm_evidence": metrics,
        "failure_categories": failure_counts,
        "failure_decomposition_note": (
            "Evidence, extraction, and normalization counts are overlapping diagnostics. "
            "CORRECT_ABSTENTION is the controlled pipeline disposition for non-sufficient "
            "evidence and remains subject to the exported manual audit."
        ),
        "latency": {
            "average_ms": statistics.fmean(row["latency_ms"] for row in results) if results else None,
            "total_input_tokens": sum(row["input_tokens"] for row in results),
            "total_output_tokens": sum(row["output_tokens"] for row in results),
        },
        "integrity": {
            "full_and_evidence_ids_reconciled": reconcile_evidence_ids(audits, evidence_rows),
            "gold_excluded_from_audit_decisions_and_verifier_prompts": True,
            "atomic_fact_support_status_excluded": True,
        },
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "artifacts": {
            "felm_full": str(full_path), "felm_evidence": str(evidence_path),
            "verifier_results": str(result_path), "manual_audit_jsonl": str(audit_jsonl),
            "manual_audit_csv": str(audit_csv),
        },
        "blockers": [
            "Bundled evidence is absent for most claims.",
            "Automated evidence sufficiency and extraction checks require validation with the exported 50-claim manual audit.",
            "FELM factual_error is binary and does not distinguish explicit contradiction from unsupported content.",
        ],
    }
    summary_path = PROJECT_ROOT / "results" / "raw" / "felm_phase8_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
