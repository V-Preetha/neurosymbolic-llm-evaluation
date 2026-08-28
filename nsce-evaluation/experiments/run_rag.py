"""Run the Phase 6 SentenceTransformers + FAISS + Qwen evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.baselines.rag import evaluate_proofwriter_record
from src.config import ExperimentConfig, PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl
from src.models.qwen import QwenRunner
from src.retrieval.corpus import build_documents, build_queries
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.index import build_global_index, search_candidate_pool


PROMPT_PATH = PROJECT_ROOT / "prompts" / "proofwriter_rag.txt"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "raw" / "rag_phase6.jsonl")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "raw" / "rag_phase6_summary.json")
    args = parser.parse_args()

    import faiss

    config = ExperimentConfig.load()
    dev_path = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
    all_records = list(iter_jsonl(dev_path))
    selected = all_records[: args.limit]
    documents = build_documents(all_records)
    queries = build_queries(all_records, documents, config.seed, config.rag_distractors)
    query_by_id = {query["query_id"]: query for query in queries}
    document_by_id = {document["doc_id"]: document for document in documents}
    document_position = {document["doc_id"]: index for index, document in enumerate(documents)}
    write_jsonl(PROJECT_ROOT / "data" / "rag" / "corpus.jsonl", documents)
    write_jsonl(PROJECT_ROOT / "data" / "rag" / "queries.jsonl", queries)

    embedding_path = PROJECT_ROOT / ".model_cache" / "all-MiniLM-L6-v2"
    embedding_model = EmbeddingModel(embedding_path)
    encoding_started = time.perf_counter()
    document_embeddings = embedding_model.encode_documents([document["text"] for document in documents])
    corpus_encoding_ms = round((time.perf_counter() - encoding_started) * 1000, 3)
    global_index = build_global_index(document_embeddings)
    faiss.write_index(global_index, str(PROJECT_ROOT / "data" / "rag" / "corpus.faiss"))

    retrievals: dict[str, tuple[list[dict], float]] = {}
    for record in selected:
        query = query_by_id[record["sample_id"]]
        started = time.perf_counter()
        query_embedding = embedding_model.encode_queries([query["query"]])[0]
        candidates = [document_position[doc_id] for doc_id in query["candidate_doc_ids"]]
        hits = search_candidate_pool(
            document_embeddings, query_embedding, candidates, config.rag_top_k
        )
        retrieved = [
            {**document_by_id[documents[position]["doc_id"]], "score": round(score, 8)}
            for position, score in hits
        ]
        retrievals[record["sample_id"]] = (
            retrieved,
            round((time.perf_counter() - started) * 1000, 3),
        )

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)
    results: list[dict] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(selected, start=1):
            retrieved, retrieval_latency = retrievals[record["sample_id"]]
            result = evaluate_proofwriter_record(
                runner,
                record,
                retrieved,
                query_by_id[record["sample_id"]],
                prompt,
                retrieval_latency,
            )
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({
                "completed": index, "sample_id": result["sample_id"],
                "prediction": result["final_prediction"], "correct": result["correct_finally"],
                "recall_at_5": result["retrieval_recall_at_k"],
                "error": result["error_category"],
            }))

    proof_rows = [row for row in results if row["proof_recall_at_k"] is not None]
    summary = {
        "phase": 6, "method": "rag", "model_id": config.model_id,
        "embedding_model_id": config.embedding_model_id,
        "embedding_model_revision": config.embedding_model_revision,
        "settings": {
            "temperature": config.temperature, "max_new_tokens": config.max_new_tokens,
            "seed": config.seed, "quantization": config.quantization,
            "top_k": config.rag_top_k, "distractors": config.rag_distractors,
        },
        "model_runtime": runner.runtime_info,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "corpus_documents": len(documents), "queries_built": len(queries),
        "corpus_encoding_ms": corpus_encoding_ms,
        "samples": len(results), "correct": sum(row["correct_finally"] for row in results),
        "accuracy": sum(row["correct_finally"] for row in results) / len(results),
        "average_recall_at_5": statistics.fmean(row["retrieval_recall_at_k"] for row in results),
        "average_precision_at_5": statistics.fmean(row["retrieval_precision_at_k"] for row in results),
        "proof_evaluable_samples": len(proof_rows),
        "average_proof_recall_at_5": (
            statistics.fmean(row["proof_recall_at_k"] for row in proof_rows) if proof_rows else None
        ),
        "complete_proof_retrievals": sum(row["retrieval_success"] is True for row in results),
        "retrieval_failures": sum(row["error_category"] == "retrieval_failure" for row in results),
        "reasoning_failures_after_successful_retrieval": sum(
            row["error_category"] == "reasoning_failure_after_successful_retrieval" for row in results
        ),
        "unknown_incomplete_coverage_failures": sum(
            row["error_category"] == "unknown_query_incomplete_theory_coverage" for row in results
        ),
        "distractors_retrieved": sum(row["distractors_retrieved"] for row in results),
        "format_valid": sum(row["final_prediction"] is not None for row in results),
        "error_categories": dict(Counter(row["error_category"] or "none" for row in results)),
        "average_retrieval_latency_ms": statistics.fmean(row["retrieval_latency_ms"] for row in results),
        "average_generation_latency_ms": statistics.fmean(row["generation_latency_ms"] for row in results),
        "total_input_tokens": sum(row["input_tokens"] for row in results),
        "total_output_tokens": sum(row["output_tokens"] for row in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

