"""Architecture 3: retrieved evidence followed by the shared Qwen model."""

from __future__ import annotations

from src.models.qwen import QwenRunner


def render_rag_prompt(record: dict, retrieved: list[dict], template: str) -> str:
    evidence = "\n".join(
        f"- [{item['type']}] {item['text']}" for item in retrieved
    )
    return template.format(evidence=evidence, question=record["input"]["question"])


def evaluate_proofwriter_record(
    runner: QwenRunner,
    record: dict,
    retrieved: list[dict],
    query_metadata: dict,
    template: str,
    retrieval_latency_ms: float,
) -> dict:
    generation = runner.generate_label(render_rag_prompt(record, retrieved, template))
    retrieved_ids = [item["doc_id"] for item in retrieved]
    relevant = set(query_metadata["relevant_doc_ids"])
    required = set(query_metadata["required_doc_ids"])
    relevant_hits = len(set(retrieved_ids) & relevant)
    required_hits = len(set(retrieved_ids) & required)
    instance_recall = relevant_hits / len(relevant) if relevant else None
    instance_precision = relevant_hits / len(retrieved_ids) if retrieved_ids else None
    if query_metadata["required_scope"] == "minimal_symbolic_proof":
        proof_recall = required_hits / len(required) if required else 1.0
        retrieval_success = required.issubset(retrieved_ids)
    else:
        proof_recall = None
        retrieval_success = None

    prediction = generation.prediction
    correct = prediction == record["ground_truth"]["label"]
    if correct:
        error_category = None
    elif prediction is None:
        error_category = "output_format_error"
    elif retrieval_success is False:
        error_category = "retrieval_failure"
    elif retrieval_success is True:
        error_category = "reasoning_failure_after_successful_retrieval"
    else:
        error_category = "unknown_query_incomplete_theory_coverage"

    return {
        "sample_id": record["sample_id"], "dataset": "proofwriter", "method": "rag",
        "initial_prediction": None, "final_prediction": prediction,
        "ground_truth": record["ground_truth"]["label"],
        "correct_initially": None, "correct_finally": correct,
        "retrieved_docs": retrieved, "retrieved_doc_ids": retrieved_ids,
        "relevant_doc_ids": query_metadata["relevant_doc_ids"],
        "required_doc_ids": query_metadata["required_doc_ids"],
        "required_scope": query_metadata["required_scope"],
        "retrieval_recall_at_k": instance_recall,
        "retrieval_precision_at_k": instance_precision,
        "proof_recall_at_k": proof_recall,
        "retrieval_success": retrieval_success,
        "distractors_retrieved": len(retrieved_ids) - relevant_hits,
        "error_category": error_category,
        "raw_output": generation.raw_text,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation.latency_ms,
        "latency_ms": round(retrieval_latency_ms + generation.latency_ms, 3),
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "reasoning_depth": record["metadata"]["reasoning_depth"],
    }

