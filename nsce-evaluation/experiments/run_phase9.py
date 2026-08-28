"""Run the frozen, controlled 50-claim FELM Phase 9 comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import ExperimentConfig
from src.evaluation.felm_audit import audit_evidence
from src.evaluation.phase9_metrics import failure_summary, score
from src.nsce.factual_structured import check as structured_check
from src.retrieval.factual_corpus import file_hash

EXPECTED_HASH = "5b4a1ffb1b011dfbe7ee9459ac83eec4b7b20a62027999e5e0844a62dfb887c7"
EXPECTED_COUNT = 162696
LABEL_CHOICES = ("Supported", "Contradicted", "Unknown")
OUT = PROJECT / "results" / "phase9"
DATA = PROJECT / "data" / "processed"
CORPUS_DIR = PROJECT / "data" / "factual_corpus"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def retrieval_query(question: str, claim: str) -> str:
    return f"{question.strip()}\nClaim: {claim.strip()}"


def evidence_text(documents: list[dict]) -> str:
    return "\n\n".join(f"[{d['rank']}] {d['title']} ({d['doc_id']})\n{d['text']}" for d in documents)


def llm_prompt(question: str, claim: str) -> str:
    return f"""Classify one factual claim using your parametric knowledge only.
Question: {question}
Claim: {claim}
Return exactly one label: Supported, Contradicted, or Unknown.
Supported means the claim is true; Contradicted means an incompatible fact is established; Unknown means you cannot reliably establish either. Do not guess."""


def self_prompt(question: str, claim: str, initial: str) -> str:
    return f"""Recheck your factual classification without external evidence.
Question: {question}
Claim: {claim}
Initial prediction: {initial}
Return exactly one label: Supported, Contradicted, or Unknown. Revise only when your own factual knowledge reliably warrants it; otherwise use Unknown."""


def rag_prompt(question: str, claim: str, documents: list[dict]) -> str:
    return f"""Verify one factual claim using only the retrieved evidence.
Question: {question}
Claim: {claim}
Retrieved evidence:
{evidence_text(documents)}
Return exactly one label: Supported, Contradicted, or Unknown.
Supported requires clear support. Contradicted requires an explicit incompatible fact. Insufficient, irrelevant, or ambiguous evidence is Unknown. Lack of support is not contradiction."""


def nsce_prompt(question: str, claim: str, documents: list[dict]) -> str:
    return f"""Act as an evidence-constrained factual verifier.
Question: {question}
Claim: {claim}
Evidence:
{evidence_text(documents)}
Return exactly one label: Supported, Contradicted, or Unknown. Use only this evidence. An explicit incompatible fact is required for Contradicted; missing or insufficient evidence is Unknown."""


def feedback_prompt(question: str, claim: str, initial: str, verifier: str,
                    documents: list[dict], finding: str) -> str:
    return f"""Reconsider a factual prediction using only the supplied evidence and verifier finding.
Question: {question}
Claim: {claim}
Initial prediction: {initial}
Evidence:
{evidence_text(documents)}
Deterministic finding: {finding or 'none'}
Evidence verifier conclusion: {verifier}
Return exactly one final label: Supported, Contradicted, or Unknown. Missing evidence must be Unknown, never Contradicted."""


def validate_frozen() -> dict:
    import faiss
    corpus = CORPUS_DIR / "corpus.jsonl"
    digest = file_hash(corpus)
    index = faiss.read_index(str(CORPUS_DIR / "corpus.faiss"))
    if digest != EXPECTED_HASH:
        raise RuntimeError(f"STOP: corpus hash changed: {digest}")
    if index.ntotal != EXPECTED_COUNT or index.d != 384:
        raise RuntimeError(f"STOP: index mismatch ntotal={index.ntotal}, d={index.d}")
    return {"corpus_sha256": digest, "index_ntotal": index.ntotal, "index_dimension": index.d}


def retrieve() -> list[dict]:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    frozen = validate_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA unavailable for frozen embedding model")
    config = ExperimentConfig.load()
    pilot = read_jsonl(DATA / "felm_phase9_pilot.jsonl")
    ids = json.loads((DATA / "felm_phase9_pilot_ids.json").read_text(encoding="utf-8"))["claim_ids"]
    if [r["claim_id"] for r in pilot] != ids:
        raise RuntimeError("STOP: pilot IDs differ from frozen pre-inference IDs")
    model = SentenceTransformer(str(PROJECT / ".model_cache" / "all-MiniLM-L6-v2"),
                                device="cuda", local_files_only=True)
    queries = [retrieval_query(r["question"], r["claim_text"]) for r in pilot]
    if hasattr(model, "encode_query"):
        vectors = model.encode_query(queries, batch_size=50, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=True)
    else:
        vectors = model.encode(queries, batch_size=50, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
    index = faiss.read_index(str(CORPUS_DIR / "corpus.faiss"))
    scores, positions = index.search(vectors.astype("float32"), config.rag_top_k)
    corpus = read_jsonl(CORPUS_DIR / "corpus.jsonl")
    rows = []
    for item, query, item_scores, item_positions in zip(pilot, queries, scores, positions):
        docs = []
        for rank, (position, similarity) in enumerate(zip(item_positions, item_scores), 1):
            source = corpus[int(position)]
            docs.append({"doc_id": source["doc_id"], "rank": rank, "score": float(similarity),
                         "title": source["title"], "section": source["section"], "text": source["text"]})
        audit = audit_evidence(item["claim_text"], docs, float(item_scores[0]))
        rows.append({"claim_id": item["claim_id"], "query": query,
                     "retrieved_documents": docs, "evidence_audit": audit})
    write_jsonl(OUT / "shared_retrieval.jsonl", rows)
    write_json(OUT / "retrieval_runtime.json", {**frozen, "device": "cuda",
               "gpu_name": torch.cuda.get_device_name(0), "top_k": config.rag_top_k,
               "embedding_model": config.embedding_model_id,
               "embedding_revision": config.embedding_model_revision})
    return rows


def _generation_fields(result) -> dict:
    return {"prediction": (result.prediction or "unknown").upper(), "raw_output": result.raw_text,
            "latency_ms": result.latency_ms, "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens}


def _base(method: str, item: dict, generated: dict) -> dict:
    row = {"claim_id": item["claim_id"], "method": method, "gold_label": item["gold_label"], **generated}
    row["correct"] = row["prediction"] == row["gold_label"]
    return row


def infer() -> None:
    import gc
    import os
    import psutil
    import torch
    from src.models.qwen import QwenRunner

    validate_frozen()
    config = ExperimentConfig.load()
    pilot = read_jsonl(DATA / "felm_phase9_pilot.jsonl")
    retrieval = read_jsonl(OUT / "shared_retrieval.jsonl")
    if [r["claim_id"] for r in pilot] != [r["claim_id"] for r in retrieval]:
        raise RuntimeError("STOP: shared retrieval does not match pilot IDs")
    names = ("llm_only", "self_check", "rag", "nsce")
    existing = {
        name: read_jsonl(OUT / f"{name}.jsonl") if (OUT / f"{name}.jsonl").exists() else []
        for name in names
    }
    pilot_ids = [row["claim_id"] for row in pilot]
    for name, rows in existing.items():
        row_ids = [row.get("claim_id") for row in rows]
        if len(row_ids) != len(set(row_ids)) or not set(row_ids).issubset(pilot_ids):
            raise RuntimeError(f"STOP: invalid resumable {name} records")
        if any(row.get("prediction") not in {"SUPPORTED", "CONTRADICTED", "UNKNOWN"} for row in rows):
            raise RuntimeError(f"STOP: invalid prediction in resumable {name} records")
    if all(len(existing[name]) == len(pilot) for name in names):
        print(json.dumps({"status": "already_complete", "claims": len(pilot)}), flush=True)
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    runner = QwenRunner(config.model_id, config.seed, 8, config.quantization)
    if not torch.cuda.is_available() or "RTX 5050" not in torch.cuda.get_device_name(0):
        raise RuntimeError("STOP: expected RTX 5050 CUDA device is unavailable")
    if runner.runtime_info.get("quantization") != "nf4":
        raise RuntimeError("STOP: NF4 quantization is not active")
    placements = set(runner.runtime_info.get("device_map_summary", {}))
    if not placements or placements - {"0", "cuda:0"}:
        raise RuntimeError(f"STOP: CPU fallback detected: {sorted(placements)}")
    gate = runner.generate_choice("Return exactly Supported.", LABEL_CHOICES)
    if gate.prediction != "supported":
        raise RuntimeError(f"STOP: trivial GPU inference gate failed: {gate.raw_text!r}")
    peak_process_rss = psutil.Process(os.getpid()).memory_info().rss
    write_json(OUT / "model_runtime.json", {**runner.runtime_info,
               "model_loaded": True, "trivial_inference_passed": True,
               "process_rss_after_load_bytes": peak_process_rss})
    llm_by_id = {row["claim_id"]: row for row in existing["llm_only"]}
    self_by_id = {row["claim_id"]: row for row in existing["self_check"]}
    rag_by_id = {row["claim_id"]: row for row in existing["rag"]}
    nsce_by_id = {row["claim_id"]: row for row in existing["nsce"]}
    started = time.perf_counter()
    for number, (item, shared) in enumerate(zip(pilot, retrieval), 1):
        q, claim, docs = item["question"], item["claim_text"], shared["retrieved_documents"]
        llm = llm_by_id.get(item["claim_id"])
        if llm is None:
            first = runner.generate_choice(llm_prompt(q, claim), LABEL_CHOICES)
            llm = _base("llm_only", item, _generation_fields(first)); llm_by_id[item["claim_id"]] = llm
        if item["claim_id"] not in self_by_id:
            second = runner.generate_choice(self_prompt(q, claim, llm["prediction"]), LABEL_CHOICES)
            self_row = _base("self_check", item, _generation_fields(second))
            self_row.update({"initial_prediction": llm["prediction"], "revised": self_row["prediction"] != llm["prediction"],
                             "correction_success": llm["prediction"] != item["gold_label"] and self_row["prediction"] == item["gold_label"],
                             "harmful_revision": llm["prediction"] == item["gold_label"] and self_row["prediction"] != item["gold_label"]})
            self_by_id[item["claim_id"]] = self_row
        if item["claim_id"] not in rag_by_id:
            rag_result = runner.generate_choice(rag_prompt(q, claim, docs), LABEL_CHOICES)
            rag = _base("rag", item, _generation_fields(rag_result))
            rag.update({"retrieved_doc_ids": [d["doc_id"] for d in docs], "evidence_audit": shared["evidence_audit"]})
            rag_by_id[item["claim_id"]] = rag
        if item["claim_id"] not in nsce_by_id:
            finding = structured_check(claim, docs)
            if finding.status:
                verifier, verifier_raw, verify_latency, verify_in, verify_out = finding.status, None, 0.0, 0, 0
            else:
                verified = runner.generate_choice(nsce_prompt(q, claim, docs), LABEL_CHOICES)
                verifier, verifier_raw = (verified.prediction or "unknown").upper(), verified.raw_text
                verify_latency, verify_in, verify_out = verified.latency_ms, verified.input_tokens, verified.output_tokens
            apply_feedback = verifier != llm["prediction"]
            if apply_feedback:
                reconsidered = runner.generate_choice(feedback_prompt(q, claim, llm["prediction"], verifier, docs, finding.justification or ""), LABEL_CHOICES)
                final_fields = _generation_fields(reconsidered)
                final_fields["latency_ms"] += verify_latency; final_fields["input_tokens"] += verify_in; final_fields["output_tokens"] += verify_out
            else:
                final_fields = {"prediction": verifier, "raw_output": verifier_raw,
                                "latency_ms": verify_latency, "input_tokens": verify_in, "output_tokens": verify_out}
            nsce = _base("nsce", item, final_fields)
            nsce.update({"initial_prediction": llm["prediction"], "verifier_prediction": verifier,
                         "verifier_method": finding.kind or "qwen_evidence_verifier", "structured_finding": finding.representation,
                         "justification": finding.justification or f"Evidence-constrained verifier returned {verifier}.",
                         "feedback_applied": apply_feedback,
                         "correction_success": llm["prediction"] != item["gold_label"] and nsce["prediction"] == item["gold_label"],
                         "feedback_ignored": apply_feedback and nsce["prediction"] == llm["prediction"],
                         "retrieved_doc_ids": [d["doc_id"] for d in docs]})
            nsce_by_id[item["claim_id"]] = nsce
        method_maps = {"llm_only": llm_by_id, "self_check": self_by_id, "rag": rag_by_id, "nsce": nsce_by_id}
        for name, rows_by_id in method_maps.items():
            write_jsonl(OUT / f"{name}.jsonl", [rows_by_id[x] for x in pilot_ids if x in rows_by_id])
        if number % 5 == 0 or number == len(pilot):
            peak_process_rss = max(peak_process_rss, psutil.Process(os.getpid()).memory_info().rss)
            elapsed = round(time.perf_counter() - started, 1)
            with (OUT / "phase9_run.log").open("a", encoding="utf-8", newline="\n") as log:
                for name, rows_by_id in method_maps.items():
                    completed_rows = list(rows_by_id.values())
                    accuracy = sum(row["prediction"] == row["gold_label"] for row in completed_rows) / len(completed_rows)
                    log.write(f"method={name} completed={len(completed_rows)}/50 accuracy_so_far={accuracy:.6f} elapsed={elapsed} gpu=cuda\n")
            print(json.dumps({"completed": number, "total": len(pilot), "elapsed_seconds": elapsed}), flush=True)
    write_json(OUT / "model_runtime.json", {**runner.runtime_info,
               "model_loaded": True, "trivial_inference_passed": True,
               "peak_process_rss_bytes": peak_process_rss,
               "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())})


def _rates(rows: list[dict], field: str, denominator_field: str | None = None) -> float | None:
    denominator = [r for r in rows if denominator_field is None or r.get(denominator_field)]
    return sum(bool(r.get(field)) for r in denominator) / len(denominator) if denominator else None


def report() -> dict:
    frozen = validate_frozen()
    pilot = read_jsonl(DATA / "felm_phase9_pilot.jsonl")
    shared = read_jsonl(OUT / "shared_retrieval.jsonl")
    methods = {name: read_jsonl(OUT / f"{name}.jsonl") for name in ("llm_only", "self_check", "rag", "nsce")}
    ids = [r["claim_id"] for r in pilot]
    if any([r["claim_id"] for r in rows] != ids for rows in methods.values()):
        raise RuntimeError("STOP: method claim IDs differ")
    if any(r["retrieved_doc_ids"] != n["retrieved_doc_ids"] for r, n in zip(methods["rag"], methods["nsce"])):
        raise RuntimeError("STOP: RAG and NSCE evidence differs")
    for name, rows in methods.items():
        for row in rows:
            if row["prediction"] == row["gold_label"]:
                category = "NSCE_CORRECT_ABSTENTION" if name == "nsce" and row["prediction"] == "UNKNOWN" else "CORRECT"
            elif name == "self_check":
                category = "SELF_CHECK_HARMFUL_REVISION" if row.get("harmful_revision") else "SELF_CHECK_FAILURE"
            elif name == "rag":
                status = row["evidence_audit"]["audit_status"]
                category = {"NO_EVIDENCE": "NO_EVIDENCE_RETRIEVED", "PARTIAL_EVIDENCE": "PARTIAL_EVIDENCE_RETRIEVED",
                            "IRRELEVANT_EVIDENCE": "IRRELEVANT_EVIDENCE"}.get(status, "RAG_REASONING_FAILURE")
            elif name == "nsce":
                if row.get("feedback_ignored"): category = "NSCE_FEEDBACK_IGNORED"
                elif row["prediction"] == "SUPPORTED": category = "NSCE_FALSE_SUPPORT"
                elif row["prediction"] == "CONTRADICTED": category = "NSCE_FALSE_CONTRADICTION"
                else: category = "NSCE_UNNECESSARY_UNKNOWN"
            else: category = "LLM_HALLUCINATION"
            row["failure_category"] = category
        write_jsonl(OUT / f"{name}.jsonl", rows)
    evidence_statuses = Counter(r["evidence_audit"]["audit_status"] for r in shared)
    summary = {"eligible_claims": json.loads((DATA / "felm_phase9_pool_summary.json").read_text())["eligible_claims"],
               "gold_ambiguous": json.loads((DATA / "felm_phase9_pool_summary.json").read_text())["gold_ambiguous"],
               "pilot_distribution": dict(Counter(r["gold_label"] for r in pilot)), **frozen,
               "methods": {name: score(rows) for name, rows in methods.items()},
               "self_check": {"correction_success_rate": _rates(methods["self_check"], "correction_success"),
                              "harmful_revision_rate": _rates(methods["self_check"], "harmful_revision")},
               "rag": {"retrieval_coverage": sum(x != "NO_EVIDENCE" for x in evidence_statuses.elements()) / len(shared),
                       "retrieval_failure_rate": (evidence_statuses["NO_EVIDENCE"] + evidence_statuses["IRRELEVANT_EVIDENCE"]) / len(shared),
                       "evidence_statuses": dict(evidence_statuses), "recall_at_5": None, "precision_at_5": None,
                       "ranking_metric_note": "Unavailable: FELM provides no gold IDs into this independent frozen corpus."},
               "nsce": {"deterministic_check_coverage": sum(r["verifier_method"] != "qwen_evidence_verifier" for r in methods["nsce"]) / len(pilot),
                        "verifier_coverage": sum(r["verifier_prediction"] != "UNKNOWN" for r in methods["nsce"]) / len(pilot),
                        "correction_success_rate": _rates(methods["nsce"], "correction_success"),
                        "justification_availability": sum(bool(r["justification"]) for r in methods["nsce"]) / len(pilot)},
               "failures": {name: failure_summary(rows) for name, rows in methods.items()}, "blockers": []}
    write_json(OUT / "phase9_report.json", summary)
    audit_rows = []
    for i, item in enumerate(pilot):
        audit_rows.append({"claim_id": item["claim_id"], "claim": item["claim_text"], "gold_label": item["gold_label"],
                           "retrieved_evidence": shared[i]["retrieved_documents"],
                           "llm_only_prediction": methods["llm_only"][i]["prediction"],
                           "self_check_prediction": methods["self_check"][i]["prediction"],
                           "rag_prediction": methods["rag"][i]["prediction"],
                           "nsce_prediction": methods["nsce"][i]["prediction"],
                           "nsce_justification": methods["nsce"][i]["justification"],
                           "failure_category": methods["nsce"][i]["failure_category"]})
    audit_directory = PROJECT / "reports"
    write_jsonl(audit_directory / "phase9_manual_audit_50.jsonl", audit_rows)
    with (audit_directory / "phase9_manual_audit_50.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [x for x in audit_rows[0] if x != "retrieved_evidence"] + ["retrieved_evidence"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in audit_rows:
            writer.writerow({**row, "retrieved_evidence": json.dumps(row["retrieved_evidence"], ensure_ascii=False)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("retrieval", "inference", "report", "all"), default="all")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"retrieval", "all"}: retrieve()
    if args.stage in {"inference", "all"}: infer()
    if args.stage in {"report", "all"}: print(json.dumps(report(), indent=2), flush=True)


if __name__ == "__main__":
    main()
