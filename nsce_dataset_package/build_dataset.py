import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260814


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def base_record(sample_id, dataset, task_type):
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "task_type": task_type,
        "input": {"question": "", "context": None, "facts": [], "rules": []},
        "candidate_response": {"text": None, "claims": []},
        "ground_truth": {
            "label": None,
            "reference_answer": None,
            "acceptable_answers": [],
            "incorrect_answers": [],
            "claim_labels": [],
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
        "provenance": {},
    }


def normalize_proofwriter(path, split):
    out = []
    for i, raw in enumerate(read_jsonl(path)):
        source_id = str(raw["id"])
        sample_id = f"proofwriter_{split}_{i:06d}"
        stmts = sentences(raw["theory"])
        facts = [s for s in stmts if not s.lower().startswith("if ")]
        rules = [s for s in stmts if s.lower().startswith("if ")]
        label = {"True": "true", "False": "false", "Uncertain": "unknown"}[raw["answer"]]
        expected = {"true": "entailed", "false": "contradicted", "unknown": "unknown"}[label]
        rec = base_record(sample_id, "proofwriter", "logical_reasoning")
        rec["input"] = {"question": raw["question"], "context": raw["theory"], "facts": facts, "rules": rules}
        rec["ground_truth"].update({
            "label": label,
            "reference_answer": raw["answer"],
            "acceptable_answers": {"true": ["true", "yes"], "false": ["false", "no"], "unknown": ["unknown", "uncertain"]}[label],
            "incorrect_answers": [],
        })
        docs = []
        for j, text in enumerate(stmts):
            docs.append({"doc_id": f"{sample_id}_d{j:03d}", "type": "rule" if text.lower().startswith("if ") else "fact", "text": text})
        rec["evidence"] = {"documents": docs, "atomic_facts": [], "source_available": True}
        rec["symbolic"].update({"query": raw["question"], "expected_status": expected})
        rec["metadata"].update({
            "domain": "synthetic_logic", "reasoning_depth": int(raw["QDep"]),
            "hallucination_type": "logical_error", "source_split": split,
        })
        rec["provenance"] = {
            "source_id": source_id,
            "source_dataset": "ProofWriter OWA balanced derivative",
            "source_file": path.name,
            "transformation": "sentence split; label normalization; no generated content",
        }
        out.append(rec)
    return out


def normalize_felm(path):
    out = []
    for i, raw in enumerate(read_jsonl(path)):
        sample_id = f"felm_test_{i:06d}"
        segments = raw.get("segmented_response", [])
        labels = raw.get("labels", [])
        comments = raw.get("comment", [])
        types = raw.get("type", [])
        refs = raw.get("ref", [])
        claims, claim_labels, atomic = [], [], []
        for j, segment in enumerate(segments):
            cid = f"{sample_id}_c{j:03d}"
            claims.append({"claim_id": cid, "text": segment})
            source_label = bool(labels[j]) if j < len(labels) else None
            label = "supported" if source_label is True else "factual_error" if source_label is False else "unlabeled"
            claim_labels.append({
                "claim_id": cid,
                "label": label,
                "source_binary_label": source_label,
                "error_type": types[j] if j < len(types) else None,
                "annotator_comment": comments[j] if j < len(comments) else "",
            })
            atomic.append({"claim_id": cid, "support_status": label})
        docs = []
        ref_contents = raw.get("ref_contents")
        if isinstance(ref_contents, list):
            for j, content in enumerate(ref_contents):
                if content:
                    docs.append({"doc_id": f"{sample_id}_e{j:03d}", "type": "reference_content", "text": str(content)})
        elif isinstance(ref_contents, str) and ref_contents.strip():
            docs.append({"doc_id": f"{sample_id}_e000", "type": "reference_content", "text": ref_contents})
        rec = base_record(sample_id, "felm", "claim_factuality")
        rec["input"]["question"] = raw["prompt"]
        rec["candidate_response"] = {"text": raw["response"], "claims": claims}
        overall = "fully_supported" if labels and all(labels) else "contains_factual_error"
        rec["ground_truth"].update({"label": overall, "claim_labels": claim_labels})
        rec["evidence"] = {"documents": docs, "atomic_facts": atomic, "source_available": bool(docs or refs)}
        rec["metadata"].update({
            "domain": raw.get("domain"), "hallucination_type": "factual_error",
            "source_split": "test", "source": raw.get("source"),
        })
        rec["provenance"] = {
            "source_id": str(raw.get("index", i)), "source_dataset": "FELM",
            "source_file": path.name, "reference_urls": refs,
            "transformation": "segment-to-claim mapping; source binary labels preserved",
        }
        out.append(rec)
    return out


def build_rag(proof_test, out_dir):
    corpus = []
    by_sample = {}
    for rec in proof_test:
        by_sample[rec["sample_id"]] = []
        for doc in rec["evidence"]["documents"]:
            row = {**doc, "sample_id": rec["sample_id"], "provenance_role": "in_instance"}
            corpus.append(row)
            by_sample[rec["sample_id"]].append(doc["doc_id"])
    rng = random.Random(SEED)
    all_ids = [d["doc_id"] for d in corpus]
    queries = []
    for rec in proof_test:
        own = set(by_sample[rec["sample_id"]])
        pool = [x for x in all_ids if x not in own]
        distractors = rng.sample(pool, min(40, len(pool)))
        candidates = list(own) + distractors
        rng.shuffle(candidates)
        queries.append({
            "query_id": rec["sample_id"], "query": rec["input"]["question"],
            "ground_truth": rec["ground_truth"]["label"],
            "relevant_doc_ids": sorted(own),
            "candidate_doc_ids": candidates,
            "relevance_scope": "all statements from the originating theory; not minimal-proof relevance",
        })
    write_jsonl(out_dir / "rag" / "proofwriter_corpus.jsonl", corpus)
    write_jsonl(out_dir / "rag" / "proofwriter_queries.jsonl", queries)
    write_jsonl(out_dir / "rag" / "retrieval_results_template.jsonl", [
        {"query_id": q["query_id"], "retrieved_doc_ids": [], "top_k": 5, "retriever": None}
        for q in queries
    ])
    return len(corpus), len(queries)


def validate(pw_dev, pw_test, felm):
    all_rows = pw_dev + pw_test + felm
    ids = [r["sample_id"] for r in all_rows]
    source_pairs = [(r["dataset"], r["metadata"]["source_split"], r["provenance"]["source_id"]) for r in all_rows]
    pw_dist = defaultdict(Counter)
    for r in pw_dev + pw_test:
        pw_dist[r["metadata"]["source_split"]][(r["ground_truth"]["label"], r["metadata"]["reasoning_depth"])] += 1
    felm_segments = sum(len(r["ground_truth"]["claim_labels"]) for r in felm)
    felm_labels = Counter(c["label"] for r in felm for c in r["ground_truth"]["claim_labels"])
    errors = []
    if len(ids) != len(set(ids)): errors.append("duplicate normalized sample_id")
    if len(source_pairs) != len(set(source_pairs)): errors.append("duplicate source record within split")
    for r in all_rows:
        if not r["input"]["question"]: errors.append(f"missing question: {r['sample_id']}")
        if r["dataset"] == "felm" and len(r["candidate_response"]["claims"]) != len(r["ground_truth"]["claim_labels"]):
            errors.append(f"claim-label mismatch: {r['sample_id']}")
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "counts": {"proofwriter_dev": len(pw_dev), "proofwriter_test": len(pw_test), "felm_test": len(felm), "master": len(all_rows), "felm_segments": felm_segments},
        "proofwriter_label_depth_distribution": {split: {f"{k[0]}|depth_{k[1]}": v for k, v in sorted(counts.items())} for split, counts in pw_dist.items()},
        "felm_claim_label_distribution": dict(felm_labels),
        "checks": {"unique_normalized_ids": len(ids) == len(set(ids)), "unique_source_ids_per_split": len(source_pairs) == len(set(source_pairs)), "claim_label_alignment": not any("claim-label" in e for e in errors)},
    }


def write_csv_summary(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "split", "domain", "label", "reasoning_depth", "claim_count"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "sample_id": r["sample_id"], "dataset": r["dataset"], "split": r["metadata"]["source_split"],
                "domain": r["metadata"]["domain"], "label": r["ground_truth"]["label"],
                "reasoning_depth": r["metadata"]["reasoning_depth"], "claim_count": len(r["candidate_response"]["claims"]),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("work/raw"))
    ap.add_argument("--out", type=Path, default=Path("outputs/nsce_dataset_package"))
    args = ap.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "normalized").mkdir(parents=True)
    pw_dev = normalize_proofwriter(args.raw / "proofwriter-train-balanced.jsonl", "dev")
    pw_test = normalize_proofwriter(args.raw / "proofwriter-test-balanced.jsonl", "test")
    felm = normalize_felm(args.raw / "felm-all.jsonl")
    write_jsonl(out / "normalized" / "proofwriter_dev.jsonl", pw_dev)
    write_jsonl(out / "normalized" / "proofwriter_test.jsonl", pw_test)
    write_jsonl(out / "normalized" / "felm_test.jsonl", felm)
    write_jsonl(out / "normalized" / "master.jsonl", pw_dev + pw_test + felm)
    write_csv_summary(out / "sample_index.csv", pw_dev + pw_test + felm)
    corpus_n, queries_n = build_rag(pw_test, out)
    report = validate(pw_dev, pw_test, felm)
    report["counts"].update({"rag_corpus_documents": corpus_n, "rag_queries": queries_n})
    (out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit("Validation failed")
    docs = Path(__file__).with_name("package_docs")
    for path in docs.iterdir():
        shutil.copy2(path, out / path.name)
    shutil.copy2(__file__, out / "build_dataset.py")
    checksums = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
