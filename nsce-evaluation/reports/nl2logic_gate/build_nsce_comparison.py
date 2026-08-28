"""Combine Branch A (Qwen3-8B) + Branch B (NL2Logic+Gemma+Z3) into the full NSCE
verdict and report Qwen-only vs full-NSCE, on the frozen 20-item ProofWriter sample.

verifier logic (per spec):
    if symbolic translation/Z3 failed:  ABSTAIN, keep Qwen answer as unverified output
    elif qwen_pred == symbolic_pred:    RETAIN
    else:                               CORRECT using symbolic verdict
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_PATH = REPO_ROOT / "results" / "raw" / "llm_only_qwen8b_proofwriter.jsonl"
SYMBOLIC_PATH = REPO_ROOT / "reports" / "nl2logic_gate" / "step_proofwriter_frozen20.jsonl"
OUT_PATH = REPO_ROOT / "reports" / "nl2logic_gate" / "nsce_final_comparison.json"
LOG_PATH = REPO_ROOT / "reports" / "nl2logic_gate" / "nsce_final_log.jsonl"

QWEN_LABEL_MAP = {"true": "True", "false": "False", "unknown": "Uncertain"}
LABELS = ("True", "False", "Uncertain")


def load_qwen():
    rows = {}
    with open(QWEN_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows[rec["sample_id"]] = rec
    return rows


def load_symbolic():
    rows = {}
    with open(SYMBOLIC_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[rec["sample_id"]] = rec
    return rows


def macro_f1(gold: list[str], pred: list[str]) -> dict:
    per_class = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": gold.count(label)}
    macro = sum(v["f1"] for v in per_class.values()) / len(LABELS)
    return {"per_class": per_class, "macro_f1": macro}


def main() -> None:
    qwen_rows = load_qwen()
    symbolic_rows = load_symbolic()
    sample_ids = sorted(qwen_rows.keys() & symbolic_rows.keys())
    assert len(sample_ids) == 20, f"expected 20 shared items, got {len(sample_ids)}"

    gold_list, qwen_pred_list, nsce_pred_list = [], [], []
    verdict_counts = Counter()
    corrections_helped = corrections_harmed = 0
    retained_correct = retained_incorrect = 0
    log = []

    for sid in sample_ids:
        qwen = qwen_rows[sid]
        symb = symbolic_rows[sid]
        gold = QWEN_LABEL_MAP[qwen["ground_truth"]]
        qwen_pred_raw = qwen["final_prediction"]
        qwen_pred = QWEN_LABEL_MAP.get(qwen_pred_raw, "Uncertain") if qwen_pred_raw else None
        bridge_ok = symb["translation_ok"] and symb["verdict"] in LABELS
        symbolic_pred = symb["verdict"] if bridge_ok else None

        if not bridge_ok or qwen_pred is None:
            status = "ABSTAIN"
            final_pred = qwen_pred
        elif qwen_pred == symbolic_pred:
            status = "RETAIN"
            final_pred = qwen_pred
        else:
            status = "CORRECT"
            final_pred = symbolic_pred

        verdict_counts[status] += 1
        qwen_correct = qwen_pred == gold
        final_correct = final_pred == gold

        if status == "CORRECT":
            if final_correct and not qwen_correct:
                corrections_helped += 1
            elif not final_correct and qwen_correct:
                corrections_harmed += 1
        if status == "RETAIN":
            if final_correct:
                retained_correct += 1
            else:
                retained_incorrect += 1

        gold_list.append(gold)
        qwen_pred_list.append(qwen_pred or "NONE")
        nsce_pred_list.append(final_pred or "NONE")

        log.append({
            "sample_id": sid,
            "gold": gold,
            "qwen_prediction": qwen_pred,
            "symbolic_prediction": symbolic_pred,
            "bridge_ok": bridge_ok,
            "status": status,
            "final_prediction": final_pred,
            "qwen_correct": qwen_correct,
            "final_correct": final_correct,
        })

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")

    qwen_acc = sum(1 for g, p in zip(gold_list, qwen_pred_list) if g == p) / len(gold_list)
    nsce_acc = sum(1 for g, p in zip(gold_list, nsce_pred_list) if g == p) / len(gold_list)
    qwen_f1 = macro_f1(gold_list, qwen_pred_list)
    nsce_f1 = macro_f1(gold_list, nsce_pred_list)

    summary = {
        "n_items": len(sample_ids),
        "qwen_only": {"accuracy": qwen_acc, "macro_f1": qwen_f1["macro_f1"], "per_class": qwen_f1["per_class"]},
        "full_nsce": {"accuracy": nsce_acc, "macro_f1": nsce_f1["macro_f1"], "per_class": nsce_f1["per_class"]},
        "verifier_status_counts": dict(verdict_counts),
        "corrections_helped": corrections_helped,
        "corrections_harmed": corrections_harmed,
        "retained_correct": retained_correct,
        "retained_incorrect": retained_incorrect,
        "bridge_failures": sum(1 for row in log if not row["bridge_ok"]),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
