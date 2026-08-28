"""Three-way metrics and failure decomposition for the controlled FELM pilot."""

from __future__ import annotations

from collections import Counter


LABELS = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")


def _safe(num: int | float, den: int | float) -> float | None:
    return num / den if den else None


def score(rows: list[dict]) -> dict:
    classes = {}
    for label in LABELS:
        tp = sum(r["gold_label"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["gold_label"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["gold_label"] == label and r["prediction"] != label for r in rows)
        precision, recall = _safe(tp, tp + fp), _safe(tp, tp + fn)
        f1 = 0.0 if not precision or not recall else 2 * precision * recall / (precision + recall)
        classes[label] = {"support": sum(r["gold_label"] == label for r in rows),
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1}
    latencies = [float(r.get("latency_ms", 0)) for r in rows]
    tokens = [int(r.get("input_tokens", 0)) + int(r.get("output_tokens", 0)) for r in rows]
    return {
        "claims": len(rows),
        "accuracy": _safe(sum(r["prediction"] == r["gold_label"] for r in rows), len(rows)),
        "macro_f1": sum(classes[x]["f1"] for x in LABELS) / len(LABELS),
        "per_class": classes,
        "abstention_rate": _safe(sum(r["prediction"] == "UNKNOWN" for r in rows), len(rows)),
        "false_support_rate": _safe(sum(r["prediction"] == "SUPPORTED" and r["gold_label"] != "SUPPORTED" for r in rows), len(rows)),
        "false_contradiction_rate": _safe(sum(r["prediction"] == "CONTRADICTED" and r["gold_label"] != "CONTRADICTED" for r in rows), len(rows)),
        "total_latency_ms": sum(latencies), "mean_latency_ms": _safe(sum(latencies), len(latencies)),
        "total_tokens": sum(tokens), "mean_tokens": _safe(sum(tokens), len(tokens)),
    }


def failure_category(method: str, row: dict) -> str:
    pred, gold = row["prediction"], row["gold_label"]
    if pred == gold:
        if pred == "UNKNOWN":
            return "CORRECT_ABSTENTION"
        return "CORRECT"
    if pred == "SUPPORTED":
        return "FALSE_SUPPORT"
    if pred == "CONTRADICTED":
        return "FALSE_CONTRADICTION"
    if method == "self_check" and row.get("initial_prediction") == gold:
        return "SELF_CHECK_HARM"
    if method == "nsce" and row.get("feedback_applied") and row.get("prediction") == row.get("initial_prediction"):
        return "FEEDBACK_IGNORED"
    if pred == "UNKNOWN":
        return "UNNECESSARY_UNKNOWN"
    return "METHOD_ERROR"


def failure_summary(rows: list[dict]) -> dict:
    return dict(sorted(Counter(r["failure_category"] for r in rows).items()))
