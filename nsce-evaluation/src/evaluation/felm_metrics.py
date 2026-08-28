"""Metrics for evidence-conditional FELM claim verification."""

from __future__ import annotations


LABEL_MAP = {"supported": "SUPPORTED", "factual_error": "CONTRADICTED"}


def class_metrics(rows: list[dict], label: str) -> dict:
    true_positive = sum(row["prediction"] == label and row["gold_label"] == label for row in rows)
    false_positive = sum(row["prediction"] == label and row["gold_label"] != label for row in rows)
    false_negative = sum(row["prediction"] != label and row["gold_label"] == label for row in rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
    return {
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "precision": precision, "recall": recall, "f1": f1,
    }


def verification_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "claims": 0, "coverage": None, "selective_accuracy": None,
            "accuracy_including_unknown": None, "abstention_rate": None,
            "supported": class_metrics([], "SUPPORTED"),
            "contradicted": class_metrics([], "CONTRADICTED"),
        }
    decided = [row for row in rows if row["prediction"] != "UNKNOWN"]
    unknown = [row for row in rows if row["prediction"] == "UNKNOWN"]
    return {
        "claims": len(rows), "coverage": len(decided) / len(rows),
        "selective_accuracy": sum(row["correct"] for row in decided) / len(decided) if decided else None,
        "accuracy_including_unknown": sum(row["correct"] for row in rows) / len(rows),
        "supported": class_metrics(rows, "SUPPORTED"),
        "contradicted": class_metrics(rows, "CONTRADICTED"),
        "abstention_rate": len(unknown) / len(rows),
        "false_support_rate": sum(
            row["prediction"] == "SUPPORTED" and row["gold_label"] == "CONTRADICTED" for row in rows
        ) / len(rows),
        "false_contradiction_rate": sum(
            row["prediction"] == "CONTRADICTED" and row["gold_label"] == "SUPPORTED" for row in rows
        ) / len(rows),
        "abstention_calibration": {
            "unknown_count": len(unknown),
            "unnecessary_unknown_rate_on_sufficient_evidence": len(unknown) / len(rows),
            "note": "FELM gold is binary, so Unknown is never scored correct on FELM-EVIDENCE.",
        },
    }
