"""Strict multiclass metrics and paired analyses for the canonical run."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .canonical import LABELS


PREDICTION_COLUMNS = LABELS + ("Invalid",)


def _prediction(row: dict[str, Any]) -> str:
    prediction = row.get("predicted_label")
    return prediction if prediction in LABELS else "Invalid"


def _is_correct(row: dict[str, Any]) -> bool:
    return _prediction(row) == row.get("gold_label")


def condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = [[0 for _ in PREDICTION_COLUMNS] for _ in LABELS]
    for row in rows:
        matrix[LABELS.index(row["gold_label"])][PREDICTION_COLUMNS.index(_prediction(row))] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[other][index] for other in range(len(LABELS)) if other != index)
        fn = sum(matrix[index][column] for column in range(len(PREDICTION_COLUMNS)) if column != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(rows)
    correct = sum(_is_correct(row) for row in rows)
    valid = sum(_prediction(row) != "Invalid" for row in rows)
    macro_precision = sum(float(value["precision"]) for value in per_class.values()) / len(LABELS)
    macro_recall = sum(float(value["recall"]) for value in per_class.values()) / len(LABELS)
    macro_f1 = sum(float(value["f1"]) for value in per_class.values()) / len(LABELS)
    weighted_f1 = sum(float(value["f1"]) * int(value["support"]) for value in per_class.values()) / total
    total_ms = [float(row.get("timing", {}).get("total_ms", row.get("latency_ms", 0.0))) for row in rows]

    return {
        "n": total,
        "correct": correct,
        "incorrect": total - correct,
        "valid": valid,
        "accuracy": correct / total,
        "valid_rate": valid / total,
        "valid_only_accuracy": correct / valid if valid else None,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_columns": list(PREDICTION_COLUMNS),
        "latency_ms": {
            "mean": float(np.mean(total_ms)),
            "median": float(np.median(total_ms)),
            "p95": float(np.percentile(total_ms, 95)),
            "sum": float(np.sum(total_ms)),
        },
    }


def paired_analysis(c3: list[dict[str, Any]], c4: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["example_id"] for row in c3] != [row["example_id"] for row in c4]:
        raise ValueError("paired analysis requires identical ordered example IDs")
    transitions = Counter()
    for direct, bridge in zip(c3, c4):
        direct_correct = _is_correct(direct)
        bridge_correct = _is_correct(bridge)
        if not direct_correct and bridge_correct:
            transitions["wrong_to_correct"] += 1
        elif direct_correct and bridge_correct:
            transitions["correct_to_correct"] += 1
        elif direct_correct and not bridge_correct:
            transitions["correct_to_wrong"] += 1
        else:
            transitions["wrong_to_wrong"] += 1

    ordered = {
        "wrong_to_correct": transitions["wrong_to_correct"],
        "correct_to_correct": transitions["correct_to_correct"],
        "correct_to_wrong": transitions["correct_to_wrong"],
        "wrong_to_wrong": transitions["wrong_to_wrong"],
    }
    c3_wrong = ordered["wrong_to_correct"] + ordered["wrong_to_wrong"]
    c3_correct = ordered["correct_to_correct"] + ordered["correct_to_wrong"]
    discordant = ordered["wrong_to_correct"] + ordered["correct_to_wrong"]
    p_value = float(binomtest(min(ordered["wrong_to_correct"], ordered["correct_to_wrong"]), discordant, 0.5).pvalue)
    return {
        "transitions": ordered,
        "correction_rate": ordered["wrong_to_correct"] / c3_wrong,
        "harm_rate": ordered["correct_to_wrong"] / c3_correct,
        "net_corrections": ordered["wrong_to_correct"] - ordered["correct_to_wrong"],
        "discordant": discordant,
        "mcnemar_exact_p": p_value,
    }


def depth_analysis(conditions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    reference = conditions["C1"]
    for group in ("0", "1", "2", "3", "4+"):
        indices = [
            index
            for index, row in enumerate(reference)
            if (str(row["qdep"]) == group if group != "4+" else int(row["qdep"]) >= 4)
        ]
        entry: dict[str, Any] = {"qdep": group, "n": len(indices)}
        for name, rows in conditions.items():
            subset = [rows[index] for index in indices]
            metrics = condition_metrics(subset)
            entry[f"{name}_accuracy"] = metrics["accuracy"]
            entry[f"{name}_valid_rate"] = metrics["valid_rate"]
        result.append(entry)
    return result


def _macro_f1_from_arrays(gold: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for label in LABELS:
        tp = np.sum((gold == label) & (pred == label))
        fp = np.sum((gold != label) & (pred == label))
        fn = np.sum((gold == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def bootstrap_intervals(
    conditions: dict[str, list[dict[str, Any]]], seed: int = 42, replicates: int = 10000
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(conditions["C1"])
    samples = rng.integers(0, n, size=(replicates, n))
    gold = np.asarray([row["gold_label"] for row in conditions["C1"]], dtype=object)
    predictions = {
        name: np.asarray([_prediction(row) for row in rows], dtype=object)
        for name, rows in conditions.items()
    }
    draws: dict[str, dict[str, np.ndarray]] = {}
    for name, pred in predictions.items():
        accuracies = np.mean(pred[samples] == gold[samples], axis=1)
        macro = np.asarray([_macro_f1_from_arrays(gold[index], pred[index]) for index in samples])
        draws[name] = {"accuracy": accuracies, "macro_f1": macro}

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    result: dict[str, Any] = {
        name: {metric: interval(values) for metric, values in metrics.items()}
        for name, metrics in draws.items()
    }
    result["C4_minus_C3"] = {
        "accuracy": interval(draws["C4"]["accuracy"] - draws["C3"]["accuracy"]),
        "macro_f1": interval(draws["C4"]["macro_f1"] - draws["C3"]["macro_f1"]),
    }
    result["seed"] = seed
    result["replicates"] = replicates
    return result
