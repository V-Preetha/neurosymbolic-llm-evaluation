from pathlib import Path

import pytest

from final_experiment.analysis.canonical import load_canonical_run
from final_experiment.analysis.metrics import condition_metrics, depth_analysis, paired_analysis


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "final200_proofwriter"


def test_invalid_prediction_is_wrong_but_not_a_fourth_gold_class():
    rows = [
        {"gold_label": "True", "predicted_label": "True"},
        {"gold_label": "False", "predicted_label": None},
        {"gold_label": "Unknown", "predicted_label": "False"},
    ]

    result = condition_metrics(rows)

    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["valid_rate"] == pytest.approx(2 / 3)
    assert result["valid_only_accuracy"] == pytest.approx(1 / 2)
    assert result["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ]
    assert set(result["per_class"]) == {"True", "False", "Unknown"}


def test_canonical_metrics_and_bridge_transitions_are_recomputed_from_rows():
    run = load_canonical_run(CANONICAL)
    c1 = condition_metrics(run.conditions["C1"])
    c3 = condition_metrics(run.conditions["C3"])
    c4 = condition_metrics(run.conditions["C4"])
    paired = paired_analysis(run.conditions["C3"], run.conditions["C4"])

    assert (c1["accuracy"], c3["accuracy"], c4["accuracy"]) == pytest.approx((0.485, 0.825, 0.495))
    assert (c1["macro_f1"], c3["macro_f1"], c4["macro_f1"]) == pytest.approx(
        (0.2510636557, 0.8262677737, 0.4589251517)
    )
    assert c4["valid_rate"] == pytest.approx(0.895)
    assert paired["transitions"] == {
        "wrong_to_correct": 12,
        "correct_to_correct": 87,
        "correct_to_wrong": 78,
        "wrong_to_wrong": 23,
    }
    assert paired["correction_rate"] == pytest.approx(12 / 35)
    assert paired["harm_rate"] == pytest.approx(78 / 165)
    assert paired["net_corrections"] == -66
    assert paired["mcnemar_exact_p"] == pytest.approx(5.202430003083186e-13)


def test_reasoning_depth_groups_sparse_tail_as_four_plus():
    run = load_canonical_run(CANONICAL)
    rows = depth_analysis(run.conditions)

    assert [row["qdep"] for row in rows] == ["0", "1", "2", "3", "4+"]
    assert [row["n"] for row in rows] == [104, 53, 24, 15, 4]
    assert rows[0]["C3_accuracy"] == pytest.approx(100 / 104)
    assert rows[-1]["C3_accuracy"] == pytest.approx(3 / 4)
