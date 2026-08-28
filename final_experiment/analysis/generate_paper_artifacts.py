"""Generate all numerical, tabular, and graphical paper artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .bridge_diagnostics import bridge_diagnostics
from .canonical import CanonicalRun, LABELS, load_canonical_run, read_json, read_jsonl
from .metrics import bootstrap_intervals, condition_metrics, depth_analysis, paired_analysis
from .tables import write_tables


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = PACKAGE_ROOT / "results" / "final200_proofwriter"


def build_results(run: CanonicalRun, bootstrap_replicates: int = 10000) -> dict[str, Any]:
    conditions = {name: condition_metrics(rows) for name, rows in run.conditions.items()}
    paired = paired_analysis(run.conditions["C3"], run.conditions["C4"])
    paired["accuracy_delta"] = conditions["C4"]["accuracy"] - conditions["C3"]["accuracy"]
    paired["macro_f1_delta"] = conditions["C4"]["macro_f1"] - conditions["C3"]["macro_f1"]
    paired["relative_error_reduction"] = (
        (1 - conditions["C3"]["accuracy"]) - (1 - conditions["C4"]["accuracy"])
    ) / (1 - conditions["C3"]["accuracy"])
    c4_root = run.root / "c4_finetuned_bridge"
    bridge = bridge_diagnostics(run.conditions["C4"], read_jsonl(c4_root / "parser_traces.jsonl"))
    proof_audit = read_json(PACKAGE_ROOT / "data" / "processed" / "proofwriter" / "audit.json")
    folio_audit = read_json(PACKAGE_ROOT / "data" / "processed" / "folio" / "audit.json")
    config = read_json(run.root / "config.json")
    gold_distribution = {label: sum(row["gold_label"] == label for row in run.conditions["C1"]) for label in LABELS}

    return {
        "provenance": {
            "canonical_root": str(run.root),
            "subset_hash": run.subset_hash,
            "n": 200,
            "checkpoint": run.checkpoint,
            "ordered_ids_verified": True,
            "gold_labels_verified": True,
        },
        "conditions": conditions,
        "paired": paired,
        "bootstrap": bootstrap_intervals(run.conditions, seed=42, replicates=bootstrap_replicates),
        "depth": depth_analysis(run.conditions),
        "bridge": bridge,
        "gold_distribution": gold_distribution,
        "datasets": {
            "proofwriter": {
                "source": proof_audit["counts"],
                "selected": {
                    "train": proof_audit["manifests"]["train"]["size"],
                    "validation": proof_audit["manifests"]["training_validation"]["size"],
                    "heldout": proof_audit["manifests"]["test"]["size"],
                },
                "canonical_n": 200,
            },
            "folio": {
                "train": folio_audit["counts"]["train"],
                "development": folio_audit["counts"]["validation"],
                "heldout": folio_audit["counts"]["test"],
                "performance_available": False,
            },
        },
        "configuration": {
            "model": config["model"],
            "bridge": config["bridge"],
            "training": {
                **config["training"],
                "train_examples": proof_audit["manifests"]["train"]["size"],
                "validation_examples": proof_audit["manifests"]["training_validation"]["size"],
                "executed_best_metric": "eval_loss",
                "selected_checkpoint": "checkpoint-250",
            },
            "experiment": config["experiment"],
        },
    }


def _valid_prediction(row: dict[str, Any]) -> bool:
    return row.get("predicted_label") in LABELS


def _transition(c3: dict[str, Any], c4: dict[str, Any]) -> str:
    a = _valid_prediction(c3) and c3["predicted_label"] == c3["gold_label"]
    b = _valid_prediction(c4) and c4["predicted_label"] == c4["gold_label"]
    return {(
        False,
        True,
    ): "wrong_to_correct", (True, True): "correct_to_correct", (True, False): "correct_to_wrong", (False, False): "wrong_to_wrong"}[(a, b)]


def write_generated_artifacts(
    results: dict[str, Any], run: CanonicalRun, analysis_dir: Path, paper_dir: Path
) -> list[Path]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    results_path = analysis_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(results_path)

    per_example_path = analysis_dir / "per_example.csv"
    fields = [
        "example_id", "gold_label", "qdep", "C1_prediction", "C1_valid", "C1_correct",
        "C3_prediction", "C3_valid", "C3_correct", "C4_prediction", "C4_valid", "C4_correct",
        "transition", "C4_error_category",
    ]
    with per_example_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c1, c3, c4 in zip(run.conditions["C1"], run.conditions["C3"], run.conditions["C4"]):
            row = {"example_id": c1["example_id"], "gold_label": c1["gold_label"], "qdep": c1["qdep"]}
            for name, source in (("C1", c1), ("C3", c3), ("C4", c4)):
                row[f"{name}_prediction"] = source.get("predicted_label") or "Invalid"
                row[f"{name}_valid"] = _valid_prediction(source)
                row[f"{name}_correct"] = _valid_prediction(source) and source.get("predicted_label") == source["gold_label"]
            row["transition"] = _transition(c3, c4)
            row["C4_error_category"] = c4.get("error_category") or "correct"
            writer.writerow(row)
    written.append(per_example_path)

    depth_path = analysis_dir / "depth.csv"
    with depth_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results["depth"][0]))
        writer.writeheader()
        writer.writerows(results["depth"])
    written.append(depth_path)

    failure_path = analysis_dir / "bridge_failures.csv"
    failure_fields = ["example_id", "gold_label", "predicted_label", "error_category", "parser_status", "compiler_status", "solver_status", "executable"]
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        for row in run.conditions["C4"]:
            if row.get("error_category"):
                writer.writerow({
                    "example_id": row["example_id"],
                    "gold_label": row["gold_label"],
                    "predicted_label": row.get("predicted_label") or "Invalid",
                    "error_category": row["error_category"],
                    "parser_status": row.get("parser_status"),
                    "compiler_status": row.get("compiler_status"),
                    "solver_status": row.get("solver_status"),
                    "executable": row.get("translation", {}).get("executable", False),
                })
    written.append(failure_path)
    written.extend(write_tables(results, paper_dir / "generated_tables"))
    return written


def main() -> None:
    run = load_canonical_run(CANONICAL_ROOT)
    results = build_results(run)
    generated = PACKAGE_ROOT / "analysis" / "generated"
    paper = PACKAGE_ROOT / "paper"
    written = write_generated_artifacts(results, run, generated, paper)
    from .figures import generate_all_figures

    written.extend(generate_all_figures(results, paper / "figures"))
    print(f"Canonical source: {run.root}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
