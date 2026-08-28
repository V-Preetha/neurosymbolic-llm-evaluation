import csv
import json
from pathlib import Path

from final_experiment.analysis.canonical import load_canonical_run
from final_experiment.analysis.generate_paper_artifacts import build_results, write_generated_artifacts


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "final200_proofwriter"


def test_generated_outputs_share_one_canonical_results_object(tmp_path):
    run = load_canonical_run(CANONICAL)
    results = build_results(run, bootstrap_replicates=100)
    analysis_dir = tmp_path / "analysis"
    paper_dir = tmp_path / "paper"

    written = write_generated_artifacts(results, run, analysis_dir, paper_dir)

    loaded = json.loads((analysis_dir / "results.json").read_text(encoding="utf-8"))
    assert loaded["provenance"]["subset_hash"] == run.subset_hash
    assert loaded["conditions"]["C3"]["accuracy"] == 0.825
    assert loaded["paired"]["transitions"]["correct_to_wrong"] == 78
    assert loaded["datasets"]["folio"]["performance_available"] is False
    with (analysis_dir / "per_example.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert set(rows[0]) >= {"example_id", "gold_label", "qdep", "C1_prediction", "C3_prediction", "C4_prediction", "transition"}
    table = (paper_dir / "generated_tables" / "overall_results.tex").read_text(encoding="utf-8")
    assert "82.5" in table and "49.5" in table and "89.5" in table
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
