from pathlib import Path

import pytest

from final_experiment.analysis.canonical import EXPECTED_SUBSET_HASH, load_canonical_run


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "final200_proofwriter"


def test_load_canonical_run_rejects_any_loss_of_matched_provenance():
    run = load_canonical_run(CANONICAL)

    assert list(run.conditions) == ["C1", "C3", "C4"]
    assert {name: len(rows) for name, rows in run.conditions.items()} == {
        "C1": 200,
        "C3": 200,
        "C4": 200,
    }
    assert run.subset_hash == EXPECTED_SUBSET_HASH
    assert run.example_ids == tuple(row["example_id"] for row in run.conditions["C1"])
    assert run.checkpoint.endswith("checkpoint-250")
    assert run.condition_metadata["C3"]["adapter_checkpoint"] == run.condition_metadata["C4"]["adapter_checkpoint"]


def test_loader_rejects_noncanonical_directory(tmp_path):
    with pytest.raises(ValueError, match="missing canonical file"):
        load_canonical_run(tmp_path)
