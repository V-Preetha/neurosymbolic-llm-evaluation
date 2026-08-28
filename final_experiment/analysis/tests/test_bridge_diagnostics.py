from pathlib import Path

from final_experiment.analysis.bridge_diagnostics import bridge_diagnostics
from final_experiment.analysis.canonical import load_canonical_run, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "final200_proofwriter"


def test_bridge_diagnostics_follow_prediction_and_trace_stage_fields():
    run = load_canonical_run(CANONICAL)
    traces = read_jsonl(CANONICAL / "c4_finetuned_bridge" / "parser_traces.jsonl")

    result = bridge_diagnostics(run.conditions["C4"], traces)

    assert result["total"] == 200
    assert result["schema_valid"] == 180
    assert result["parser_valid"] == 180
    assert result["ast_valid"] == 180
    assert result["compile_success"] == 180
    assert result["executable"] == 179
    assert result["solver_success"] == 179
    assert result["error_categories"] == {
        "correct": 99,
        "incorrect_final_symbolic_answer": 80,
        "malformed_output": 11,
        "schema_invalid": 9,
        "inconsistent_premises": 1,
    }
    assert result["semantic_equivalence"] == {"different": 158, "not_available": 42}
    assert result["fallback"]["available"] is True
    assert result["fallback"]["trace_step_count"] >= 1
