"""Build the gold-FOL oracle verdict for each of the 30 frozen prototype items.

Uses the dataset's own gold `premises-FOL` / `conclusion-FOL` strings (parsed
by reports/folio_feasibility/gold_fol_parser.py into our JSON IR, then run
through src/folio/solver.py) -- never Qwen's output. This is the reference
point item 9 of the phase deliverable compares Qwen-generated verdicts
against ("generated-logic verdict vs gold-FOL oracle agreement").
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "folio_feasibility"))

from gold_fol_parser import parse_gold_fol  # noqa: E402
from src.folio.solver import classify_theory  # noqa: E402

manifest = json.loads((Path(__file__).parent / "sample_manifest.json").read_text(encoding="utf-8"))

oracle = []
for item in manifest["items"]:
    try:
        premise_dicts = [parse_gold_fol(p) for p in item["premises_fol_gold"]]
        hyp_dict = parse_gold_fol(item["conclusion_fol_gold"])
        verdict = classify_theory(premise_dicts, hyp_dict)
        oracle.append({
            "idx": item["idx"],
            "gold_label": item["label"],
            "oracle_status": verdict.status,
            "oracle_label": verdict.label,
            "matches_gold": verdict.label == item["label"],
        })
    except Exception as exc:
        oracle.append({
            "idx": item["idx"],
            "gold_label": item["label"],
            "oracle_status": "parse_error",
            "oracle_label": None,
            "matches_gold": False,
            "parse_error": str(exc),
        })

correct = sum(1 for o in oracle if o["matches_gold"])
print(f"{correct}/{len(oracle)} gold-FOL oracle predictions match official label")
for o in oracle:
    print(o)

out_path = Path(__file__).parent / "gold_oracle.json"
out_path.write_text(json.dumps({
    "description": "Gold-FOL -> our JSON IR (via gold_fol_parser.py) -> src/folio/solver.py, "
                    "for the 30-item prototype sample. Reference oracle only -- never used as translator input.",
    "accuracy_vs_official_label": correct / len(oracle),
    "items": oracle,
}, indent=2), encoding="utf-8")
print(f"\nWrote {out_path}")
