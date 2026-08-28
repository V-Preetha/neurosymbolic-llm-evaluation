"""Fast 5-item gate for the T5 dedicated NL->FOL bridge (Translation-v3).

Selects 5 items from the SAME frozen-30 sample used by V1/V2
(reports/folio_translation_prototype/sample_manifest.json) -- none of them
were used to tune the adapter grammar (the adapter was built from separate
smoke-test sentences, not from this sample). Covers: universal implication
(idx 51), existential (idx 6), negation+disjunction (idx 9), xor/complex (idx
78), binary relation/multi-variable (idx 119).

Usage: python reports/folio_translation_t5/run_gate5.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.folio.io_guard import extract_allowed_folio_input  # noqa: E402
from src.folio.solver import classify_theory  # noqa: E402
from src.folio.t5_bridge.model import T5FolBridge  # noqa: E402
from src.folio.t5_bridge.pipeline import translate_theory_t5  # noqa: E402

MANIFEST = json.loads(
    (ROOT / "reports" / "folio_translation_prototype" / "sample_manifest.json").read_text(encoding="utf-8")
)
GATE_IDX = [51, 6, 9, 78, 119]
HERE = Path(__file__).parent


def sentence_result_to_dict(r):
    return {
        "sentence": r.sentence,
        "raw_output": r.translation.raw_output,
        "latency_ms": r.translation.latency_ms,
        "valid": r.valid,
        "reason": r.reason,
        "detail": r.detail,
    }


def main():
    by_idx = {i["idx"]: i for i in MANIFEST["items"]}
    items = [by_idx[i] for i in GATE_IDX]

    print("Loading fvossel/t5-base-nl-to-fol ...", flush=True)
    t0 = time.perf_counter()
    bridge = T5FolBridge()
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s. Runtime info: {bridge.runtime_info}", flush=True)

    out_path = HERE / "gate5_results.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for i, item in enumerate(items):
            idx = item["idx"]
            print(f"[{i + 1}/5] idx={idx} translating ...", flush=True)
            allowed = extract_allowed_folio_input({"premises": item["premises"], "conclusion": item["conclusion"]})

            t_start = time.perf_counter()
            result = translate_theory_t5(bridge, allowed)
            elapsed_s = time.perf_counter() - t_start

            if result.all_valid:
                verdict = classify_theory(list(result.premise_dicts), result.hypothesis_dict)
                generated_label = verdict.label
                solver_status = verdict.status
            else:
                generated_label = None
                solver_status = None

            record = {
                "idx": idx,
                "constructs": item["constructs"],
                "premises_raw": item["premises"],
                "hypothesis_raw": item["conclusion"],
                "premise_results": [sentence_result_to_dict(r) for r in result.premise_results],
                "hypothesis_result": sentence_result_to_dict(result.hypothesis_result) if result.hypothesis_result else None,
                "theory_level_reason": result.theory_level_reason,
                "theory_level_detail": result.theory_level_detail,
                "premise_dicts": list(result.premise_dicts) if result.premise_dicts else None,
                "hypothesis_dict": result.hypothesis_dict,
                "all_valid": result.all_valid,
                "solver_status": solver_status,
                "generated_label": generated_label,
                "gold_label": item["label"],
                "elapsed_s": round(elapsed_s, 2),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"    all_valid={result.all_valid} generated_label={generated_label} "
                f"gold_label={item['label']} ({elapsed_s:.2f}s)",
                flush=True,
            )

    print(f"\nDone. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
