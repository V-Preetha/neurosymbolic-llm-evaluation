"""Debug/development run of Translation-v2 over the 18-item dev subset
(dev_manifest.json -- disjoint from the frozen 30). Never used for a
reported metric; purpose is finding pipeline/prompt/schema defects.

Usage: python reports/folio_translation_v2/run_dev.py [--variant A|B|D] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.folio.io_guard import extract_allowed_folio_input  # noqa: E402
from src.folio.solver import classify_theory  # noqa: E402
from src.folio.v2.pipeline import translate_theory_v2  # noqa: E402
from src.models.qwen import QwenRunner  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "experiment_qwen8b.json").read_text(encoding="utf-8"))
HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "dev_manifest.json").read_text(encoding="utf-8"))

VARIANTS = {
    "A": {"use_global_vocabulary": False, "enable_repair": False},
    "B": {"use_global_vocabulary": True, "enable_repair": False},
    "C": {"use_global_vocabulary": True, "enable_repair": False},
    "D": {"use_global_vocabulary": True, "enable_repair": True},
}


def sentence_result_to_dict(r):
    return {
        "sentence": r.sentence,
        "valid": r.valid,
        "reason": r.reason,
        "detail": r.detail,
        "pre_repair_valid": r.pre_repair_valid,
        "repaired": r.repair_attempt is not None,
        "initial_raw": r.initial_attempt.raw_output,
        "repair_raw": r.repair_attempt.raw_output if r.repair_attempt else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="C", choices=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = VARIANTS[args.variant]

    print(f"Loading Qwen3-8B ... (variant {args.variant}: {cfg})", flush=True)
    t0 = time.perf_counter()
    runner = QwenRunner(
        model_id=CONFIG["model_id"], seed=CONFIG["seed"],
        max_new_tokens=CONFIG["max_new_tokens"], quantization=CONFIG["quantization"],
    )
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. Runtime info: {runner.runtime_info}", flush=True)

    items = MANIFEST["items"][: args.limit] if args.limit else MANIFEST["items"]
    out_path = HERE / f"dev_run_variant_{args.variant}.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for i, item in enumerate(items):
            idx = item["idx"]
            print(f"[{i+1}/{len(items)}] idx={idx} translating ...", flush=True)
            allowed = extract_allowed_folio_input({"premises": item["premises"], "conclusion": item["conclusion"]})

            t_start = time.perf_counter()
            result = translate_theory_v2(runner, allowed, max_new_tokens=500, **cfg)
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
                "variant": args.variant,
                "premises_raw": item["premises"],
                "hypothesis_raw": item["conclusion"],
                "theory_level_reason": result.theory_level_reason,
                "theory_level_detail": result.theory_level_detail,
                "premise_results": [sentence_result_to_dict(r) for r in result.premise_results],
                "hypothesis_result": sentence_result_to_dict(result.hypothesis_result) if result.hypothesis_result else None,
                "all_valid": result.all_valid,
                "generated_label": generated_label,
                "solver_status": solver_status,
                "gold_label": item["label"],
                "elapsed_s": round(elapsed_s, 2),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"    all_valid={result.all_valid} generated_label={generated_label} "
                f"gold_label={item['label']} ({elapsed_s:.1f}s)",
                flush=True,
            )

    print(f"\nDone. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
