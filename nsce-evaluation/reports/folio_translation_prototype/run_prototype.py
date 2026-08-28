"""Run the frozen Qwen3-8B translation prototype over the 30-item sample.

Raw English -> Qwen3-8B semantic translation -> validated JSON IR -> Z3 verdict.
Never attaches gold information until AFTER inference (see the "gold_label"/
"gold_fol_oracle" fields being merged in only at the very end of each record).

Writes incrementally to translations.jsonl (one line per item, flushed after
each item) so a long run can be inspected/resumed-in-spirit even if it's
interrupted -- there is no resume logic, this is a one-shot frozen run.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.folio.io_guard import extract_allowed_folio_input  # noqa: E402
from src.folio.solver import classify_theory  # noqa: E402
from src.folio.translator import translate_theory  # noqa: E402
from src.models.qwen import QwenRunner  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "experiment_qwen8b.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((Path(__file__).parent / "sample_manifest.json").read_text(encoding="utf-8"))
OUT_PATH = Path(__file__).parent / "translations.jsonl"


def attempt_to_dict(a):
    return {
        "sentence": a.sentence,
        "raw_output": a.raw_output,
        "json_ast": a.json_ast,
        "schema_valid": a.schema_valid,
        "error_reason": a.error_reason,
        "error_detail": a.error_detail,
        "latency_ms": a.latency_ms,
    }


def main():
    print("Loading Qwen3-8B ...", flush=True)
    t0 = time.perf_counter()
    runner = QwenRunner(
        model_id=CONFIG["model_id"],
        seed=CONFIG["seed"],
        max_new_tokens=CONFIG["max_new_tokens"],
        quantization=CONFIG["quantization"],
    )
    print(f"Loaded in {time.perf_counter() - t0:.1f}s. Runtime info: {runner.runtime_info}", flush=True)

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as out:
        for i, item in enumerate(MANIFEST["items"]):
            idx = item["idx"]
            print(f"[{i+1}/{len(MANIFEST['items'])}] idx={idx} translating ...", flush=True)
            # extract_allowed_folio_input structurally cannot see label/premises-FOL/conclusion-FOL --
            # it only ever reads record["premises"] / record["conclusion"].
            allowed = extract_allowed_folio_input({"premises": item["premises"], "conclusion": item["conclusion"]})

            item_start = time.perf_counter()
            translation = translate_theory(runner, allowed, max_new_tokens=400)
            elapsed_s = time.perf_counter() - item_start

            if translation.all_schema_valid:
                solver_start = time.perf_counter()
                verdict = classify_theory(list(translation.premise_dicts), translation.hypothesis_dict)
                solver_latency_ms = (time.perf_counter() - solver_start) * 1000
            else:
                verdict = None
                solver_latency_ms = None

            record = {
                "idx": idx,
                "premises_raw": item["premises"],
                "hypothesis_raw": item["conclusion"],
                "premise_attempts": [attempt_to_dict(a) for a in translation.premise_attempts],
                "hypothesis_attempt": attempt_to_dict(translation.hypothesis_attempt),
                "all_schema_valid": translation.all_schema_valid,
                "compiled_and_solved": verdict is not None and verdict.status not in ("translation_failure", "compilation_error"),
                "solver_status": verdict.status if verdict else None,
                "generated_label": verdict.label if verdict else None,
                "solver_statuses": verdict.solver_statuses if verdict else {},
                "solver_failure_reason": verdict.failure_reason if verdict else None,
                "translation_latency_s": round(elapsed_s, 3),
                "solver_latency_ms": solver_latency_ms,
                # gold information attached AFTER inference only, for scoring:
                "gold_label": item["label"],
                "gold_constructs": item["constructs"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"    all_schema_valid={translation.all_schema_valid} "
                f"generated_label={record['generated_label']} gold_label={item['label']} "
                f"({elapsed_s:.1f}s)",
                flush=True,
            )

    print(f"\nDone. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
