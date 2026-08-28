"""Final ProofWriter 200-sample evaluation: C1, C3, C4.

Standalone runner. Adds no changes to evaluation.py, prompts.py, the NL2Logic parser, the
Z3 compiler/solver, configs, checkpoints, or any existing manifest -- everything below is
imported and used as-is.

  C1  base Qwen3-1.7B, direct, answer-only prompt (prompts.direct_prompt_answer_only)
  C3  checkpoint-250 LoRA, direct, same answer-only prompt
  C4  checkpoint-250 LoRA + frozen NL2Logic -> AST -> Z3 bridge, schema variant "descriptive"

C2 is deliberately not run.

The 200-ID subset is drawn once with data.stratified_sample(seed=42) over the existing
test_main.json manifest rows, written to <run>/frozen200_manifest.json with a sha256 of the
ID list, and reused verbatim by every condition. A rerun that finds that file reuses the IDs
rather than resampling, so the frozen subset stays frozen.

Run:
    python -m final_experiment.scripts.final_200
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from ..core import (
    FROZEN_BRIDGE_ID,
    LABELS,
    ROOT,
    PredictionRow,
    read_jsonl,
    resolved_config,
    runtime_metadata,
    seed_everything,
    write_json,
    write_jsonl,
)
from ..data import _manifest, load_manifest_split, stratified_sample

DATASET = "proofwriter"
SOURCE_MODE = "main"
ADAPTER = ROOT / "checkpoints" / "proofwriter_direct_qwen3_1_7b_lora" / "main" / "latest" / "checkpoint-250"
SCHEMA_VARIANT_C4 = "descriptive"
CONDITION_NAMES = {1: "base_direct", 3: "finetuned_direct", 4: "finetuned_bridge"}
BRIDGE_CONDITIONS = (4,)


# ---------------------------------------------------------------- frozen subset

def frozen_subset(out: Path, size: int, seed: int) -> tuple[list[dict], dict]:
    """Select (or reload) the frozen ID subset. Never touches data/processed/."""
    pool, _ = load_manifest_split(ROOT / "data" / "processed" / DATASET, "test", SOURCE_MODE)
    index = {row["example_id"]: row for row in pool}
    path = out / "frozen200_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("size") != size or manifest.get("seed") != seed:
            raise ValueError(f"Existing frozen manifest disagrees with requested size/seed: {path}")
        missing = [i for i in manifest["ids"] if i not in index]
        if missing:
            raise ValueError(f"Frozen manifest references IDs absent from the test pool: {missing[:5]}")
        print(f"[Subset] Reusing frozen {manifest['size']} IDs sha256={manifest['ids_sha256'][:16]}", flush=True)
        return [index[i] for i in manifest["ids"]], manifest
    if size > len(pool):
        raise ValueError(f"Requested {size} examples but test_{SOURCE_MODE} holds {len(pool)}")
    rows = stratified_sample(pool, size, DATASET, seed)
    manifest = _manifest(DATASET, "test", f"frozen{size}", seed, rows) | {
        "source_manifest": f"test_{SOURCE_MODE}.json",
        "source_size": len(pool),
        "selection": f"data.stratified_sample(size={size}, seed={seed}) over test_{SOURCE_MODE} rows",
    }
    write_json(path, manifest)
    print(f"[Subset] Selected {size} IDs sha256={manifest['ids_sha256'][:16]} -> {path}", flush=True)
    return rows, manifest


# ---------------------------------------------------------------- direct path (C1, C3)

def run_direct(condition: int, runner, rows: list[dict], out: Path, cfg: dict) -> list[dict]:
    from ..prompts import direct_prompt_answer_only

    results: list[dict] = []
    started = time.perf_counter()
    print(f"[Eval] C{condition} direct answer-only examples={len(rows)}", flush=True)
    for n, row in enumerate(rows, 1):
        t0 = time.perf_counter()
        item = PredictionRow(
            dataset=DATASET, run_id=out.name, example_id=row["example_id"], condition=condition,
            gold_label=row["label"], qdep=row.get("QDep"), maxD=row.get("maxD"),
            NFact=row.get("NFact"), NRule=row.get("NRule"),
        )
        try:
            g = runner.generate(direct_prompt_answer_only(row), False)
            item.raw_model_response = g["raw"]
            item.latency_ms = g["latency_ms"]
            item.token_count = g["input_tokens"] + g["output_tokens"]
            item.predicted_label = g["label"]
            item.error_category = None if g["label"] else ("empty_output" if not g["raw"] else "malformed_direct_output")
            item.correctness = item.predicted_label == item.gold_label
            if not item.correctness and item.error_category is None:
                item.error_category = "incorrect_direct_llm_prediction"
        except Exception as exc:
            item.error_category = "model_failure"
            item.raw_model_response = (item.raw_model_response or "") + f"\n[EXCEPTION] {type(exc).__name__}: {exc}"
        item.timing["total_ms"] = (time.perf_counter() - t0) * 1000
        results.append(item.to_dict())
        if n % 20 == 0 or n == len(rows):
            write_jsonl(out / "predictions.jsonl", results)
            elapsed = time.perf_counter() - started
            print(f"[Eval] C{condition} {n}/{len(rows)} {100*n/len(rows):.0f}% "
                  f"elapsed={elapsed/60:.1f}m ETA={elapsed/n*(len(rows)-n)/60:.1f}m", flush=True)
    write_jsonl(out / "predictions.jsonl", results)
    return results


# ---------------------------------------------------------------- bridge path (C4)

def run_bridge(condition: int, runner, rows: list[dict], out: Path, cfg: dict) -> list[dict]:
    from .. import prompts
    from ..evaluation import evaluate_rows

    previous = prompts.ACTIVE_SCHEMA_VARIANT
    prompts.ACTIVE_SCHEMA_VARIANT = SCHEMA_VARIANT_C4
    try:
        evaluate_rows(DATASET, condition, f"frozen{len(rows)}", rows, runner, out, out.name, cfg)
    finally:
        prompts.ACTIVE_SCHEMA_VARIANT = previous
    return read_jsonl(out / "predictions.jsonl")


# ---------------------------------------------------------------- metrics

def _f1(gold: list[str], pred: list[str]) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score
    macro = f1_score(gold, pred, average="macro", labels=list(LABELS), zero_division=0)
    per = f1_score(gold, pred, average=None, labels=list(LABELS), zero_division=0)
    return float(macro), {label: float(score) for label, score in zip(LABELS, per)}


def _latency(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    ordered = sorted(clean)
    return {
        "mean_ms": statistics.fmean(clean),
        "median_ms": statistics.median(clean),
        "p95_ms": ordered[max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))],
        "total_s": sum(clean) / 1000,
    }


def _counts(rows: list[dict], key: str) -> dict[str, int]:
    from collections import Counter
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def bridge_diagnostics(rows: list[dict]) -> dict:
    from collections import Counter
    n = len(rows)
    t = [r.get("translation") or {} for r in rows]

    def rate(field):
        return sum(1 for x in t if x.get(field)) / n if n else 0.0

    calls = [x.get("recursive_llm_call_count") for x in t if x.get("recursive_llm_call_count") is not None]
    lookups = sum(x.get("cache_lookups", 0) for x in t)
    hits = sum(x.get("cache_hits", 0) for x in t)
    scored = [x for x in t if x.get("exact_match") is not None]
    stage_means = {}
    for stage in ("parser_ms", "compiler_ms", "z3_ms", "solver_ms", "selector_ms", "subparser_ms"):
        vals = [r["timing"][stage] for r in rows if stage in (r.get("timing") or {})]
        if vals:
            stage_means[stage] = statistics.fmean(vals)
    return {
        "schema_valid_rate": rate("schema_valid"),
        "parser_valid_rate": rate("parser_valid"),
        "ast_valid_rate": rate("ast_valid"),
        "compile_success_rate": rate("compile_success"),
        "executable_rate": rate("executable"),
        "parser_status": _counts(rows, "parser_status"),
        "compiler_status": _counts(rows, "compiler_status"),
        "solver_status": _counts(rows, "solver_status"),
        "semantic_equivalence": dict(sorted(Counter(str(x.get("semantic_equivalence")) for x in t).items())),
        "exact_match_rate": (sum(1 for x in scored if x.get("exact_match")) / len(scored)) if scored else None,
        "normalized_logic_match_rate": (sum(1 for x in scored if x.get("normalized_logic_match")) / len(scored)) if scored else None,
        "llm_calls": {
            "total": sum(calls),
            "mean": statistics.fmean(calls) if calls else None,
            "max": max(calls) if calls else None,
        },
        "cache_hit_rate": hits / lookups if lookups else None,
        "stage_ms_mean": stage_means,
    }


def summarize(condition: int, rows: list[dict], wall_s: float) -> dict:
    n = len(rows)
    gold = [r["gold_label"] for r in rows]
    raw_pred = [r["predicted_label"] for r in rows]
    scored = [p if p in LABELS else "INVALID" for p in raw_pred]
    valid = sum(1 for p in raw_pred if p in LABELS)
    correct = sum(1 for g, p in zip(gold, scored) if g == p)
    macro, per_class = _f1(gold, scored)
    valid_pairs = [(g, p) for g, p in zip(gold, raw_pred) if p in LABELS]
    out = {
        "condition": condition,
        "condition_name": CONDITION_NAMES[condition],
        "n": n,
        "valid_output_count": valid,
        "valid_output_rate": valid / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "accuracy_on_valid": (sum(1 for g, p in valid_pairs if g == p) / len(valid_pairs)) if valid_pairs else None,
        "macro_f1": macro,
        "classwise_f1": per_class,
        "gold_distribution": _counts(rows, "gold_label"),
        "prediction_distribution": dict(sorted({str(p): scored.count(p) for p in set(scored)}.items())),
        "error_categories": _counts(rows, "error_category"),
        "latency": _latency([r.get("latency_ms") for r in rows]),
        "total_ms": _latency([(r.get("timing") or {}).get("total_ms") for r in rows]),
        "tokens": {
            "total": sum(r.get("token_count") or 0 for r in rows),
            "mean": statistics.fmean([r.get("token_count") or 0 for r in rows]) if n else None,
        },
        "wall_clock_s": wall_s,
    }
    if condition in BRIDGE_CONDITIONS:
        out["bridge_diagnostics"] = bridge_diagnostics(rows)
    return out


def write_comparison(out: Path, summaries: dict[int, dict], manifest: dict) -> None:
    import csv
    write_json(out / "comparison.json", {
        "dataset": DATASET,
        "frozen_subset": {k: manifest[k] for k in ("size", "seed", "ids_sha256", "selection", "strata")},
        "conditions_run": sorted(summaries),
        "c2_excluded": "C2 (base + bridge) intentionally not run at 200",
        "summaries": {str(k): v for k, v in sorted(summaries.items())},
    })
    fields = ["condition", "condition_name", "n", "valid_output_rate", "accuracy", "accuracy_on_valid",
              "macro_f1", *[f"f1_{label.lower()}" for label in LABELS], "executable_rate",
              "compile_success_rate", "parser_valid_rate", "exact_match_rate", "mean_latency_ms",
              "p95_latency_ms", "mean_total_ms", "mean_tokens", "wall_clock_s"]
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, s in sorted(summaries.items()):
            d = s.get("bridge_diagnostics", {})
            writer.writerow({
                "condition": condition, "condition_name": s["condition_name"], "n": s["n"],
                "valid_output_rate": round(s["valid_output_rate"], 4),
                "accuracy": round(s["accuracy"], 4),
                "accuracy_on_valid": None if s["accuracy_on_valid"] is None else round(s["accuracy_on_valid"], 4),
                "macro_f1": round(s["macro_f1"], 4),
                **{f"f1_{label.lower()}": round(s["classwise_f1"][label], 4) for label in LABELS},
                "executable_rate": None if not d else round(d["executable_rate"], 4),
                "compile_success_rate": None if not d else round(d["compile_success_rate"], 4),
                "parser_valid_rate": None if not d else round(d["parser_valid_rate"], 4),
                "exact_match_rate": None if not d or d["exact_match_rate"] is None else round(d["exact_match_rate"], 4),
                "mean_latency_ms": round(s["latency"].get("mean_ms", 0)),
                "p95_latency_ms": round(s["latency"].get("p95_ms", 0)),
                "mean_total_ms": round(s["total_ms"].get("mean_ms", 0)),
                "mean_tokens": round(s["tokens"]["mean"] or 0),
                "wall_clock_s": round(s["wall_clock_s"], 1),
            })


# ---------------------------------------------------------------- driver

def condition_dir(out: Path, condition: int) -> Path:
    path = out / f"c{condition}_{CONDITION_NAMES[condition]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def already_done(path: Path, expected: int) -> list[dict] | None:
    target = path / "predictions.jsonl"
    if not target.exists():
        return None
    rows = read_jsonl(target)
    return rows if len(rows) == expected else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Final ProofWriter 200-sample evaluation (C1, C3, C4)")
    ap.add_argument("--run-id", default="final200_proofwriter")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", default="1,3,4", help="comma-separated subset of 1,3,4")
    ap.add_argument("--force", action="store_true", help="re-run conditions that already have complete predictions")
    args = ap.parse_args()

    conditions = [int(c) for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITION_NAMES]
    if bad:
        raise SystemExit(f"Unsupported conditions {bad}; this runner covers 1, 3 and 4 only (C2 is excluded by design)")

    cfg = resolved_config(DATASET)
    seed_everything(args.seed)
    out = ROOT / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = frozen_subset(out, args.size, args.seed)

    if not ADAPTER.exists() and any(c in (3, 4) for c in conditions):
        raise FileNotFoundError(f"Required adapter checkpoint not found: {ADAPTER}")

    write_json(out / "config.json", cfg)
    write_json(out / "metadata.json", runtime_metadata(cfg) | {
        "dataset": DATASET, "mode": f"frozen{args.size}", "runner": "scripts.final_200",
        "conditions": conditions, "c2_excluded": True,
        "evaluation_size": len(rows), "seed": args.seed,
        "base_model": cfg["model"]["id"], "base_revision": cfg["model"].get("revision"),
        "adapter_checkpoint": str(ADAPTER),
        "direct_prompt_variant": "direct_answer_only",
        "schema_variant_c4": SCHEMA_VARIANT_C4,
        "frozen_bridge_id": FROZEN_BRIDGE_ID,
        "frozen_subset_sha256": manifest["ids_sha256"],
        "test_ids": manifest["ids"],
    })

    from ..modeling import ModelRunner
    summaries: dict[int, dict] = {}
    runner = None
    loaded_adapter = "<unset>"

    try:
        for condition in conditions:
            path = condition_dir(out, condition)
            cached = None if args.force else already_done(path, len(rows))
            if cached is not None:
                print(f"[Eval] C{condition} already complete ({len(cached)} rows) -- skipping (use --force to redo)", flush=True)
                summaries[condition] = summarize(condition, cached, 0.0)
                write_json(path / "summary.json", summaries[condition])
                continue

            wanted = ADAPTER if condition in (3, 4) else None
            if runner is None or loaded_adapter != wanted:
                if runner is not None:
                    del runner
                    runner = None
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                seed_everything(args.seed)
                runner = ModelRunner(cfg, wanted)
                loaded_adapter = wanted

            seed_everything(args.seed)
            write_json(path / "metadata.json", {
                "dataset": DATASET, "condition": condition, "condition_name": CONDITION_NAMES[condition],
                "mode": f"frozen{args.size}", "evaluation_size": len(rows),
                "base_model": cfg["model"]["id"], "base_revision": cfg["model"].get("revision"),
                "adapter_checkpoint": str(wanted) if wanted else None,
                "direct_prompt_variant": None if condition in BRIDGE_CONDITIONS else "direct_answer_only",
                "schema_variant": SCHEMA_VARIANT_C4 if condition in BRIDGE_CONDITIONS else None,
                "frozen_bridge_id": FROZEN_BRIDGE_ID if condition in BRIDGE_CONDITIONS else None,
                "frozen_subset_sha256": manifest["ids_sha256"],
                "test_ids": manifest["ids"],
            })
            started = time.perf_counter()
            produced = (run_bridge if condition in BRIDGE_CONDITIONS else run_direct)(condition, runner, rows, path, cfg)
            wall = time.perf_counter() - started
            summaries[condition] = summarize(condition, produced, wall)
            write_json(path / "summary.json", summaries[condition])
            s = summaries[condition]
            print(f"[Eval] C{condition} done wall={wall/60:.1f}m valid={s['valid_output_rate']:.1%} "
                  f"acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.4f}", flush=True)
            write_comparison(out, summaries, manifest)
    finally:
        if summaries:
            write_comparison(out, summaries, manifest)

    print(f"\n[Final] {out}")
    print(f"[Final] frozen subset sha256={manifest['ids_sha256']}")
    for condition, s in sorted(summaries.items()):
        d = s.get("bridge_diagnostics", {})
        extra = f" exec={d['executable_rate']:.1%}" if d else ""
        print(f"  C{condition} {s['condition_name']:<18} valid={s['valid_output_rate']:.1%} "
              f"acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.4f}{extra}")


if __name__ == "__main__":
    main()
