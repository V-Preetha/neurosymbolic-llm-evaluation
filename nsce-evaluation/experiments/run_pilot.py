"""Frozen 72-example ProofWriter development pilot on Qwen3-8B.

Runs LLM-only, self-check, and NS-bridge conditions in a single pass, loading
Qwen3-8B exactly once and reusing the shared initial prediction across all three
conditions (self-check and the NS bridge both build on the same initial answer
rather than re-generating it). Gold labels are read only for post-hoc scoring,
after all three conditions' predictions have already been produced -- never
passed into any prompt-rendering or verification call (see src/io_guard.py and
tests/test_field_whitelisting.py for the enforced guarantee).

Does not touch the 360-example held-out test split.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import ExperimentConfig, PROJECT_ROOT
from src.models.qwen import QwenRunner
from src.nsce.feedback import render_feedback
from src.nsce.verifier import verify_record
from src.prompts import (
    PROOFWRITER_LLM_ONLY_TEMPLATE,
    PROOFWRITER_SELF_CHECK_TEMPLATE,
    load_template,
    render_proofwriter_prompt,
    render_self_check_prompt,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_qwen8b.json"
PILOT_PATH = PROJECT_ROOT / "data" / "revised_study" / "proofwriter_dev_pilot.jsonl"
FEEDBACK_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "proofwriter_nsce_feedback.txt"
OUT_DIR = PROJECT_ROOT / "reports" / "revised_study" / "pilot"
LABELS = ("true", "false", "unknown")
LABEL_DISPLAY = {"true": "TRUE/Supported", "false": "FALSE/Contradicted", "unknown": "UNKNOWN"}
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42


def load_pilot_records() -> list[dict]:
    records = []
    with PILOT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_pilot() -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    initial_template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
    self_check_template = load_template(PROOFWRITER_SELF_CHECK_TEMPLATE)
    feedback_template = load_template(FEEDBACK_TEMPLATE_PATH)
    records = load_pilot_records()
    assert len(records) == 72, f"expected frozen 72-example pilot, got {len(records)}"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_free_before = torch.cuda.mem_get_info()[0] / (1024 ** 2)

    load_started = time.perf_counter()
    runner = QwenRunner(
        model_id=config.model_id,
        seed=config.seed,
        max_new_tokens=config.max_new_tokens,
        quantization=config.quantization,
    )
    model_load_time_s = round(time.perf_counter() - load_started, 3)

    llm_only_records: list[dict] = []
    self_check_records: list[dict] = []
    ns_bridge_records: list[dict] = []
    vram_trace: list[dict] = []

    calls_initial = 0
    calls_self_check_verification = 0
    calls_ns_bridge_correction = 0

    run_started = time.perf_counter()
    for index, record in enumerate(records):
        example_id = record["sample_id"]
        depth = record["metadata"]["reasoning_depth"]

        # --- Shared initial generation (used by all three conditions) ---
        initial_prompt = render_proofwriter_prompt(record, initial_template)
        initial = runner.generate_label(initial_prompt)
        calls_initial += 1

        # A. LLM-only
        llm_only_records.append({
            "example_id": example_id,
            "method": "llm_only",
            "reasoning_depth": depth,
            "initial_prediction": initial.prediction,
            "final_prediction": initial.prediction,
            "parse_success": initial.prediction is not None,
            "raw_output": initial.raw_text,
            "latency_ms": initial.latency_ms,
            "input_tokens": initial.input_tokens,
            "output_tokens": initial.output_tokens,
        })

        # B. Self-check (reuses the shared initial; one additional generation)
        if initial.prediction is not None:
            verification_prompt = render_self_check_prompt(record, initial.prediction, self_check_template)
            verification = runner.generate_label(verification_prompt)
            calls_self_check_verification += 1
            final_sc = verification.prediction
            self_check_changed = final_sc is not None and final_sc != initial.prediction
            sc_verification_raw = verification.raw_text
            sc_verification_latency = verification.latency_ms
            sc_parse_success = final_sc is not None
        else:
            final_sc = None
            self_check_changed = False
            sc_verification_raw = None
            sc_verification_latency = 0.0
            sc_parse_success = False

        self_check_records.append({
            "example_id": example_id,
            "method": "self_check",
            "reasoning_depth": depth,
            "initial_prediction": initial.prediction,
            "final_prediction": final_sc,
            "parse_success": sc_parse_success,
            "self_check_changed": self_check_changed,
            "initial_raw_output": initial.raw_text,
            "verification_raw_output": sc_verification_raw,
            "initial_latency_ms": initial.latency_ms,
            "verification_latency_ms": sc_verification_latency,
            "latency_ms": round(initial.latency_ms + sc_verification_latency, 3),
        })

        # C. NS bridge (reuses the shared initial; correction only on conflict)
        verification_result = verify_record(record)
        if verification_result.reasoning is None:
            bridge_action = "ABSTAIN"
            abstain_reason = "translation_failure"
            symbolic_verdict = None
            proof_trace: list[str] = []
            final_bridge = initial.prediction
            correction_raw = None
            correction_latency = 0.0
            bridge_parse_success = initial.prediction is not None
        else:
            reasoning = verification_result.reasoning
            symbolic_verdict = reasoning.prediction
            proof_trace = list(reasoning.proof_trace)
            conflict = initial.prediction != symbolic_verdict
            if conflict:
                feedback_prompt = render_feedback(record, initial.prediction or "Unknown", reasoning, feedback_template)
                correction = runner.generate_label(feedback_prompt)
                calls_ns_bridge_correction += 1
                final_bridge = correction.prediction
                correction_raw = correction.raw_text
                correction_latency = correction.latency_ms
                bridge_action = "CORRECT"
                abstain_reason = None
                bridge_parse_success = final_bridge is not None
            else:
                final_bridge = initial.prediction
                correction_raw = None
                correction_latency = 0.0
                bridge_action = "RETAIN"
                abstain_reason = None
                bridge_parse_success = initial.prediction is not None

        ns_bridge_records.append({
            "example_id": example_id,
            "method": "ns_bridge",
            "reasoning_depth": depth,
            "initial_prediction": initial.prediction,
            "symbolic_verdict": symbolic_verdict,
            "final_prediction": final_bridge,
            "parse_success": bridge_parse_success,
            "bridge_action": bridge_action,
            "abstain_reason": abstain_reason,
            "proof_trace": proof_trace,
            "parser_error": verification_result.parser_error,
            "initial_raw_output": initial.raw_text,
            "correction_raw_output": correction_raw,
            "initial_latency_ms": initial.latency_ms,
            "symbolic_latency_ms": verification_result.latency_ms,
            "correction_latency_ms": correction_latency,
            "latency_ms": round(initial.latency_ms + verification_result.latency_ms + correction_latency, 3),
        })

        if index % 10 == 0 or index == len(records) - 1:
            vram_trace.append({
                "index": index,
                "example_id": example_id,
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 1),
            })

    total_run_time_s = round(time.perf_counter() - run_started, 3)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

    # --- Attach gold labels now, strictly for scoring (evaluator step, post-inference) ---
    gold_map = {record["sample_id"]: record["ground_truth"]["label"] for record in records}
    for rec_list in (llm_only_records, self_check_records, ns_bridge_records):
        for rec in rec_list:
            rec["gold_label"] = gold_map[rec["example_id"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "llm_only.jsonl", llm_only_records)
    write_jsonl(OUT_DIR / "self_check.jsonl", self_check_records)
    write_jsonl(OUT_DIR / "ns_bridge.jsonl", ns_bridge_records)

    runtime_info = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "quantization": runner.runtime_info["quantization"],
        "torch_version": runner.runtime_info["torch_version"],
        "gpu_name": runner.runtime_info["gpu_name"],
        "vram_free_before_load_mb": round(vram_free_before, 1),
        "peak_vram_mb": peak_vram_mb,
        "model_load_time_s": model_load_time_s,
        "total_run_time_s": total_run_time_s,
        "pilot_size": len(records),
        "generation_calls": {
            "shared_initial": calls_initial,
            "self_check_verification": calls_self_check_verification,
            "ns_bridge_correction": calls_ns_bridge_correction,
            "total": calls_initial + calls_self_check_verification + calls_ns_bridge_correction,
        },
        "vram_trace": vram_trace,
    }
    (OUT_DIR / "runtime_info.json").write_text(json.dumps(runtime_info, indent=2), encoding="utf-8")

    metrics = compute_all_metrics(llm_only_records, self_check_records, ns_bridge_records)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_report(OUT_DIR / "pilot_report.md", runtime_info, metrics)

    print(json.dumps({"status": "ok", "pilot_size": len(records), "runtime": runtime_info, "metrics_summary": {
        "llm_only_accuracy": metrics["classification"]["llm_only"]["accuracy"],
        "self_check_accuracy": metrics["classification"]["self_check"]["accuracy"],
        "ns_bridge_accuracy": metrics["classification"]["ns_bridge"]["accuracy"],
        "symbolic_verdict_accuracy": metrics["symbolic_validity"]["accuracy"],
    }}, indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(rows: list[dict], pred_key: str = "final_prediction") -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r[pred_key] == r["gold_label"])
    accuracy = correct / total if total else 0.0

    per_class = {}
    f1s = []
    supports = []
    for label in LABELS:
        tp = sum(1 for r in rows if r[pred_key] == label and r["gold_label"] == label)
        fp = sum(1 for r in rows if r[pred_key] == label and r["gold_label"] != label)
        fn = sum(1 for r in rows if r[pred_key] != label and r["gold_label"] == label)
        support = sum(1 for r in rows if r["gold_label"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
        f1s.append(f1)
        supports.append(support)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = (
        sum(f1 * support for f1, support in zip(f1s, supports)) / sum(supports)
        if sum(supports) else 0.0
    )

    confusion = {gold: {pred: 0 for pred in (*LABELS, "none")} for gold in LABELS}
    unparsed = 0
    for r in rows:
        pred = r[pred_key] if r[pred_key] in LABELS else "none"
        if pred == "none":
            unparsed += 1
        confusion[r["gold_label"]][pred] += 1

    return {
        "n": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "unparsed_predictions": unparsed,
    }


def depth_breakdown(rows: list[dict], pred_key: str = "final_prediction") -> dict:
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_depth[r["reasoning_depth"]].append(r)
    return {
        str(depth): classification_metrics(depth_rows, pred_key)
        for depth, depth_rows in sorted(by_depth.items())
    }


def self_check_change_analysis(rows: list[dict]) -> dict:
    changed = wrong_to_correct = correct_to_wrong = wrong_to_different_wrong = unchanged_wrong = 0
    for r in rows:
        initial, final, gold = r["initial_prediction"], r["final_prediction"], r["gold_label"]
        if r["self_check_changed"]:
            changed += 1
            if initial != gold and final == gold:
                wrong_to_correct += 1
            elif initial == gold and final != gold:
                correct_to_wrong += 1
            elif initial != gold and final != gold and final != initial:
                wrong_to_different_wrong += 1
        else:
            if initial != gold:
                unchanged_wrong += 1
    return {
        "changed": changed,
        "changed_pct": round(100 * changed / len(rows), 2) if rows else 0.0,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_different_wrong": wrong_to_different_wrong,
        "unchanged_wrong": unchanged_wrong,
    }


def bridge_action_analysis(rows: list[dict]) -> dict:
    retain = [r for r in rows if r["bridge_action"] == "RETAIN"]
    correct_action = [r for r in rows if r["bridge_action"] == "CORRECT"]
    abstain = [r for r in rows if r["bridge_action"] == "ABSTAIN"]

    correct_wrong_to_correct = correct_correct_to_wrong = correct_wrong_to_different_wrong = 0
    for r in correct_action:
        initial, final, gold = r["initial_prediction"], r["final_prediction"], r["gold_label"]
        if initial != gold and final == gold:
            correct_wrong_to_correct += 1
        elif initial == gold and final != gold:
            correct_correct_to_wrong += 1
        elif initial != gold and final != gold and final != initial:
            correct_wrong_to_different_wrong += 1

    retain_correct = sum(1 for r in retain if r["final_prediction"] == r["gold_label"])
    retain_incorrect = len(retain) - retain_correct

    return {
        "retain_count": len(retain),
        "correct_count": len(correct_action),
        "abstain_count": len(abstain),
        "among_correct": {
            "wrong_qwen_to_correct_bridge": correct_wrong_to_correct,
            "correct_qwen_to_wrong_bridge": correct_correct_to_wrong,
            "wrong_qwen_to_different_wrong_bridge": correct_wrong_to_different_wrong,
        },
        "among_retain": {
            "correctly_retained": retain_correct,
            "incorrectly_retained": retain_incorrect,
        },
    }


def case_analysis(llm_only: list[dict], self_check: list[dict], ns_bridge: list[dict]) -> dict:
    llm_map = {r["example_id"]: r for r in llm_only}
    sc_map = {r["example_id"]: r for r in self_check}
    ns_map = {r["example_id"]: r for r in ns_bridge}

    case_a, case_b, case_c, case_d, case_e = [], [], [], [], []
    for example_id, llm_row in llm_map.items():
        gold = llm_row["gold_label"]
        qwen_pred = llm_row["final_prediction"]
        qwen_correct = qwen_pred == gold
        sc_final = sc_map[example_id]["final_prediction"]
        sc_correct = sc_final == gold
        ns_final = ns_map[example_id]["final_prediction"]
        ns_correct = ns_final == gold

        if not qwen_correct and ns_correct:
            case_a.append(example_id)
        if qwen_correct and not ns_correct:
            case_b.append(example_id)
        if not qwen_correct and sc_correct:
            case_c.append(example_id)
        if not qwen_correct and not sc_correct and ns_correct:
            case_d.append(example_id)
        if qwen_correct and not sc_correct:
            case_e.append(example_id)

    return {
        "case_a_qwen_wrong_bridge_correct": case_a,
        "case_b_qwen_correct_bridge_wrong": case_b,
        "case_c_qwen_wrong_selfcheck_correct": case_c,
        "case_d_qwen_wrong_selfcheck_wrong_bridge_correct": case_d,
        "case_e_qwen_correct_selfcheck_wrong": case_e,
        "counts": {
            "case_a": len(case_a), "case_b": len(case_b), "case_c": len(case_c),
            "case_d": len(case_d), "case_e": len(case_e),
        },
    }


def symbolic_validity_check(ns_bridge_rows: list[dict]) -> dict:
    evaluated = [r for r in ns_bridge_rows if r["symbolic_verdict"] is not None]
    mismatches = [r["example_id"] for r in evaluated if r["symbolic_verdict"] != r["gold_label"]]
    parser_failures = [r["example_id"] for r in ns_bridge_rows if r["bridge_action"] == "ABSTAIN"]
    accuracy = (len(evaluated) - len(mismatches)) / len(evaluated) if evaluated else None
    return {
        "n_evaluated": len(evaluated),
        "n_parser_failures": len(parser_failures),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "mismatching_example_ids": mismatches,
        "parser_failure_example_ids": parser_failures,
        "reasoner_exceptions": [],  # none observed: classify() is total over any parsed theory
        "abstain_example_ids": parser_failures,
    }


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a_map = {r["example_id"]: r["final_prediction"] == r["gold_label"] for r in rows_a}
    b_map = {r["example_id"]: r["final_prediction"] == r["gold_label"] for r in rows_b}
    both_correct = both_wrong = only_a_correct = only_b_correct = 0
    for example_id in a_map:
        a_ok, b_ok = a_map[example_id], b_map[example_id]
        if a_ok and b_ok:
            both_correct += 1
        elif not a_ok and not b_ok:
            both_wrong += 1
        elif a_ok and not b_ok:
            only_a_correct += 1
        else:
            only_b_correct += 1

    b, c = only_a_correct, only_b_correct
    n_discordant = b + c
    if n_discordant == 0:
        chi2 = 0.0
        exact_p = 1.0
    else:
        chi2 = ((abs(b - c) - 1) ** 2) / n_discordant
        # exact two-sided binomial p-value, min(b,c) as k, n=b+c, p=0.5
        k = min(b, c)
        n = n_discordant
        cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        exact_p = min(1.0, 2 * cumulative)

    return {
        "contingency": {
            "both_correct": both_correct, "both_wrong": both_wrong,
            "only_first_correct": only_a_correct, "only_second_correct": only_b_correct,
        },
        "mcnemar_chi2_continuity_corrected": round(chi2, 4),
        "exact_two_sided_p_value": round(exact_p, 4),
        "note": "n=72 pilot -- treat as preliminary, not confirmatory.",
    }


def bootstrap_accuracy_diff(rows_a: list[dict], rows_b: list[dict]) -> dict:
    ids = [r["example_id"] for r in rows_a]
    a_correct = {r["example_id"]: r["final_prediction"] == r["gold_label"] for r in rows_a}
    b_correct = {r["example_id"]: r["final_prediction"] == r["gold_label"] for r in rows_b}
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = []
    n = len(ids)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        acc_a = sum(a_correct[i] for i in sample) / n
        acc_b = sum(b_correct[i] for i in sample) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    lo = diffs[int(0.025 * BOOTSTRAP_ITERATIONS)]
    hi = diffs[int(0.975 * BOOTSTRAP_ITERATIONS) - 1]
    observed = (
        sum(b_correct[i] for i in ids) / n - sum(a_correct[i] for i in ids) / n
    )
    return {
        "observed_accuracy_diff_b_minus_a": round(observed, 4),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "note": "n=72 pilot -- treat as preliminary, not confirmatory.",
    }


def compute_all_metrics(llm_only: list[dict], self_check: list[dict], ns_bridge: list[dict]) -> dict:
    classification = {
        "llm_only": classification_metrics(llm_only),
        "self_check": classification_metrics(self_check),
        "ns_bridge": classification_metrics(ns_bridge),
    }
    depth = {
        "llm_only": depth_breakdown(llm_only),
        "self_check": depth_breakdown(self_check),
        "ns_bridge": depth_breakdown(ns_bridge),
    }
    return {
        "classification": classification,
        "by_reasoning_depth": depth,
        "self_check_change_analysis": self_check_change_analysis(self_check),
        "bridge_action_analysis": bridge_action_analysis(ns_bridge),
        "case_analysis": case_analysis(llm_only, self_check, ns_bridge),
        "symbolic_validity": symbolic_validity_check(ns_bridge),
        "paired_statistics": {
            "llm_only_vs_self_check": {
                "mcnemar": mcnemar(llm_only, self_check),
                "bootstrap": bootstrap_accuracy_diff(llm_only, self_check),
            },
            "llm_only_vs_ns_bridge": {
                "mcnemar": mcnemar(llm_only, ns_bridge),
                "bootstrap": bootstrap_accuracy_diff(llm_only, ns_bridge),
            },
        },
        "latency": latency_summary(llm_only, self_check, ns_bridge),
    }


def latency_summary(llm_only: list[dict], self_check: list[dict], ns_bridge: list[dict]) -> dict:
    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "mean_llm_only_latency_ms": mean([r["latency_ms"] for r in llm_only]),
        "mean_self_check_latency_ms": mean([r["latency_ms"] for r in self_check]),
        "mean_ns_bridge_latency_ms": mean([r["latency_ms"] for r in ns_bridge]),
        "mean_symbolic_reasoning_latency_ms": mean([r["symbolic_latency_ms"] for r in ns_bridge]),
    }


def write_report(path: Path, runtime_info: dict, metrics: dict) -> None:
    md = []
    md.append("# Frozen 72-Example ProofWriter Development Pilot -- Qwen3-8B\n")
    md.append(f"Model: `{runtime_info['model_id']}` (revision `{runtime_info['model_revision']}`), "
              f"quantization `{runtime_info['quantization']}`, GPU `{runtime_info['gpu_name']}`.\n")
    md.append(f"Pilot size: {runtime_info['pilot_size']}. Total run time: {runtime_info['total_run_time_s']} s "
              f"(model load {runtime_info['model_load_time_s']} s). Peak VRAM: {runtime_info['peak_vram_mb']} MB.\n")
    calls = runtime_info["generation_calls"]
    md.append(f"Qwen generation calls: shared initial={calls['shared_initial']}, "
              f"self-check verification={calls['self_check_verification']}, "
              f"NS-bridge correction={calls['ns_bridge_correction']} (only on conflict), "
              f"total={calls['total']}.\n")

    md.append("## Primary metrics\n")
    md.append("| Method | Accuracy | Macro F1 | Weighted F1 |")
    md.append("|---|---:|---:|---:|")
    for method in ("llm_only", "self_check", "ns_bridge"):
        c = metrics["classification"][method]
        md.append(f"| {method} | {c['accuracy']} | {c['macro_f1']} | {c['weighted_f1']} |")
    md.append("")

    md.append("## Per-class metrics\n")
    for method in ("llm_only", "self_check", "ns_bridge"):
        md.append(f"### {method}\n")
        md.append("| Label | Precision | Recall | F1 | Support |")
        md.append("|---|---:|---:|---:|---:|")
        for label in LABELS:
            pc = metrics["classification"][method]["per_class"][label]
            md.append(f"| {LABEL_DISPLAY[label]} | {pc['precision']} | {pc['recall']} | {pc['f1']} | {pc['support']} |")
        md.append("")

    md.append("## Confusion matrices\n")
    for method in ("llm_only", "self_check", "ns_bridge"):
        md.append(f"### {method} (rows=gold, cols=predicted)\n")
        cm = metrics["classification"][method]["confusion_matrix"]
        header = "| gold\\pred | " + " | ".join(LABELS) + " | none |"
        md.append(header)
        md.append("|" + "---|" * (len(LABELS) + 2))
        for gold in LABELS:
            row = cm[gold]
            md.append(f"| {gold} | " + " | ".join(str(row[p]) for p in LABELS) + f" | {row['none']} |")
        md.append("")

    md.append("## Reasoning-depth breakdown (accuracy)\n")
    md.append("| Depth | LLM-only | Self-check | NS Bridge |")
    md.append("|---|---:|---:|---:|")
    for depth in map(str, range(6)):
        row = [metrics["by_reasoning_depth"][m].get(depth, {}).get("accuracy", "-") for m in ("llm_only", "self_check", "ns_bridge")]
        md.append(f"| {depth} | {row[0]} | {row[1]} | {row[2]} |")
    md.append("")
    md.append("### Reasoning-depth breakdown (macro F1)\n")
    md.append("| Depth | LLM-only | Self-check | NS Bridge |")
    md.append("|---|---:|---:|---:|")
    for depth in map(str, range(6)):
        row = [metrics["by_reasoning_depth"][m].get(depth, {}).get("macro_f1", "-") for m in ("llm_only", "self_check", "ns_bridge")]
        md.append(f"| {depth} | {row[0]} | {row[1]} | {row[2]} |")
    md.append("")

    sc = metrics["self_check_change_analysis"]
    md.append("## Self-check change analysis\n")
    md.append(f"- Changed: {sc['changed']} ({sc['changed_pct']}%)")
    md.append(f"- Wrong -> correct: {sc['wrong_to_correct']}")
    md.append(f"- Correct -> wrong: {sc['correct_to_wrong']}")
    md.append(f"- Wrong -> different wrong: {sc['wrong_to_different_wrong']}")
    md.append(f"- Unchanged wrong: {sc['unchanged_wrong']}\n")

    ba = metrics["bridge_action_analysis"]
    md.append("## Symbolic bridge action analysis\n")
    md.append(f"- RETAIN: {ba['retain_count']}, CORRECT: {ba['correct_count']}, ABSTAIN: {ba['abstain_count']}")
    md.append(f"- Among CORRECT -- wrong Qwen -> correct bridge: {ba['among_correct']['wrong_qwen_to_correct_bridge']}, "
              f"correct Qwen -> wrong bridge: {ba['among_correct']['correct_qwen_to_wrong_bridge']}, "
              f"wrong Qwen -> different wrong bridge: {ba['among_correct']['wrong_qwen_to_different_wrong_bridge']}")
    md.append(f"- Among RETAIN -- correctly retained: {ba['among_retain']['correctly_retained']}, "
              f"incorrectly retained: {ba['among_retain']['incorrectly_retained']}\n")

    ca = metrics["case_analysis"]["counts"]
    md.append("## Case A-E counts\n")
    md.append(f"- Case A (Qwen wrong -> bridge correct): {ca['case_a']}")
    md.append(f"- Case B (Qwen correct -> bridge wrong): {ca['case_b']}")
    md.append(f"- Case C (Qwen wrong -> self-check correct): {ca['case_c']}")
    md.append(f"- Case D (Qwen wrong, self-check still wrong -> bridge correct): {ca['case_d']}")
    md.append(f"- Case E (Qwen correct -> self-check wrong): {ca['case_e']}")
    md.append("\nExact example IDs are in `metrics.json` -> `case_analysis`.\n")

    sv = metrics["symbolic_validity"]
    md.append("## Symbolic verdict validity check\n")
    md.append(f"- Evaluated: {sv['n_evaluated']}/{runtime_info['pilot_size']}, accuracy: {sv['accuracy']}")
    md.append(f"- Parser failures (ABSTAIN): {sv['n_parser_failures']}")
    md.append(f"- Mismatching example IDs: {sv['mismatching_example_ids']}")
    md.append(f"- Parser failure example IDs: {sv['parser_failure_example_ids']}\n")

    lat = metrics["latency"]
    md.append("## Latency\n")
    md.append(f"- Mean LLM-only latency/example: {lat['mean_llm_only_latency_ms']} ms")
    md.append(f"- Mean self-check latency/example: {lat['mean_self_check_latency_ms']} ms")
    md.append(f"- Mean NS-bridge latency/example: {lat['mean_ns_bridge_latency_ms']} ms")
    md.append(f"- Mean symbolic-reasoning latency/example: {lat['mean_symbolic_reasoning_latency_ms']} ms")
    md.append(f"- Peak VRAM: {runtime_info['peak_vram_mb']} MB\n")

    ps = metrics["paired_statistics"]
    md.append("## Preliminary paired statistics (pilot, n=72 -- not confirmatory)\n")
    for pair_name, label in (("llm_only_vs_self_check", "LLM-only vs Self-check"), ("llm_only_vs_ns_bridge", "LLM-only vs NS Bridge")):
        mc = ps[pair_name]["mcnemar"]
        bs = ps[pair_name]["bootstrap"]
        md.append(f"### {label}\n")
        md.append(f"- Contingency: {mc['contingency']}")
        md.append(f"- McNemar chi2 (continuity-corrected): {mc['mcnemar_chi2_continuity_corrected']}, exact two-sided p={mc['exact_two_sided_p_value']}")
        md.append(f"- Bootstrap accuracy diff (95% CI): {bs['observed_accuracy_diff_b_minus_a']} [{bs['ci_95_low']}, {bs['ci_95_high']}] ({bs['bootstrap_iterations']} resamples, seed={bs['seed']})\n")

    path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    run_pilot()
