"""ProofWriter gate + frozen-sample runner through the same NL2Logic+Gemma->Z3 bridge
used for the FOLIO gate (run_gate.py). Reuses that module's pipeline/sanitizer/Z3
helpers unchanged; only the item loader and premise/hypothesis extraction differ
(ProofWriter's facts+rules+question format vs FOLIO's premises+hypothesis format).

Two stages:
  gate  -- 5 items from the non-frozen dev pilot (data/revised_study/proofwriter_dev_pilot.jsonl)
  final -- the existing frozen 20-item sample (data/processed/proofwriter.jsonl,
           matching configs/*.json's proofwriter_phase1_samples=20), only run if gate passes
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import (  # noqa: E402
    OUT_DIR as _BASE_OUT_DIR,
    REPO_ROOT,
    DEFAULT_Z3_TIMEOUT_MS,
    build_pipeline,
    checkpoint,
    compile_sentence_to_z3,
    translate_sentence,
    Pipeline,
)

OUT_DIR = REPO_ROOT / "reports" / "nl2logic_gate"

# 5 items from the non-frozen dev pilot, chosen for construct diversity + small size
# (kept short so the gate stays fast). ProofWriter has no existential/disjunctive
# constructs by dataset design (Horn-clause-style facts + universally-quantified
# "if something is X then it Y" rules) -- diversity here is over negation /
# conjunction / rule depth / label instead.
GATE_SAMPLE_IDS = {
    "proofwriter_dev_000061": "conditional_rule only (baseline universal implication), depth0",
    "proofwriter_dev_000213": "conjunction + conditional_rule, depth0",
    "proofwriter_dev_000190": "negation + conditional_rule, depth3",
    "proofwriter_dev_000284": "conjunction + conditional_rule, depth1",
    "proofwriter_dev_000016": "negation + conjunction + conditional_rule, depth3",
}

LABEL_MAP = {"true": "True", "false": "False", "unknown": "Uncertain"}


def load_pilot_items():
    path = REPO_ROOT / "data" / "revised_study" / "proofwriter_dev_pilot.jsonl"
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["sample_id"] in GATE_SAMPLE_IDS:
                items[rec["sample_id"]] = rec
    return [items[sid] for sid in GATE_SAMPLE_IDS]


def load_frozen_items():
    path = REPO_ROOT / "data" / "processed" / "proofwriter.jsonl"
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def run_one_item(p: Pipeline, sample_id: str, premises_raw: list[str], hypothesis_raw: str,
                  gold_label: str, tag: str | None, out_path: Path):
    t_item0 = time.time()
    ns: dict = {"__builtins__": __builtins__}
    exec("from z3 import *\nEntity = DeclareSort('Entity')\n", ns)  # noqa: S102

    premise_bools = []
    premise_translations = []
    translation_ok = True
    for sent in premises_raw:
        tr = translate_sentence(p, sent)
        if tr.get("_ast_obj") is not None:
            try:
                expr, decl_code, expr_code = compile_sentence_to_z3(tr["_ast_obj"], ns)
                premise_bools.append(expr)
                tr["z3_compiles"] = True
            except Exception as e:  # noqa: BLE001
                tr["z3_compiles"] = False
                tr["z3_error"] = f"{type(e).__name__}: {e}"
                translation_ok = False
        else:
            translation_ok = False
        tr.pop("_ast_obj", None)
        premise_translations.append(tr)

    hyp_tr = translate_sentence(p, hypothesis_raw)
    hyp_bool = None
    if hyp_tr.get("_ast_obj") is not None:
        try:
            hyp_bool, decl_code, expr_code = compile_sentence_to_z3(hyp_tr["_ast_obj"], ns)
            hyp_tr["z3_compiles"] = True
        except Exception as e:  # noqa: BLE001
            hyp_tr["z3_compiles"] = False
            hyp_tr["z3_error"] = f"{type(e).__name__}: {e}"
            translation_ok = False
    else:
        translation_ok = False
    hyp_tr.pop("_ast_obj", None)

    verdict = "ABSTAIN_BRIDGE_FAILURE"
    solver_statuses = {}
    if translation_ok and hyp_bool is not None:
        try:
            solver = ns["Solver"]()
            solver.set("timeout", DEFAULT_Z3_TIMEOUT_MS)
            for b in premise_bools:
                solver.add(b)
            base_check = str(solver.check())
            solver_statuses["premises_consistent"] = base_check
            if base_check == "unsat":
                verdict = "INCONSISTENT_PREMISES"
            else:
                solver.push()
                solver.add(ns["Not"](hyp_bool))
                r_not_h = str(solver.check())
                solver_statuses["premises_and_not_hyp"] = r_not_h
                solver.pop()

                solver.push()
                solver.add(hyp_bool)
                r_h = str(solver.check())
                solver_statuses["premises_and_hyp"] = r_h
                solver.pop()

                if r_not_h == "unsat":
                    verdict = "True"
                elif r_h == "unsat":
                    verdict = "False"
                else:
                    verdict = "Uncertain"
        except Exception as e:  # noqa: BLE001
            verdict = "ABSTAIN_SOLVER_ERROR"
            solver_statuses["error"] = f"{type(e).__name__}: {e}"

    record = {
        "sample_id": sample_id,
        "construct_tag": tag,
        "gold_label": gold_label,
        "premises_raw": premises_raw,
        "hypothesis_raw": hypothesis_raw,
        "premise_translations": premise_translations,
        "hypothesis_translation": hyp_tr,
        "translation_ok": translation_ok,
        "verdict": verdict,
        "verdict_correct": (verdict == gold_label) if translation_ok else None,
        "solver_statuses": solver_statuses,
        "elapsed_s": time.time() - t_item0,
    }
    checkpoint(out_path, record)
    n_premise_ok = sum(1 for t in premise_translations if t.get("z3_compiles"))
    print(f"[{sample_id}] tag={tag} translation_ok={translation_ok} "
          f"premises_z3_ok={n_premise_ok}/{len(premise_translations)} hyp_z3_ok={hyp_tr.get('z3_compiles')} "
          f"verdict={verdict} gold={gold_label} correct={record['verdict_correct']} "
          f"elapsed={record['elapsed_s']:.1f}s")
    return record


def run_gate(p: Pipeline):
    out_path = OUT_DIR / "step_proofwriter_gate.jsonl"
    if out_path.exists():
        out_path.unlink()
    results = []
    for item in load_pilot_items():
        sid = item["sample_id"]
        premises = item["input"]["facts"] + item["input"]["rules"]
        hypothesis = item["input"]["question"]
        gold = LABEL_MAP[item["ground_truth"]["label"]]
        tag = GATE_SAMPLE_IDS[sid]
        results.append(run_one_item(p, sid, premises, hypothesis, gold, tag, out_path))
    return results


def run_final(p: Pipeline):
    out_path = OUT_DIR / "step_proofwriter_frozen20.jsonl"
    if out_path.exists():
        out_path.unlink()
    results = []
    for item in load_frozen_items():
        sid = item["sample_id"]
        premises = item["input"]["facts"] + item["input"]["rules"]
        hypothesis = item["input"]["question"]
        gold = LABEL_MAP[item["ground_truth"]["label"]]
        results.append(run_one_item(p, sid, premises, hypothesis, gold, None, out_path))
    return results


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "gate"
    p = build_pipeline()
    if stage == "gate":
        run_gate(p)
    elif stage == "final":
        run_final(p)
