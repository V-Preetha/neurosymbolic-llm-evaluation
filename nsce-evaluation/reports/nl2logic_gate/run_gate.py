"""Fast validation harness for official NL2Logic + Gemma-2-9B (Ollama backend).

Runs, in order, and checkpoints each result to disk immediately:
  1. the repo's own README example
  2. 3 smoke tests (universal / existential / negated-universal)
  3. a 5-item FOLIO gate (raw text only, no gold FOL/labels given to the model)

Never modifies external/nl2logic. Never touches src/folio/v2 or the T5 bridge.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[2]
NL2LOGIC_DIR = REPO_ROOT / "external" / "nl2logic"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NL2LOGIC_DIR))

import re  # noqa: E402

import z3  # noqa: E402
from ast_rl import RelationalLogic  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# Fields whose string value convert_to_z3() splices verbatim into generated
# Python/Z3 source as an identifier (Constant/Variable names, predicate names).
_IDENTIFIER_FIELDS = {"name", "adjective", "verb"}


def sanitize_identifiers(node):
    """NL2Logic's entity/predicate names can be raw multi-word phrases
    (e.g. "to be addicted to caffeine", "a bird"), which convert_to_z3()
    emits verbatim as Python identifiers -> SyntaxError. This is a code-gen
    artifact, not a semantic-parsing failure, so we rewrite names to valid
    identifiers in place (harness-side only; external/nl2logic is untouched).
    """
    if isinstance(node, BaseModel):
        for field in node.__class__.model_fields:
            value = getattr(node, field)
            if field in _IDENTIFIER_FIELDS and isinstance(value, str):
                setattr(node, field, re.sub(r"\W|^(?=\d)", "_", value.strip()) or "_x")
            else:
                sanitize_identifiers(value)
    elif isinstance(node, list):
        for item in node:
            sanitize_identifiers(item)

MODEL = "gemma2:9b-instruct-q4_K_M"
CALL_TIMEOUT_S = 90  # per-LLM-call timeout
ITEM_TIMEOUT_S = 240  # per-example (whole translation) timeout

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def with_timeout(fn, *args, timeout=ITEM_TIMEOUT_S, **kwargs):
    fut = _executor.submit(fn, *args, **kwargs)
    return fut.result(timeout=timeout)


def build_pipeline():
    return Pipeline(llm="ollama", model=MODEL, logging=False)


def try_parse(p: Pipeline, text: str):
    """Returns (ast_or_none, error_str_or_none, latency_s)."""
    t0 = time.time()
    try:
        ast = with_timeout(p.rephrase_and_parse, text, timeout=ITEM_TIMEOUT_S)
        return ast, None, time.time() - t0
    except concurrent.futures.TimeoutError:
        return None, f"TIMEOUT after {ITEM_TIMEOUT_S}s", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}", time.time() - t0


def try_z3(ast, original_text: str = "") -> tuple[bool, str | None, str | None]:
    """Returns (compiles, z3_code_or_none, error_or_none).

    `ast` is a raw Sentence node from Pipeline.parse(); it must be wrapped in
    RelationalLogic (never done by Pipeline itself) before convert_to_z3() exists.
    """
    try:
        sanitize_identifiers(ast)
        theory = RelationalLogic(original_sentence=original_text, sentences=[ast])
        code = theory.convert_to_z3()
    except Exception as e:  # noqa: BLE001
        return False, None, f"convert_to_z3 failed: {type(e).__name__}: {e}"
    try:
        ns: dict = {}
        exec(compile(code, "<z3_generated>", "exec"), ns)  # noqa: S102
        return True, code, None
    except Exception as e:  # noqa: BLE001
        return False, code, f"exec failed: {type(e).__name__}: {e}"


def checkpoint(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_official_example(p: Pipeline):
    text = "No one who does not want to be addicted to caffeine is unaware that caffeine is a drug."
    out_path = OUT_DIR / "step3_official_example.jsonl"
    if out_path.exists():
        out_path.unlink()
    ast, err, latency = try_parse(p, text)
    record = {"text": text, "error": err, "latency_s": latency}
    if ast is not None:
        record["ast"] = ast.to_dict()
        compiles, code, z3err = try_z3(ast, text)
        record["z3_compiles"] = compiles
        record["z3_code"] = code
        record["z3_error"] = z3err
    else:
        record["z3_compiles"] = False
    checkpoint(out_path, record)
    print(f"[official example] error={err} z3_compiles={record.get('z3_compiles')} latency={latency:.1f}s")
    return record


SMOKE_TESTS = [
    {"id": "smoke1", "text": "All humans are mammals.", "expected": "ForAll(x, Human(x) -> Mammal(x))"},
    {"id": "smoke2", "text": "Some doctors are researchers.", "expected": "Exists(x, Doctor(x) & Researcher(x))"},
    {"id": "smoke3", "text": "No birds are mammals.", "expected": "ForAll(x, Bird(x) -> ~Mammal(x))"},
]


def run_smoke_tests(p: Pipeline):
    out_path = OUT_DIR / "step4_smoke_tests.jsonl"
    if out_path.exists():
        out_path.unlink()
    results = []
    for item in SMOKE_TESTS:
        ast, err, latency = try_parse(p, item["text"])
        record = {**item, "error": err, "latency_s": latency}
        if ast is not None:
            record["ast"] = ast.to_dict()
            record["ast_str"] = str(ast)
            compiles, code, z3err = try_z3(ast, item["text"])
            record["z3_compiles"] = compiles
            record["z3_code"] = code
            record["z3_error"] = z3err
        else:
            record["z3_compiles"] = False
        checkpoint(out_path, record)
        print(f"[{item['id']}] '{item['text']}' -> error={err} z3_compiles={record.get('z3_compiles')} "
              f"ast_str={record.get('ast_str')!r} latency={latency:.1f}s")
        results.append(record)
    return results


# 5 items from the frozen-30 FOLIO sample (reports/folio_translation_v2/frozen30_variant_C.jsonl),
# chosen for construct diversity per the gate spec. Only premises_raw/hypothesis_raw are read;
# gold_label/gold_constructs are loaded separately, only for post-hoc scoring, never given to the model.
GATE_IDS = {
    180: "binary_relation + multi_variable + universal (+xor)",
    51: "universal implication + negation",
    203: "existential + binary_relation",
    6: "conjunction/disjunction + existential + universal",
    199: "negation + binary_relation + universal implication",
}

DEFAULT_Z3_TIMEOUT_MS = 10_000


def load_gate_items():
    frozen_path = REPO_ROOT / "reports" / "folio_translation_v2" / "frozen30_variant_C.jsonl"
    items = []
    with open(frozen_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["idx"] in GATE_IDS:
                items.append(rec)
    items.sort(key=lambda r: list(GATE_IDS).index(r["idx"]))
    return items


def compile_sentence_to_z3(ast, ns: dict):
    """Registers this sentence's declarations into `ns` and returns its z3 BoolRef expr."""
    sanitize_identifiers(ast)
    decl_code = ast.z3declaration_pass()
    exec(compile(decl_code, "<decl>", "exec"), ns)  # noqa: S102
    expr_code = ast.z3expression_pass()
    return eval(compile(expr_code, "<expr>", "eval"), ns), decl_code, expr_code  # noqa: S307


def translate_sentence(p: Pipeline, text: str) -> dict:
    ast, err, latency = try_parse(p, text)
    rec = {"text": text, "error": err, "latency_s": latency}
    if ast is None:
        rec["z3_compiles"] = False
        return rec
    rec["ast"] = ast.to_dict()
    rec["ast_str"] = str(ast)
    rec["_ast_obj"] = ast
    rec["z3_compiles"] = None  # filled in once we know if it plugs into the shared solver ns
    return rec


def run_folio_gate(p: Pipeline):
    out_path = OUT_DIR / "step5_folio_gate.jsonl"
    if out_path.exists():
        out_path.unlink()

    items = load_gate_items()
    results = []
    for rec in items:
        idx = rec["idx"]
        construct_tag = GATE_IDS[idx]
        t_item0 = time.time()
        ns: dict = {"__builtins__": __builtins__}
        exec("from z3 import *\nEntity = DeclareSort('Entity')\n", ns)  # noqa: S102

        premise_bools = []
        premise_translations = []
        translation_ok = True
        for sent in rec["premises_raw"]:
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

        hyp_tr = translate_sentence(p, rec["hypothesis_raw"])
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
            "idx": idx,
            "construct_tag": construct_tag,
            "gold_label": rec["gold_label"],
            "gold_constructs": rec.get("gold_constructs"),
            "premises_raw": rec["premises_raw"],
            "hypothesis_raw": rec["hypothesis_raw"],
            "premise_translations": premise_translations,
            "hypothesis_translation": hyp_tr,
            "translation_ok": translation_ok,
            "verdict": verdict,
            "verdict_correct": (verdict == rec["gold_label"]) if translation_ok else None,
            "solver_statuses": solver_statuses,
            "elapsed_s": time.time() - t_item0,
        }
        checkpoint(out_path, record)
        n_premise_ok = sum(1 for t in premise_translations if t.get("z3_compiles"))
        print(f"[idx={idx}] construct={construct_tag} translation_ok={translation_ok} "
              f"premises_z3_ok={n_premise_ok}/{len(premise_translations)} hyp_z3_ok={hyp_tr.get('z3_compiles')} "
              f"verdict={verdict} gold={rec['gold_label']} correct={record['verdict_correct']} "
              f"elapsed={record['elapsed_s']:.1f}s")
        results.append(record)
    return results


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = build_pipeline()
    if stage in ("all", "example"):
        run_official_example(p)
    if stage in ("all", "smoke"):
        run_smoke_tests(p)
    if stage in ("all", "gate"):
        run_folio_gate(p)
