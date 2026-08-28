"""Validate src/folio/{ir,schema,compiler,solver}.py against the same 18-item
FOLIO feasibility sample used by gold_fol_oracle_z3.py -- but this time routed
through the full JSON-IR pipeline (parse -> validate -> compile -> solve)
instead of hand-written z3 assertions.

Purpose: isolate compiler/solver bugs from translation-quality questions.
The official comparison point for the feasibility sample remains the direct
z3 oracle result recorded in oracle_results.txt (16/18) -- this script must
reproduce that result exactly (same 2 known mismatches, same reason), not
improve on it, since "improving" it would mean silently changing what the
gold annotations say.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.folio.solver import classify_theory  # noqa: E402


def pred(name, *args):
    return {"type": "predicate", "name": name, "args": list(args)}


def NOT(e):
    return {"type": "not", "body": e}


def AND(*es):
    return {"type": "and", "args": list(es)}


def OR(*es):
    return {"type": "or", "args": list(es)}


def IMPLIES(a, c):
    return {"type": "implies", "antecedent": a, "consequent": c}


def IFF(l, r):
    return {"type": "iff", "left": l, "right": r}


def FORALL(variables, body):
    return {"type": "forall", "variables": variables, "body": body}


def EXISTS(variables, body):
    return {"type": "exists", "variables": variables, "body": body}


results = []


def check(idx, gold, premises, hypothesis):
    verdict = classify_theory(premises, hypothesis)
    got = verdict.label if verdict.label else verdict.status.upper()
    results.append((idx, gold, got, verdict.status, gold == got))


# --- Group A: Zelda (idx 12, 13, 14) ----------------------------------------
p_zelda = [
    EXISTS(["x"], AND(pred("japanese", "x"), pred("vgcompany", "x"), pred("created", "x", "zelda"))),
    pred("game", "zelda"),
    FORALL(["x", "y"], IMPLIES(AND(pred("game", "x"), pred("top10", "x"), pred("created", "x", "y")), pred("japanese", "y"))),
    FORALL(["x"], IMPLIES(AND(pred("game", "x"), pred("sellsmorethan", "x", "onemillion")), pred("top10", "x"))),
    pred("sellsmorethan", "zelda", "onemillion"),
]
check(12, "True", p_zelda, pred("top10", "zelda"))
check(13, "Uncertain", p_zelda, FORALL(["x"], IMPLIES(pred("created", "fifa22", "x"), AND(pred("japanese", "x"), pred("vgcompany", "x")))))
check(14, "False", p_zelda, NOT(pred("top10", "zelda")))

# --- Group B: Beijing (idx 24, 25, 26) --------------------------------------
p_beijing = [
    pred("innorth", "beijing"),
    pred("hosted", "beijing", "s08o"), pred("hosted", "beijing", "s08p"),
    pred("hosted", "beijing", "so"), pred("hosted", "beijing", "wo"),
    pred("hosted", "beijing", "sp"), pred("hosted", "beijing", "wp"),
]
check(24, "True", p_beijing, AND(pred("hosted", "beijing", "s08o"), pred("hosted", "beijing", "wo")))
check(25, "Uncertain", p_beijing, pred("insouth", "beijing"))
check(26, "Uncertain", p_beijing, pred("secondlargest", "beijing"))

# --- Group C: Montana/Pierre (idx 58, 59) -----------------------------------
p_montana = [
    pred("cityin", "billings", "montana"),
    pred("cityin", "butte", "montana"), pred("cityin", "helena", "montana"), pred("cityin", "missoula", "montana"),
    EXISTS(["x"], AND(pred("cityin", "wss", "x"), pred("cityin", "butte", "x"))),
    NOT(pred("cityin", "pierre", "montana")),
    FORALL(["x"], IMPLIES(pred("cityin", "butte", "x"), NOT(pred("cityin", "pierre", "x")))),
]
check(58, "False", p_montana, EXISTS(["x"], AND(pred("cityin", "butte", "x"), pred("cityin", "pierre", "x"))))
check(59, "Uncertain", p_montana, EXISTS(["x"], AND(pred("cityin", "pierre", "x"), pred("cityin", "bismarck", "x"))))

# --- Group D: Badults (idx 69, 70) ------------------------------------------
p_badults = [
    EXISTS(["x"], AND(pred("memberof", "x", "pappys"), pred("starring", "badults", "x"))),
    pred("series", "badults"),
    pred("pilotedon", "badults", "bbcthree"),
    pred("workingtitle", "tsds", "badults"),
    pred("scripteditorfor", "andrewcollins", "badults"),
]
check(69, "True", p_badults, EXISTS(["x"], AND(pred("scripteditorfor", "andrewcollins", "x"), pred("series", "x"), pred("workingtitle", "tsds", "x"))))
check(70, "False", p_badults, FORALL(["x", "y"], IMPLIES(
    AND(pred("memberof", "x", "pappys"), pred("starring", "y", "x")),
    NOT(OR(pred("pilotedon", "y", "bbctwo"), pred("pilotedon", "y", "bbcthree"))),
)))

# --- Group E: Space/Vacuum (idx 135, 136, 137) ------------------------------
p_space = [
    FORALL(["x"], IMPLIES(pred("plunger", "x"), pred("suck", "x"))),
    FORALL(["x"], IMPLIES(pred("vacuum", "x"), pred("suck", "x"))),
    FORALL(["x"], IMPLIES(pred("vampire", "x"), pred("suck", "x"))),
    pred("vacuum", "space"),
    AND(pred("householdapp", "duster"), NOT(pred("suck", "duster"))),
]
check(135, "True", p_space, pred("suck", "space"))
check(136, "Uncertain", p_space, pred("vampire", "space"))
check(137, "False", p_space, FORALL(["x"], IMPLIES(pred("householdapp", "x"), pred("suck", "x"))))

# --- Group F: ML algorithms (idx 138, 139) -- literal disjunctive reading ---
p_ml = [
    OR(pred("mlalgorithm", "sup"), pred("mlalgorithm", "unsup"), pred("mlalgorithm", "reinf")),
    NOT(pred("requirelabelled", "unsup")),
    OR(pred("trainsummodel", "sup"), pred("trainsummodel", "unsup"), pred("trainsummodel", "reinf")),
    NOT(pred("trainsummodel", "reinf")),
    FORALL(["x"], IMPLIES(AND(pred("mlalgorithm", "x"), pred("trainsummodel", "x")), pred("requirelabelled", "x"))),
]
check(138, "True", p_ml, pred("trainsummodel", "sup"))
check(139, "False", p_ml, pred("trainsummodel", "unsup"))

# --- Group G: Heck cattle (idx 143, 144) ------------------------------------
p_heck = [
    FORALL(["x", "y"], IMPLIES(pred("breedingback", "x", "y"), pred("artificiallyselect", "x"))),
    EXISTS(["x", "y"], AND(pred("heckcattle", "x"), pred("aurochs", "y"), pred("breedingback", "x", "y"))),
    FORALL(["x"], IMPLIES(pred("heckcattle", "x"), pred("animal", "x"))),
    FORALL(["x"], IMPLIES(pred("aurochs", "x"), pred("animal", "x"))),
    EXISTS(["x", "y"], AND(pred("animal", "x"), pred("animal", "y"), pred("breedingback", "x", "y"), pred("extinct", "y"))),
]
check(143, "True", p_heck, EXISTS(["x"], AND(pred("heckcattle", "x"), pred("artificiallyselect", "x"))))
check(144, "Uncertain", p_heck, FORALL(["x"], IMPLIES(pred("aurochs", "x"), pred("extinct", "x"))))

# --- Group H: Remy (idx 154) ------------------------------------------------
p_remy = [
    FORALL(["x"], IMPLIES(pred("desktop", "x"), pred("workhome", "x"))),
    FORALL(["x"], IMPLIES(pred("homeoffice", "x"), pred("desktop", "x"))),
    FORALL(["x"], IMPLIES(pred("workhome", "x"), pred("tidying", "x"))),
    FORALL(["x"], IMPLIES(pred("workhome", "x"), pred("officestore", "x"))),
    FORALL(["x"], IMPLIES(pred("tidying", "x"), NOT(pred("late", "x")))),
    NOT(IFF(AND(pred("late", "remy"), pred("tidying", "remy")), AND(NOT(pred("late", "remy")), NOT(pred("tidying", "remy"))))),
]
check(154, "False", p_remy, AND(pred("homeoffice", "remy"), pred("officestore", "remy")))

# --- Report ------------------------------------------------------------------
correct = sum(1 for r in results if r[4])
print(f"{correct}/{len(results)} IR-pipeline predictions matched gold label\n")
for idx, gold, got, status, ok in results:
    print(f"idx={idx:>4}  gold={gold:<10} ir_pipeline={got:<12} status={status:<20} {'OK' if ok else 'MISMATCH'}")
