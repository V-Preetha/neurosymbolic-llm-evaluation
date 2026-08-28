"""Z3 execution and three-way logical label mapping."""
from __future__ import annotations
import time
from typing import Any
from .fol_ast import BridgeError
from .z3_compiler import CompiledTheory,compile_theory

def _unknown(reason):return "solver_timeout" if "timeout" in (reason or "").lower() else "solver_unknown"
def solve_compiled(compiled:CompiledTheory,timeout_ms:int=10000)->dict[str,Any]:
    import z3
    started=time.perf_counter();solver=z3.Solver();solver.set(timeout=timeout_ms);statuses={}
    def check(items):
        solver.push();solver.add(*items);result=solver.check();reason=solver.reason_unknown() if result==z3.unknown else None;solver.pop();return str(result),reason
    statuses["premises"],reason=check(compiled.premises)
    if statuses["premises"]=="unknown":raise BridgeError(_unknown(reason),reason or "unknown")
    if statuses["premises"]=="unsat":raise BridgeError("inconsistent_premises","premises inconsistent")
    statuses["entailment"],reason=check([*compiled.premises,z3.Not(compiled.hypothesis)])
    if statuses["entailment"]=="unsat":label="True"
    elif statuses["entailment"]=="unknown":raise BridgeError(_unknown(reason),reason or "unknown")
    else:
        statuses["contradiction"],reason=check([*compiled.premises,compiled.hypothesis])
        if statuses["contradiction"]=="unsat":label="False"
        elif statuses["contradiction"]=="unknown":raise BridgeError(_unknown(reason),reason or "unknown")
        else:label="Unknown"
    return {"label":label,"statuses":statuses,"solver_ms":(time.perf_counter()-started)*1000}
def solve(theory:dict[str,Any],timeout_ms:int=10000):return solve_compiled(compile_theory(theory),timeout_ms)
def semantic_agreement(predicted:dict[str,Any],gold:dict[str,Any],timeout_ms:int=10000)->dict[str,Any]:
    import z3
    p,g=compile_theory(predicted),compile_theory(gold);solver=z3.Solver();solver.set(timeout=timeout_ms)
    comparisons={"premises":z3.Xor(z3.And(*p.premises),z3.And(*g.premises)),"hypothesis":z3.Xor(p.hypothesis,g.hypothesis)};statuses={}
    for name,expr in comparisons.items():
        solver.push();solver.add(expr);result=solver.check();statuses[name]=str(result)
        if result==z3.unknown:statuses[f"{name}_reason"]=solver.reason_unknown()
        solver.pop()
    status="solver_unknown" if "unknown" in statuses.values() else "equivalent" if all(statuses[n]=="unsat" for n in comparisons) else "different"
    return {"status":status,"checks":statuses}
