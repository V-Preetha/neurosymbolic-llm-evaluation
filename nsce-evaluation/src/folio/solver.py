"""FOLIO verdict engine: compiled Z3 theory -> True / False / Uncertain.

    P = conjunction of compiled premises
    H = compiled hypothesis

    unsat(P AND NOT H)  -> "True"      (H is entailed by P)
    unsat(P AND H)      -> "False"     (H is contradicted by P)
    else                 -> "Uncertain" (both extensions of P are satisfiable)

Premise consistency is checked *first* and separately: if P itself is
unsat, the knowledge base is self-contradictory and no verdict about H is
meaningful -- this is reported as its own status ("inconsistent_premises"),
never silently collapsed into True/False/Uncertain. Every individual
solver.check() call result is logged (sat/unsat/unknown), including
`unknown` (Z3 gave up / timed out) which is reported honestly rather than
treated as any particular verdict.

No LLM calls anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import z3

from src.folio import ir, schema
from src.folio.compiler import CompilationError, compile_theory


DEFAULT_TIMEOUT_MS = 10_000


@dataclass(frozen=True)
class Verdict:
    status: str  # entailed | contradicted | uncertain | inconsistent_premises | translation_failure | compilation_error
    label: str | None  # "True" | "False" | "Uncertain" | None (for failure statuses)
    solver_statuses: dict[str, str] = field(default_factory=dict)
    failure_reason: str | None = None
    failure_detail: str | None = None


def _check(solver: z3.Solver, assertions: list[z3.BoolRef]) -> str:
    solver.push()
    for a in assertions:
        solver.add(a)
    result = solver.check()
    solver.pop()
    if result == z3.sat:
        return "sat"
    if result == z3.unsat:
        return "unsat"
    return "unknown"


def classify_compiled(
    premises: list[z3.BoolRef],
    hypothesis: z3.BoolRef,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Verdict:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    statuses: dict[str, str] = {}

    consistency_status = _check(solver, premises)
    statuses["premises_consistency"] = consistency_status
    if consistency_status == "unsat":
        return Verdict(status="inconsistent_premises", label=None, solver_statuses=statuses)

    entail_status = _check(solver, premises + [z3.Not(hypothesis)])
    statuses["entailment_check"] = entail_status
    if entail_status == "unsat":
        return Verdict(status="entailed", label="True", solver_statuses=statuses)

    contradiction_status = _check(solver, premises + [hypothesis])
    statuses["contradiction_check"] = contradiction_status
    if contradiction_status == "unsat":
        return Verdict(status="contradicted", label="False", solver_statuses=statuses)

    return Verdict(status="uncertain", label="Uncertain", solver_statuses=statuses)


def classify_theory(
    premise_dicts: list[dict],
    hypothesis_dict: dict,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Verdict:
    """Full pipeline: raw JSON dicts -> parse -> validate -> compile -> solve.

    Any TranslationFailure (ir.py / schema.py) or CompilationError is caught
    here and reported as a failure Verdict rather than propagating -- this is
    the boundary the translator/pipeline code calls, so callers get a single
    uniform result type instead of having to catch multiple exception types.
    """
    try:
        premise_exprs = [ir.parse_expr(p) for p in premise_dicts]
        hypothesis_expr = ir.parse_expr(hypothesis_dict)
        schema.validate_theory(premise_exprs, hypothesis_expr)
    except ir.TranslationFailure as exc:
        return Verdict(
            status="translation_failure", label=None,
            failure_reason=exc.reason, failure_detail=exc.detail,
        )

    try:
        compiled_premises, compiled_hypothesis, _symbols = compile_theory(premise_exprs, hypothesis_expr)
    except CompilationError as exc:
        return Verdict(
            status="compilation_error", label=None,
            failure_reason="compilation_error", failure_detail=exc.detail,
        )

    return classify_compiled(compiled_premises, compiled_hypothesis, timeout_ms=timeout_ms)
