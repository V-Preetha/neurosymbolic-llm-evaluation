"""Cross-tree validation for the FOLIO IR: variable scope and predicate arity.

Node-local structural checks (required fields, non-empty lists, unknown node
types) already happened in ir.parse_expr. This module adds the two checks
that need context beyond a single node:

1. Variable scope — every variable-shaped predicate argument must be bound
   by an enclosing forall/exists on the path from the root.
2. Predicate arity consistency — the same predicate name must have the same
   argument count everywhere it's used across an entire theory (all
   premises + the hypothesis together, not just within one sentence).

Both raise ir.TranslationFailure with a specific `reason` code. Nothing here
repairs a structure; a failure always propagates.
"""

from __future__ import annotations

from src.folio.ir import (
    And, Exists, Expr, ForAll, Iff, Implies, Not, Or, Predicate,
    TranslationFailure, is_variable,
)


def validate_scope(expr: Expr, bound: frozenset[str] = frozenset()) -> None:
    """Walk `expr`, raising TranslationFailure on any unbound variable use."""
    if isinstance(expr, Predicate):
        for arg in expr.args:
            if is_variable(arg) and arg not in bound:
                raise TranslationFailure(
                    "unbound_variable",
                    f"variable {arg!r} used in predicate {expr.name!r} is not bound by an enclosing forall/exists",
                )
        return
    if isinstance(expr, Not):
        validate_scope(expr.body, bound)
        return
    if isinstance(expr, (And, Or)):
        for sub in expr.args:
            validate_scope(sub, bound)
        return
    if isinstance(expr, Implies):
        validate_scope(expr.antecedent, bound)
        validate_scope(expr.consequent, bound)
        return
    if isinstance(expr, Iff):
        validate_scope(expr.left, bound)
        validate_scope(expr.right, bound)
        return
    if isinstance(expr, (ForAll, Exists)):
        shadowed = set(expr.variables) & bound
        if shadowed:
            raise TranslationFailure(
                "variable_shadowing",
                f"quantifier rebinds already-bound variable(s) {sorted(shadowed)}",
            )
        validate_scope(expr.body, bound | set(expr.variables))
        return
    raise TranslationFailure("malformed_node", f"unrecognized IR node during scope check: {expr!r}")


def collect_predicate_arities(expr: Expr, registry: dict[str, int]) -> None:
    """Populate/check a shared {predicate_name: arity} registry across a whole theory."""
    if isinstance(expr, Predicate):
        arity = len(expr.args)
        existing = registry.get(expr.name)
        if existing is not None and existing != arity:
            raise TranslationFailure(
                "arity_conflict",
                f"predicate {expr.name!r} used with {existing} arg(s) elsewhere but {arity} here",
            )
        registry[expr.name] = arity
        return
    if isinstance(expr, Not):
        collect_predicate_arities(expr.body, registry)
        return
    if isinstance(expr, (And, Or)):
        for sub in expr.args:
            collect_predicate_arities(sub, registry)
        return
    if isinstance(expr, Implies):
        collect_predicate_arities(expr.antecedent, registry)
        collect_predicate_arities(expr.consequent, registry)
        return
    if isinstance(expr, Iff):
        collect_predicate_arities(expr.left, registry)
        collect_predicate_arities(expr.right, registry)
        return
    if isinstance(expr, (ForAll, Exists)):
        collect_predicate_arities(expr.body, registry)
        return
    raise TranslationFailure("malformed_node", f"unrecognized IR node during arity check: {expr!r}")


def validate_theory(premises: list[Expr], hypothesis: Expr) -> None:
    """Full cross-tree validation for one FOLIO example: scope + arity, over premises+hypothesis together."""
    for premise in premises:
        validate_scope(premise)
    validate_scope(hypothesis)

    registry: dict[str, int] = {}
    for premise in premises:
        collect_predicate_arities(premise, registry)
    collect_predicate_arities(hypothesis, registry)
