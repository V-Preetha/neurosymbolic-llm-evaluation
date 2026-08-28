"""Deterministic JSON-AST -> Z3 compiler. No LLM calls anywhere in this module.

Pipeline (per FOLIO_SYMBOLIC_IR.md / the translation-prototype spec):

    validated IR (src/folio/ir.py + schema.py)
            |
            v
    this module: deterministic compiler
            |
            v
    Z3 BoolRef expression(s)

Predicates and entity constants are compiled against one shared, per-theory
symbol table so the same name always maps to the same Z3 function/constant
across every premise and the hypothesis. Quantifier variables get a fresh
z3.Const per (compile call, variable occurrence) bound by z3's own
ForAll/Exists — shadowing is already rejected upstream by schema.py, so no
name-collision handling is needed here.

Fails loudly (raises CompilationError) on anything the 8-node IR cannot
represent; there is no best-effort fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import z3

from src.folio.ir import And, Exists, Expr, ForAll, Iff, Implies, Not, Or, Predicate, is_variable


FolioObject = z3.DeclareSort("FolioObject")


class CompilationError(ValueError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass
class SymbolTable:
    """Shared predicate/entity registry for one theory (premises + hypothesis)."""

    predicates: dict[tuple[str, int], z3.FuncDeclRef] = field(default_factory=dict)
    entities: dict[str, z3.ExprRef] = field(default_factory=dict)

    def predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = (name, arity)
        existing = self.predicates.get(key)
        if existing is not None:
            return existing
        # Reject silent arity collisions here too -- schema.py should already
        # have caught this, but the compiler must never trust that blindly.
        for (other_name, other_arity) in self.predicates:
            if other_name == name and other_arity != arity:
                raise CompilationError(
                    f"predicate {name!r} compiled with arity {other_arity} earlier, now {arity}"
                )
        func = z3.Function(name, *([FolioObject] * arity), z3.BoolSort())
        self.predicates[key] = func
        return func

    def entity(self, name: str) -> z3.ExprRef:
        existing = self.entities.get(name)
        if existing is not None:
            return existing
        const = z3.Const(name, FolioObject)
        self.entities[name] = const
        return const


def compile_expr(expr: Expr, symbols: SymbolTable, var_env: dict[str, z3.ExprRef]) -> z3.BoolRef:
    if isinstance(expr, Predicate):
        func = symbols.predicate(expr.name, len(expr.args))
        resolved = []
        for arg in expr.args:
            if is_variable(arg):
                if arg not in var_env:
                    raise CompilationError(f"variable {arg!r} not bound during compilation (schema.py should have caught this)")
                resolved.append(var_env[arg])
            else:
                resolved.append(symbols.entity(arg))
        return func(*resolved)

    if isinstance(expr, Not):
        return z3.Not(compile_expr(expr.body, symbols, var_env))

    if isinstance(expr, And):
        return z3.And(*[compile_expr(a, symbols, var_env) for a in expr.args])

    if isinstance(expr, Or):
        return z3.Or(*[compile_expr(a, symbols, var_env) for a in expr.args])

    if isinstance(expr, Implies):
        return z3.Implies(
            compile_expr(expr.antecedent, symbols, var_env),
            compile_expr(expr.consequent, symbols, var_env),
        )

    if isinstance(expr, Iff):
        return compile_expr(expr.left, symbols, var_env) == compile_expr(expr.right, symbols, var_env)

    if isinstance(expr, (ForAll, Exists)):
        fresh_consts = [z3.Const(v, FolioObject) for v in expr.variables]
        inner_env = dict(var_env)
        inner_env.update(zip(expr.variables, fresh_consts))
        body = compile_expr(expr.body, symbols, inner_env)
        return z3.ForAll(fresh_consts, body) if isinstance(expr, ForAll) else z3.Exists(fresh_consts, body)

    raise CompilationError(f"unsupported IR node for compilation: {expr!r}")


def compile_theory(premises: list[Expr], hypothesis: Expr) -> tuple[list[z3.BoolRef], z3.BoolRef, SymbolTable]:
    """Compile a full theory (shared symbol table across premises + hypothesis)."""
    symbols = SymbolTable()
    compiled_premises = [compile_expr(p, symbols, {}) for p in premises]
    compiled_hypothesis = compile_expr(hypothesis, symbols, {})
    return compiled_premises, compiled_hypothesis, symbols
