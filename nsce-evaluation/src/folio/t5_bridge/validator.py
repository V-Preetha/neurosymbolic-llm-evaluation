"""Cross-theory validation for T5-bridge output. Reuses src/folio/schema.py
entirely (no new scope/arity logic), same pattern as src/folio/v2/validator.py.

Per-sentence structural/scope validity is already established by
translator.py (adapter.py raises TranslationFailure on malformed structure;
ir.parse_expr + schema.validate_scope catch unbound variables). This module
adds only the check that needs the whole theory at once: predicate arity
consistency across every premise + the hypothesis together.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.folio import ir, schema
from src.folio.ir import Expr, TranslationFailure


@dataclass(frozen=True)
class TheoryValidation:
    valid: bool
    premises: tuple[Expr, ...] | None
    hypothesis: Expr | None
    reason: str | None
    detail: str | None


def parse_and_validate_sentence(parsed_dict: dict) -> tuple[Expr | None, str | None, str | None]:
    """dict -> (Expr, None, None) on success, or (None, reason, detail) on failure.
    ir.parse_expr does node-local structural checks; schema.validate_scope
    catches unbound variables, both attributable to exactly this sentence.
    """
    try:
        expr = ir.parse_expr(parsed_dict)
        schema.validate_scope(expr)
    except TranslationFailure as exc:
        return None, exc.reason, exc.detail
    return expr, None, None


def validate_theory(premise_exprs: list[Expr], hypothesis_expr: Expr) -> TheoryValidation:
    try:
        schema.validate_theory(list(premise_exprs), hypothesis_expr)
    except TranslationFailure as exc:
        return TheoryValidation(
            valid=False, premises=None, hypothesis=None, reason=exc.reason, detail=exc.detail,
        )
    return TheoryValidation(
        valid=True, premises=tuple(premise_exprs), hypothesis=hypothesis_expr, reason=None, detail=None,
    )
