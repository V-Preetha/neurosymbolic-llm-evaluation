"""Step G: static semantic checks, run before anything touches Z3.

Composes (does not reimplement) three layers, in order, and never calls Z3:

1. ast_builder.build_expr per sentence -- catches dangling/duplicate/cyclic
   id references and unknown operator types (checks that only make sense at
   the single-sentence decomposition level).
2. schema.validate_scope per sentence -- catches unbound variables /
   shadowing, attributable to exactly one sentence (this is what makes a
   failure here eligible for repair.py's one-shot, sentence-local repair).
3. schema.validate_theory over the whole theory (premises + hypothesis
   together) -- catches cross-sentence predicate arity conflicts, which are
   NOT attributable to one sentence and are therefore never repaired.

Reuses src/folio/schema.py entirely; adds no new scope/arity logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.folio import schema
from src.folio.ir import Expr, TranslationFailure
from src.folio.v2 import ast_builder
from src.folio.v2.vocabulary import EntityTable, PredicateTable


@dataclass(frozen=True)
class SentenceValidation:
    expr: Expr | None
    valid: bool
    reason: str | None
    detail: str | None


def validate_sentence(
    decomposition: dict, entity_table: EntityTable, predicate_table: PredicateTable
) -> SentenceValidation:
    """Build + locally scope-check one sentence. Does not check cross-theory arity."""
    try:
        expr = ast_builder.build_expr(decomposition, entity_table, predicate_table)
        schema.validate_scope(expr)
    except TranslationFailure as exc:
        return SentenceValidation(expr=None, valid=False, reason=exc.reason, detail=exc.detail)
    return SentenceValidation(expr=expr, valid=True, reason=None, detail=None)


@dataclass(frozen=True)
class TheoryValidation:
    valid: bool
    premises: tuple[Expr, ...] | None
    hypothesis: Expr | None
    reason: str | None
    detail: str | None
    # "premise:<index>" | "hypothesis" | "theory" | None (only set when valid)
    failed_at: str | None


def validate_theory(premise_exprs: list[Expr], hypothesis_expr: Expr) -> TheoryValidation:
    """Cross-theory check only (arity consistency across all sentences), assuming
    every individual sentence has already passed validate_sentence. Per-sentence
    scope has already been checked by validate_sentence, but schema.validate_theory
    re-checks scope too (cheap, and keeps this function safe to call standalone).
    """
    try:
        schema.validate_theory(list(premise_exprs), hypothesis_expr)
    except TranslationFailure as exc:
        return TheoryValidation(
            valid=False, premises=None, hypothesis=None,
            reason=exc.reason, detail=exc.detail, failed_at="theory",
        )
    return TheoryValidation(
        valid=True, premises=tuple(premise_exprs), hypothesis=hypothesis_expr,
        reason=None, detail=None, failed_at=None,
    )
