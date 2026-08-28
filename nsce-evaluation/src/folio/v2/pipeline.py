"""Translation-v2 end-to-end orchestration.

    RAW ENGLISH
        |  decomposer.py            (LLM, per sentence)
        v
    semantic decomposition (entities / atoms / nodes / root)
        |  vocabulary.py            (deterministic)
        v
    theory-shared entity/predicate tables
        |  ast_builder.py           (deterministic)
        v
    FOL AST (src/folio/ir.py Expr)
        |  validator.py             (deterministic, reuses schema.py)
        v
    [repair.py -- LLM, at most once per sentence, only on repairable errors]
        |
        v
    validated theory -> ir.to_dict() -> src/folio/solver.classify_theory
                                          (deterministic, unchanged from v1)

Ablation surface (see TRANSLATION_V2_DESIGN.md for why these two flags,
rather than a literal V2-A/B/C/D switch, are what's actually toggled):

    use_global_vocabulary=False -> V2-A (decomposition only: every sentence
        gets its own throwaway entity/predicate table, no cross-sentence
        identity sharing)
    use_global_vocabulary=True, enable_repair=False -> V2-B / V2-C (full
        AST-guided pipeline with theory-shared vocabulary)
    use_global_vocabulary=True, enable_repair=True -> V2-D (+ constrained repair)

Never reads a gold label or gold FOL. Premises and the hypothesis are each
decomposed independently, exactly like Translation-v1's translate_theory.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.folio import ir
from src.folio.io_guard import FolioAllowedInput
from src.folio.ir import Expr
from src.folio.v2 import ast_builder, repair, validator
from src.folio.v2.decomposer import DecompositionAttempt, decompose_sentence
from src.folio.v2.vocabulary import EntityTable, PredicateTable


@dataclass(frozen=True)
class SentencePipelineResult:
    sentence: str
    initial_attempt: DecompositionAttempt
    repair_attempt: DecompositionAttempt | None
    final_attempt: DecompositionAttempt
    valid: bool
    expr: Expr | None
    reason: str | None
    detail: str | None
    pre_repair_valid: bool | None  # None if never reached validation (e.g. bad JSON pre-repair too)


@dataclass(frozen=True)
class TheoryPipelineResult:
    premise_results: tuple[SentencePipelineResult, ...]
    hypothesis_result: SentencePipelineResult | None
    theory_level_reason: str | None
    theory_level_detail: str | None
    all_valid: bool
    premise_dicts: tuple[dict, ...] | None
    hypothesis_dict: dict | None
    entity_table: EntityTable
    predicate_table: PredicateTable
    config: dict


def _decompose_all(runner, sentences: list[str], max_new_tokens: int) -> list[DecompositionAttempt]:
    return [decompose_sentence(runner, s, max_new_tokens=max_new_tokens) for s in sentences]


def _atom_pairs_for(attempts: list[DecompositionAttempt]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for a in attempts:
        if a.shape_valid and a.decomposition is not None:
            pairs.extend(ast_builder.collect_atom_predicate_arities(a.decomposition))
    return pairs


def _validate_decomposition_attempt(
    attempt: DecompositionAttempt, entity_table: EntityTable, predicate_table: PredicateTable, embed_fn=None
) -> validator.SentenceValidation:
    """Validate one attempt (initial or repaired), building the (local-mode)
    predicate table from its own atoms first if it hasn't been built yet.
    """
    if not attempt.shape_valid:
        return validator.SentenceValidation(expr=None, valid=False, reason=attempt.error_reason, detail=attempt.error_detail)
    if not predicate_table.built:
        try:
            predicate_table.build(ast_builder.collect_atom_predicate_arities(attempt.decomposition), embed_fn=embed_fn)
        except ir.TranslationFailure as exc:
            return validator.SentenceValidation(expr=None, valid=False, reason=exc.reason, detail=exc.detail)
    return validator.validate_sentence(attempt.decomposition, entity_table, predicate_table)


def _validate_and_maybe_repair(
    runner,
    attempt: DecompositionAttempt,
    entity_table: EntityTable,
    predicate_table: PredicateTable,
    enable_repair: bool,
    max_new_tokens: int,
    embed_fn=None,
) -> SentencePipelineResult:
    sv = _validate_decomposition_attempt(attempt, entity_table, predicate_table, embed_fn=embed_fn)
    current = attempt
    pre_repair_valid = sv.valid

    repair_attempt = None
    if not sv.valid and enable_repair and repair.is_repairable(sv.reason):
        reason = sv.reason or "invalid_json"
        detail = sv.detail or ""
        repair_attempt = repair.attempt_repair(runner, current.sentence, current, reason, detail, max_new_tokens=max_new_tokens)
        current = repair_attempt
        sv = _validate_decomposition_attempt(repair_attempt, entity_table, predicate_table, embed_fn=embed_fn)
        if not sv.valid:
            sv = validator.SentenceValidation(
                expr=None, valid=False, reason="repair_failed",
                detail=f"post-repair error: {sv.reason} -- {sv.detail}",
            )

    return SentencePipelineResult(
        sentence=attempt.sentence,
        initial_attempt=attempt,
        repair_attempt=repair_attempt,
        final_attempt=current,
        valid=sv.valid,
        expr=sv.expr,
        reason=sv.reason,
        detail=sv.detail,
        pre_repair_valid=pre_repair_valid,
    )


def translate_theory_v2(
    runner,
    allowed: FolioAllowedInput,
    *,
    use_global_vocabulary: bool = True,
    enable_repair: bool = False,
    embed_fn=None,
    max_new_tokens: int = 500,
) -> TheoryPipelineResult:
    config = {"use_global_vocabulary": use_global_vocabulary, "enable_repair": enable_repair}

    premise_attempts = _decompose_all(runner, list(allowed.premises), max_new_tokens)
    hypothesis_attempt = decompose_sentence(runner, allowed.hypothesis, max_new_tokens=max_new_tokens)
    all_attempts = premise_attempts + [hypothesis_attempt]

    entity_table = EntityTable()
    predicate_table = PredicateTable()
    theory_level_reason = None
    theory_level_detail = None

    if use_global_vocabulary:
        try:
            predicate_table.build(_atom_pairs_for(all_attempts), embed_fn=embed_fn)
        except ir.TranslationFailure as exc:
            theory_level_reason, theory_level_detail = exc.reason, exc.detail
    else:
        # Ablation V2-A: no build() call here -- each sentence below gets its
        # own fresh PredicateTable/EntityTable built from just its own atoms.
        pass

    if theory_level_reason is not None:
        # Theory-wide vocabulary itself failed (e.g. a genuine cross-sentence
        # arity conflict) -- not attributable to one sentence, not repairable.
        return TheoryPipelineResult(
            premise_results=(), hypothesis_result=None,
            theory_level_reason=theory_level_reason, theory_level_detail=theory_level_detail,
            all_valid=False, premise_dicts=None, hypothesis_dict=None,
            entity_table=entity_table, predicate_table=predicate_table, config=config,
        )

    def validate_one(attempt: DecompositionAttempt) -> SentencePipelineResult:
        if use_global_vocabulary:
            et, pt = entity_table, predicate_table
        else:
            et, pt = EntityTable(), PredicateTable()
            if attempt.shape_valid and attempt.decomposition is not None:
                try:
                    pt.build(ast_builder.collect_atom_predicate_arities(attempt.decomposition), embed_fn=embed_fn)
                except ir.TranslationFailure as exc:
                    # Arity conflict within this one sentence's own atoms --
                    # attributable to this sentence, so report it directly
                    # rather than masking it behind an empty vocabulary.
                    return SentencePipelineResult(
                        sentence=attempt.sentence, initial_attempt=attempt, repair_attempt=None,
                        final_attempt=attempt, valid=False, expr=None,
                        reason=exc.reason, detail=exc.detail, pre_repair_valid=False,
                    )
        return _validate_and_maybe_repair(runner, attempt, et, pt, enable_repair, max_new_tokens, embed_fn=embed_fn)

    premise_results = tuple(validate_one(a) for a in premise_attempts)
    hypothesis_result = validate_one(hypothesis_attempt)

    per_sentence_valid = all(r.valid for r in premise_results) and hypothesis_result.valid
    theory_reason = None
    theory_detail = None
    premise_dicts = None
    hypothesis_dict = None
    all_valid = False
    if per_sentence_valid:
        # Final cross-theory pass -- catches arity conflicts that a
        # per-sentence-only vocabulary (use_global_vocabulary=False) cannot
        # see, since it never looks at more than one sentence's atoms at a
        # time. In global mode this is normally already guaranteed by
        # PredicateTable.build() scanning every atom up front; kept here too
        # so the guarantee holds under either ablation setting.
        premise_exprs = [r.expr for r in premise_results]
        theory = validator.validate_theory(premise_exprs, hypothesis_result.expr)  # type: ignore[arg-type]
        if theory.valid:
            all_valid = True
            premise_dicts = tuple(ir.to_dict(e) for e in premise_exprs)  # type: ignore[arg-type]
            hypothesis_dict = ir.to_dict(hypothesis_result.expr)  # type: ignore[arg-type]
        else:
            theory_reason, theory_detail = theory.reason, theory.detail

    return TheoryPipelineResult(
        premise_results=premise_results,
        hypothesis_result=hypothesis_result,
        theory_level_reason=theory_reason,
        theory_level_detail=theory_detail,
        all_valid=all_valid,
        premise_dicts=premise_dicts,
        hypothesis_dict=hypothesis_dict,
        entity_table=entity_table,
        predicate_table=predicate_table,
        config=config,
    )
