"""Translation-v3 (dedicated T5 bridge) end-to-end orchestration.

    RAW ENGLISH (premises + hypothesis, one sentence at a time)
        |  translator.py            (T5 model call, deterministic beam search)
        v
    raw FOL string per sentence
        |  adapter.py               (deterministic parser)
        v
    IR dict per sentence
        |  validator.py             (reuses schema.py: scope, then arity)
        v
    validated theory -> ir.to_dict()-shaped premise_dicts/hypothesis_dict
                          -> src/folio/solver.classify_theory (unchanged)

Sentence-level translation only (Step 6 of the integration spec): each
premise and the hypothesis are translated independently, then
deterministically assembled into one theory. No entity/predicate identity
resolution across sentences is performed here -- the model's own output
already reuses consistent PascalCase predicate names and lowercase entity
names for the same concept across sentences (observed in smoke testing), so
no v2-style vocabulary table is needed for the fast gate. If cross-sentence
identity turns out to be a real problem on the frozen 30, add only
lightweight deterministic normalization -- do not restart a semantic-parsing
project.

Never reads a gold label or gold FOL -- same io_guard discipline as v1/v2.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.folio import ir
from src.folio.io_guard import FolioAllowedInput
from src.folio.ir import Expr
from src.folio.t5_bridge.model import T5FolBridge
from src.folio.t5_bridge.translator import SentenceTranslation, translate_sentence
from src.folio.t5_bridge.validator import parse_and_validate_sentence, validate_theory


@dataclass(frozen=True)
class SentencePipelineResult:
    sentence: str
    translation: SentenceTranslation
    expr: Expr | None
    valid: bool
    reason: str | None
    detail: str | None


@dataclass(frozen=True)
class TheoryPipelineResult:
    premise_results: tuple[SentencePipelineResult, ...]
    hypothesis_result: SentencePipelineResult | None
    theory_level_reason: str | None
    theory_level_detail: str | None
    all_valid: bool
    premise_dicts: tuple[dict, ...] | None
    hypothesis_dict: dict | None


def _translate_and_validate_one(bridge: T5FolBridge, sentence: str) -> SentencePipelineResult:
    translation = translate_sentence(bridge, sentence)
    if not translation.valid:
        return SentencePipelineResult(
            sentence=sentence, translation=translation, expr=None,
            valid=False, reason=translation.reason, detail=translation.detail,
        )
    expr, reason, detail = parse_and_validate_sentence(translation.parsed)
    return SentencePipelineResult(
        sentence=sentence, translation=translation, expr=expr,
        valid=expr is not None, reason=reason, detail=detail,
    )


def translate_theory_t5(bridge: T5FolBridge, allowed: FolioAllowedInput) -> TheoryPipelineResult:
    premise_results = tuple(_translate_and_validate_one(bridge, s) for s in allowed.premises)
    hypothesis_result = _translate_and_validate_one(bridge, allowed.hypothesis)

    per_sentence_valid = all(r.valid for r in premise_results) and hypothesis_result.valid
    if not per_sentence_valid:
        return TheoryPipelineResult(
            premise_results=premise_results, hypothesis_result=hypothesis_result,
            theory_level_reason=None, theory_level_detail=None,
            all_valid=False, premise_dicts=None, hypothesis_dict=None,
        )

    premise_exprs = [r.expr for r in premise_results]  # type: ignore[misc]
    theory = validate_theory(premise_exprs, hypothesis_result.expr)  # type: ignore[arg-type]
    if not theory.valid:
        return TheoryPipelineResult(
            premise_results=premise_results, hypothesis_result=hypothesis_result,
            theory_level_reason=theory.reason, theory_level_detail=theory.detail,
            all_valid=False, premise_dicts=None, hypothesis_dict=None,
        )

    premise_dicts = tuple(ir.to_dict(e) for e in premise_exprs)  # type: ignore[arg-type]
    hypothesis_dict = ir.to_dict(hypothesis_result.expr)  # type: ignore[arg-type]
    return TheoryPipelineResult(
        premise_results=premise_results, hypothesis_result=hypothesis_result,
        theory_level_reason=None, theory_level_detail=None,
        all_valid=True, premise_dicts=premise_dicts, hypothesis_dict=hypothesis_dict,
    )
