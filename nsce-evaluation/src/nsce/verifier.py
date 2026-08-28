"""Deterministic verification of a normalized ProofWriter record."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.io_guard import extract_allowed_input
from src.nsce.parser import ParserError, parse_query, parse_theory
from src.nsce.symbolic_reasoner import ReasoningResult, classify


@dataclass(frozen=True)
class VerificationResult:
    reasoning: ReasoningResult | None
    parser_error: str | None
    latency_ms: float


def verify_record(record: dict) -> VerificationResult:
    """Verify a record using only its whitelisted facts/rules/query.

    `record` is narrowed to an `AllowedInput` immediately -- the symbolic engine
    never sees `ground_truth` or any other evaluation-only field.
    """
    started = time.perf_counter()
    allowed = extract_allowed_input(record)
    try:
        theory = parse_theory(list(allowed.facts) + list(allowed.rules))
        reasoning = classify(parse_query(allowed.query), theory.facts, theory.rules)
        return VerificationResult(
            reasoning=reasoning,
            parser_error=None,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    except ParserError as exc:
        return VerificationResult(
            reasoning=None,
            parser_error=str(exc),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

