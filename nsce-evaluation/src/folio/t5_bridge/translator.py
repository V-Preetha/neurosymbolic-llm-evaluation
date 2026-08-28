"""Sentence-level translation: raw NL string -> (parsed IR dict | failure).

Composes model.py (LLM call) and adapter.py (deterministic parse) into one
result type, mirroring the shape of src/folio/v2/decomposer.py's
DecompositionAttempt so pipeline.py can report per-sentence outcomes the
same way run_frozen30.py already does for Translation-v2.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.folio.ir import TranslationFailure
from src.folio.t5_bridge.adapter import parse_t5_fol
from src.folio.t5_bridge.model import T5FolBridge


@dataclass(frozen=True)
class SentenceTranslation:
    sentence: str
    raw_output: str
    latency_ms: float
    parsed: dict | None
    valid: bool
    reason: str | None
    detail: str | None


def translate_sentence(bridge: T5FolBridge, sentence: str) -> SentenceTranslation:
    """Translate exactly one NL sentence. Never sends the gold label or gold
    FOL -- only the raw sentence text reaches the model, same io_guard
    discipline as Translation-v1/v2."""
    generation = bridge.translate(sentence)
    try:
        parsed = parse_t5_fol(generation.raw_text)
    except TranslationFailure as exc:
        return SentenceTranslation(
            sentence=sentence,
            raw_output=generation.raw_text,
            latency_ms=generation.latency_ms,
            parsed=None,
            valid=False,
            reason=exc.reason,
            detail=exc.detail,
        )
    return SentenceTranslation(
        sentence=sentence,
        raw_output=generation.raw_text,
        latency_ms=generation.latency_ms,
        parsed=parsed,
        valid=True,
        reason=None,
        detail=None,
    )
