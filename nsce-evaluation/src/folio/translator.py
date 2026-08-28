"""Raw English -> FOLIO JSON IR, via the frozen Qwen3-8B checkpoint.

This is semantic translation only -- the prompt never asks "is this true?"
and the model is never shown a label, gold FOL, or Z3 result. Premises and
the hypothesis are each translated independently, one sentence per model
call, so translating the hypothesis cannot be influenced by having just
been told "and here's what the premises imply."

No auto-repair: a malformed model output is reported as a TranslationAttempt
with schema_valid=False and a structured error_reason; nothing here retries
with a second, differently-worded LLM call to fix it up.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from src.folio import ir, schema
from src.folio.io_guard import FolioAllowedInput


SCHEMA_DESCRIPTION = """You are a semantic parser. Translate ONE English sentence into a constrained JSON logical form. Output ONLY the JSON object -- no explanation, no markdown fences, no extra text.

Allowed node types (exactly these 8, nothing else):
- {"type": "predicate", "name": "<lowercase_predicate>", "args": ["<term>", ...]}
- {"type": "not", "body": <node>}
- {"type": "and", "args": [<node>, <node>, ...]}   (2 or more)
- {"type": "or", "args": [<node>, <node>, ...]}    (2 or more)
- {"type": "implies", "antecedent": <node>, "consequent": <node>}
- {"type": "iff", "left": <node>, "right": <node>}
- {"type": "forall", "variables": ["x", ...], "body": <node>}
- {"type": "exists", "variables": ["x", ...], "body": <node>}

Rules for terms in "args":
- A VARIABLE is a single lowercase letter, optionally with a digit ("x", "y", "x1"). Every variable used inside a predicate MUST be bound by an enclosing "forall" or "exists" that declares it in "variables".
- An ENTITY (a specific named person/place/thing) is any other lowercase, underscore-joined string (e.g. "velite", "andrew_collins"). Entities need no quantifier.
- Do not invent XOR; express "exactly one of A, B" as {"type": "not", "body": {"type": "iff", "left": A, "right": B}} if needed.

Example 1.
Sentence: "All dogs bark."
JSON: {"type": "forall", "variables": ["x"], "body": {"type": "implies", "antecedent": {"type": "predicate", "name": "dog", "args": ["x"]}, "consequent": {"type": "predicate", "name": "bark", "args": ["x"]}}}

Example 2.
Sentence: "Some cat is sleeping and is not hungry."
JSON: {"type": "exists", "variables": ["x"], "body": {"type": "and", "args": [{"type": "predicate", "name": "cat", "args": ["x"]}, {"type": "predicate", "name": "sleeping", "args": ["x"]}, {"type": "not", "body": {"type": "predicate", "name": "hungry", "args": ["x"]}}]}}

Example 3.
Sentence: "Fluffy is a cat."
JSON: {"type": "predicate", "name": "cat", "args": ["fluffy"]}

Now translate this sentence. Output only the JSON object.
Sentence: "{sentence}"
JSON:"""


@dataclass(frozen=True)
class TranslationAttempt:
    sentence: str
    raw_output: str
    json_ast: dict | None
    schema_valid: bool
    error_reason: str | None
    error_detail: str | None
    latency_ms: float


def render_sentence_translation_prompt(sentence: str) -> str:
    """Pure function: no record access, no gold fields -- just the one sentence."""
    return SCHEMA_DESCRIPTION.replace("{sentence}", sentence.strip())


def _extract_json_object(raw_text: str) -> dict:
    """Best-effort but deterministic extraction of the first JSON object in the
    model's output (strips ```json fences if present). Raises json.JSONDecodeError
    or ValueError on anything that isn't a clean single JSON object -- this is
    detection of malformed output, not repair of it.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text, start)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def translate_sentence(runner, sentence: str, max_new_tokens: int = 400) -> TranslationAttempt:
    prompt = render_sentence_translation_prompt(sentence)
    generation = runner.generate_text(prompt, max_new_tokens=max_new_tokens)
    try:
        candidate = _extract_json_object(generation.raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return TranslationAttempt(
            sentence=sentence, raw_output=generation.raw_text, json_ast=None,
            schema_valid=False, error_reason="invalid_json", error_detail=str(exc),
            latency_ms=generation.latency_ms,
        )
    try:
        expr = ir.parse_expr(candidate)
        schema.validate_scope(expr)  # per-sentence scope check; cross-sentence arity is checked at theory level
    except ir.TranslationFailure as exc:
        return TranslationAttempt(
            sentence=sentence, raw_output=generation.raw_text, json_ast=candidate,
            schema_valid=False, error_reason=exc.reason, error_detail=exc.detail,
            latency_ms=generation.latency_ms,
        )
    return TranslationAttempt(
        sentence=sentence, raw_output=generation.raw_text, json_ast=candidate,
        schema_valid=True, error_reason=None, error_detail=None,
        latency_ms=generation.latency_ms,
    )


@dataclass(frozen=True)
class TheoryTranslation:
    premise_attempts: tuple[TranslationAttempt, ...]
    hypothesis_attempt: TranslationAttempt
    all_schema_valid: bool
    premise_dicts: tuple[dict, ...] | None
    hypothesis_dict: dict | None


def translate_theory(runner, allowed: FolioAllowedInput, max_new_tokens: int = 400) -> TheoryTranslation:
    """Translate every premise and the hypothesis independently. Never receives
    or reads a gold label / gold FOL -- `allowed` is a FolioAllowedInput, which
    structurally cannot carry those fields (see io_guard.py).
    """
    premise_attempts = tuple(
        translate_sentence(runner, sentence, max_new_tokens=max_new_tokens) for sentence in allowed.premises
    )
    hypothesis_attempt = translate_sentence(runner, allowed.hypothesis, max_new_tokens=max_new_tokens)

    all_valid = all(a.schema_valid for a in premise_attempts) and hypothesis_attempt.schema_valid
    premise_dicts = tuple(a.json_ast for a in premise_attempts) if all_valid else None
    hypothesis_dict = hypothesis_attempt.json_ast if all_valid else None

    return TheoryTranslation(
        premise_attempts=premise_attempts,
        hypothesis_attempt=hypothesis_attempt,
        all_schema_valid=all_valid,
        premise_dicts=premise_dicts,
        hypothesis_dict=hypothesis_dict,
    )
