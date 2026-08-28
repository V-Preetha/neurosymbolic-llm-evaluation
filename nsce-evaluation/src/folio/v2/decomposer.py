"""Raw English sentence -> semantic-decomposition JSON, via the frozen Qwen3-8B checkpoint.

This is Translation-v2's *only* LLM-facing stage that asks for a complete
sentence structure (repair.py issues a second, more constrained call, but
only on failure). It intentionally does NOT ask for a nested Z3-shaped AST
the way Translation-v1 did -- it asks for:

  - entities: proper-name constants mentioned in the sentence, with a
    canonical (snake_case) form (Step B/C input material)
  - atoms: flat predicate applications ("clauses" in the task's Step D
    sense), each with its own id
  - nodes: boolean/quantifier structure, referencing atom ids or other node
    ids by id rather than embedding them inline -- this is Step E (explicit,
    non-heuristic quantifier scope) and Step F (component references for a
    deterministic builder) at once
  - root: the id (atom or node) that is the whole sentence

The deterministic assembly of this into src/folio/ir.py's Expr tree happens
in ast_builder.py, not here. This module only extracts and shallow-checks
JSON; it never calls Z3, never sees a gold label, and never sees gold FOL.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.folio.ir import TranslationFailure


DECOMPOSE_PROMPT = """You are a semantic parser for first-order logic. Decompose ONE English sentence into a structured JSON object. Do NOT try to write nested logic in one shot -- break it into small pieces first.

Output JSON with these fields:
- "entities": list of {"surface": "<exact text as it appears>", "canonical": "<snake_case name>"} for every specific named person/place/thing (proper noun) in the sentence. Omit this field (or use []) if there are none.
- "atoms": list of {"id": "A1", "predicate": "<lowercase_predicate_name>", "args": ["<term>", ...]}. Each atom is ONE simple predicate application. A term is either a single-letter variable ("x", "y", optionally with one trailing digit like "x1") or an entity's canonical name from "entities".
- "nodes": list of boolean/quantifier structure nodes, each with its own "id", referencing atom ids or other node ids (never inlining another node). Allowed node shapes:
  - {"id": "N1", "op": "not", "body": "<id>"}
  - {"id": "N1", "op": "and", "args": ["<id>", "<id>", ...]}   (2 or more)
  - {"id": "N1", "op": "or", "args": ["<id>", "<id>", ...]}    (2 or more)
  - {"id": "N1", "op": "implies", "antecedent": "<id>", "consequent": "<id>"}
  - {"id": "N1", "op": "iff", "left": "<id>", "right": "<id>"}
  - {"id": "N1", "op": "forall", "variables": ["x", ...], "body": "<id>"}
  - {"id": "N1", "op": "exists", "variables": ["x", ...], "body": "<id>"}
- "root": the id (an atom id or a node id) of the top-level structure -- the whole sentence.

Quantifier scope matters and is NOT interchangeable. Consider:
"Some scientist admires every researcher." means one specific scientist admires ALL researchers:
  atoms: A1=scientist(y), A2=researcher(x), A3=admires(y,x)
  nodes: N1={"op":"implies","antecedent":"A2","consequent":"A3"}, N2={"op":"forall","variables":["x"],"body":"N1"}, N3={"op":"and","args":["A1","N2"]}, N4={"op":"exists","variables":["y"],"body":"N3"}
  root: "N4"
This is different from "Every researcher is admired by some scientist" read with the existential outermost -- do not silently flatten quantifiers into one list; nest them exactly where they scope.

Example 1.
Sentence: "Every doctor who treats a musician admires that musician."
JSON: {"entities": [], "atoms": [{"id": "A1", "predicate": "doctor", "args": ["x"]}, {"id": "A2", "predicate": "musician", "args": ["y"]}, {"id": "A3", "predicate": "treats", "args": ["x", "y"]}, {"id": "A4", "predicate": "admires", "args": ["x", "y"]}], "nodes": [{"id": "N1", "op": "and", "args": ["A1", "A2", "A3"]}, {"id": "N2", "op": "implies", "antecedent": "N1", "consequent": "A4"}, {"id": "N3", "op": "forall", "variables": ["x", "y"], "body": "N2"}], "root": "N3"}

Example 2.
Sentence: "Fluffy is a cat."
JSON: {"entities": [{"surface": "Fluffy", "canonical": "fluffy"}], "atoms": [{"id": "A1", "predicate": "cat", "args": ["fluffy"]}], "nodes": [], "root": "A1"}

Example 3.
Sentence: "Some cat is sleeping and is not hungry."
JSON: {"entities": [], "atoms": [{"id": "A1", "predicate": "cat", "args": ["x"]}, {"id": "A2", "predicate": "sleeping", "args": ["x"]}, {"id": "A3", "predicate": "hungry", "args": ["x"]}], "nodes": [{"id": "N1", "op": "not", "body": "A3"}, {"id": "N2", "op": "and", "args": ["A1", "A2", "N1"]}, {"id": "N3", "op": "exists", "variables": ["x"], "body": "N2"}], "root": "N3"}

Output ONLY the JSON object -- no explanation, no markdown fences, no extra text.
Now decompose this sentence.
Sentence: "{sentence}"
JSON:"""


@dataclass(frozen=True)
class DecompositionAttempt:
    sentence: str
    raw_output: str
    decomposition: dict | None
    shape_valid: bool
    error_reason: str | None
    error_detail: str | None
    latency_ms: float
    repaired: bool = False


def render_decompose_prompt(sentence: str) -> str:
    return DECOMPOSE_PROMPT.replace("{sentence}", sentence.strip())


def _extract_json_object(raw_text: str) -> dict:
    """Deterministic extraction of the first JSON object in the model's output."""
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


def _check_shape(decomposition: dict) -> None:
    """Cheap, local structural checks -- not the full Step G validation
    (that needs the global vocabulary and lives in validator.py). This only
    guards against a decomposition so malformed that ast_builder cannot even
    start (e.g. "atoms" not a list).
    """
    for key in ("atoms", "nodes"):
        value = decomposition.get(key, [])
        if not isinstance(value, list):
            raise TranslationFailure("malformed_node", f"decomposition.{key} must be a list")
    if "root" not in decomposition or not isinstance(decomposition["root"], str) or not decomposition["root"]:
        raise TranslationFailure("missing_field", "decomposition.root (a non-empty string id) is required")
    entities = decomposition.get("entities", [])
    if not isinstance(entities, list):
        raise TranslationFailure("malformed_node", "decomposition.entities must be a list")
    for atom in decomposition.get("atoms", []):
        if not isinstance(atom, dict) or "id" not in atom or "predicate" not in atom or "args" not in atom:
            raise TranslationFailure("missing_field", f"atom {atom!r} missing id/predicate/args")
        if not isinstance(atom["args"], list):
            raise TranslationFailure("malformed_node", f"atom {atom['id']!r}.args must be a list")
    for node in decomposition.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node or "op" not in node:
            raise TranslationFailure("missing_field", f"node {node!r} missing id/op")


def decompose_sentence(runner, sentence: str, max_new_tokens: int = 500) -> DecompositionAttempt:
    prompt = render_decompose_prompt(sentence)
    generation = runner.generate_text(prompt, max_new_tokens=max_new_tokens)
    return _parse_generation(sentence, generation.raw_text, generation.latency_ms)


def _parse_generation(sentence: str, raw_text: str, latency_ms: float, repaired: bool = False) -> DecompositionAttempt:
    candidate = None
    try:
        candidate = _extract_json_object(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return DecompositionAttempt(
            sentence=sentence, raw_output=raw_text, decomposition=None,
            shape_valid=False, error_reason="invalid_json", error_detail=str(exc),
            latency_ms=latency_ms, repaired=repaired,
        )
    try:
        _check_shape(candidate)
    except TranslationFailure as exc:
        return DecompositionAttempt(
            sentence=sentence, raw_output=raw_text, decomposition=candidate,
            shape_valid=False, error_reason=exc.reason, error_detail=exc.detail,
            latency_ms=latency_ms, repaired=repaired,
        )
    return DecompositionAttempt(
        sentence=sentence, raw_output=raw_text, decomposition=candidate,
        shape_valid=True, error_reason=None, error_detail=None,
        latency_ms=latency_ms, repaired=repaired,
    )
