"""Typed-dictionary FOL AST parsing, normalization, and validation."""

from __future__ import annotations

import json
import re
from typing import Any

NODES = {"predicate", "equals", "not", "and", "or", "implies", "iff", "forall", "exists"}
IDENTIFIER = re.compile(r"(?:[^\W\d]|_)[\w.\-]*\Z", re.UNICODE)
TERM = re.compile(r"[\w][\w.'\-]*\Z", re.UNICODE)
VARIABLE = re.compile(r"[tuvwxyz](?:[0-9]+)?\Z")


class BridgeError(ValueError):
    def __init__(self, status: str, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def extract_json(text: str) -> dict[str, Any]:
    if not text.strip():
        raise BridgeError("empty_output", "model returned no text")
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start = clean.find("{")
    if start < 0:
        try:
            value = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise BridgeError("malformed_output", str(exc)) from exc
    else:
        decoder = json.JSONDecoder()
        try:
            value, end = decoder.raw_decode(clean, start)
        except json.JSONDecodeError as exc:
            raise BridgeError("malformed_output", str(exc)) from exc
        tail = clean[end:]
        for offset, character in enumerate(tail):
            if character != "{":
                continue
            try:
                extra, _ = decoder.raw_decode(tail, offset)
            except json.JSONDecodeError:
                continue
            if isinstance(extra, dict):
                raise BridgeError("malformed_output", "multiple JSON objects in model output")
    if not isinstance(value, dict):
        raise BridgeError("schema_invalid", "top level must be object")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise BridgeError("ast_invalid", f"invalid {label}: {value!r}")
    return value


def _term(value: Any, label: str) -> str:
    if not isinstance(value, str) or not TERM.fullmatch(value):
        raise BridgeError("ast_invalid", f"invalid {label}: {value!r}")
    return value


def validate_expr(
    node: Any,
    arities: dict[str, set[int]] | None = None,
    bound: set[str] | None = None,
) -> None:
    arities = {} if arities is None else arities
    bound = set() if bound is None else set(bound)
    if not isinstance(node, dict) or node.get("type") not in NODES:
        raise BridgeError("ast_invalid", "unknown/missing node type")
    kind = node["type"]
    if kind == "predicate":
        name, args = node.get("name"), node.get("args")
        _identifier(name, "predicate name")
        if not isinstance(args, list):
            raise BridgeError("ast_invalid", "predicate arguments must be a list")
        for arg in args:
            _term(arg, "term")
            if VARIABLE.fullmatch(arg) and arg not in bound:
                raise BridgeError("ast_invalid", f"free variable: {arg}")
        arities.setdefault(name,set()).add(len(args))
        return
    if kind == "equals":
        for label in ("left", "right"):
            term = _term(node.get(label), f"equality {label} term")
            if VARIABLE.fullmatch(term) and term not in bound:
                raise BridgeError("ast_invalid", f"free variable: {term}")
        return
    if kind == "not":
        validate_expr(node.get("body"), arities, bound)
        return
    if kind in {"and", "or"}:
        args = node.get("args")
        if not isinstance(args, list) or len(args) < 2:
            raise BridgeError("ast_invalid", f"{kind} requires >=2 args")
        for child in args:
            validate_expr(child, arities, bound)
        return
    if kind == "implies":
        validate_expr(node.get("antecedent"), arities, bound)
        validate_expr(node.get("consequent"), arities, bound)
        return
    if kind == "iff":
        validate_expr(node.get("left"), arities, bound)
        validate_expr(node.get("right"), arities, bound)
        return
    variables = node.get("variables")
    if not isinstance(variables, list) or not variables:
        raise BridgeError("ast_invalid", "quantifier requires variables")
    checked = [_identifier(variable, "variable") for variable in variables]
    if len(set(checked)) != len(checked) or not all(VARIABLE.fullmatch(x) for x in checked):
        raise BridgeError("ast_invalid", "bad variables")
    validate_expr(node.get("body"), arities, bound | set(checked))


def validate_theory(value: Any) -> None:
    if not isinstance(value, dict):
        raise BridgeError("schema_invalid", "theory must be object")
    premises, hypothesis = value.get("premises"), value.get("hypothesis")
    if not isinstance(premises, list) or not premises or not isinstance(hypothesis, dict):
        raise BridgeError("schema_invalid", "premises list and hypothesis required")
    arities: dict[str, set[int]] = {}
    for expression in [*premises, hypothesis]:
        validate_expr(expression, arities, set())


def normalized_logic(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
