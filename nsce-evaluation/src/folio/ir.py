"""FOLIO symbolic IR: 8 node types, frozen per FOLIO_SYMBOLIC_IR.md.

Parses raw JSON (dicts) into typed, immutable dataclasses. Performs only the
*local*, per-node structural checks (required fields present, correct types,
non-empty arg lists) — cross-tree checks that need whole-theory context
(variable scope binding, predicate arity consistency across premises +
hypothesis) live in schema.py, not here.

No LLM calls anywhere in this module. No auto-repair: a malformed dict
raises TranslationFailure immediately; callers must not retry with another
unconstrained generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


VARIABLE_PATTERN = re.compile(r"^[a-z][0-9]?$")
# At most one trailing digit -- every quantifier variable observed in the FOLIO
# feasibility audit and the gold-FOL oracle sample is "x", "y", "x1", "y2", etc.
# The looser ^[a-z][0-9]*$ (unbounded digits) was tried first and found to
# misclassify entity constants such as "y1995" (a year value, e.g.
# BornIn(ailtonsilva, y1995)) as a bound variable during gold-FOL oracle
# construction (reports/folio_translation_prototype/build_gold_oracle.py,
# idx 199-203) -- a real implementation defect, fixed here rather than
# worked around, since the same collision would affect real Qwen output.

NODE_TYPES = frozenset(
    {"predicate", "not", "and", "or", "implies", "iff", "forall", "exists"}
)


class TranslationFailure(ValueError):
    """Raised whenever a candidate structure cannot become valid IR.

    Carries a machine-readable `reason` code (used by the error taxonomy in
    the prototype report) alongside a human-readable detail string. Never
    caught-and-repaired internally — it always propagates to the caller as
    a hard TRANSLATION_FAILURE.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def is_variable(term: str) -> bool:
    return bool(VARIABLE_PATTERN.fullmatch(term))


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Not:
    body: "Expr"


@dataclass(frozen=True)
class And:
    args: tuple["Expr", ...]


@dataclass(frozen=True)
class Or:
    args: tuple["Expr", ...]


@dataclass(frozen=True)
class Implies:
    antecedent: "Expr"
    consequent: "Expr"


@dataclass(frozen=True)
class Iff:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class ForAll:
    variables: tuple[str, ...]
    body: "Expr"


@dataclass(frozen=True)
class Exists:
    variables: tuple[str, ...]
    body: "Expr"


Expr = Union[Predicate, Not, And, Or, Implies, Iff, ForAll, Exists]


def _require(condition: bool, reason: str, detail: str) -> None:
    if not condition:
        raise TranslationFailure(reason, detail)


def _require_dict(data: object, where: str) -> dict:
    _require(isinstance(data, dict), "malformed_node", f"{where}: expected an object, got {type(data).__name__}")
    return data  # type: ignore[return-value]


def _require_str(value: object, where: str) -> str:
    _require(isinstance(value, str) and value != "", "malformed_node", f"{where}: expected a non-empty string")
    return value  # type: ignore[return-value]


def _require_list(value: object, where: str, min_len: int) -> list:
    _require(isinstance(value, list), "malformed_node", f"{where}: expected a list")
    _require(len(value) >= min_len, "malformed_node", f"{where}: expected at least {min_len} item(s), got {len(value)}")
    return value  # type: ignore[return-value]


def parse_expr(data: object) -> Expr:
    """Parse one raw JSON node (and its subtree) into typed IR, or raise TranslationFailure."""
    node = _require_dict(data, "node")
    node_type = node.get("type")
    _require(isinstance(node_type, str), "missing_field", "node.type is required")
    _require(node_type in NODE_TYPES, "unknown_node_type", f"{node_type!r} is not one of {sorted(NODE_TYPES)}")

    if node_type == "predicate":
        name = _require_str(node.get("name"), "predicate.name")
        args = _require_list(node.get("args"), "predicate.args", min_len=1)
        clean_args = tuple(_require_str(a, "predicate.args[]") for a in args)
        return Predicate(name=name, args=clean_args)

    if node_type == "not":
        body = _require_dict(node.get("body"), "not.body")
        return Not(body=parse_expr(body))

    if node_type in ("and", "or"):
        raw_args = _require_list(node.get("args"), f"{node_type}.args", min_len=2)
        parsed_args = tuple(parse_expr(a) for a in raw_args)
        return And(args=parsed_args) if node_type == "and" else Or(args=parsed_args)

    if node_type == "implies":
        antecedent = parse_expr(_require_dict(node.get("antecedent"), "implies.antecedent"))
        consequent = parse_expr(_require_dict(node.get("consequent"), "implies.consequent"))
        return Implies(antecedent=antecedent, consequent=consequent)

    if node_type == "iff":
        left = parse_expr(_require_dict(node.get("left"), "iff.left"))
        right = parse_expr(_require_dict(node.get("right"), "iff.right"))
        return Iff(left=left, right=right)

    # forall / exists
    variables_raw = _require_list(node.get("variables"), f"{node_type}.variables", min_len=1)
    variables = tuple(_require_str(v, f"{node_type}.variables[]") for v in variables_raw)
    for v in variables:
        _require(is_variable(v), "entity_shaped_variable", f"{node_type}.variables contains non-variable-shaped name {v!r}")
    body = parse_expr(_require_dict(node.get("body"), f"{node_type}.body"))
    return ForAll(variables=variables, body=body) if node_type == "forall" else Exists(variables=variables, body=body)


def to_dict(expr: Expr) -> dict:
    """Inverse of parse_expr, for logging/round-tripping."""
    if isinstance(expr, Predicate):
        return {"type": "predicate", "name": expr.name, "args": list(expr.args)}
    if isinstance(expr, Not):
        return {"type": "not", "body": to_dict(expr.body)}
    if isinstance(expr, And):
        return {"type": "and", "args": [to_dict(a) for a in expr.args]}
    if isinstance(expr, Or):
        return {"type": "or", "args": [to_dict(a) for a in expr.args]}
    if isinstance(expr, Implies):
        return {"type": "implies", "antecedent": to_dict(expr.antecedent), "consequent": to_dict(expr.consequent)}
    if isinstance(expr, Iff):
        return {"type": "iff", "left": to_dict(expr.left), "right": to_dict(expr.right)}
    if isinstance(expr, ForAll):
        return {"type": "forall", "variables": list(expr.variables), "body": to_dict(expr.body)}
    if isinstance(expr, Exists):
        return {"type": "exists", "variables": list(expr.variables), "body": to_dict(expr.body)}
    raise TypeError(f"Unknown IR node: {expr!r}")
