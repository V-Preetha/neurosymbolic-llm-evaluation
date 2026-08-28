"""Tests for src/folio/ir.py and src/folio/schema.py."""
from __future__ import annotations

import pytest

from src.folio import ir, schema


def test_unary_predicate_parses():
    node = {"type": "predicate", "name": "suck", "args": ["space"]}
    expr = ir.parse_expr(node)
    assert expr == ir.Predicate(name="suck", args=("space",))


def test_binary_predicate_parses():
    node = {"type": "predicate", "name": "created", "args": ["x", "zelda"]}
    expr = ir.parse_expr(node)
    assert expr == ir.Predicate(name="created", args=("x", "zelda"))


def test_negation_parses():
    node = {"type": "not", "body": {"type": "predicate", "name": "top10", "args": ["zelda"]}}
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.Not)
    assert expr.body == ir.Predicate(name="top10", args=("zelda",))


def test_conjunction_requires_two_args():
    with pytest.raises(ir.TranslationFailure) as exc:
        ir.parse_expr({"type": "and", "args": [{"type": "predicate", "name": "p", "args": ["x"]}]})
    assert exc.value.reason == "malformed_node"


def test_disjunction_parses():
    node = {
        "type": "or",
        "args": [
            {"type": "predicate", "name": "sup", "args": ["m"]},
            {"type": "predicate", "name": "unsup", "args": ["m"]},
        ],
    }
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.Or)
    assert len(expr.args) == 2


def test_implies_parses():
    node = {
        "type": "implies",
        "antecedent": {"type": "predicate", "name": "researcher", "args": ["x"]},
        "consequent": {"type": "predicate", "name": "careful", "args": ["x"]},
    }
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.Implies)


def test_iff_parses():
    node = {
        "type": "iff",
        "left": {"type": "predicate", "name": "a", "args": ["x"]},
        "right": {"type": "predicate", "name": "b", "args": ["x"]},
    }
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.Iff)


def test_forall_single_variable():
    node = {
        "type": "forall",
        "variables": ["x"],
        "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "researcher", "args": ["x"]},
            "consequent": {"type": "predicate", "name": "careful", "args": ["x"]},
        },
    }
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.ForAll)
    assert expr.variables == ("x",)


def test_exists_nested_inside_forall_preserves_structure():
    """'Some scientist admires every researcher.' -- nested quantifiers, not flattened."""
    node = {
        "type": "exists",
        "variables": ["y"],
        "body": {
            "type": "and",
            "args": [
                {"type": "predicate", "name": "scientist", "args": ["y"]},
                {
                    "type": "forall",
                    "variables": ["x"],
                    "body": {
                        "type": "implies",
                        "antecedent": {"type": "predicate", "name": "researcher", "args": ["x"]},
                        "consequent": {"type": "predicate", "name": "admires", "args": ["y", "x"]},
                    },
                },
            ],
        },
    }
    expr = ir.parse_expr(node)
    assert isinstance(expr, ir.Exists)
    inner_forall = expr.body.args[1]
    assert isinstance(inner_forall, ir.ForAll)
    assert inner_forall.variables == ("x",)


def test_multiple_variables_in_one_quantifier():
    node = {
        "type": "forall",
        "variables": ["x", "y"],
        "body": {"type": "predicate", "name": "breedingback", "args": ["x", "y"]},
    }
    expr = ir.parse_expr(node)
    assert expr.variables == ("x", "y")


def test_unknown_node_type_fails():
    with pytest.raises(ir.TranslationFailure) as exc:
        ir.parse_expr({"type": "xor", "left": {}, "right": {}})
    assert exc.value.reason == "unknown_node_type"


def test_missing_type_field_fails():
    with pytest.raises(ir.TranslationFailure) as exc:
        ir.parse_expr({"name": "p", "args": ["x"]})
    assert exc.value.reason == "missing_field"


def test_empty_predicate_args_fails():
    with pytest.raises(ir.TranslationFailure) as exc:
        ir.parse_expr({"type": "predicate", "name": "p", "args": []})
    assert exc.value.reason == "malformed_node"


def test_non_dict_node_fails():
    with pytest.raises(ir.TranslationFailure):
        ir.parse_expr("not a node")


def test_entity_shaped_name_in_variables_fails():
    node = {
        "type": "forall",
        "variables": ["velite"],
        "body": {"type": "predicate", "name": "p", "args": ["velite"]},
    }
    with pytest.raises(ir.TranslationFailure) as exc:
        ir.parse_expr(node)
    assert exc.value.reason == "entity_shaped_variable"


def test_round_trip_to_dict():
    node = {
        "type": "forall",
        "variables": ["x"],
        "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "researcher", "args": ["x"]},
            "consequent": {"type": "predicate", "name": "careful", "args": ["x"]},
        },
    }
    expr = ir.parse_expr(node)
    assert ir.to_dict(expr) == node


# --- schema.py: scope + arity -----------------------------------------------

def test_unbound_variable_in_query_fails():
    expr = ir.parse_expr({"type": "predicate", "name": "top10", "args": ["x"]})
    with pytest.raises(ir.TranslationFailure) as exc:
        schema.validate_scope(expr)
    assert exc.value.reason == "unbound_variable"


def test_bound_variable_passes_scope_check():
    expr = ir.parse_expr({
        "type": "forall",
        "variables": ["x"],
        "body": {"type": "predicate", "name": "suck", "args": ["x"]},
    })
    schema.validate_scope(expr)  # should not raise


def test_entity_arg_never_needs_binding():
    expr = ir.parse_expr({"type": "predicate", "name": "top10", "args": ["zelda"]})
    schema.validate_scope(expr)  # should not raise


def test_shadowed_variable_fails():
    node = {
        "type": "forall",
        "variables": ["x"],
        "body": {
            "type": "exists",
            "variables": ["x"],
            "body": {"type": "predicate", "name": "p", "args": ["x"]},
        },
    }
    expr = ir.parse_expr(node)
    with pytest.raises(ir.TranslationFailure) as exc:
        schema.validate_scope(expr)
    assert exc.value.reason == "variable_shadowing"


def test_arity_conflict_across_theory_fails():
    p1 = ir.parse_expr({"type": "predicate", "name": "created", "args": ["zelda", "japan"]})
    p2 = ir.parse_expr({"type": "predicate", "name": "created", "args": ["zelda"]})
    with pytest.raises(ir.TranslationFailure) as exc:
        schema.validate_theory([p1], p2)
    assert exc.value.reason == "arity_conflict"


def test_consistent_arity_across_theory_passes():
    p1 = ir.parse_expr({"type": "predicate", "name": "created", "args": ["nintendo", "zelda"]})
    h = ir.parse_expr({
        "type": "exists",
        "variables": ["x"],
        "body": {"type": "predicate", "name": "created", "args": ["x", "zelda"]},
    })
    schema.validate_theory([p1], h)  # should not raise: arity is 2 everywhere, all vars bound


def test_validate_theory_checks_scope_per_expression():
    p1 = ir.parse_expr({"type": "predicate", "name": "created", "args": ["x"]})  # unbound x
    h = ir.parse_expr({"type": "predicate", "name": "top10", "args": ["zelda"]})
    with pytest.raises(ir.TranslationFailure) as exc:
        schema.validate_theory([p1], h)
    assert exc.value.reason == "unbound_variable"
