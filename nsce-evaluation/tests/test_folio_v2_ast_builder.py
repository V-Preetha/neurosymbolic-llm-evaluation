"""Tests for src/folio/v2/ast_builder.py -- deterministic decomposition -> ir.Expr."""
from __future__ import annotations

import pytest

from src.folio import ir
from src.folio.ir import TranslationFailure
from src.folio.v2 import ast_builder
from src.folio.v2.vocabulary import EntityTable, PredicateTable


def _tables(atom_pairs):
    entity_table = EntityTable()
    predicate_table = PredicateTable()
    predicate_table.build(atom_pairs)
    return entity_table, predicate_table


def test_build_expr_simple_fact():
    decomposition = {
        "entities": [{"surface": "Fluffy", "canonical": "fluffy"}],
        "atoms": [{"id": "A1", "predicate": "cat", "args": ["fluffy"]}],
        "nodes": [],
        "root": "A1",
    }
    entity_table, predicate_table = _tables([("cat", 1)])
    expr = ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert expr == ir.Predicate(name="cat", args=("fluffy",))


def test_build_expr_forall_implies_and():
    """Every doctor who treats a musician admires that musician."""
    decomposition = {
        "atoms": [
            {"id": "A1", "predicate": "doctor", "args": ["x"]},
            {"id": "A2", "predicate": "musician", "args": ["y"]},
            {"id": "A3", "predicate": "treats", "args": ["x", "y"]},
            {"id": "A4", "predicate": "admires", "args": ["x", "y"]},
        ],
        "nodes": [
            {"id": "N1", "op": "and", "args": ["A1", "A2", "A3"]},
            {"id": "N2", "op": "implies", "antecedent": "N1", "consequent": "A4"},
            {"id": "N3", "op": "forall", "variables": ["x", "y"], "body": "N2"},
        ],
        "root": "N3",
    }
    entity_table, predicate_table = _tables([("doctor", 1), ("musician", 1), ("treats", 2), ("admires", 2)])
    expr = ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert isinstance(expr, ir.ForAll)
    assert expr.variables == ("x", "y")
    assert isinstance(expr.body, ir.Implies)
    assert isinstance(expr.body.antecedent, ir.And)
    assert expr.body.consequent == ir.Predicate(name="admires", args=("x", "y"))


def test_build_expr_nested_quantifier_scope_some_scientist_admires_every_researcher():
    """Distinguishes exists-outer (one scientist admiring all researchers) from a
    flattened reading -- the whole point of Step E/F not silently prenexing quantifiers.
    """
    decomposition = {
        "atoms": [
            {"id": "A1", "predicate": "scientist", "args": ["y"]},
            {"id": "A2", "predicate": "researcher", "args": ["x"]},
            {"id": "A3", "predicate": "admires", "args": ["y", "x"]},
        ],
        "nodes": [
            {"id": "N1", "op": "implies", "antecedent": "A2", "consequent": "A3"},
            {"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"},
            {"id": "N3", "op": "and", "args": ["A1", "N2"]},
            {"id": "N4", "op": "exists", "variables": ["y"], "body": "N3"},
        ],
        "root": "N4",
    }
    entity_table, predicate_table = _tables([("scientist", 1), ("researcher", 1), ("admires", 2)])
    expr = ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert isinstance(expr, ir.Exists)
    assert expr.variables == ("y",)
    assert isinstance(expr.body, ir.And)
    inner_forall = expr.body.args[1]
    assert isinstance(inner_forall, ir.ForAll)
    assert inner_forall.variables == ("x",)


def test_build_expr_entity_args_resolve_through_entity_table():
    decomposition = {
        "entities": [{"surface": "Ailton Silva", "canonical": "ailton_silva"}],
        "atoms": [{"id": "A1", "predicate": "doctor", "args": ["Ailton Silva"]}],
        "nodes": [],
        "root": "A1",
    }
    entity_table, predicate_table = _tables([("doctor", 1)])
    expr = ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert expr.args == ("ailton_silva",)


def test_build_expr_dangling_reference_raises():
    decomposition = {"atoms": [], "nodes": [{"id": "N1", "op": "not", "body": "A_MISSING"}], "root": "N1"}
    entity_table, predicate_table = _tables([])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "dangling_reference"


def test_build_expr_cyclic_reference_raises():
    decomposition = {
        "atoms": [],
        "nodes": [{"id": "N1", "op": "not", "body": "N2"}, {"id": "N2", "op": "not", "body": "N1"}],
        "root": "N1",
    }
    entity_table, predicate_table = _tables([])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "cyclic_reference"


def test_build_expr_unknown_operator_raises():
    decomposition = {"atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}], "nodes": [{"id": "N1", "op": "xor", "args": ["A1"]}], "root": "N1"}
    entity_table, predicate_table = _tables([("dog", 1)])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "unknown_operator"


def test_build_expr_duplicate_id_raises():
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}],
        "nodes": [{"id": "A1", "op": "not", "body": "A1"}],
        "root": "A1",
    }
    entity_table, predicate_table = _tables([("dog", 1)])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "duplicate_id"


def test_build_expr_entity_shaped_variable_in_quantifier_raises():
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "dog", "args": ["fido"]}],
        "nodes": [{"id": "N1", "op": "forall", "variables": ["fido"], "body": "A1"}],
        "root": "N1",
    }
    entity_table, predicate_table = _tables([("dog", 1)])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "entity_shaped_variable"


def test_build_expr_and_with_fewer_than_two_args_raises():
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}],
        "nodes": [{"id": "N1", "op": "and", "args": ["A1"]}],
        "root": "N1",
    }
    entity_table, predicate_table = _tables([("dog", 1)])
    with pytest.raises(TranslationFailure) as exc:
        ast_builder.build_expr(decomposition, entity_table, predicate_table)
    assert exc.value.reason == "malformed_node"


def test_collect_atom_predicate_arities():
    decomposition = {"atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}, {"id": "A2", "predicate": "treats", "args": ["x", "y"]}]}
    assert ast_builder.collect_atom_predicate_arities(decomposition) == [("dog", 1), ("treats", 2)]
