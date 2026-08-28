"""Tests for src/folio/compiler.py -- deterministic JSON-AST -> Z3, no LLM calls."""
from __future__ import annotations

import z3

from src.folio import ir
from src.folio.compiler import SymbolTable, compile_expr, compile_theory


def _compile(node: dict, symbols: SymbolTable | None = None, var_env: dict | None = None):
    return compile_expr(ir.parse_expr(node), symbols or SymbolTable(), var_env or {})


def test_unary_predicate_compiles_and_is_satisfiable():
    expr = _compile({"type": "predicate", "name": "suck", "args": ["space"]})
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat


def test_binary_predicate_compiles():
    expr = _compile({"type": "predicate", "name": "created", "args": ["nintendo", "zelda"]})
    assert expr.decl().arity() == 2


def test_negation_compiles_and_is_unsat_when_conjoined_with_positive():
    symbols = SymbolTable()
    pos = _compile({"type": "predicate", "name": "suck", "args": ["space"]}, symbols)
    neg = _compile({"type": "not", "body": {"type": "predicate", "name": "suck", "args": ["space"]}}, symbols)
    s = z3.Solver()
    s.add(pos, neg)
    assert s.check() == z3.unsat


def test_conjunction_compiles():
    expr = _compile({
        "type": "and",
        "args": [
            {"type": "predicate", "name": "a", "args": ["obj"]},
            {"type": "predicate", "name": "b", "args": ["obj"]},
        ],
    })
    assert z3.is_and(expr)


def test_disjunction_compiles():
    expr = _compile({
        "type": "or",
        "args": [
            {"type": "predicate", "name": "a", "args": ["obj"]},
            {"type": "predicate", "name": "b", "args": ["obj"]},
        ],
    })
    assert z3.is_or(expr)


def test_implication_modus_ponens():
    symbols = SymbolTable()
    rule = _compile({
        "type": "forall",
        "variables": ["x"],
        "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "vacuum", "args": ["x"]},
            "consequent": {"type": "predicate", "name": "suck", "args": ["x"]},
        },
    }, symbols)
    fact = _compile({"type": "predicate", "name": "vacuum", "args": ["space"]}, symbols)
    negated_conclusion = _compile({"type": "not", "body": {"type": "predicate", "name": "suck", "args": ["space"]}}, symbols)
    s = z3.Solver()
    s.add(rule, fact, negated_conclusion)
    assert s.check() == z3.unsat  # rule + fact entail suck(space); negating it is unsat


def test_universal_quantification_compiles():
    expr = _compile({
        "type": "forall",
        "variables": ["x"],
        "body": {"type": "predicate", "name": "suck", "args": ["x"]},
    })
    assert z3.is_quantifier(expr) and expr.is_forall()


def test_existential_quantification_compiles():
    expr = _compile({
        "type": "exists",
        "variables": ["x"],
        "body": {"type": "predicate", "name": "suck", "args": ["x"]},
    })
    assert z3.is_quantifier(expr) and not expr.is_forall()


def test_nested_quantifiers_compile():
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
    expr = _compile(node)
    assert z3.is_quantifier(expr) and not expr.is_forall()


def test_multiple_variables_in_one_quantifier_compile():
    expr = _compile({
        "type": "forall",
        "variables": ["x", "y"],
        "body": {"type": "predicate", "name": "breedingback", "args": ["x", "y"]},
    })
    assert z3.is_quantifier(expr) and expr.num_vars() == 2


def test_iff_compiles():
    expr = _compile({
        "type": "iff",
        "left": {"type": "predicate", "name": "a", "args": ["obj"]},
        "right": {"type": "predicate", "name": "b", "args": ["obj"]},
    })
    assert z3.is_eq(expr)


def test_shared_symbol_table_reuses_predicate_across_premises_and_hypothesis():
    premises = [ir.parse_expr({"type": "predicate", "name": "top10", "args": ["zelda"]})]
    hypothesis = ir.parse_expr({"type": "not", "body": {"type": "predicate", "name": "top10", "args": ["zelda"]}})
    compiled_premises, compiled_hypothesis, symbols = compile_theory(premises, hypothesis)
    assert symbols.predicates[("top10", 1)] is not None
    s = z3.Solver()
    s.add(*compiled_premises, compiled_hypothesis)
    assert s.check() == z3.unsat  # top10(zelda) and not top10(zelda) can't both hold
