"""Tests for src/folio/v2/validator.py -- Step G static checks (reuses schema.py)."""
from __future__ import annotations

from src.folio.v2 import validator
from src.folio.v2.vocabulary import EntityTable, PredicateTable


def _tables(atom_pairs):
    entity_table = EntityTable()
    predicate_table = PredicateTable()
    predicate_table.build(atom_pairs)
    return entity_table, predicate_table


def test_validate_sentence_valid_forall():
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}, {"id": "A2", "predicate": "bark", "args": ["x"]}],
        "nodes": [
            {"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"},
            {"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"},
        ],
        "root": "N2",
    }
    entity_table, predicate_table = _tables([("dog", 1), ("bark", 1)])
    result = validator.validate_sentence(decomposition, entity_table, predicate_table)
    assert result.valid is True
    assert result.expr is not None


def test_validate_sentence_unbound_variable_is_repairable_error():
    """'Every researcher is careful' with a missing forall -- the canonical
    repair-eligible failure from the phase spec."""
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "researcher", "args": ["x"]}, {"id": "A2", "predicate": "careful", "args": ["x"]}],
        "nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}],
        "root": "N1",  # missing the wrapping forall
    }
    entity_table, predicate_table = _tables([("researcher", 1), ("careful", 1)])
    result = validator.validate_sentence(decomposition, entity_table, predicate_table)
    assert result.valid is False
    assert result.reason == "unbound_variable"


def test_validate_sentence_variable_shadowing_rejected():
    decomposition = {
        "atoms": [{"id": "A1", "predicate": "p", "args": ["x"]}],
        "nodes": [
            {"id": "N1", "op": "forall", "variables": ["x"], "body": "A1"},
            {"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"},
        ],
        "root": "N2",
    }
    entity_table, predicate_table = _tables([("p", 1)])
    result = validator.validate_sentence(decomposition, entity_table, predicate_table)
    assert result.valid is False
    assert result.reason == "variable_shadowing"


def test_validate_theory_catches_arity_conflict_between_two_valid_sentences():
    """Defense in depth: even if a per-sentence-only vocabulary let two
    different arities for the same predicate name through, theory-level
    validate_theory (schema.py, reused unchanged) still catches it."""
    entity_table_a, predicate_table_a = _tables([("treats", 2)])
    entity_table_b, predicate_table_b = _tables([("treats", 1)])
    d1 = {"atoms": [{"id": "A1", "predicate": "treats", "args": ["x", "y"]}], "nodes": [{"id": "N1", "op": "forall", "variables": ["x", "y"], "body": "A1"}], "root": "N1"}
    d2 = {"atoms": [{"id": "A1", "predicate": "treats", "args": ["z"]}], "nodes": [{"id": "N1", "op": "forall", "variables": ["z"], "body": "A1"}], "root": "N1"}
    v1 = validator.validate_sentence(d1, entity_table_a, predicate_table_a)
    v2 = validator.validate_sentence(d2, entity_table_b, predicate_table_b)
    assert v1.valid and v2.valid
    theory = validator.validate_theory([v1.expr], v2.expr)
    assert theory.valid is False
    assert theory.reason == "arity_conflict"


def test_validate_theory_reused_from_schema_on_consistent_theory():
    entity_table, predicate_table = _tables([("dog", 1), ("bark", 1)])
    premise = {
        "atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}, {"id": "A2", "predicate": "bark", "args": ["x"]}],
        "nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}, {"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}],
        "root": "N2",
    }
    hypothesis = {"atoms": [{"id": "A1", "predicate": "dog", "args": ["fido"]}], "nodes": [], "root": "A1"}
    premise_v = validator.validate_sentence(premise, entity_table, predicate_table)
    hypothesis_v = validator.validate_sentence(hypothesis, entity_table, predicate_table)
    assert premise_v.valid and hypothesis_v.valid
    theory = validator.validate_theory([premise_v.expr], hypothesis_v.expr)
    assert theory.valid is True
    assert theory.premises == (premise_v.expr,)
