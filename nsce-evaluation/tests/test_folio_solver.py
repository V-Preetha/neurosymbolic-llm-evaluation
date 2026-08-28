"""Tests for src/folio/solver.py -- verdict engine (TRUE/FALSE/UNKNOWN) + failure handling."""
from __future__ import annotations

from src.folio.solver import classify_theory


def test_true_classification_modus_ponens():
    premises = [
        {"type": "forall", "variables": ["x"], "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "vacuum", "args": ["x"]},
            "consequent": {"type": "predicate", "name": "suck", "args": ["x"]},
        }},
        {"type": "predicate", "name": "vacuum", "args": ["space"]},
    ]
    hypothesis = {"type": "predicate", "name": "suck", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "entailed"
    assert verdict.label == "True"
    assert verdict.solver_statuses["entailment_check"] == "unsat"


def test_false_classification_via_counterexample_rule():
    premises = [
        {"type": "forall", "variables": ["x"], "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "householdapp", "args": ["x"]},
            "consequent": {"type": "not", "body": {"type": "predicate", "name": "suck", "args": ["x"]}},
        }},
        {"type": "predicate", "name": "householdapp", "args": ["duster"]},
    ]
    hypothesis = {
        "type": "forall", "variables": ["x"],
        "body": {"type": "implies",
                 "antecedent": {"type": "predicate", "name": "householdapp", "args": ["x"]},
                 "consequent": {"type": "predicate", "name": "suck", "args": ["x"]}},
    }
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "contradicted"
    assert verdict.label == "False"


def test_unknown_classification_no_relevant_rule():
    premises = [{"type": "predicate", "name": "vacuum", "args": ["space"]}]
    hypothesis = {"type": "predicate", "name": "vampire", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "uncertain"
    assert verdict.label == "Uncertain"
    assert verdict.solver_statuses["entailment_check"] == "sat"
    assert verdict.solver_statuses["contradiction_check"] == "sat"


def test_inconsistent_premises_reported_explicitly_not_forced():
    premises = [
        {"type": "predicate", "name": "suck", "args": ["space"]},
        {"type": "not", "body": {"type": "predicate", "name": "suck", "args": ["space"]}},
    ]
    hypothesis = {"type": "predicate", "name": "vampire", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "inconsistent_premises"
    assert verdict.label is None
    assert verdict.solver_statuses["premises_consistency"] == "unsat"


def test_malformed_translator_output_becomes_translation_failure_not_exception():
    premises = [{"type": "predicate", "name": "vacuum", "args": []}]  # empty args, invalid
    hypothesis = {"type": "predicate", "name": "suck", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "translation_failure"
    assert verdict.label is None
    assert verdict.failure_reason == "malformed_node"


def test_unsupported_node_type_becomes_translation_failure():
    premises = [{"type": "xor", "left": {}, "right": {}}]
    hypothesis = {"type": "predicate", "name": "suck", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "translation_failure"
    assert verdict.failure_reason == "unknown_node_type"


def test_unbound_variable_becomes_translation_failure():
    premises = [{"type": "predicate", "name": "vacuum", "args": ["x"]}]  # x never quantified
    hypothesis = {"type": "predicate", "name": "suck", "args": ["space"]}
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "translation_failure"
    assert verdict.failure_reason == "unbound_variable"


def test_existential_and_conjunctive_query():
    premises = [
        {"type": "forall", "variables": ["x", "y"], "body": {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "breedingback", "args": ["x", "y"]},
            "consequent": {"type": "predicate", "name": "artificiallyselect", "args": ["x"]},
        }},
        {"type": "exists", "variables": ["x", "y"], "body": {
            "type": "and",
            "args": [
                {"type": "predicate", "name": "heckcattle", "args": ["x"]},
                {"type": "predicate", "name": "aurochs", "args": ["y"]},
                {"type": "predicate", "name": "breedingback", "args": ["x", "y"]},
            ],
        }},
    ]
    hypothesis = {
        "type": "exists", "variables": ["x"],
        "body": {"type": "and", "args": [
            {"type": "predicate", "name": "heckcattle", "args": ["x"]},
            {"type": "predicate", "name": "artificiallyselect", "args": ["x"]},
        ]},
    }
    verdict = classify_theory(premises, hypothesis)
    assert verdict.status == "entailed"
    assert verdict.label == "True"
