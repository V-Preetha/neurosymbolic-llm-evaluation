"""Deterministic Horn-rule forward chaining with explicit negation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.nsce.parser import Atom, Rule, VARIABLE


@dataclass(frozen=True)
class Derivation:
    atom: Atom
    rule: Rule | None
    premises: tuple[Atom, ...]


@dataclass(frozen=True)
class ReasoningResult:
    status: str
    prediction: str
    query: Atom
    derivations: dict[Atom, Derivation]
    proof_trace: tuple[str, ...]
    inconsistent: bool


def _match(pattern: Atom, fact: Atom, substitution: dict[str, str]) -> dict[str, str] | None:
    if pattern.predicate != fact.predicate or pattern.negated != fact.negated:
        return None
    if len(pattern.arguments) != len(fact.arguments):
        return None
    result = dict(substitution)
    for expected, actual in zip(pattern.arguments, fact.arguments):
        if expected == VARIABLE:
            existing = result.get(VARIABLE)
            if existing is not None and existing != actual:
                return None
            result[VARIABLE] = actual
        elif expected != actual:
            return None
    return result


def _instantiate(atom: Atom, substitution: dict[str, str]) -> Atom | None:
    arguments: list[str] = []
    for argument in atom.arguments:
        if argument == VARIABLE:
            if VARIABLE not in substitution:
                return None
            arguments.append(substitution[VARIABLE])
        else:
            arguments.append(argument)
    return Atom(atom.predicate, tuple(arguments), atom.negated)


def _substitutions(rule: Rule, knowledge: set[Atom]) -> Iterable[tuple[dict[str, str], tuple[Atom, ...]]]:
    states: list[tuple[dict[str, str], tuple[Atom, ...]]] = [({}, ())]
    for premise in rule.premises:
        next_states: list[tuple[dict[str, str], tuple[Atom, ...]]] = []
        for substitution, matched in states:
            for fact in sorted(knowledge):
                unified = _match(premise, fact, substitution)
                if unified is not None:
                    next_states.append((unified, matched + (fact,)))
        states = next_states
        if not states:
            break
    return states


def forward_chain(facts: Iterable[Atom], rules: Iterable[Rule]) -> dict[Atom, Derivation]:
    derivations = {fact: Derivation(fact, None, ()) for fact in facts}
    knowledge = set(derivations)
    changed = True
    ordered_rules = tuple(rules)
    while changed:
        changed = False
        for rule in ordered_rules:
            for substitution, premises in _substitutions(rule, knowledge):
                conclusion = _instantiate(rule.conclusion, substitution)
                if conclusion is not None and conclusion not in knowledge:
                    knowledge.add(conclusion)
                    derivations[conclusion] = Derivation(conclusion, rule, premises)
                    changed = True
    return derivations


def _trace(target: Atom, derivations: dict[Atom, Derivation]) -> tuple[str, ...]:
    output: list[str] = []
    visited: set[Atom] = set()

    def visit(atom: Atom) -> None:
        if atom in visited:
            return
        visited.add(atom)
        derivation = derivations[atom]
        for premise in derivation.premises:
            visit(premise)
        if derivation.rule is None:
            output.append(f"Fact: {atom.render()}")
        else:
            joined = ", ".join(premise.render() for premise in derivation.premises)
            output.append(f"Rule: {joined} -> {atom.render()}")

    visit(target)
    return tuple(output)


def classify(query: Atom, facts: Iterable[Atom], rules: Iterable[Rule]) -> ReasoningResult:
    derivations = forward_chain(facts, rules)
    supported = query in derivations
    contradicted = query.opposite() in derivations
    inconsistent = supported and contradicted
    if supported:
        target, status, prediction = query, "supported", "true"
    elif contradicted:
        target, status, prediction = query.opposite(), "contradicted", "false"
    else:
        return ReasoningResult("unknown", "unknown", query, derivations, (), False)
    return ReasoningResult(status, prediction, query, derivations, _trace(target, derivations), inconsistent)

