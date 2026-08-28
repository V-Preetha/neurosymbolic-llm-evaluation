"""Deterministic parser for the controlled ProofWriter English grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass


VARIABLE = "?x"
VERB_FORMS = {
    "chase": "chase", "chases": "chase",
    "eat": "eat", "eats": "eat",
    "like": "like", "likes": "like",
    "need": "need", "needs": "need",
    "see": "see", "sees": "see",
    "visit": "visit", "visits": "visit",
}
VERB_PATTERN = "|".join(sorted(VERB_FORMS, key=len, reverse=True))


class ParserError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def opposite(self) -> "Atom":
        return Atom(self.predicate, self.arguments, not self.negated)

    def render(self) -> str:
        name = "Not " + self.predicate if self.negated else self.predicate
        args = ", ".join(self.arguments)
        return f"{name}({args})"


@dataclass(frozen=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom
    source: str


@dataclass(frozen=True)
class ParsedTheory:
    facts: tuple[Atom, ...]
    rules: tuple[Rule, ...]


def _clean(text: str) -> str:
    return text.strip().rstrip(".").strip()


def _entity(text: str) -> str:
    value = text.strip().lower()
    if value in {"someone", "something", "they", "it", "them"}:
        return VARIABLE
    if value.startswith("the "):
        value = value[4:]
    return value.replace(" ", "_")


def _adjectives(text: str) -> list[tuple[str, bool]]:
    parts = [part.strip() for part in re.split(r",\s*|\s+and\s+", text) if part.strip()]
    output: list[tuple[str, bool]] = []
    for part in parts:
        negated = part.startswith("not ")
        predicate = part[4:] if negated else part
        if not re.fullmatch(r"[a-z]+", predicate, flags=re.IGNORECASE):
            raise ParserError(f"Unsupported adjective phrase: {text!r}")
        output.append((predicate.lower(), negated))
    return output


def parse_clause(text: str, default_subject: str | None = None) -> tuple[list[Atom], str]:
    clause = _clean(text).strip(",")
    if default_subject and re.fullmatch(r"(?:not\s+)?[A-Za-z]+", clause):
        return [Atom(predicate, (default_subject,), negated) for predicate, negated in _adjectives(clause)], default_subject

    unary = re.fullmatch(r"(.+?)\s+(?:is|are)\s+(.+)", clause, flags=re.IGNORECASE)
    if unary:
        subject = _entity(unary.group(1))
        atoms = [Atom(predicate, (subject,), negated) for predicate, negated in _adjectives(unary.group(2))]
        return atoms, subject

    negative_binary = re.fullmatch(
        rf"(.+?)\s+(?:do|does)\s+not\s+({VERB_PATTERN})\s+(.+)", clause, flags=re.IGNORECASE
    )
    if negative_binary:
        subject = _entity(negative_binary.group(1))
        predicate = VERB_FORMS[negative_binary.group(2).lower()]
        return [Atom(predicate, (subject, _entity(negative_binary.group(3))), True)], subject

    binary = re.fullmatch(rf"(.+?)\s+({VERB_PATTERN})\s+(.+)", clause, flags=re.IGNORECASE)
    if binary:
        subject = _entity(binary.group(1))
        predicate = VERB_FORMS[binary.group(2).lower()]
        obj = _entity(binary.group(3))
        return [Atom(predicate, (subject, obj), False)], subject

    if default_subject:
        negative_shorthand = re.fullmatch(
            rf"(?:do|does)\s+not\s+({VERB_PATTERN})\s+(.+)", clause, flags=re.IGNORECASE
        )
        if negative_shorthand:
            predicate = VERB_FORMS[negative_shorthand.group(1).lower()]
            return [Atom(predicate, (default_subject, _entity(negative_shorthand.group(2))), True)], default_subject
        shorthand = re.fullmatch(rf"({VERB_PATTERN})\s+(.+)", clause, flags=re.IGNORECASE)
        if shorthand:
            predicate = VERB_FORMS[shorthand.group(1).lower()]
            return [Atom(predicate, (default_subject, _entity(shorthand.group(2))), False)], default_subject
    raise ParserError(f"Unsupported clause: {text!r}")


def parse_sentence(sentence: str) -> Atom | tuple[Rule, ...]:
    text = _clean(sentence)
    if text.lower().startswith("if "):
        match = re.fullmatch(r"If\s+(.+?)\s+then\s+(.+)", text, flags=re.IGNORECASE)
        if not match:
            raise ParserError(f"Malformed conditional: {sentence!r}")
        premise_text, conclusion_text = match.groups()
        premises: list[Atom] = []
        last_subject: str | None = None
        for part in re.split(r"\s+and\s+", premise_text, flags=re.IGNORECASE):
            atoms, last_subject = parse_clause(part, last_subject)
            premises.extend(atoms)
        conclusions, _ = parse_clause(conclusion_text, last_subject)
        return tuple(Rule(tuple(premises), conclusion, sentence) for conclusion in conclusions)

    universal = re.fullmatch(
        r"(?:All\s+)?(.+?)\s+(?:people|things)\s+are\s+(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if universal:
        premises = tuple(Atom(predicate, (VARIABLE,), negated) for predicate, negated in _adjectives(universal.group(1)))
        conclusions = [Atom(predicate, (VARIABLE,), negated) for predicate, negated in _adjectives(universal.group(2))]
        return tuple(Rule(premises, conclusion, sentence) for conclusion in conclusions)

    atoms, _ = parse_clause(text)
    if len(atoms) != 1:
        raise ParserError(f"Fact expanded to multiple atoms: {sentence!r}")
    return atoms[0]


def parse_theory(sentences: list[str]) -> ParsedTheory:
    facts: list[Atom] = []
    rules: list[Rule] = []
    for sentence in sentences:
        parsed = parse_sentence(sentence)
        if isinstance(parsed, Atom):
            facts.append(parsed)
        else:
            rules.extend(parsed)
    return ParsedTheory(tuple(facts), tuple(rules))


def parse_query(text: str) -> Atom:
    parsed = parse_sentence(text)
    if not isinstance(parsed, Atom):
        raise ParserError(f"Query is not atomic: {text!r}")
    return parsed
