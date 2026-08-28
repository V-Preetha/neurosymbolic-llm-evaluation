"""Deterministic parser: fvossel/t5-base-nl-to-fol raw output -> src/folio/ir.py dict AST.

The model does NOT emit our JSON AST -- it emits a flat string in its own
FOL surface syntax, observed directly from the loaded checkpoint (smoke
tests, 2026-08-17):

    "FORALLx (Human(x) IMPLIES Mammal(x))"
    "EXISTSx (Doctor(x) AND Researcher(x))"
    "NOTEXISTSx (Bird(x) AND Mammal(x))"
    "NOTStudent(john)"
    "FORALLx (OwnsDog(x) IMPLIES EXISTSy (Animal(y) AND Loves(x, y)))"
    "FORALLx (Person(x) IMPLIES (Happy(x) IFF (Healthy(x) AND Wealthy(x))))"

Grammar notes that drive the tokenizer/parser below (all confirmed by
smoke-testing the actual checkpoint, not assumed from the model card):

  * A quantifier and its bound variable are fused into one token with no
    space: "FORALLx", "EXISTSy", "NOTFORALLx", "NOTEXISTSx". The leading
    NOT negates the whole quantified formula (Not(ForAll(...))).
  * A bare "NOT" directly prefixes a predicate name with no space:
    "NOTGraduate(x)" -> Not(Graduate(x)). NOT is never seen as its own
    whitespace-separated token in this model's output.
  * Predicates are PascalCase names followed by a parenthesized,
    comma-separated argument list: "Loves(x, y)".
  * Binary connectives AND / OR / XOR / IMPLIES / IFF appear as their own
    whitespace-separated tokens between operands, sometimes with explicit
    parens and sometimes flat inside one enclosing paren (e.g.
    "Lawyer(x) AND PassBarExam(x) IMPLIES QualifiedAttorney(x)"), so a
    precedence-climbing parser is used for the flat case: AND binds
    tighter than OR/XOR, which bind tighter than IMPLIES/IFF.

This module performs no semantic changes -- only syntactic normalization
(symbol synonyms below) and structural translation into our IR node shapes.
A string that doesn't fit this grammar raises TranslationFailure with
reason "t5_parse_error"; callers must not silently repair it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.folio.ir import TranslationFailure, is_variable

# Synonym normalization (Step 5 of the integration spec): accept both this
# model's fused-token grammar and the more conventional symbol/word forms,
# without changing meaning. Only the fused-token forms have actually been
# observed from this checkpoint; the rest are defensive.
_SYMBOL_SYNONYMS = [
    ("<->", "IFF"), ("↔", "IFF"),
    ("->", "IMPLIES"), ("→", "IMPLIES"),
    ("&", "AND"), ("∧", "AND"),
    ("|", "OR"), ("∨", "OR"),
    ("⊕", "XOR"),
    ("~", "NOT"), ("¬", "NOT"),
    ("∀", "FORALL"),
    ("∃", "EXISTS"),
]

_TOKEN_RE = re.compile(
    r"""
    (?P<QUANT>(?:NOT)?(?:FORALL|EXISTS)[a-z][0-9]{0,1}) |
    (?P<NOTPRED>NOT[A-Z][A-Za-z0-9_]*(?=\()) |
    (?P<PRED>[A-Z][A-Za-z0-9_]*(?=\())            |
    (?P<OP>AND|OR|XOR|IMPLIES|IFF|NOT)            |
    (?P<LPAREN>\()                                |
    (?P<RPAREN>\))                                |
    (?P<COMMA>,)                                  |
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)             |
    (?P<WS>\s+)
    """,
    re.VERBOSE,
)

_QUANT_PATTERN = re.compile(r"^(NOT)?(FORALL|EXISTS)([a-z][0-9]{0,1})$")

_BINARY_PRECEDENCE = {"IFF": 1, "IMPLIES": 1, "OR": 2, "XOR": 2, "AND": 3}


@dataclass
class _Token:
    kind: str
    text: str


def _normalize(raw: str) -> str:
    text = raw.strip()
    for symbol, replacement in _SYMBOL_SYNONYMS:
        text = text.replace(symbol, f" {replacement} ")
    return text


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    n = len(text)
    while pos < n:
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise TranslationFailure(
                "t5_parse_error", f"unrecognized character at position {pos} in {text!r}"
            )
        pos = match.end()
        kind = match.lastgroup
        if kind == "WS":
            continue
        tokens.append(_Token(kind=kind, text=match.group()))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token], source: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self.source = source

    def _peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise TranslationFailure("t5_parse_error", f"unexpected end of input in {self.source!r}")
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> _Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            got = tok.text if tok else "<eof>"
            raise TranslationFailure(
                "t5_parse_error", f"expected {kind} but got {got!r} in {self.source!r}"
            )
        return self._advance()

    def parse_theory(self) -> dict:
        expr = self.parse_expr(0)
        if self._peek() is not None:
            raise TranslationFailure(
                "t5_parse_error", f"trailing tokens after complete expression in {self.source!r}"
            )
        return expr

    def parse_expr(self, min_prec: int) -> dict:
        left = self.parse_unary()
        while True:
            tok = self._peek()
            if tok is None or tok.kind != "OP" or tok.text not in _BINARY_PRECEDENCE:
                break
            prec = _BINARY_PRECEDENCE[tok.text]
            if prec < min_prec:
                break
            self._advance()
            right = self.parse_expr(prec + 1)
            left = self._combine(tok.text, left, right)
        return left

    def _combine(self, op: str, left: dict, right: dict) -> dict:
        if op == "AND":
            return {"type": "and", "args": [left, right]}
        if op == "OR":
            return {"type": "or", "args": [left, right]}
        if op == "XOR":
            # Not one of the 8 frozen IR node types -- expand to the
            # logically equivalent (A OR B) AND NOT(A AND B), a pure
            # structural rewrite that changes no semantics.
            return {
                "type": "and",
                "args": [
                    {"type": "or", "args": [left, right]},
                    {"type": "not", "body": {"type": "and", "args": [left, right]}},
                ],
            }
        if op == "IMPLIES":
            return {"type": "implies", "antecedent": left, "consequent": right}
        if op == "IFF":
            return {"type": "iff", "left": left, "right": right}
        raise TranslationFailure("t5_parse_error", f"unknown binary operator {op!r}")

    def parse_unary(self) -> dict:
        tok = self._peek()
        if tok is None:
            raise TranslationFailure("t5_parse_error", f"unexpected end of input in {self.source!r}")

        if tok.kind == "OP" and tok.text == "NOT":
            self._advance()
            body = self.parse_unary()
            return {"type": "not", "body": body}

        if tok.kind == "QUANT":
            self._advance()
            match = _QUANT_PATTERN.match(tok.text)
            if match is None:
                raise TranslationFailure("t5_parse_error", f"malformed quantifier token {tok.text!r}")
            negated, quant_word, variable = match.groups()
            if not is_variable(variable):
                raise TranslationFailure(
                    "entity_shaped_variable", f"quantifier bound non-variable-shaped name {variable!r}"
                )
            body = self.parse_expr(0)
            node_type = "forall" if quant_word == "FORALL" else "exists"
            node = {"type": node_type, "variables": [variable], "body": body}
            if negated:
                node = {"type": "not", "body": node}
            return node

        if tok.kind == "NOTPRED":
            self._advance()
            pred_name = tok.text[len("NOT") :]
            args = self._parse_arglist()
            return {"type": "not", "body": {"type": "predicate", "name": pred_name, "args": args}}

        if tok.kind == "PRED":
            self._advance()
            args = self._parse_arglist()
            return {"type": "predicate", "name": tok.text, "args": args}

        if tok.kind == "LPAREN":
            self._advance()
            inner = self.parse_expr(0)
            self._expect("RPAREN")
            return inner

        raise TranslationFailure("t5_parse_error", f"unexpected token {tok.text!r} in {self.source!r}")

    def _parse_arglist(self) -> list[str]:
        self._expect("LPAREN")
        args: list[str] = []
        if self._peek() is not None and self._peek().kind != "RPAREN":
            args.append(self._expect("IDENT").text)
            while self._peek() is not None and self._peek().kind == "COMMA":
                self._advance()
                args.append(self._expect("IDENT").text)
        self._expect("RPAREN")
        if not args:
            raise TranslationFailure("malformed_node", f"predicate with no arguments in {self.source!r}")
        return args


def parse_t5_fol(raw_text: str) -> dict:
    """Parse one T5-generated FOL string into a src/folio/ir.py-shaped dict.

    Raises TranslationFailure (never returns a best-effort/partial result)
    if the string doesn't fit the grammar this model actually produces.
    """
    if not raw_text or not raw_text.strip():
        raise TranslationFailure("empty_output", "model produced an empty translation")
    normalized = _normalize(raw_text)
    tokens = _tokenize(normalized)
    if not tokens:
        raise TranslationFailure("empty_output", f"no tokens extracted from {raw_text!r}")
    parser = _Parser(tokens, source=raw_text)
    return parser.parse_theory()
