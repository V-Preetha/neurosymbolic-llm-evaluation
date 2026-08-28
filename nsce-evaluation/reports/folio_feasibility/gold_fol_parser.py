"""Small recursive-descent parser: FOLIO's own Unicode gold-FOL strings -> our JSON IR.

Used ONLY to build the gold-FOL oracle verdict for comparison metrics -- never
fed to or seen by Qwen, never used as translator input. This is a parser for a
fixed, regular annotation format (prenex quantifiers + standard connectives),
not a general FOL parser.
"""
import re

TOKEN_RE = re.compile(
    r"""
    (?P<QUANT_FORALL>∀)
  | (?P<QUANT_EXISTS>∃)
  | (?P<NOT>¬)
  | (?P<AND>∧)
  | (?P<OR>∨)
  | (?P<IMPLIES>→)
  | (?P<IFF>↔|≡|⟷)
  | (?P<XOR>⊕)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<COMMA>,)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<WS>\s+)
    """,
    re.VERBOSE,
)


def tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"cannot tokenize at {pos!r}: {text[pos:pos+20]!r}")
        kind = m.lastgroup
        if kind != "WS":
            tokens.append((kind, m.group()))
        pos = m.end()
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind):
        tok = self.advance()
        if tok[0] != kind:
            raise ValueError(f"expected {kind}, got {tok}")
        return tok

    def parse(self):
        expr = self.parse_iff()
        if self.pos != len(self.tokens):
            raise ValueError(f"trailing tokens: {self.tokens[self.pos:]}")
        return expr

    def parse_iff(self):
        left = self.parse_implies()
        while self.peek()[0] in ("IFF", "XOR"):
            op = self.advance()[0]
            right = self.parse_implies()
            if op == "IFF":
                left = {"type": "iff", "left": left, "right": right}
            else:  # XOR(A,B) = NOT(IFF(A,B))
                left = {"type": "not", "body": {"type": "iff", "left": left, "right": right}}
        return left

    def parse_implies(self):
        left = self.parse_or()
        if self.peek()[0] == "IMPLIES":
            self.advance()
            right = self.parse_implies()  # right-assoc
            return {"type": "implies", "antecedent": left, "consequent": right}
        return left

    def parse_or(self):
        args = [self.parse_and()]
        while self.peek()[0] == "OR":
            self.advance()
            args.append(self.parse_and())
        return args[0] if len(args) == 1 else {"type": "or", "args": args}

    def parse_and(self):
        args = [self.parse_not()]
        while self.peek()[0] == "AND":
            self.advance()
            args.append(self.parse_not())
        return args[0] if len(args) == 1 else {"type": "and", "args": args}

    def parse_not(self):
        if self.peek()[0] == "NOT":
            self.advance()
            return {"type": "not", "body": self.parse_not()}
        return self.parse_atom()

    def parse_atom(self):
        kind, text = self.peek()
        if kind in ("QUANT_FORALL", "QUANT_EXISTS"):
            self.advance()
            var = self.expect("IDENT")[1]
            body = self.parse_not()  # allows chained ∀x ∀y (...) via recursive nesting
            node_type = "forall" if kind == "QUANT_FORALL" else "exists"
            return {"type": node_type, "variables": [var], "body": body}
        if kind == "LPAREN":
            self.advance()
            inner = self.parse_iff()
            self.expect("RPAREN")
            return inner
        if kind == "IDENT":
            name = self.advance()[1]
            if self.peek()[0] == "LPAREN":
                self.advance()
                args = []
                if self.peek()[0] != "RPAREN":
                    args.append(self.expect("IDENT")[1])
                    while self.peek()[0] == "COMMA":
                        self.advance()
                        args.append(self.expect("IDENT")[1])
                self.expect("RPAREN")
                return {"type": "predicate", "name": name.lower(), "args": args}
            raise ValueError(f"bare identifier {name!r} not followed by '('")
        raise ValueError(f"unexpected token {kind, text}")


def parse_gold_fol(text: str) -> dict:
    return Parser(tokenize(text)).parse()


if __name__ == "__main__":
    tests = [
        "∀x (Drinks(x) → Dependent(x))",
        "∀x (Drinks(x) ⊕ Jokes(x))",
        "(Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina))",
        "¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))",
        "∃x (Japanese(x) ∧ VideoGameCompany(x) ∧ Game(thelegendofzelda) ∧ Created(x, thelegendofzelda))",
        "∀x ∀y (Game(x) ∧ InTop10(x) ∧ Created(x, y) → Japanese(y))",
        "HouseHoldApp(duster) ∧ ¬ Suck(duster)",
        "∃x ∃y (Animal(x) ∧ Animal(y) ∧ BreedingBack(x, y) ∧ Extinct(y))",
    ]
    import json
    for t in tests:
        print(t)
        print(json.dumps(parse_gold_fol(t)))
        print()
