# FOLIO Symbolic IR — Frozen Specification

Status: **frozen for the translation prototype.** Scoped strictly to constructs demonstrated necessary by `FOLIO_REPRESENTATION_FEASIBILITY.md`'s 18-item audit (§1–4 there). Nothing here is speculative first-order logic support — every node type maps to at least one observed FOLIO construct. This is the target language Qwen3-8B translates into; it is never Python, never a Z3 API call, never a raw FOL string.

Implemented by `src/folio/ir.py` (dataclasses + parsing) and `src/folio/schema.py` (validation). Compiled to Z3 by `src/folio/compiler.py` (§ separate doc-comment there — no LLM involved in compilation).

## 1. Node types (8, matching the audit's construct list exactly)

| Node | Maps to audit construct |
|---|---|
| `predicate` | unary/binary (n-ary) atoms — every group |
| `not` | negation — 6/8 groups |
| `and` | conjunction — all 8 groups |
| `or` | disjunction — ML-algorithms group |
| `implies` | universal rules (`∀x(P(x)→Q(x))`) — 7/8 groups |
| `iff` | biconditional; also used to express XOR as `not(iff(...))` — Remy group |
| `forall` | universal quantification, incl. multi-variable — 7/8 groups |
| `exists` | existential quantification, incl. multi-variable, in facts and queries — 5/8 groups |

No `xor` node: XOR(A, B) ≡ ¬(A ↔ B), so the Remy item's `⊕` is expressed as `{"type": "not", "body": {"type": "iff", ...}}` rather than adding a ninth node for one construct. No function symbols, no equality/inequality node — **zero occurrences in the 18-item sample**; if the full FOLIO set turns out to need them, add then, not now (per the audit's explicit "don't force it, don't over-build" rule).

## 2. Grammar

```
Expr :=
    Predicate(name: str, args: [Term])
  | Not(body: Expr)
  | And(args: [Expr, ...])          # arity >= 2
  | Or(args: [Expr, ...])           # arity >= 2
  | Implies(antecedent: Expr, consequent: Expr)
  | Iff(left: Expr, right: Expr)
  | ForAll(variables: [VarName, ...], body: Expr)   # arity >= 1
  | Exists(variables: [VarName, ...], body: Expr)   # arity >= 1

Term := VarName | EntityName
```

JSON shape (matches the task's own example exactly):

```json
{
  "type": "forall",
  "variables": ["x"],
  "body": {
    "type": "implies",
    "antecedent": {"type": "predicate", "name": "researcher", "args": ["x"]},
    "consequent": {"type": "predicate", "name": "careful", "args": ["x"]}
  }
}
```

`"Some scientist admires every researcher."` preserves nested-quantifier structure (not flattened):

```json
{
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
          "consequent": {"type": "predicate", "name": "admires", "args": ["y", "x"]}
        }
      }
    ]
  }
}
```

## 3. Variables vs. entities (Term resolution)

A `Term` string is a **variable** iff it matches `^[a-z][0-9]*$` (single lowercase letter, optional digit suffix — `x`, `y`, `x1`, `z2`; matches every variable name observed in the FOLIO gold FOL sample). Anything else (`velite`, `andrewcollins`, `thelegendofzelda`) is an **entity constant**. This is a fixed lexical rule, not inferred per-example — the translator (§ translator.py) is instructed to only ever use single-letter variable names and multi-character entity names, so the rule is checkable without needing declared-symbol tables from the source text.

## 4. Well-formedness rules (enforced by `schema.py`, not the compiler)

1. `type` must be one of the 8 node names in §1. Anything else → `TRANSLATION_FAILURE` (`SCHEMA_ERROR: unknown node type`).
2. Every node's required fields must be present and correctly typed (`predicate.name`: non-empty str; `predicate.args`: list of ≥1 `Term`; `and`/`or`.args`: list of ≥2 `Expr`; `implies`: both `antecedent`/`consequent` present; `iff`: both `left`/`right`; `forall`/`exists`: `variables` non-empty list of variable-shaped strings + `body` present).
3. **Variable-shaped strings only inside `variables` lists** — an entity-shaped string in a `variables` list is a `SCHEMA_ERROR`.
4. **Scope check**: every variable-shaped `Term` appearing inside a `predicate.args` must be bound by an enclosing `forall`/`exists` on the path from the root — walked with an explicit scope stack during validation, not inferred after the fact. An unbound variable-shaped term → `SCHEMA_ERROR: unbound_variable`.
5. **Predicate arity consistency**: the same `predicate.name` must have the same `len(args)` everywhere it's used within one theory (all premises + hypothesis, checked together). A mismatch → `SCHEMA_ERROR: arity_conflict`.
6. **No empty expressions**: `and`/`or` with `< 2` args, `forall`/`exists` with `0` variables, any node missing a required child → `SCHEMA_ERROR: malformed_node`.
7. Validation is total and side-effect-free: it returns either a validated, typed IR tree or a `TRANSLATION_FAILURE` with a specific reason — **never** a partially-repaired tree. No auto-repair step exists anywhere in this pipeline (explicit instruction).

## 5. What is deliberately NOT in this IR

- Function symbols (`f(x)`), equality/inequality (`x = y`) — unobserved in the audit sample; adding them now would be speculative, contrary to the "keep the IR as small as possible" instruction.
- More than the 8 node types — no separate `xor` node (§1), no modal/temporal operators, no arithmetic.
- Anything executable — this is inert JSON. It cannot express a side effect; the only thing that "runs" it is the deterministic compiler (`compiler.py`) turning it into Z3 terms, and the only thing that decides truth is Z3 itself.
