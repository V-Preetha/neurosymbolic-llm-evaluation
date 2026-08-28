# Symbolic Bridge Specification — ProofWriter Path

Covers `src/nsce/parser.py` → `src/nsce/symbolic_reasoner.py` → `src/nsce/verifier.py`. This is Condition C's core deterministic component for the revised study. Not rewritten for this phase — documented as-is, with only the additive changes noted in §7 (field whitelisting) and §8 (ABSTAIN logging).

## 1. Fact representation

A fact is an `Atom(predicate, arguments, negated)`:

```python
Atom(predicate: str, arguments: tuple[str, ...], negated: bool = False)
```

- Unary facts (`"The cat is red."`) → `Atom("red", ("cat",), False)`.
- Binary facts (`"The cat chases the mouse."`) → `Atom("chase", ("cat", "mouse"), False)`, using the fixed verb table in `VERB_FORMS` (chase/eat/like/need/see/visit, both singular and plural inflections mapped to one canonical predicate name).
- Entities are lowercased and underscored (`_entity()`); the definite article `"the "` is stripped.

## 2. Negated fact representation

Negation is a boolean flag on the same `Atom`, not a separate predicate:

- `"The cat is not red."` → `Atom("red", ("cat",), True)`.
- `"The cat does not chase the mouse."` → `Atom("chase", ("cat", "mouse"), True)`.
- `Atom.opposite()` returns the same predicate/arguments with `negated` flipped — this is how the reasoner checks for contradiction (§5).

## 3. Rule representation

```python
Rule(premises: tuple[Atom, ...], conclusion: Atom, source: str)
```

- Parsed from `"If <premise[, and premise...]> then <conclusion>."` or the universal-quantifier form `"All <adj[, adj...]> people/things are <adj>."`.
- Premises are conjunctive only (`"and"`-joined); there is no disjunction in a premise or conclusion.
- A single `If...then` sentence with multiple conjunctive conclusions expands to multiple `Rule` objects, one per conclusion.
- `source` retains the original sentence text for traceability, but is not used in reasoning.

## 4. Query representation

A query is parsed the same way as a fact (`parse_sentence` → single `Atom`); `parse_query` raises `ParserError` if the sentence doesn't reduce to exactly one atom (e.g. a rule can't be a query).

## 5. Polarity handling

There is one implicit variable slot per rule, written `?x` (module constant `VARIABLE`). Pronouns/indefinites (`someone`, `something`, `they`, `it`, `them`) map to `?x` during parsing; unification (`_match`) binds `?x` to a concrete entity per instantiation. Predicate arity, predicate name, and the `negated` flag must all match exactly for a rule premise to unify against a fact — polarity is never coerced or inferred, only matched literally.

## 6. Supported operators

- Conjunction in rule premises (`"X and Y then Z"`).
- Explicit negation (`"is not"`, `"does not <verb>"`) on facts, rule premises, rule conclusions, and queries.
- Universal quantification over a single implicit variable (`"All ... people/things are ..."`, and `"If someone/something is ... then ..."`).
- Multi-hop chaining via repeated forward-chaining passes (transitive closure over rules, not just one substitution step) — see `forward_chain`'s `while changed:` fixed-point loop.
- One-hop and multi-hop derivations of arbitrary depth (bounded only by fixed-point termination, not a fixed hop limit).

## 7. Unsupported operators (fail loudly, not silently)

- **Disjunction** — `"is red or blue"` raises `ParserError` (the adjective-list regex only splits on `,`/`and`; anything else fails the `[a-z]+` fullmatch and is rejected).
- **Multiple distinct variables per rule** — only one variable slot (`?x`) exists; a rule needing two independently-bound variables (e.g. relating two different unknowns) cannot be expressed.
- **Existential quantification** — no `"there exists"` construct.
- **Verbs outside the fixed six-verb table** (`chase/eat/like/need/see/visit`) — any other verb raises `ParserError: Unsupported clause`.
- **Malformed conditionals** (missing `"then"`, ill-formed clause structure) raise `ParserError` rather than a best-effort partial parse.

This is a function-free, single-variable, conjunctive-Horn-clause fragment with explicit negation and a closed-world-style trichotomous verdict — essentially the ProofWriter dataset's own controlled grammar, not general first-order logic. It was scoped to exactly what the benchmark requires (per the project's own non-negotiable rule: "use the simplest formal representation capable of expressing the benchmark correctly"), not extended further.

## 8. How UNKNOWN is derived

`classify()` runs `forward_chain()` to a fixed point (saturating all derivable facts from the given facts + rules), then checks two independent memberships in the derived-fact set:

```python
supported = query in derivations
contradicted = query.opposite() in derivations
```

- If `supported` → status `"supported"`, prediction `"true"`.
- Else if `contradicted` → status `"contradicted"`, prediction `"false"`.
- Else (neither the query nor its negation was derivable) → status `"unknown"`, prediction `"unknown"`, with an **empty** proof trace (there is nothing to prove).

This is the open-world-assumption trichotomy ProofWriter itself is built around — `UNKNOWN` is not "parsing failed" or "insufficient confidence," it is a proven negative result: exhaustive forward chaining terminated without deriving either polarity.

If both `supported` and `contradicted` hold simultaneously (contradictory input knowledge), `ReasoningResult.inconsistent = True` is set; `classify()` currently resolves the returned status in favor of `"supported"` (checked first), but the `inconsistent` flag surfaces the conflict for any caller that wants to treat it specially (validated by `test_symbolic_bridge_validation.ContradictoryFactsTests`).

## 9. Proof trace provenance

`_trace()` walks the actual `Derivation` records built during `forward_chain` — each derived atom stores which `Rule` (if any) produced it and which premise atoms were consumed. The trace is generated by recursively visiting this derivation graph and rendering `"Fact: ..."` / `"Rule: p1, p2 -> conclusion"` lines strictly from that stored state. **The proof trace is never asked of or generated by Qwen** — `feedback.py` only ever *receives* this trace as pre-rendered text to include in the correction prompt (see `render_feedback`); it is read-only from the LLM's perspective. Validated directly in `test_symbolic_bridge_validation.py` (every reasoning test asserts the exact `proof_trace` tuple).

---

## 10. What is neural about the bridge? (Phase E clarification)

**The parser translates the benchmark's own natural-language facts/rules/query — not Qwen's output.** Concretely, `verify_record()` (`src/nsce/verifier.py`) calls:

```python
theory = parse_theory(allowed.facts + allowed.rules)   # from record["input"], not from Qwen
reasoning = classify(parse_query(allowed.query), theory.facts, theory.rules)
```

`allowed.facts`, `allowed.rules`, and `allowed.query` come from `AllowedInput` (§7 below), which is extracted directly from the benchmark record's `input` block. Qwen's generated text never enters `parse_theory`, `parse_query`, or `classify` anywhere in this codebase — confirmed by tracing every call site of these three functions (`src/nsce/verifier.py`, `tests/test_symbolic.py`, `tests/test_symbolic_bridge_validation.py`).

So the accurate architecture statement is:

```text
Qwen (neural)                    Deterministic parser (symbolic translation)
     |                                        |
     v                                        v
 initial label prediction        parses the SAME benchmark facts/rules/query
     |                                        |
     |                                        v
     |                          deterministic Horn-clause reasoner + proof trace
     |                                        |
     +-----------------> verifier compares Qwen's label against the symbolic verdict
                                               |
                                               v
                              RETAIN / CORRECT / ABSTAIN (bridge_action, §8 below)
```

**Do not describe this as "Qwen translates its own output into first-order logic."** It does not. What is genuinely neural is the *task-level prediction being verified* — Qwen answers the ProofWriter question directly from the natural-language prompt, exactly as it would in the LLM-only condition. What is genuinely symbolic is an *independent, deterministic re-derivation* of the same underlying facts/rules, used to check and, on disagreement, attempt to correct Qwen's answer. This is a legitimate, accurately-described external verification bridge — it is just not a bridge that performs neural-to-symbolic translation of free-form model output. There is currently no component in this codebase that translates arbitrary natural language (e.g. Qwen's own generated text) into logic; if that specific capability is wanted later, it would need to be built and audited separately (and, per the Phase A finding, the closest existing attempt — `src/nsce/factual_structured.py`/`factual_verifier.py` for FELM claims — mostly delegates back to an LLM NLI call rather than doing this deterministically).

## 11. Field whitelisting (Phase C3, cross-referenced here)

`src/io_guard.py`'s `extract_allowed_input()` is the single narrowing step every prompt/verifier function performs before touching a record — see `NEW_STUDY_DESIGN.md` and `tests/test_field_whitelisting.py`. `ground_truth`, `evidence`, `metadata`, and `candidate_response` are structurally unreachable from `parse_theory`/`parse_query`/`classify`/`render_*` because those functions only ever receive `AllowedInput.facts`/`.rules`/`.query`.

## 12. UNKNOWN vs ABSTAIN (Phase D4)

These are deliberately kept separate in `src/nsce/pipeline.py`:

- **`UNKNOWN`** is a **task label** — one of the three ProofWriter ground-truth values. A symbolic verdict of `"unknown"` (§8) is a legitimate, proven result and can legitimately become the bridge's `final_prediction`, via either the `RETAIN` or `CORRECT` `bridge_action` (below).
- **`ABSTAIN`** is a **system action** — the bridge declines to verify/correct because translation itself failed (`parse_theory`/`parse_query` raised `ParserError`; `verify_record` returns `reasoning=None`). In this case there is no symbolic verdict to compare against or correct with, so the pipeline falls back to Qwen's initial answer and logs `bridge_action: "ABSTAIN"`, `abstain_reason: "translation_failure"` — distinct from the two verified branches.

Three possible `bridge_action` values, all logged explicitly in every NSCE pipeline result:

| `bridge_action` | Condition | `final_prediction` source |
|---|---|---|
| `RETAIN` | Symbolic verdict available, agrees with Qwen's initial answer | Qwen's initial answer (unchanged) |
| `CORRECT` | Symbolic verdict available, disagrees with Qwen's initial answer | Qwen's re-answer after seeing the proof trace (feedback prompt) |
| `ABSTAIN` | Translation failed (`ParserError`) | Qwen's initial answer (no verified alternative exists) |
