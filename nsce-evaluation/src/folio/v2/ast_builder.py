"""Deterministic decomposition-JSON -> src/folio/ir.py Expr builder. Step F.

Takes one sentence's already-shape-checked decomposition (decomposer.py's
output: atoms + nodes + root, referencing each other by id) plus the
theory-shared EntityTable/PredicateTable (vocabulary.py) and recursively
resolves `root` into the exact same typed IR the rest of the backend
(schema.py, compiler.py, solver.py) already consumes.

No LLM calls anywhere in this module. This is the piece that lets the LLM
stay at "which components does this sentence have and how do they relate"
(decomposer.py) instead of "get every brace and quantifier position right
in one shot" (what Translation-v1 asked for).
"""

from __future__ import annotations

from src.folio import ir
from src.folio.ir import Expr, TranslationFailure, is_variable
from src.folio.v2.vocabulary import EntityTable, PredicateTable


def collect_atom_predicate_arities(decomposition: dict) -> list[tuple[str, int]]:
    """Every (raw_predicate_name, arity) pair mentioned by this sentence's atoms,
    in atom order -- feeds vocabulary.PredicateTable.build() across a whole theory.
    """
    pairs = []
    for atom in decomposition.get("atoms", []):
        pairs.append((atom["predicate"], len(atom["args"])))
    return pairs


def collect_entity_surfaces(decomposition: dict) -> list[str]:
    """Every entity surface form this sentence declares, in declaration order."""
    return [e["surface"] for e in decomposition.get("entities", []) if isinstance(e, dict) and "surface" in e]


def _build_atom(atom: dict, entity_table: EntityTable, predicate_table: PredicateTable) -> ir.Predicate:
    name = atom.get("predicate")
    if not isinstance(name, str) or not name:
        raise TranslationFailure("missing_field", f"atom {atom.get('id')!r} has no predicate name")
    raw_args = atom.get("args")
    if not isinstance(raw_args, list) or not raw_args:
        raise TranslationFailure("malformed_node", f"atom {atom.get('id')!r}.args must be a non-empty list")
    canonical_name = predicate_table.canonical_name(name)
    resolved_args = []
    for arg in raw_args:
        if not isinstance(arg, str) or not arg:
            raise TranslationFailure("malformed_node", f"atom {atom.get('id')!r} has a non-string/empty arg")
        if is_variable(arg):
            resolved_args.append(arg)
        else:
            resolved_args.append(entity_table.resolve(arg))
    return ir.Predicate(name=canonical_name, args=tuple(resolved_args))


def _build_node(
    node_id: str,
    atoms_by_id: dict[str, dict],
    nodes_by_id: dict[str, dict],
    entity_table: EntityTable,
    predicate_table: PredicateTable,
    visiting: frozenset[str],
) -> Expr:
    if not isinstance(node_id, str) or not node_id:
        raise TranslationFailure("dangling_reference", f"reference {node_id!r} is not a valid id")
    if node_id in atoms_by_id:
        return _build_atom(atoms_by_id[node_id], entity_table, predicate_table)
    if node_id not in nodes_by_id:
        raise TranslationFailure("dangling_reference", f"id {node_id!r} does not match any atom or node")
    if node_id in visiting:
        raise TranslationFailure("cyclic_reference", f"id {node_id!r} is part of a reference cycle")

    node = nodes_by_id[node_id]
    op = node.get("op")
    next_visiting = visiting | {node_id}

    def resolve(ref: object) -> Expr:
        return _build_node(ref, atoms_by_id, nodes_by_id, entity_table, predicate_table, next_visiting)  # type: ignore[arg-type]

    if op == "not":
        if "body" not in node:
            raise TranslationFailure("missing_field", f"not node {node_id!r} missing body")
        return ir.Not(body=resolve(node["body"]))

    if op in ("and", "or"):
        arg_ids = node.get("args")
        if not isinstance(arg_ids, list) or len(arg_ids) < 2:
            raise TranslationFailure("malformed_node", f"{op} node {node_id!r} needs a list of >=2 args")
        parsed = tuple(resolve(a) for a in arg_ids)
        return ir.And(args=parsed) if op == "and" else ir.Or(args=parsed)

    if op == "implies":
        if "antecedent" not in node or "consequent" not in node:
            raise TranslationFailure("missing_field", f"implies node {node_id!r} needs antecedent and consequent")
        return ir.Implies(antecedent=resolve(node["antecedent"]), consequent=resolve(node["consequent"]))

    if op == "iff":
        if "left" not in node or "right" not in node:
            raise TranslationFailure("missing_field", f"iff node {node_id!r} needs left and right")
        return ir.Iff(left=resolve(node["left"]), right=resolve(node["right"]))

    if op in ("forall", "exists"):
        variables = node.get("variables")
        if not isinstance(variables, list) or not variables:
            raise TranslationFailure("malformed_node", f"{op} node {node_id!r} needs a non-empty variables list")
        for v in variables:
            if not isinstance(v, str) or not is_variable(v):
                raise TranslationFailure(
                    "entity_shaped_variable", f"{op} node {node_id!r}.variables contains non-variable-shaped {v!r}"
                )
        if "body" not in node:
            raise TranslationFailure("missing_field", f"{op} node {node_id!r} missing body")
        body = resolve(node["body"])
        variables_t = tuple(variables)
        return ir.ForAll(variables=variables_t, body=body) if op == "forall" else ir.Exists(variables=variables_t, body=body)

    raise TranslationFailure("unknown_operator", f"node {node_id!r} has unknown op {op!r}")


def build_expr(decomposition: dict, entity_table: EntityTable, predicate_table: PredicateTable) -> Expr:
    """Resolve one sentence's decomposition dict into a validated-shape (but not
    yet scope/arity-checked -- see validator.py) ir.Expr tree.
    """
    atoms_by_id: dict[str, dict] = {}
    for atom in decomposition.get("atoms", []):
        aid = atom.get("id")
        if not isinstance(aid, str) or not aid:
            raise TranslationFailure("missing_field", f"atom {atom!r} missing a non-empty id")
        if aid in atoms_by_id:
            raise TranslationFailure("duplicate_id", f"duplicate atom id {aid!r}")
        atoms_by_id[aid] = atom

    nodes_by_id: dict[str, dict] = {}
    for node in decomposition.get("nodes", []):
        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            raise TranslationFailure("missing_field", f"node {node!r} missing a non-empty id")
        if nid in nodes_by_id or nid in atoms_by_id:
            raise TranslationFailure("duplicate_id", f"duplicate id {nid!r} (reused across atoms/nodes)")
        nodes_by_id[nid] = node

    root = decomposition.get("root")
    if not isinstance(root, str) or not root:
        raise TranslationFailure("missing_field", "decomposition.root (a non-empty string id) is required")

    return _build_node(root, atoms_by_id, nodes_by_id, entity_table, predicate_table, frozenset())
