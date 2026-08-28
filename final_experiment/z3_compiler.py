"""Two-pass compiler from validated FOL AST dictionaries to Z3 expressions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .fol_ast import BridgeError, VARIABLE, validate_theory

@dataclass
class DeclarationTable:
    sort: Any
    constants: dict[str, Any]
    predicates: dict[str, Any]
    predicate_arities: dict[str, int]
    predicate_keys: dict[tuple[str, int], str]
    variables: set[str]

@dataclass
class CompiledTheory:
    premises: list[Any]
    hypothesis: Any
    declarations: DeclarationTable

def collect_declarations(theory: dict[str, Any]) -> DeclarationTable:
    try:
        import z3
    except ImportError as exc:
        raise BridgeError("compiler_failure", "z3-solver not installed") from exc
    validate_theory(theory)
    constants: set[str] = set(); arities: dict[str, set[int]] = {}; variables: set[str] = set()
    def visit(node: dict[str, Any], bound: set[str]) -> None:
        kind=node["type"]
        if kind=="predicate":
            name,args=node["name"],node["args"];arities.setdefault(name,set()).add(len(args));constants.update(a for a in args if a not in bound and not VARIABLE.fullmatch(a))
        elif kind=="equals":constants.update(node[key] for key in ("left","right") if node[key] not in bound and not VARIABLE.fullmatch(node[key]))
        elif kind=="not":visit(node["body"],bound)
        elif kind in {"and","or"}:
            for child in node["args"]:visit(child,bound)
        elif kind=="implies":visit(node["antecedent"],bound);visit(node["consequent"],bound)
        elif kind=="iff":visit(node["left"],bound);visit(node["right"],bound)
        else:
            names=set(node["variables"]);variables.update(names);visit(node["body"],bound|names)
    for expression in [*theory["premises"],theory["hypothesis"]]:visit(expression,set())
    object_sort=z3.DeclareSort("Object");keys={(name,arity):(name if len(values)==1 else f"{name}/{arity}") for name,values in arities.items() for arity in values};flat={keys[(name,arity)]:arity for name,values in arities.items() for arity in values}
    return DeclarationTable(object_sort,{n:z3.Const(n,object_sort) for n in sorted(constants)},{key:z3.Function(key,*([object_sort]*arity),z3.BoolSort()) for key,arity in sorted(flat.items())},dict(sorted(flat.items())),keys,variables)

def _compile_expression(node:dict[str,Any],d:DeclarationTable,env:dict[str,Any]):
    import z3
    kind=node["type"]
    if kind=="predicate":
        try:return d.predicates[d.predicate_keys[(node["name"],len(node["args"]))]](*[env[a] if a in env else d.constants[a] for a in node["args"]])
        except KeyError as exc:raise BridgeError("compiler_failure",f"undeclared symbol: {exc.args[0]}") from exc
    if kind=="equals":
        try:return (env[node["left"]] if node["left"] in env else d.constants[node["left"]])==(env[node["right"]] if node["right"] in env else d.constants[node["right"]])
        except KeyError as exc:raise BridgeError("compiler_failure",f"undeclared symbol: {exc.args[0]}") from exc
    if kind=="not":return z3.Not(_compile_expression(node["body"],d,env))
    if kind=="and":return z3.And(*[_compile_expression(x,d,env) for x in node["args"]])
    if kind=="or":return z3.Or(*[_compile_expression(x,d,env) for x in node["args"]])
    if kind=="implies":return z3.Implies(_compile_expression(node["antecedent"],d,env),_compile_expression(node["consequent"],d,env))
    if kind=="iff":return _compile_expression(node["left"],d,env)==_compile_expression(node["right"],d,env)
    vs=[z3.Const(n,d.sort) for n in node["variables"]];body=_compile_expression(node["body"],d,env|dict(zip(node["variables"],vs)))
    return z3.ForAll(vs,body) if kind=="forall" else z3.Exists(vs,body)

def compile_theory(theory:dict[str,Any])->CompiledTheory:
    declarations=collect_declarations(theory)
    try:return CompiledTheory([_compile_expression(x,declarations,{}) for x in theory["premises"]],_compile_expression(theory["hypothesis"],declarations,{}),declarations)
    except BridgeError:raise
    except Exception as exc:raise BridgeError("compiler_failure",str(exc)) from exc
