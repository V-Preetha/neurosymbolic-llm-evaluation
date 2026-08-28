"""Archived component supervision; inactive in the frozen-bridge experiment."""
from __future__ import annotations
import json,re
from typing import Any
from .gold_targets import folio_target,proofwriter_target
from .prompts import component_prompt,direct_prompt

def canonical_clause(node:dict[str,Any])->str:
    kind=node["type"]
    if kind=="predicate":
        name=node["name"].replace("_"," ");args=node["args"]
        if not args:return name
        if len(args)==1:return f"{args[0]} is {name}"
        if len(args)==2:return f"{args[0]} {name} {args[1]}"
        return f"{args[0]} {name} {' '.join(args[1:])}"
    if kind=="equals":return f"{node['left']} equals {node['right']}"
    if kind=="not":return f"It is not true that {canonical_clause(node['body'])}"
    if kind in {"and","or"}:return f" {kind} ".join(canonical_clause(x) for x in node["args"])
    if kind=="implies":return f"If {canonical_clause(node['antecedent'])} then {canonical_clause(node['consequent'])}"
    if kind=="iff":return f"{canonical_clause(node['left'])} if and only if {canonical_clause(node['right'])}"
    prefix="For every" if kind=="forall" else "There exists"
    return f"{prefix} {', '.join(node['variables'])}, {canonical_clause(node['body'])}"

def _record(component:str,clause:str,target:dict[str,Any])->dict[str,str]:
    return {"component":component,"clause":clause,"prompt":component_prompt(component,clause),"target":json.dumps(target,ensure_ascii=False,separators=(",",":"))}

def _relation_kind(node:dict[str,Any],clause:str)->str:
    arity=len(node["args"])
    if arity==0:return "proposition"
    if arity==1:return "adjective" if re.search(r"\b(?:is|are|was|were)\b",clause,re.I) else "intransitive"
    return "transitive" if arity==2 else "ditransitive"

def _walk(node:dict[str,Any],clause:str,out:list[dict[str,str]])->None:
    kind=node["type"]
    decision="atomic" if kind in {"predicate","equals"} else "unary" if kind=="not" else "quantified" if kind in {"forall","exists"} else "binary"
    out.append(_record("parser_selector",clause,{"parser":decision}))
    if kind=="equals":
        out.append(_record("relation_selector",clause,{"relation":"equality"}));out.append(_record("equality",clause,{"left":node["left"],"right":node["right"]}));return
    if kind=="predicate":
        relation=_relation_kind(node,clause);out.append(_record("relation_selector",clause,{"relation":relation}));args=node["args"]
        target={"predicate":node["name"]}
        if relation!="proposition":target["subject"]=args[0]
        if relation=="transitive":target["object"]=args[1]
        elif relation=="ditransitive":target|={"indirect_object":args[1],"direct_object":args[2]}
        out.append(_record(relation,clause,target));return
    if kind=="not":
        child=node["body"];child_clause=canonical_clause(child);out.append(_record("unary",clause,{"operator":"not","child_clause":child_clause}));_walk(child,child_clause,out);return
    if kind in {"forall","exists"}:
        child=node["body"];child_clause=canonical_clause(child);out.append(_record("quantified",clause,{"quantifier":kind,"variable":node["variables"][0],"child_clause":child_clause}));_walk(child,child_clause,out);return
    if kind in {"and","or"}:
        left=node["args"][0];right=node["args"][1] if len(node["args"])==2 else {"type":kind,"args":node["args"][1:]};operator=kind
    elif kind=="implies":left,right,operator=node["antecedent"],node["consequent"],"implies"
    else:left,right,operator=node["left"],node["right"],"iff"
    left_clause,right_clause=canonical_clause(left),canonical_clause(right);out.append(_record("binary",clause,{"operator":operator,"left_clause":left_clause,"right_clause":right_clause}));_walk(left,left_clause,out);_walk(right,right_clause,out)

def component_records(row:dict[str,Any])->list[dict[str,str]]:
    theory=folio_target(row) if row["dataset"]=="folio" else proofwriter_target(row);out=[]
    clauses=list(row.get("premises") or [x.strip() for x in re.split(r"(?<=[.!?])\s+",row["context"]) if x.strip()])
    for index,node in enumerate(theory["premises"]):_walk(node,clauses[index] if index<len(clauses) else canonical_clause(node),out)
    _walk(theory["hypothesis"],row["question"],out);return out

def direct_training_examples(rows:list[dict[str,Any]])->list[dict[str,str]]:
    """Return the one active direct-label record for each manifest row."""
    return [{"component":"direct","prompt":direct_prompt(row),"prefix":direct_prompt(row)+"\n\nAnswer:\n","target":row["label"],"text":direct_prompt(row)+"\n\nAnswer:\n"+row["label"]} for row in rows]

def training_examples(rows:list[dict[str,Any]],task:str)->list[dict[str,str]]:
    """Legacy component expansion retained only for archival reproducibility."""
    if task=="direct":
        return direct_training_examples(rows)
    examples=[]
    for row in rows:
        for record in component_records(row):
            prefix=record["prompt"]+"\nOUTPUT:\n";examples.append(record|{"prefix":prefix,"text":prefix+record["target"],"example_id":row["example_id"]})
    return examples

def expanded_record_counts(records:list[dict[str,str]])->dict[str,int]:
    selectors={"parser_selector","relation_selector"};atomic={"proposition","adjective","intransitive","transitive","ditransitive","equality"};logical={"unary","binary"}
    counts={"total_expanded_records":len(records),"selector_records":0,"atomic_records":0,"quantified_records":0,"logical_records":0}
    for record in records:
        component=record.get("component")
        if component in selectors:counts["selector_records"]+=1
        elif component in atomic:counts["atomic_records"]+=1
        elif component=="quantified":counts["quantified_records"]+=1
        elif component in logical:counts["logical_records"]+=1
        else:raise ValueError(f"Unknown NL2Logic training component: {component}")
    return counts
