from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from .core import LABELS, ROOT, normalize_label, read_jsonl, resolved_config, stable_id, write_json, write_jsonl

SCHEMAS={"proofwriter":{"id","maxD","NFact","NRule","theory","question","answer","QDep","allProofs"},"folio":{"premises","premises-FOL","conclusion","conclusion-FOL","label"}}

def normalize_row(dataset:str,row:dict[str,Any],split:str,index:int)->dict[str,Any]:
    missing=SCHEMAS[dataset]-row.keys()
    if missing: raise ValueError(f"{dataset}/{split}[{index}] missing fields: {sorted(missing)}")
    if dataset=="proofwriter":
        theory=str(row["theory"])
        return {"dataset":dataset,"split":split,"example_id":f"{row['id']}::q{index}","world_id":stable_id({"theory":theory.strip()}),"context":theory,"question":str(row["question"]),"label":normalize_label(row["answer"]),"QDep":row.get("QDep"),"maxD":row.get("maxD"),"NFact":row.get("NFact"),"NRule":row.get("NRule"),"proof":row.get("allProofs"),"config":row.get("config")}
    premises=row["premises"] if isinstance(row["premises"],list) else str(row["premises"]).splitlines(); fol=row["premises-FOL"] if isinstance(row["premises-FOL"],list) else str(row["premises-FOL"]).splitlines()
    return {"dataset":dataset,"split":split,"example_id":str(row.get("example-id",row.get("example_id",index))),"world_id":str(row.get("story-id",row.get("story_id",stable_id(premises)))),"premises":premises,"context":"\n".join(premises),"question":str(row["conclusion"]),"label":normalize_label(row["label"]),"gold_premises_fol":fol,"gold_conclusion_fol":str(row["conclusion-FOL"]),"source":row.get("source")}

def validate_splits(parts:dict[str,list[dict[str,Any]]])->dict[str,Any]:
    ids={s:{r["example_id"] for r in rows} for s,rows in parts.items()}; worlds={s:{r["world_id"] for r in rows} for s,rows in parts.items()}; overlap={}
    for s,rows in parts.items():
        if len(ids[s])!=len(rows): raise ValueError(f"Duplicate IDs in {s}")
        if any(r["label"] not in LABELS for r in rows): raise ValueError(f"Invalid labels in {s}")
    names=list(parts)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if ids[a]&ids[b]: raise ValueError(f"Example leakage between {a} and {b}")
            if worlds[a]&worlds[b]:
                overlap[f"{a}:{b}"]=sorted(worlds[a]&worlds[b])
                raise ValueError(f"World leakage between {a} and {b}: {len(worlds[a]&worlds[b])}")
    return {"counts":{k:len(v) for k,v in parts.items()},"world_overlap":overlap,"labels":{k:dict(Counter(r["label"] for r in v)) for k,v in parts.items()}}

def stratified_sample(rows:list[dict[str,Any]],size:int|None,dataset:str,seed:int)->list[dict[str,Any]]:
    if size is None or size>=len(rows): return list(rows)
    fields=("label","QDep") if dataset=="proofwriter" else ("label",); groups=defaultdict(list)
    for row in rows: groups[tuple(row.get(x) for x in fields)].append(row)
    rng=random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    quotas={k:int(size*len(v)/len(rows)) for k,v in groups.items()}
    while sum(quotas.values())<size:
        choices=[k for k in groups if quotas[k]<len(groups[k])]; key=max(choices,key=lambda k:len(groups[k])/(quotas[k]+1)); quotas[key]+=1
    selected=[row for k,g in groups.items() for row in g[:quotas[k]]]; rng.shuffle(selected); return selected

def _strata_counts(rows:list[dict[str,Any]],dataset:str)->dict[str,int]:
    fields=("label","QDep") if dataset=="proofwriter" else("label",)
    counts=Counter("|".join(str(row.get(field)) for field in fields) for row in rows)
    return dict(sorted(counts.items()))

def _manifest(dataset:str,split:str,mode:str,seed:int,rows:list[dict[str,Any]])->dict[str,Any]:
    ids=[row["example_id"] for row in rows]
    digest=hashlib.sha256(json.dumps(ids,ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()
    return {"dataset":dataset,"split":split,"mode":mode,"seed":seed,"size":len(ids),"stratify_by":["label","QDep"] if dataset=="proofwriter" else["label"],"strata":_strata_counts(rows,dataset),"ids_sha256":digest,"ids":ids}

def training_pool_rows(rows:list[dict[str,Any]],dataset:str,policy:str|None)->list[dict[str,Any]]:
    if policy is None:return list(rows)
    if dataset=="proofwriter" and policy=="controlled_language":
        return [row for row in rows if "natlang" not in str(row.get("config") or "").lower()]
    raise ValueError(f"Unsupported training-pool policy for {dataset}: {policy}")

def _add_pool_metadata(manifest:dict[str,Any],policy:str|None,eligible:int,total:int)->dict[str,Any]:
    if policy is None:return manifest
    return manifest|{"sampling_pool":policy,"eligibility":"config does not contain NatLang","eligible_pool_size":eligible,"excluded_pool_size":total-eligible}

def create_mode_manifests(parts:dict[str,list[dict[str,Any]]],dataset:str,mode:str,train_size:int|None,evaluation_size:int|None,seed:int,validation_size:int|None=None,training_pool:str|None=None)->dict[str,dict[str,Any]]:
    eligible_train=training_pool_rows(parts["train"],dataset,training_pool)
    selected_train=stratified_sample(eligible_train,train_size,dataset,seed)
    selected_test=stratified_sample(parts["test"],evaluation_size,dataset,seed)
    manifests={"train":_add_pool_metadata(_manifest(dataset,"train",mode,seed,selected_train),training_pool,len(eligible_train),len(parts["train"])),"validation":_manifest(dataset,"validation","fixed",seed,parts["validation"]),"test":_manifest(dataset,"test",mode,seed,selected_test)}
    if validation_size is not None:
        eligible_validation=training_pool_rows(parts["validation"],dataset,training_pool)
        selected_validation=stratified_sample(eligible_validation,validation_size,dataset,seed)
        manifests["training_validation"]=_add_pool_metadata(_manifest(dataset,"validation",mode,seed,selected_validation),training_pool,len(eligible_validation),len(parts["validation"]))
    id_sets={split:set(value["ids"]) for split,value in manifests.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if id_sets[a]&id_sets[b]:raise ValueError(f"Manifest leakage between {a} and {b}")
    return manifests

def materialize_manifest_rows(rows:list[dict[str,Any]],manifest:dict[str,Any])->list[dict[str,Any]]:
    ids=manifest.get("ids")
    if not isinstance(ids,list):raise ValueError("Manifest ids must be a list")
    if len(ids)!=len(set(ids)):raise ValueError("Manifest contains duplicate IDs")
    index={row["example_id"]:row for row in rows}
    missing=[example_id for example_id in ids if example_id not in index]
    if missing:raise ValueError(f"Manifest references missing IDs: {missing[:5]}")
    return [index[example_id] for example_id in ids]

def write_frozen_manifest(path:Path,payload:dict[str,Any],available_ids:set[str])->dict[str,Any]:
    if path.exists():
        existing=json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset","split","mode","seed"):
            if existing.get(key)!=payload.get(key):raise ValueError(f"Frozen manifest metadata mismatch for {path}: {key}")
        ids=existing.get("ids",[])
        if existing.get("size")!=payload.get("size"):raise ValueError(f"Frozen manifest requested size mismatch for {path}: existing={existing.get('size')} requested={payload.get('size')}")
        if len(ids)!=len(set(ids)):raise ValueError(f"Frozen manifest contains duplicate IDs: {path}")
        missing=set(ids)-available_ids
        if missing:raise ValueError(f"Frozen manifest references missing IDs: {path}")
        if existing.get("size")!=len(ids):raise ValueError(f"Frozen manifest size mismatch: {path}")
        return existing
    write_json(path,payload)
    return payload

def save_mode_manifests(base:Path,parts:dict[str,list[dict[str,Any]]],dataset:str,mode:str,train_size:int|None,evaluation_size:int|None,seed:int,validation_size:int|None=None,training_pool:str|None=None)->dict[str,dict[str,Any]]:
    candidates=create_mode_manifests(parts,dataset,mode,train_size,evaluation_size,seed,validation_size,training_pool)
    manifest_dir=base/"manifests"
    paths={"train":manifest_dir/f"train_{mode}.json","validation":manifest_dir/"validation.json","test":manifest_dir/f"test_{mode}.json"}
    if "training_validation" in candidates:paths["training_validation"]=manifest_dir/f"validation_train_{mode}.json"
    saved={split:write_frozen_manifest(paths[split],candidate,{row["example_id"] for row in parts["validation" if split=="training_validation" else split]}) for split,candidate in candidates.items()}
    id_sets={split:set(value["ids"]) for split,value in saved.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if id_sets[a]&id_sets[b]:raise ValueError(f"Frozen manifest leakage between {a} and {b}")
    return saved

def load_manifest_split(base:Path,split:str,mode:str)->tuple[list[dict[str,Any]],dict[str,Any]]:
    manifest_path=base/"manifests"/("validation.json" if split=="validation" else f"{split}_{mode}.json")
    if not manifest_path.exists():raise FileNotFoundError(f"Prepare the dataset for mode={mode} first; missing {manifest_path}")
    data_path=base/f"{split}.jsonl"
    if not data_path.exists():raise FileNotFoundError(f"Prepared split is missing: {data_path}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split")!=split:raise ValueError(f"Manifest split mismatch: expected {split}, got {manifest.get('split')}")
    if split!="validation" and manifest.get("mode")!=mode:raise ValueError(f"Manifest mode mismatch: expected {mode}, got {manifest.get('mode')}")
    rows=materialize_manifest_rows(read_jsonl(data_path),manifest)
    if manifest.get("size")!=len(rows):raise ValueError(f"Manifest size mismatch: {manifest_path}")
    return rows,manifest

def load_training_validation(base:Path,mode:str)->tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    full_rows,full_manifest=load_manifest_split(base,"validation",mode)
    subset_path=base/"manifests"/f"validation_train_{mode}.json"
    if not subset_path.exists():return full_rows,full_manifest,full_manifest
    subset=json.loads(subset_path.read_text(encoding="utf-8"))
    if subset.get("split")!="validation" or subset.get("mode")!=mode:raise ValueError(f"Training validation manifest metadata mismatch: {subset_path}")
    if not set(subset.get("ids",[]))<=set(full_manifest["ids"]):raise ValueError(f"Training validation is not a subset of full validation: {subset_path}")
    rows=materialize_manifest_rows(full_rows,subset)
    if subset.get("size")!=len(rows):raise ValueError(f"Training validation manifest size mismatch: {subset_path}")
    return rows,subset,full_manifest

def mode_sample_sizes(config:dict[str,Any],mode:str)->tuple[int|None,int|None]:
    sampling=config.get("sampling",{})
    training=sampling.get("training",{})
    evaluation=sampling.get("evaluation",{})
    if mode not in training or mode not in evaluation:raise ValueError(f"Sampling mode is not configured: {mode}")
    return training[mode],evaluation[mode]

def training_validation_size(config:dict[str,Any],mode:str)->int|None:
    return config.get("sampling",{}).get("training_validation",{}).get(mode)

def training_pool_policy(config:dict[str,Any],mode:str)->str|None:
    return config.get("sampling",{}).get("training_pool",{}).get(mode)

def prepare(dataset:str,mode:str,cache_dir:str|None=None):
    try: from datasets import load_dataset
    except ImportError as exc: raise RuntimeError("Install final_experiment/requirements.txt first") from exc
    cfg=resolved_config(dataset); dc=cfg["dataset"]; print(f"[Dataset] Loading {dc['id']} revision={dc.get('revision') or 'default'}",flush=True); raw=load_dataset(dc["id"],revision=dc.get("revision"),cache_dir=cache_dir)
    train=[normalize_row(dataset,dict(r),"train",i) for i,r in enumerate(raw[dc["train_split"]])]
    if dataset=="folio":
        worlds=sorted({r["world_id"] for r in train}); rng=random.Random(cfg["experiment"]["seed"]); rng.shuffle(worlds); dev=set(worlds[:max(1,round(len(worlds)*.1))]); parts={"train":[r for r in train if r["world_id"] not in dev],"validation":[r|{"split":"validation"} for r in train if r["world_id"] in dev]}
    else: parts={"train":train,"validation":[normalize_row(dataset,dict(r),"validation",i) for i,r in enumerate(raw[dc["validation_split"]])]}
    parts["test"]=[normalize_row(dataset,dict(r),"test",i) for i,r in enumerate(raw[dc["test_split"]])]
    audit=validate_splits(parts);out=ROOT/"data"/"processed"/dataset
    for split,rows in parts.items():write_jsonl(out/f"{split}.jsonl",rows)
    train_size,evaluation_size=mode_sample_sizes(cfg,mode)
    manifests=save_mode_manifests(out,parts,dataset,mode,train_size,evaluation_size,cfg["experiment"]["seed"],training_validation_size(cfg,mode),training_pool_policy(cfg,mode))
    write_json(out/"audit.json",audit|{"available_splits":list(raw.keys()),"prepared_mode":mode,"manifests":{split:{"size":value["size"],"ids_sha256":value["ids_sha256"]} for split,value in manifests.items()}})
    print(f"[Dataset] {dataset}: "+", ".join(f"{k}={len(v)}" for k,v in parts.items()),flush=True)
    print(f"[Sampling] mode={mode} selected_train={manifests['train']['size']} training_validation={manifests.get('training_validation',manifests['validation'])['size']} full_validation={manifests['validation']['size']} selected_evaluation={manifests['test']['size']}",flush=True)
    print(f"[Manifests] {out/'manifests'}",flush=True)
    return out

def smoke_dataset(dataset:str,limit:int=5):
    """Stream official rows to validate live access/schema without materializing a split."""
    try:from datasets import load_dataset
    except ImportError as exc:raise RuntimeError("Install final_experiment/requirements.txt first") from exc
    cfg=resolved_config(dataset);dc=cfg["dataset"];print(f"[Dataset smoke] Loading {dc['id']} (streaming)",flush=True);raw=load_dataset(dc["id"],revision=dc.get("revision"),streaming=True);report={}
    seen=set()
    for logical,actual in (("train",dc["train_split"]),("validation",dc["validation_split"]),("test",dc["test_split"])):
        if actual in seen:continue
        seen.add(actual);rows=[];fields=[]
        for i,row in enumerate(raw[actual]):
            fields=sorted(dict(row).keys());rows.append(normalize_row(dataset,dict(row),logical,i))
            if len(rows)>=limit:break
        report[logical]={"source_split":actual,"rows_checked":len(rows),"fields":fields}
    print(f"[Dataset smoke] PASS {dataset}",flush=True);return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["folio","proofwriter"],required=True);p.add_argument("--mode",choices=["smoke","pilot","main","full"],default="main");p.add_argument("--cache-dir");a=p.parse_args();prepare(a.dataset,a.mode,a.cache_dir)
if __name__=="__main__":main()
