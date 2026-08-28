from __future__ import annotations
import argparse,csv,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
from .core import LABELS,ROOT,read_jsonl,write_json
def metrics(rows):
    scored=[r for r in rows if r.get("predicted_label") in LABELS];per={};supports=Counter(r["gold_label"] for r in scored)
    for label in LABELS:
        tp=sum(r["gold_label"]==label and r["predicted_label"]==label for r in scored);fp=sum(r["gold_label"]!=label and r["predicted_label"]==label for r in scored);fn=sum(r["gold_label"]==label and r["predicted_label"]!=label for r in scored);p=tp/(tp+fp) if tp+fp else 0.;q=tp/(tp+fn) if tp+fn else 0.;f=2*p*q/(p+q) if p+q else 0.;per[label]={"precision":p,"recall":q,"f1":f,"support":supports[label]}
    n=len(scored);total=len(rows);acc=sum(r["gold_label"]==r.get("predicted_label") for r in rows)/total if total else 0.;lat=[r.get("latency_ms") for r in rows if isinstance(r.get("latency_ms"),(int,float))];tokens=[r.get("token_count") for r in rows if isinstance(r.get("token_count"),(int,float))]
    return {"n_total":total,"n_scored":n,"coverage":n/total if total else 0.,"system_failures":total-n,"accuracy":acc,"macro_precision":statistics.mean(x["precision"] for x in per.values()),"macro_recall":statistics.mean(x["recall"] for x in per.values()),"macro_f1":statistics.mean(x["f1"] for x in per.values()),"weighted_f1":sum(per[x]["f1"]*supports[x] for x in LABELS)/n if n else 0.,"per_class":per,"confusion_matrix":[[sum(r["gold_label"]==g and r["predicted_label"]==p for r in scored) for p in LABELS] for g in LABELS],"efficiency":{"total_runtime_s":sum(lat)/1000 if lat else None,"mean_latency_ms":statistics.mean(lat) if lat else None,"median_latency_ms":statistics.median(lat) if lat else None,"p95_latency_ms":sorted(lat)[min(len(lat)-1,int(.95*len(lat)))] if lat else None,"generated_token_count":sum(tokens) if tokens else None}}
classification_metrics=metrics
def paired(a,b):
    x={r["example_id"]:r for r in a};y={r["example_id"]:r for r in b}
    if x.keys()!=y.keys():raise ValueError("Conditions do not use identical IDs")
    groups={"corrected":[],"regressed":[],"still_wrong":[],"unchanged_correct":[]}
    for k in x:
        ac=bool(x[k].get("correctness"));bc=bool(y[k].get("correctness"));groups["corrected" if not ac and bc else "regressed" if ac and not bc else "unchanged_correct" if ac else "still_wrong"].append(k)
    wrong=sum(not r.get("correctness") for r in a);correct=len(a)-wrong
    return groups|{"correction_rate":len(groups["corrected"])/wrong if wrong else 0.,"regression_rate":len(groups["regressed"])/correct if correct else 0.,"net_corrections":len(groups["corrected"])-len(groups["regressed"])}
bridge_delta=paired
def mcnemar(a,b):
    x={r["example_id"]:bool(r.get("correctness")) for r in a};y={r["example_id"]:bool(r.get("correctness")) for r in b};n01=sum(x[k] and not y[k] for k in x);n10=sum(not x[k] and y[k] for k in x);n=n01+n10;p=min(1.,2*sum(math.comb(n,k) for k in range(min(n01,n10)+1))/2**n) if n else 1.;return {"a_correct_b_wrong":n01,"a_wrong_b_correct":n10,"exact_p":p}
def bootstrap(a,b,key,n=2000,seed=42):
    rng=random.Random(seed);g=[]
    for _ in range(n):
        ix=[rng.randrange(len(a)) for _ in a];g.append(metrics([b[i] for i in ix])[key]-metrics([a[i] for i in ix])[key])
    g.sort();return {"gain":metrics(b)[key]-metrics(a)[key],"ci_low":g[int(.025*n)],"ci_high":g[min(n-1,int(.975*n))]}
def depth(rows):
    groups=defaultdict(list)
    for r in rows:
        d=r.get("qdep");groups["5+" if isinstance(d,(int,float)) and d>=5 else str(int(d)) if isinstance(d,(int,float)) else "missing"].append(r)
    return {k:{"sample_count":len(v),"accuracy":metrics(v)["accuracy"],"macro_f1":metrics(v)["macro_f1"]} for k,v in groups.items()}
def translation(rows):
    times=[r.get("timing",{}).get("solver_ms") for r in rows];times=[x for x in times if isinstance(x,(int,float))];errors=Counter(r.get("error_category") for r in rows if r.get("error_category"));total=len(rows);ast=sum(r.get("translation",{}).get("ast_valid",False) for r in rows);compiled=sum(r.get("translation",{}).get("compile_success",False) for r in rows);executable=sum(r.get("translation",{}).get("executable",False) for r in rows);agreement=Counter(r.get("translation",{}).get("semantic_equivalence","not_available") for r in rows);correct=sum(bool(r.get("correctness")) for r in rows);executable_rows=[r for r in rows if r.get("translation",{}).get("executable",False)]
    return {"attempts":total,"schema_valid":sum(r.get("translation",{}).get("schema_valid",False) for r in rows),"parser_valid":sum(r.get("translation",{}).get("parser_valid",False) for r in rows),"ast_valid":ast,"ast_valid_rate":ast/total if total else 0.,"compile_success":compiled,"compile_success_rate":compiled/total if total else 0.,"executable":executable,"executable_rate":executable/total if total else 0.,"successful_solver_executions":sum(r.get("solver_status")=="success" for r in rows),"semantic_agreement":dict(agreement),"final_solver_accuracy":correct/total if total else 0.,"executable_solver_accuracy":sum(bool(r.get("correctness")) for r in executable_rows)/len(executable_rows) if executable_rows else 0.,"failure_counts":dict(errors),"solver_time_ms":{"mean":statistics.mean(times) if times else None,"median":statistics.median(times) if times else None,"p95":sorted(times)[min(len(times)-1,int(.95*len(times)))] if times else None}}
def analyze(dataset,dirs,n=2000):
    by={int((rows:=read_jsonl(p/"predictions.jsonl"))[0]["condition"]):rows for p in dirs};out=ROOT/"results"/f"analysis_{dataset}_{max(p.name for p in dirs)}";out.mkdir(parents=True,exist_ok=False);report={"dataset":dataset,"conditions":{str(c):metrics(v) for c,v in by.items()},"depth":{str(c):depth(v) for c,v in by.items()} if dataset=="proofwriter" else {},"translation":{str(c):translation(v) for c,v in by.items() if c in(2,4)},"paired":{}}
    for a,b in ((1,2),(1,3),(3,4)):
        if a in by and b in by:report["paired"][f"C{a}_vs_C{b}"]={"mcnemar":mcnemar(by[a],by[b]),"accuracy":bootstrap(by[a],by[b],"accuracy",n),"macro_f1":bootstrap(by[a],by[b],"macro_f1",n),**({"bridge":paired(by[a],by[b])} if(a,b)in((1,2),(3,4)) else {})}
    if all(c in by for c in(1,2,3,4)):
        m={c:metrics(by[c]) for c in by};report["gains"]={k:{"bridge_gain":m[2][k]-m[1][k],"fine_tuning_gain":m[3][k]-m[1][k],"combined_gain":m[4][k]-m[1][k],"interaction_effect":m[4][k]-m[3][k]-m[2][k]+m[1][k]} for k in("accuracy","macro_f1")}
    report["error_analysis"]={str(c):{"counts":dict(Counter(r.get("error_category") for r in v if r.get("error_category"))),"representative_ids":{k:[r["example_id"] for r in v if r.get("error_category")==k][:5] for k in {r.get("error_category") for r in v if r.get("error_category")}}} for c,v in by.items()};write_json(out/"analysis.json",report)
    with(out/"paper_table.csv").open("w",newline="",encoding="utf-8")as f:
        w=csv.writer(f);w.writerow(["dataset","condition","accuracy","macro_f1","weighted_f1","system_failures"]);[w.writerow([dataset,c,m["accuracy"],m["macro_f1"],m["weighted_f1"],m["system_failures"]]) for c,m in report["conditions"].items()]
    print(json.dumps({"output":str(out),"conditions":report["conditions"],"paired":report["paired"]},indent=2));return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["proofwriter","folio"],required=True);p.add_argument("--run-dir",type=Path,action="append",required=True);p.add_argument("--bootstrap-samples",type=int,default=2000);a=p.parse_args();analyze(a.dataset,a.run_dir,a.bootstrap_samples)
if __name__=="__main__":main()
