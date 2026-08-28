import argparse,csv,json
from pathlib import Path
from .core import ROOT
def summarize(files):
    rows=[]
    for p in files:
        r=json.loads(p.read_text(encoding="utf-8"));print(f"\n{r['dataset'].upper()}")
        for c,m in r["conditions"].items():
            fs=", ".join(f"{k}={v['f1']:.3f}" for k,v in m["per_class"].items());print(f"C{c}: accuracy={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} F1[{fs}] failures={m['system_failures']}");rows.append({"dataset":r["dataset"],"condition":c,"accuracy":m["accuracy"],"macro_f1":m["macro_f1"],**{f"f1_{k.lower()}":v["f1"] for k,v in m["per_class"].items()}})
        for name,x in r.get("paired",{}).items():print(f"{name}: McNemar p={x['mcnemar']['exact_p']:.4g}"+(f" corrected={len(x['bridge']['corrected'])} regressed={len(x['bridge']['regressed'])}" if"bridge"in x else""))
        if r.get("depth"):print("Depth trend:",json.dumps(r["depth"],separators=(",",":")))
        if r.get("translation"):print("Translation:",json.dumps(r["translation"],separators=(",",":")))
    print("\nBEST TRAINING CHECKPOINTS")
    for p in (ROOT/"checkpoints").glob("*/*/metadata.json"):
        meta=json.loads(p.read_text(encoding="utf-8"));print(f"{p.parent.parent.name}/{p.parent.name}: {meta.get('best_checkpoint')}")
    out=ROOT/"results"/"final_research_summary.csv";out.parent.mkdir(exist_ok=True)
    with out.open("w",newline="",encoding="utf-8")as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f"\nPaper table: {out}")
def main():
    p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,action="append",required=True);summarize(p.parse_args().analysis)
if __name__=="__main__":main()
