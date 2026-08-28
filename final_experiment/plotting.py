import argparse,json
from pathlib import Path
from .core import LABELS,ROOT
def save(fig,path):
    for ext in("png","pdf"):fig.savefig(path.with_suffix("."+ext),bbox_inches="tight")
def plot(files):
    import matplotlib.pyplot as plt
    import numpy as np
    reports=[json.loads(p.read_text(encoding="utf-8")) for p in files];out=ROOT/"plots";out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({"font.size":10,"figure.dpi":150})
    for key,label in(("accuracy","Accuracy"),("macro_f1","Macro F1")):
        fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4);width=.8/len(reports)
        for i,r in enumerate(reports):ax.bar(x+(i-(len(reports)-1)/2)*width,[r["conditions"].get(str(c),{}).get(key,0) for c in range(1,5)],width,label=r["dataset"])
        ax.set(xticks=x,xticklabels=["C1 Base","C2 Bridge","C3 LoRA","C4 LoRA+Bridge"],ylabel=label,ylim=(0,1),title=f"{label} by Condition");ax.legend();save(fig,out/f"{key}_comparison");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4);width=.8/len(reports)
    for i,r in enumerate(reports):ax.bar(x+(i-(len(reports)-1)/2)*width,[r["conditions"].get(str(c),{}).get("efficiency",{}).get("mean_latency_ms") or 0 for c in range(1,5)],width,label=r["dataset"])
    ax.set(xticks=x,xticklabels=["C1 Base","C2 Bridge","C3 LoRA","C4 LoRA+Bridge"],ylabel="Mean latency (ms/example)",title="End-to-End Latency by Condition");ax.legend();save(fig,out/"latency_comparison");plt.close(fig)
    for r in reports:
        for c,m in r["conditions"].items():
            cm=np.array(m["confusion_matrix"]);fig,ax=plt.subplots(figsize=(4.5,4));im=ax.imshow(cm,cmap="Blues")
            for i in range(3):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
            ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel="Predicted",ylabel="Gold",title=f"{r['dataset']} C{c}");fig.colorbar(im,ax=ax);save(fig,out/f"{r['dataset']}_c{c}_confusion");plt.close(fig)
        if r["dataset"]=="proofwriter":
            for key,label in(("accuracy","Accuracy"),("macro_f1","Macro F1")):
                fig,ax=plt.subplots(figsize=(7,4))
                for c,d in r["depth"].items():
                    ks=[k for k in("0","1","2","3","4","5+") if k in d];ax.plot(ks,[d[k][key] for k in ks],marker="o",label=f"C{c}")
                ax.set(xlabel="QDep",ylabel=label,ylim=(0,1),title=f"ProofWriter {label} vs Reasoning Depth");ax.legend();save(fig,out/f"proofwriter_{key}_by_depth");plt.close(fig)
        pairs=r.get("paired",{});names=[k for k in pairs if"bridge"in pairs[k]]
        if names:
            fig,ax=plt.subplots(figsize=(6,4));x=np.arange(len(names));ax.bar(x-.18,[len(pairs[k]["bridge"]["corrected"]) for k in names],.36,label="Corrected");ax.bar(x+.18,[len(pairs[k]["bridge"]["regressed"]) for k in names],.36,label="Regressed");ax.set(xticks=x,xticklabels=names,ylabel="Examples",title=f"{r['dataset']} Bridge Corrections vs Regressions");ax.legend();save(fig,out/f"{r['dataset']}_bridge_delta");plt.close(fig)
        if r.get("translation"):
            fig,ax=plt.subplots(figsize=(7,4));names=list(r["translation"]);cats=["schema_valid","parser_valid","ast_valid","successful_solver_executions"]
            for i,k in enumerate(cats):ax.bar(np.arange(len(names))+(i-1.5)*.18,[r["translation"][n][k] for n in names],.18,label=k.replace("_"," "))
            ax.set(xticks=np.arange(len(names)),xticklabels=[f"C{x}" for x in names],ylabel="Examples",title=f"{r['dataset']} Translation Pipeline");ax.legend(fontsize=8);save(fig,out/f"{r['dataset']}_translation_pipeline");plt.close(fig)
    # Training loss/eval loss/LR curves for every adapter with available logs.
    for history in (ROOT/"checkpoints").glob("*/*/training_history.json"):
        rows=json.loads(history.read_text(encoding="utf-8"));name=f"{history.parent.parent.name}_{history.parent.name}"
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,(key,title) in zip(axes,(("loss","Training Loss"),("eval_loss","Validation Loss"),("learning_rate","Learning Rate"))):
            pts=[r for r in rows if key in r];ax.plot([r["step"] for r in pts],[r[key] for r in pts]);ax.set(xlabel="Step",ylabel=title,title=title)
        fig.suptitle(name);save(fig,out/f"{name}_training_curves");plt.close(fig)
    print(f"[Plots] PNG/PDF saved to {out}")
def main():
    p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,action="append",required=True);plot(p.parse_args().analysis)
if __name__=="__main__":main()
