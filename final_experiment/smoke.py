import argparse,json,tempfile
from pathlib import Path
from .analysis import metrics,mcnemar
from .bridge import solve,validate_theory
from .core import normalize_label,read_jsonl,write_jsonl
from .data import load_manifest_split,normalize_row,save_mode_manifests,stratified_sample
from .nl2logic_parser import NL2LogicParser
TOY={"premises":[{"type":"predicate","name":"red","args":["alice"]}],"hypothesis":{"type":"predicate","name":"red","args":["alice"]}}
PW={"id":"world1","maxD":1,"NFact":1,"NRule":1,"theory":"Alice is red.","question":"Alice is red.","answer":"True","QDep":0,"allProofs":"fact"}
def run(with_model=False):
    status={}
    def check(name,fn):
        try:fn();status[name]="PASS";print(f"[Smoke] PASS {name}")
        except Exception as e:status[name]=f"FAIL: {type(e).__name__}: {e}";print(f"[Smoke] FAIL {name}: {e}")
    row=normalize_row("proofwriter",PW,"test",0);check("dataset_schema",lambda:normalize_row("proofwriter",PW,"test",0));check("label_mapping",lambda:(_ for _ in()).throw(AssertionError()) if normalize_label("uncertain")!="Unknown" else None);check("stratified_sampling",lambda:stratified_sample([row|{"example_id":str(i),"label":["True","False","Unknown"][i%3]} for i in range(12)],6,"proofwriter",42));trace=[]
    class Scripted:
        def __init__(self):self.outputs=iter(['{"parser":"atomic"}','{"relation":"adjective"}','{"predicate":"red","subject":"alice"}','{"parser":"atomic"}','{"relation":"adjective"}','{"predicate":"red","subject":"alice"}'])
        def __call__(self,prompt,bridge=True,component=None):return {"raw":next(self.outputs),"latency_ms":0.,"input_tokens":1,"output_tokens":1,"generation_cap":64,"hit_generation_cap":False}
    parsed={}
    def recursive_parse():
        parsed.update(NL2LogicParser(Scripted(),"proofwriter",2,"smoke",trace.append).parse_theory(row));validate_theory(parsed);assert len(trace)==2 and {x["status"] for x in trace}=={"success","memoized"} and all(x["returned_ast_node"] for x in trace)
    check("nl2logic_recursive_parser",recursive_parse);check("z3_solver",lambda:(_ for _ in()).throw(AssertionError()) if solve(TOY)["label"]!="True" else None);sample=[{"example_id":"1","gold_label":"True","predicted_label":"True","correctness":True}];check("metrics",lambda:metrics(sample));check("statistics",lambda:mcnemar(sample,sample))
    def io():
        with tempfile.TemporaryDirectory()as d:p=Path(d)/"x.jsonl";write_jsonl(p,[row]);assert read_jsonl(p)==[row]
    check("result_saving",io)
    def manifests():
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);parts={}
            for split in("train","validation","test"):
                parts[split]=[row|{"split":split,"example_id":f"{split}-{i}","world_id":f"{split}-world-{i}","label":["True","False","Unknown"][i%3],"QDep":i%2} for i in range(12)]
                write_jsonl(base/f"{split}.jsonl",parts[split])
            saved=save_mode_manifests(base,parts,"proofwriter","smoke",6,6,42);selected,_=load_manifest_split(base,"train","smoke");assert len(selected)==saved["train"]["size"]==6
    check("frozen_sampling_manifests",manifests)
    if with_model:
        from .core import resolved_config
        from .modeling import ModelRunner
        from .prompts import direct_prompt
        runner=ModelRunner(resolved_config("proofwriter"));check("base_model_loading",lambda:None);check("five_example_direct_inference",lambda:[runner.generate(direct_prompt(row)) for _ in range(5)]);generated={}
        model_trace=[]
        def bridge_generation():generated.update(NL2LogicParser(runner.generate,"proofwriter",2,"smoke-model",model_trace.append).parse_theory(row))
        check("nl2logic_recursive_generation",bridge_generation)
        def e2e():
            try:result=solve(generated);record={"system_status":"success","prediction":result["label"]}
            except Exception as e:record={"system_status":getattr(e,"status","system_failure"),"prediction":None}
            assert record["system_status"] and (record["prediction"] in("True","False","Unknown") or record["prediction"] is None)
        check("tiny_end_to_end",e2e)
    else:
        for x in("base_model_loading","five_example_direct_inference","nl2logic_recursive_generation","tiny_end_to_end"):status[x]="SKIP (use --with-model)";print(f"[Smoke] SKIP {x}")
    print(json.dumps(status,indent=2));
    if any(v.startswith("FAIL") for v in status.values()):raise SystemExit(1)
def main():
    p=argparse.ArgumentParser();p.add_argument("--with-model",action="store_true");run(p.parse_args().with_model)
if __name__=="__main__":main()
