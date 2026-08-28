from __future__ import annotations
import argparse,json,time
from pathlib import Path
from .core import FROZEN_BRIDGE_ID,CONDITIONS,ROOT,PredictionRow,adapter_name_for_condition,new_run_dir,resolved_config,runtime_metadata,seed_everything,uses_frozen_bridge,write_json,write_jsonl
from .data import load_manifest_split
from .fol_ast import BridgeError,normalized_logic
from .gold_targets import folio_target,proofwriter_target
from .modeling import ModelRunner
from .nl2logic_parser import NL2LogicParser
from .prompts import direct_prompt
from .z3_compiler import compile_theory
from .z3_solver import semantic_agreement,solve_compiled

def selected_rows(dataset,mode):
    rows,_=load_manifest_split(ROOT/"data"/"processed"/dataset,"test",mode);return rows

def _gold(dataset,row):return folio_target(row) if dataset=="folio" else proofwriter_target(row)

def _bridge_error_stage(status):
    if status.startswith("compiler"):return "compiler"
    if status.startswith("solver") or status in {"inconsistent_premises"}:return "solver"
    return "parser"

def evaluate_rows(dataset:str,condition:int,mode:str,rows:list[dict],runner,out:Path,rid:str,cfg:dict)->Path:
    results=[];traces=[];started=time.perf_counter();bridge=uses_frozen_bridge(condition);print(f"[Evaluation] dataset={dataset} condition={condition} mode={mode} examples={len(rows)}",flush=True)
    for index,row in enumerate(rows,1):
        example_started=time.perf_counter();item=PredictionRow(dataset=dataset,run_id=rid,example_id=row["example_id"],condition=condition,gold_label=row["label"],qdep=row.get("QDep"),maxD=row.get("maxD"),NFact=row.get("NFact"),NRule=row.get("NRule"));trace_start=len(traces);parser=None
        try:
            if not bridge:
                generated=runner.generate(direct_prompt(row),False);item.raw_model_response=generated["raw"];item.latency_ms=generated["latency_ms"];item.token_count=generated["input_tokens"]+generated["output_tokens"];item.predicted_label=generated["label"];item.error_category=None if generated["label"] else ("empty_output" if not generated["raw"] else "malformed_direct_output")
            else:
                item.translation={"frozen_bridge_id":FROZEN_BRIDGE_ID,"schema_valid":False,"parser_valid":False,"ast_valid":False,"compile_success":False,"executable":False,"semantic_equivalence":"not_available"};parser=NL2LogicParser(runner.generate,dataset,condition,rid,traces.append);parse_started=time.perf_counter();theory=parser.parse_theory(row);item.timing["parser_ms"]=(time.perf_counter()-parse_started)*1000;item.parser_status="success";item.translation|={"schema_valid":True,"parser_valid":True,"ast_valid":True,"normalized_logic":normalized_logic(theory)};item.generated_logic=normalized_logic(theory);item.latency_ms=parser.usage["latency_ms"];item.token_count=parser.usage["token_count"]
                compile_started=time.perf_counter()
                try:compiled=compile_theory(theory)
                finally:item.timing["compiler_ms"]=(time.perf_counter()-compile_started)*1000
                item.compiler_status="success";item.translation["compile_success"]=True;z3_started=time.perf_counter()
                try:
                    gold=_gold(dataset,row);item.translation|={"exact_match":theory==gold,"normalized_logic_match":normalized_logic(theory)==normalized_logic(gold)};agreement=semantic_agreement(theory,gold,cfg["experiment"]["solver_timeout_ms"]);item.translation["semantic_equivalence"]=agreement["status"];item.translation["semantic_checks"]=agreement["checks"]
                except Exception as agreement_error:item.translation["semantic_equivalence"]="not_available";item.translation["semantic_agreement_error"]=f"{type(agreement_error).__name__}: {agreement_error}"
                try:solved=solve_compiled(compiled,cfg["experiment"]["solver_timeout_ms"])
                finally:item.timing["z3_ms"]=(time.perf_counter()-z3_started)*1000
                item.solver_status="success";item.solver_result=json.dumps(solved["statuses"]);item.predicted_label=solved["label"];item.timing["solver_ms"]=solved["solver_ms"];item.translation["executable"]=True
            item.correctness=item.predicted_label==item.gold_label
            if not item.correctness and item.error_category is None:item.error_category="incorrect_final_symbolic_answer" if bridge else "incorrect_direct_llm_prediction"
        except BridgeError as exc:
            item.error_category=exc.status;stage=_bridge_error_stage(exc.status)
            if stage=="parser":item.parser_status=exc.status
            elif stage=="compiler":item.compiler_status=exc.status
            else:item.solver_status=exc.status
            item.translation[exc.status]=True
        except Exception as exc:
            item.error_category="system_exception" if bridge else "model_failure";item.solver_status="exception" if bridge else item.solver_status;item.raw_model_response=(item.raw_model_response or "")+f"\n[EXCEPTION] {type(exc).__name__}: {exc}"
        finally:
            if bridge:
                current=traces[trace_start:];raw=[{"step_id":e["step_id"],"selector":e.get("selector_raw_output"),"subparser":e.get("subparser_raw_output")} for e in current];item.raw_model_response=json.dumps(raw,ensure_ascii=False);item.latency_ms=item.latency_ms if item.latency_ms is not None else sum(e.get("latency_ms",0) for e in current);item.token_count=item.token_count if item.token_count is not None else sum(e.get("token_count",0) for e in current)
                if parser is not None:
                    cache_lookups=parser.usage["cache_lookups"]+parser.usage["ast_cache_lookups"];cache_hits=parser.usage["cache_hits"]+parser.usage["ast_cache_hits"]
                    item.translation|={"recursive_llm_call_count":parser.usage["model_calls"],"model_invocation_count":parser.usage["model_invocations"],"batch_invocation_count":parser.usage["batch_invocations"],"batch_sizes":parser.usage["batch_sizes"],"parser_call_requests":parser.usage["parser_call_requests"],"generated_token_count":parser.usage["generated_token_count"],"unique_normalized_clauses":parser.usage["unique_normalized_clauses"],"selector_cache_hits":parser.usage["selector_cache_hits"],"subparser_cache_hits":parser.usage["subparser_cache_hits"],"prefetch_cache_hits":parser.usage["prefetch_cache_hits"],"ast_cache_hits":parser.usage["ast_cache_hits"],"cache_lookups":cache_lookups,"cache_hits":cache_hits,"cache_hit_rate":cache_hits/cache_lookups if cache_lookups else 0.0};item.timing|={"selector_ms":parser.usage["selector_ms"],"subparser_ms":parser.usage["subparser_ms"]}
                write_jsonl(out/"parser_traces.jsonl",traces)
            item.timing["total_ms"]=(time.perf_counter()-example_started)*1000
        results.append(item.to_dict());elapsed=time.perf_counter()-started;eta=elapsed/index*(len(rows)-index);print(f"[Evaluation] {dataset} C{condition} {index}/{len(rows)} {100*index/len(rows):.0f}% elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",flush=True)
        if index%10==0 or index==len(rows):write_jsonl(out/"predictions.jsonl",results)
    return out

def evaluate(dataset,condition,mode,run_id=None):
    cfg=resolved_config(dataset);seed_everything(cfg["experiment"]["seed"]);rows=selected_rows(dataset,mode);adapter_name=adapter_name_for_condition(dataset,condition);adapter=None
    if adapter_name:
        adapter=ROOT/"checkpoints"/adapter_name/mode/"final_adapter"
        if not adapter.exists():raise FileNotFoundError(f"Required adapter not found: {adapter}")
    out=new_run_dir(dataset,f"condition{condition}_{mode}",run_id);rid=out.name;write_json(out/"config.json",cfg);write_json(out/"metadata.json",runtime_metadata(cfg)|{"dataset":dataset,"condition":condition,"condition_name":CONDITIONS[condition],"mode":mode,"adapter_name":adapter_name,"frozen_bridge_id":FROZEN_BRIDGE_ID if uses_frozen_bridge(condition) else None,"modeled_factors":{"direct_lora":condition in (3,4),"frozen_bridge":uses_frozen_bridge(condition)},"evaluation_size":len(rows),"test_ids":[r["example_id"] for r in rows]});runner=ModelRunner(cfg,adapter);evaluate_rows(dataset,condition,mode,rows,runner,out,rid,cfg);print(f"[Evaluation] Results: {out}");return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["proofwriter","folio"],required=True);parser.add_argument("--condition",type=int,choices=[1,2,3,4],required=True);parser.add_argument("--mode",choices=["smoke","pilot","main","full"],default="main");parser.add_argument("--run-id");args=parser.parse_args();evaluate(args.dataset,args.condition,args.mode,args.run_id)
if __name__=="__main__":main()
