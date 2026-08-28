from __future__ import annotations
import argparse,csv,time
from pathlib import Path
from .core import ADAPTER_NAMES,ROOT,resolved_config,runtime_metadata,seed_everything,write_json
from .data import load_manifest_split,load_training_validation
from .nl2logic_training import direct_training_examples
from .modeling import model_identity,validate_model_artifact,write_model_identity
from .training_data import DynamicCausalCollator,tokenize_training_record
def latest_checkpoint(path):
    candidates=[p for p in path.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    return max(candidates,key=lambda p:int(p.name.split("-")[-1])) if candidates else None
def training_schedule_kwargs():
    return {"eval_strategy":"epoch","save_strategy":"epoch","load_best_model_at_end":True,"metric_for_best_model":"eval_loss","greater_is_better":False}
def validate_training_task(task):
    if task!="direct":raise ValueError("The NL2Logic + Z3 bridge is frozen and never fine-tuned; only --task direct is valid")
    return task
def train(dataset,task,mode,resume=None,final_full_validation=False):
    task=validate_training_task(task)
    import torch
    from datasets import Dataset
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainerCallback,TrainingArguments
    cfg=resolved_config(dataset);seed_everything(cfg["experiment"]["seed"]);t=cfg["training"];m=cfg["model"];name=ADAPTER_NAMES[dataset];out=ROOT/"checkpoints"/name/mode;out.mkdir(parents=True,exist_ok=True)
    if (out/"base_model_identity.json").exists():validate_model_artifact(out,cfg)
    else:write_model_identity(out,cfg)
    if resume:validate_model_artifact(Path(resume),cfg)
    base=ROOT/"data"/"processed"/dataset;train_rows,train_manifest=load_manifest_split(base,"train",mode);val_rows,val_manifest,full_val_manifest=load_training_validation(base,mode);_test_rows,test_manifest=load_manifest_split(base,"test",mode)
    train_records=direct_training_examples(train_rows);val_records=direct_training_examples(val_rows);print(f"[Dataset] {dataset} mode={mode} selected_train={len(train_rows)} direct_training_records={len(train_records)} training_validation={len(val_rows)} direct_validation_records={len(val_records)} full_validation={full_val_manifest['size']} heldout_evaluation={test_manifest['size']}",flush=True)
    print(f"[Manifest] train={train_manifest['ids_sha256']} training_validation={val_manifest['ids_sha256']} full_validation={full_val_manifest['ids_sha256']} test={test_manifest['ids_sha256']}",flush=True);print(f"[Training] Starting {name}/{mode}",flush=True)
    tok=AutoTokenizer.from_pretrained(m["id"],revision=m.get("revision"));tok.pad_token=tok.eos_token;q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if t["bf16"] else torch.float16);model=AutoModelForCausalLM.from_pretrained(m["id"],revision=m.get("revision"),device_map="auto",quantization_config=q);model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=t["gradient_checkpointing"]);model=get_peft_model(model,LoraConfig(r=t["lora_rank"],lora_alpha=t["lora_alpha"],lora_dropout=t["lora_dropout"],bias="none",task_type="CAUSAL_LM",target_modules="all-linear"))
    def make_dataset(records):
        return Dataset.from_list(records).map(
            lambda record:tokenize_training_record(record,tok,m["max_sequence_length"]),
            remove_columns=list(records[0]),
        )
    train_ds=make_dataset(train_records);val_ds=make_dataset(val_records);history=[]
    class Progress(TrainerCallback):
        started=time.perf_counter()
        def on_log(self,args,state,control,logs=None,**kw):
            row={"step":state.global_step,"epoch":state.epoch,**(logs or {})};elapsed=time.perf_counter()-self.started;eta=elapsed/state.global_step*(state.max_steps-state.global_step) if state.global_step else 0.;row|={"elapsed_s":elapsed,"eta_s":eta};history.append(row);pct=100*state.global_step/max(1,state.max_steps);parts=[f"Epoch {state.epoch or 0:.2f}/{args.num_train_epochs}",f"Step {state.global_step}/{state.max_steps}",f"{pct:.0f}%"]+[f"{k}={row[k]:.5g}" for k in ("loss","eval_loss","learning_rate") if k in row]+[f"elapsed={elapsed/60:.1f}m",f"ETA={eta/60:.1f}m"]
            if torch.cuda.is_available():parts.append(f"GPU={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
            print(" | ".join(parts),flush=True)
    args=TrainingArguments(output_dir=str(out/"latest"),num_train_epochs=t["epochs"],per_device_train_batch_size=t["batch_size"],per_device_eval_batch_size=t["batch_size"],gradient_accumulation_steps=t["gradient_accumulation"],learning_rate=t["learning_rate"],warmup_ratio=t["warmup_ratio"],lr_scheduler_type=t["scheduler"],optim=t["optimizer"],fp16=t["fp16"],bf16=t["bf16"],logging_steps=t["logging_steps"],save_total_limit=3,report_to="none",seed=cfg["experiment"]["seed"],**training_schedule_kwargs());trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,data_collator=DynamicCausalCollator(tok.pad_token_id),callbacks=[Progress()]);trainer.train(resume_from_checkpoint=resume or False)
    full_metrics=None
    if final_full_validation:
        full_val_rows,_=load_manifest_split(base,"validation",mode);full_records=direct_training_examples(full_val_rows);print(f"[Full validation] rows={len(full_val_rows)} direct_records={len(full_records)}",flush=True);full_metrics=trainer.evaluate(make_dataset(full_records),metric_key_prefix="full_validation");write_json(out/"full_validation_metrics.json",full_metrics)
    trainer.save_model(str(out/"final_adapter"));tok.save_pretrained(out/"final_adapter");write_model_identity(out/"final_adapter",cfg);write_json(out/"training_history.json",history);write_json(out/"metadata.json",runtime_metadata(cfg)|{"adapter":name,"training_task":"direct","bridge_training":False,"mode":mode,"base_model_identity":model_identity(cfg),"train_manifest":train_manifest,"training_validation_manifest":val_manifest,"full_validation_manifest":full_val_manifest,"heldout_test_manifest":test_manifest,"best_checkpoint":trainer.state.best_model_checkpoint,"dynamic_padding":True,"loss_masking":"target_tokens_only","final_full_validation":bool(final_full_validation),"full_validation_metrics":full_metrics,"final_adapter":str(out/"final_adapter")})
    with (out/"training_history.csv").open("w",newline="",encoding="utf-8") as f:
        keys=sorted({k for r in history for k in r});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(history)
    print(f"[Training] Complete best={trainer.state.best_model_checkpoint} final={out/'final_adapter'}",flush=True)
def build_parser():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["proofwriter","folio"],required=True);p.add_argument("--task",choices=["direct"],required=True);p.add_argument("--mode",choices=["smoke","pilot","main","full"],required=True);p.add_argument("--resume-from-checkpoint");p.add_argument("--final-full-validation",action="store_true");return p
def main():
    a=build_parser().parse_args();train(a.dataset,a.task,a.mode,a.resume_from_checkpoint,a.final_full_validation)
if __name__=="__main__":main()
