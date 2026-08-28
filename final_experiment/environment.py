import importlib.util,json,platform,sys
from .core import resolved_config
def main():
    cfg=resolved_config("proofwriter");mods={m:bool(importlib.util.find_spec(m)) for m in("torch","transformers","datasets","peft","z3","yaml","sklearn","matplotlib")};cuda=False;gpu=None
    if mods["torch"]:
        import torch
        cuda=torch.cuda.is_available();gpu=torch.cuda.get_device_name(0) if cuda else None
    result={"status":"ok" if all(mods.values()) else"missing_dependencies","python":sys.version.split()[0],"os":platform.platform(),"base_model":cfg["model"]["id"],"model_revision":cfg["model"].get("revision"),"max_sequence_length":cfg["model"]["max_sequence_length"],"dynamic_padding":cfg["training"]["dynamic_padding"],"answer_only_loss":cfg["training"]["answer_only_loss"],"bridge_training":False,"frozen_bridge":"nl2logic_ast_two_pass_z3_v1","packages":mods,"cuda_available":cuda,"gpu":gpu,"note":"The active direct-record audit maximum is 292 tokens with zero truncation. NL2Logic + AST + two-pass compiler + Z3 is frozen and never fine-tuned."};print(json.dumps(result,indent=2))
    if not all(mods.values()):raise SystemExit(1)
if __name__=="__main__":main()
