from __future__ import annotations
import re,time
import json
from pathlib import Path
from typing import Any
from .core import write_json
LABEL_RE=re.compile(r"^\s*(True|False|Unknown)\s*$",re.I)
LABEL_TOKEN_RE=re.compile(r"(?<![A-Za-z])(True|False|Unknown)(?![A-Za-z])",re.I)
IDENTITY_FILE="base_model_identity.json"
BRIDGE_COMPONENT_GROUPS={
    "parser_selector":"selector","relation_selector":"selector",
    "proposition":"atomic","equality":"atomic","adjective":"atomic",
    "intransitive":"atomic","transitive":"atomic","ditransitive":"atomic",
    "quantified":"quantified","binary":"logical","unary":"logical",
}
def bridge_component_group(component:str)->str:
    try:return BRIDGE_COMPONENT_GROUPS[component]
    except KeyError as exc:raise ValueError(f"Unknown NL2Logic component: {component}") from exc
def bridge_generation_cap(config:dict[str,Any],component:str|None)->int:
    fallback=int(config["model"]["max_new_tokens_bridge"])
    if component is None:return fallback
    return int(config.get("bridge",{}).get("generation_caps",{}).get(bridge_component_group(component),fallback))
def extract_direct_label(text:str)->str|None:
    exact=LABEL_RE.fullmatch(text)
    if exact:return exact.group(1).title()
    matches=LABEL_TOKEN_RE.findall(text)
    return matches[0].title() if len(matches)==1 else None
def model_identity(config:dict[str,Any])->dict[str,str|None]:
    return {"model_id":config["model"]["id"],"revision":config["model"].get("revision")}
def write_model_identity(path:Path,config:dict[str,Any])->Path:
    target=Path(path)/IDENTITY_FILE;write_json(target,model_identity(config));return target
def validate_model_artifact(path:Path,config:dict[str,Any])->dict[str,Any]:
    artifact=Path(path);candidates=[artifact/IDENTITY_FILE,artifact.parent/IDENTITY_FILE,artifact.parent.parent/IDENTITY_FILE]
    identity_path=next((candidate for candidate in candidates if candidate.exists()),None)
    if identity_path is None:raise ValueError(f"Model artifact missing base-model identity: {artifact}")
    identity=json.loads(identity_path.read_text(encoding="utf-8"));expected=model_identity(config)
    if identity!=expected:raise ValueError(f"Incompatible model artifact at {artifact}: expected {expected}, found {identity}")
    return identity
class ModelRunner:
    def __init__(self,config:dict[str,Any],adapter:Path|None=None):
        if adapter:validate_model_artifact(adapter,config)
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
        m=config["model"];print(f"[Model] Loading {m['id']} revision={m.get('revision') or 'default'}",flush=True);kwargs={"revision":m.get("revision"),"trust_remote_code":m.get("trust_remote_code",False),"device_map":"auto"}
        if m.get("quantization")=="nf4":print("[Model] Quantization: NF4",flush=True);kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16 if config["training"].get("bf16") else torch.float16)
        self.tokenizer=AutoTokenizer.from_pretrained(m["id"],revision=m.get("revision"),trust_remote_code=m.get("trust_remote_code",False));self.model=AutoModelForCausalLM.from_pretrained(m["id"],**kwargs)
        if adapter:
            from peft import PeftModel
            print(f"[Model] Attaching {adapter}",flush=True);self.model=PeftModel.from_pretrained(self.model,adapter)
        self.model.eval();self.model.generation_config.temperature=None;self.model.generation_config.top_p=None;self.model.generation_config.top_k=None;self.torch=torch;self.config=config;print(f"[GPU] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable (CPU)'}",flush=True)
    def generate_batch(self,prompts:list[str],bridge:bool=False,components:list[str|None]|None=None):
        if not prompts:return []
        components=components or [None]*len(prompts)
        if len(components)!=len(prompts):raise ValueError("components must match prompts")
        caps=[bridge_generation_cap(self.config,component) if bridge else int(self.config["model"]["max_new_tokens_direct"]) for component in components]
        if len(set(caps))!=1:raise ValueError(f"Batched prompts require one generation cap, found {sorted(set(caps))}")
        texts=[self.tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for prompt in prompts]
        original_padding_side=self.tokenizer.padding_side;self.tokenizer.padding_side="left"
        try:inputs=self.tokenizer(texts,return_tensors="pt",padding=True,truncation=True,max_length=self.config["model"]["max_sequence_length"]).to(self.model.device)
        finally:self.tokenizer.padding_side=original_padding_side
        if self.torch.cuda.is_available():self.torch.cuda.synchronize()
        started=time.perf_counter();generation_cap=caps[0]
        with self.torch.inference_mode():out=self.model.generate(**inputs,do_sample=False,use_cache=True,max_new_tokens=generation_cap,pad_token_id=self.tokenizer.eos_token_id)
        if self.torch.cuda.is_available():self.torch.cuda.synchronize()
        elapsed_ms=(time.perf_counter()-started)*1000;generated=out[:,inputs.input_ids.shape[1]:];results=[]
        for index,new in enumerate(generated):
            ids=new.tolist();eos=self.tokenizer.eos_token_id;output_tokens=next((position+1 for position,value in enumerate(ids) if value==eos),len(ids));tokens=new[:output_tokens];raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip();label=extract_direct_label(raw)
            input_tokens=int(inputs.attention_mask[index].sum().item()) if hasattr(inputs,"attention_mask") else int(inputs.input_ids.shape[1])
            results.append({"raw":raw,"label":label,"latency_ms":elapsed_ms/len(prompts),"batch_latency_ms":elapsed_ms,"batch_size":len(prompts),"input_tokens":input_tokens,"output_tokens":output_tokens,"generation_cap":generation_cap,"hit_generation_cap":output_tokens>=generation_cap})
        return results
    def generate(self,prompt:str,bridge:bool=False,component:str|None=None):
        return self.generate_batch([prompt],bridge,[component])[0]
