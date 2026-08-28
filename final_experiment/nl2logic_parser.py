"""Faithful recursive NL2Logic selector/subparser pipeline with full traces."""
from __future__ import annotations
import copy,difflib,re,unicodedata
from typing import Any,Callable
from .fol_ast import BridgeError,extract_json,validate_expr,validate_theory,IDENTIFIER,VARIABLE
from .modeling import bridge_component_group
from .prompts import component_prompt

Generator=Callable[...,dict[str,Any]]
TraceSink=Callable[[dict[str,Any]],None]

_PUNCTUATION_TRANSLATION=str.maketrans({"’":"'","‘":"'","“":"\"","”":"\"","–":"-","—":"-"})
def canonical_clause(value:str)->str:
    text=unicodedata.normalize("NFKC",value).translate(_PUNCTUATION_TRANSLATION)
    text=re.sub(r"\s+"," ",text.strip());text=re.sub(r"\s+([,;:!?.])",r"\1",text);text=re.sub(r"([,;:!?])(?=\S)",r"\1 ",text)
    return re.sub(r"[.!?]+$","",text).strip().casefold()

def _choice(value:dict[str,Any],key:str,allowed:set[str])->str:
    answer=value.get(key)
    if answer not in allowed:raise BridgeError("schema_invalid",f"{key} must be one of {sorted(allowed)}")
    return answer
def _text(value:dict[str,Any],key:str)->str:
    answer=value.get(key)
    if not isinstance(answer,str) or not answer.strip():raise BridgeError("schema_invalid",f"{key} must be non-empty string")
    return answer.strip().strip('"')
def _symbol(value:str)->str:
    result=re.sub(r"[^A-Za-z0-9_]+","_",value.strip().lower()).strip("_")
    if not result:raise BridgeError("ast_invalid",f"cannot normalize symbol {value!r}")
    return result

# Literal negation markers used by the reference NL2Logic unary parser to detect invented negation.
NEGATION_MARKERS=("not","do not","dont","don't","does not","doesn't")
# Reference NL2Logic ChooseRelation options A-D, in order. `proposition` and `equality` are fork
# additions with no upstream counterpart, so a locally invalid node from either (or from a
# mis-chosen valency) falls back deterministically through this ladder.
RELATION_FALLBACK_ORDER=("adjective","intransitive","transitive","ditransitive")
def _soft_text(value:dict[str,Any],key:str)->str:
    answer=value.get(key)
    return answer.strip().strip('"') if isinstance(answer,str) else ""
def _soft_choice(value:dict[str,Any],key:str,allowed:set[str])->str|None:
    answer=value.get(key)
    return answer if answer in allowed else None
def _variable_symbol(value:str)->str:
    return re.sub(r"[^A-Za-z0-9_]+","_",value.strip().lower()).strip("_")
# Copulas the parsing model sometimes carries into a predicate name ("is_green" vs "green"),
# which splits one relation into two unrelated Z3 symbols. Stripped deterministically, from
# predicate names only - never from entity terms, variables, or equality operands.
COPULA_PREFIXES=("is_","are_","was_","were_","be_","been_","am_")
def _predicate_symbol(value:str)->str:
    """_symbol for a relation name, with a leading copula removed when a valid name remains."""
    result=_symbol(value)
    for prefix in COPULA_PREFIXES:
        if result.startswith(prefix):
            stripped=result[len(prefix):].strip("_")
            if stripped and IDENTIFIER.fullmatch(stripped):return stripped
            break
    return result

# A predicate name like "not_blue" hides the negation from the solver: Z3 sees an atom unrelated
# to "blue" rather than its complement. Lift such a name into an explicit `not` node so the
# compiler and solver see the real structure. Applies to predicate names only, and only when a
# valid identifier remains after the prefix.
NEGATION_NAME_PREFIX="not_"
def _apply_name_negation(node:dict[str,Any])->tuple[dict[str,Any],str|None]:
    if node.get("type")!="predicate":return node,None
    name=node["name"]
    if not name.startswith(NEGATION_NAME_PREFIX):return node,None
    stripped=name[len(NEGATION_NAME_PREFIX):].strip("_")
    if not stripped or not IDENTIFIER.fullmatch(stripped):return node,None
    return {"type":"not","body":{**node,"name":stripped}},name

# NL2Logic requires a logical connective to combine complete clauses, not partial phrases. The
# parsing model frequently latches onto the inner conjunction of "If A and B then C", returning
# operands A and B and silently dropping consequent C - which then asserts the antecedent as a
# fact. Content words of the clause that neither operand represents mark such a decomposition
# invalid. Function words are ignored, and near-matches are tolerated so that a spelling
# normalization ("usally" -> "usually") does not count as dropped content.
BINARY_COVERAGE_STOPWORDS=frozenset("""a an the is are was were be been being that it its they them their he she his her
this those these of to in on at by with as and or if then so some all every each any anyone someone something anything
thing things people person who whom which what usually always almost also too sure guy does do did not no nor but when
there here""".split())
_WORD_RE=re.compile(r"[a-z']+")
def _tokens(text:str)->set[str]:return set(_WORD_RE.findall(text.lower()))
def _uncovered_content(clause:str,operands:list[str],cutoff:float=0.85)->list[str]:
    covered=set()
    for operand in operands:covered|=_tokens(operand)
    missing=sorted(_tokens(clause)-BINARY_COVERAGE_STOPWORDS-covered)
    return [word for word in missing if not any(difflib.SequenceMatcher(None,word,other).ratio()>=cutoff for other in covered)]

class NL2LogicParser:
    def __init__(self,generate:Generator,dataset:str,condition:int,run_id:str,trace_sink:TraceSink,max_depth:int=12):
        self.generate=generate;owner=getattr(generate,"__self__",None);self.generate_batch=getattr(owner,"generate_batch",None) or getattr(generate,"generate_batch",None);self.dataset=dataset;self.condition=condition;self.run_id=run_id;self.trace_sink=trace_sink;self.max_depth=max_depth
        self.selector_cache={};self.subparser_cache={};self.ast_cache={};self.inflight=set();self.unique_clauses=set();self._batch_index=0
        self.usage={"latency_ms":0.0,"token_count":0,"generated_token_count":0,"model_calls":0,"model_invocations":0,"batch_invocations":0,"parser_call_requests":0,"cache_lookups":0,"cache_hits":0,"selector_cache_hits":0,"subparser_cache_hits":0,"prefetch_cache_hits":0,"ast_cache_lookups":0,"ast_cache_hits":0,"unique_normalized_clauses":0,"selector_ms":0.0,"subparser_ms":0.0,"batch_sizes":[]}
    def _cache(self,component:str):return self.selector_cache if bridge_component_group(component)=="selector" else self.subparser_cache
    def _key(self,component:str,clause:str):return component,canonical_clause(clause)
    def _record(self,component:str,result:dict[str,Any],event:dict[str,Any],*,cache_hit:bool=False,prefetched:bool=False,batch_id:str|None=None):
        group=bridge_component_group(component);lat=0.0 if cache_hit else float(result.get("latency_ms") or 0);input_tokens=0 if cache_hit else int(result.get("input_tokens") or 0);output_tokens=0 if cache_hit else int(result.get("output_tokens") or 0)
        event["model_calls"].append({"component":component,"component_group":group,"latency_ms":lat,"input_token_count":input_tokens,"generated_token_count":output_tokens,"cached_generated_token_count":int(result.get("output_tokens") or 0) if cache_hit else None,"generation_cap":result.get("generation_cap"),"hit_generation_cap":bool(result.get("hit_generation_cap",False)),"cache_hit":cache_hit,"prefetched":prefetched,"batch_id":batch_id,"batch_size":int(result.get("batch_size") or 1)})
        if cache_hit:return
        tokens=input_tokens+output_tokens;event["latency_ms"]+=lat;event["token_count"]+=tokens;self.usage["latency_ms"]+=lat;self.usage["token_count"]+=tokens;self.usage["generated_token_count"]+=output_tokens;self.usage["model_calls"]+=1
        if group=="selector":self.usage["selector_ms"]+=lat
        else:self.usage["subparser_ms"]+=lat
    def _call(self,component:str,clause:str,event:dict[str,Any])->tuple[dict[str,Any],str]:
        self.usage["parser_call_requests"]+=1;self.usage["cache_lookups"]+=1;cache=self._cache(component);key=self._key(component,clause);entry=cache.get(key)
        if entry is not None:
            self.usage["cache_hits"]+=1;group=bridge_component_group(component);self.usage[f"{group if group=='selector' else 'subparser'}_cache_hits"]+=1
            if entry["prefetched"] and entry["uses"]==0:self.usage["prefetch_cache_hits"]+=1
            entry["uses"]+=1;result=entry["result"];self._record(component,result,event,cache_hit=True,prefetched=entry["prefetched"],batch_id=entry.get("batch_id"));raw=result.get("raw","")
            try:return extract_json(raw),raw
            except BridgeError:
                event["failed_raw_output"]={"component":component,"raw":raw};raise
        result=self.generate(component_prompt(component,clause),True,component=component);self.usage["model_invocations"]+=1;cache[key]={"result":result,"uses":1,"prefetched":False,"batch_id":None};self._record(component,result,event);raw=result.get("raw","")
        try:return extract_json(raw),raw
        except BridgeError:
            event["failed_raw_output"]={"component":component,"raw":raw};raise
    def _prefetch_component(self,component:str,clauses:list[str],event:dict[str,Any]|None):
        cache=self._cache(component);pending=[];seen=set()
        for clause in clauses:
            key=self._key(component,clause)
            if key in cache or key in seen:continue
            seen.add(key);pending.append((key,clause))
        if len(pending)<2 or self.generate_batch is None:return
        prompts=[component_prompt(component,clause) for _,clause in pending];components=[component]*len(pending);self._batch_index+=1;batch_id=f"batch-{self._batch_index}"
        results=self.generate_batch(prompts,True,components=components);self.usage["model_invocations"]+=1;self.usage["batch_invocations"]+=1;self.usage["batch_sizes"].append(len(pending))
        sink=event if event is not None else {"model_calls":[],"latency_ms":0.0,"token_count":0}
        for (key,_),result in zip(pending,results):cache[key]={"result":result,"uses":0,"prefetched":True,"batch_id":batch_id};self._record(component,result,sink,prefetched=True,batch_id=batch_id)
    def _peek(self,component:str,clause:str):
        entry=self._cache(component).get(self._key(component,clause));return extract_json(entry["result"].get("raw","")) if entry else None
    def _prefetch_siblings(self,clauses:list[str],event:dict[str,Any]|None=None):
        if self.generate_batch is None:return
        clauses=[" ".join(value.strip().split()) for value in clauses if value.strip()]
        if len(clauses)<2:return
        self._prefetch_component("parser_selector",clauses,event);atomic=[];structured={"quantified":[],"binary":[],"unary":[]}
        for clause in clauses:
            try:
                value=self._peek("parser_selector",clause)
                if value is None:continue
                decision=_choice(value,"parser",{"atomic","quantified","binary","unary"})
            except BridgeError:continue
            (atomic if decision=="atomic" else structured[decision]).append(clause)
        if atomic:self._prefetch_component("relation_selector",atomic,event)
        atomic_kinds={}
        for clause in atomic:
            try:
                value=self._peek("relation_selector",clause)
                if value is None:continue
                kind=_choice(value,"relation",{"proposition","adjective","intransitive","transitive","ditransitive","equality"});atomic_kinds.setdefault(kind,[]).append(clause)
            except BridgeError:continue
        for component,values in atomic_kinds.items():self._prefetch_component(component,values,event)
        for component,values in structured.items():
            if values:self._prefetch_component(component,values,event)
    def _relation_node(self,kind:str,parsed:dict[str,Any])->dict[str,Any]:
        if kind=="equality":return {"type":"equals","left":_symbol(_text(parsed,"left")),"right":_symbol(_text(parsed,"right"))}
        if kind=="proposition":return {"type":"predicate","name":_predicate_symbol(_text(parsed,"predicate")),"args":[]}
        predicate=_predicate_symbol(_text(parsed,"predicate"));args=[_symbol(_text(parsed,"subject"))]
        if kind=="transitive":args.append(_symbol(_text(parsed,"object")))
        elif kind=="ditransitive":args.extend([_symbol(_text(parsed,"indirect_object")),_symbol(_text(parsed,"direct_object"))])
        return {"type":"predicate","name":predicate,"args":args}
    def _relation_attempt(self,kind:str,normalized:str,event:dict[str,Any],bound:set[str])->tuple[dict[str,Any],str]:
        """One atomic subparser call, validated locally before it is accepted."""
        parsed,raw=self._call(kind,normalized,event);node=self._relation_node(kind,parsed);node,lifted=_apply_name_negation(node);validate_expr(node,{},bound)
        if lifted:event["negation_lifted_from_name"].append(lifted)
        return node,raw
    def _atomic_node(self,normalized:str,event:dict[str,Any],bound:set[str],fallback:tuple[str,str,Any]|None=None)->dict[str,Any]:
        """Relation/atomic parse. Reference NL2Logic falls back here whenever a decomposition node is locally invalid."""
        if fallback is not None:event["fallback_from"],event["fallback_reason"]=fallback[0],fallback[1]
        relation,relation_raw=self._call("relation_selector",normalized,event);kind=_choice(relation,"relation",{"proposition","adjective","intransitive","transitive","ditransitive","equality"});event["chosen_subparser"]=f"atomic.{kind}"
        detail={"relation_selector":relation_raw,"atomic_parser":None}
        if fallback is not None:detail["fallback_source_raw"]=fallback[2]
        try:
            node,detail["atomic_parser"]=self._relation_attempt(kind,normalized,event,bound);event["subparser_raw_output"]=detail;return node
        except BridgeError as exc:chosen_error=exc
        attempts=[]
        for candidate in RELATION_FALLBACK_ORDER:
            if candidate==kind:continue
            try:node,detail["atomic_parser"]=self._relation_attempt(candidate,normalized,event,bound)
            except BridgeError as exc:attempts.append({"relation":candidate,"error":f"{exc.status}: {exc.detail}"});continue
            event["relation_fallback_from"]=kind;event["relation_fallback_reason"]=f"{chosen_error.status}: {chosen_error.detail}";event["chosen_subparser"]=f"atomic.{candidate}"
            detail["relation_fallback"]={"from":kind,"reason":event["relation_fallback_reason"],"to":candidate,"rejected":attempts};event["subparser_raw_output"]=detail
            return node
        detail["relation_fallback"]={"from":kind,"reason":f"{chosen_error.status}: {chosen_error.detail}","to":None,"rejected":attempts};event["subparser_raw_output"]=detail
        raise chosen_error
    def parse_sentence(self,clause:str,example_id:str,role:str,sentence_id:str,*,depth:int=0,path:str="0",parent_step_id:str|None=None,bound:set[str]|None=None,ancestors:tuple[str,...]=()):
        normalized=" ".join(clause.strip().split());canonical=canonical_clause(normalized);step_id=f"{example_id}:{sentence_id}:{path}";bound=set() if bound is None else set(bound);ast_key=(canonical,tuple(sorted(bound)));self.unique_clauses.add(canonical);self.usage["unique_normalized_clauses"]=len(self.unique_clauses)
        event={"dataset":self.dataset,"condition":self.condition,"run_id":self.run_id,"example_id":example_id,"sentence_id":sentence_id,"role":role,"step_id":step_id,"parent_step_id":parent_step_id,"recursive_path":path,"depth":depth,"input_clause":normalized,"canonical_clause":canonical,"selector_raw_output":None,"selector_decision":None,"selector_validation_status":"pending","chosen_subparser":None,"subparser_raw_output":None,"failed_raw_output":None,"child_clauses":[],"returned_ast_node":None,"validation_status":"pending","validation_errors":[],"status":"started","failure_reason":None,"fallback_from":None,"fallback_reason":None,"relation_fallback_from":None,"relation_fallback_reason":None,"negation_lifted_from_name":[],"latency_ms":0.0,"token_count":0,"model_calls":[],"ast_cache_hit":False}
        node=None
        try:
            if depth>self.max_depth:raise BridgeError("parser_depth_exceeded",f"maximum recursion depth {self.max_depth} exceeded")
            if canonical in ancestors or ast_key in self.inflight:raise BridgeError("parser_cycle",f"recursive clause cycle: {normalized}")
            self.usage["ast_cache_lookups"]+=1
            if ast_key in self.ast_cache:
                self.usage["ast_cache_hits"]+=1;event["ast_cache_hit"]=True;node=copy.deepcopy(self.ast_cache[ast_key]);event["returned_ast_node"]=node;event["selector_validation_status"]="cached";event["validation_status"]="valid";event["status"]="memoized";return node
            self.inflight.add(ast_key)
            selector,event["selector_raw_output"]=self._call("parser_selector",normalized,event);decision=_choice(selector,"parser",{"atomic","quantified","binary","unary"});event["selector_decision"]=decision;event["selector_validation_status"]="valid";event["chosen_subparser"]=decision
            ancestry=ancestors+(canonical,)
            def child(text:str,index:int,child_bound:set[str]|None=None):
                clean=" ".join(text.strip().split());child_path=f"{path}.{index}";child_id=f"{example_id}:{sentence_id}:{child_path}";event["child_clauses"].append({"index":index,"clause":clean,"step_id":child_id})
                if canonical_clause(clean)==canonical:raise BridgeError("parser_cycle",f"unchanged child clause: {clean}")
                return self.parse_sentence(clean,example_id,role,sentence_id,depth=depth+1,path=child_path,parent_step_id=step_id,bound=bound if child_bound is None else child_bound,ancestors=ancestry)
            if decision=="atomic":
                node=self._atomic_node(normalized,event,bound)
            elif decision=="quantified":
                parsed,raw=self._call("quantified",normalized,event);event["subparser_raw_output"]=raw
                quantifier=_soft_choice(parsed,"quantifier",{"forall","exists"});scope=_soft_text(parsed,"child_clause");variable=_variable_symbol(_soft_text(parsed,"variable") or "x")
                reason=None if quantifier else "quantified parser returned no valid quantifier"
                if reason is None and not scope:reason="quantified parser returned an empty scope"
                if reason is None and canonical_clause(scope)==canonical:reason="quantified parser returned an unchanged scope"
                if reason is None and not VARIABLE.fullmatch(variable):reason=f"quantified parser returned an invalid variable: {variable!r}"
                if reason is not None:node=self._atomic_node(normalized,event,bound,("quantified",reason,raw))
                else:node={"type":quantifier,"variables":[variable],"body":child(scope,0,bound|{variable})}
            elif decision=="binary":
                parsed,raw=self._call("binary",normalized,event);event["subparser_raw_output"]=raw
                operator=_soft_choice(parsed,"operator",{"and","or","implies","iff"});left_text=_soft_text(parsed,"left_clause");right_text=_soft_text(parsed,"right_clause")
                reason=None if operator else "binary parser returned no valid operator"
                if reason is None and not left_text:reason="binary parser returned an empty left operand"
                if reason is None and not right_text:reason="binary parser returned an empty right operand"
                if reason is None and canonical_clause(left_text)==canonical:reason="binary parser returned a self-referential left operand"
                if reason is None and canonical_clause(right_text)==canonical:reason="binary parser returned a self-referential right operand"
                if reason is None:
                    dropped=_uncovered_content(normalized,[left_text,right_text])
                    if dropped:reason="binary parser dropped clause content: "+", ".join(dropped[:6])
                if reason is not None:node=self._atomic_node(normalized,event,bound,("binary",reason,raw))
                else:
                    self._prefetch_siblings([left_text,right_text],event);left=child(left_text,0);right=child(right_text,1)
                    node={"type":operator,"args":[left,right]} if operator in {"and","or"} else {"type":operator,"antecedent":left,"consequent":right} if operator=="implies" else {"type":"iff","left":left,"right":right}
            else:
                parsed,raw=self._call("unary",normalized,event);event["subparser_raw_output"]=raw
                operator=_soft_choice(parsed,"operator",{"not"});operand=_soft_text(parsed,"child_clause");lowered=normalized.lower();operand_lowered=operand.lower()
                reason=None if operator else "unary parser returned no valid operator"
                if reason is None:reason=next((f"unary parser invented negation marker {marker!r}" for marker in NEGATION_MARKERS if marker not in lowered and marker in operand_lowered),None)
                if reason is None and not operand:reason="unary parser returned an empty operand"
                if reason is None and canonical_clause(operand)==canonical:reason="unary parser returned a self-referential operand"
                if reason is not None:node=self._atomic_node(normalized,event,bound,("unary",reason,raw))
                else:node={"type":"not","body":child(operand,0)}
            validate_expr(node,{},bound);self.ast_cache[ast_key]=copy.deepcopy(node);event["returned_ast_node"]=node;event["validation_status"]="valid";event["status"]="success";return node
        except BridgeError as exc:
            if event["selector_validation_status"]=="pending":event["selector_validation_status"]="invalid"
            event["validation_status"]="invalid";event["validation_errors"].append(exc.detail);event["failure_reason"]=f"{exc.status}: {exc.detail}";event["status"]="partial" if event["child_clauses"] or node is not None else "failed";raise
        except Exception as exc:
            event["validation_status"]="invalid";event["validation_errors"].append(str(exc));event["failure_reason"]=f"parser_exception: {type(exc).__name__}: {exc}";event["status"]="partial" if event["child_clauses"] else "failed";raise BridgeError("parser_exception",str(exc)) from exc
        finally:self.inflight.discard(ast_key);self.trace_sink(event)
    def parse_theory(self,row:dict[str,Any])->dict[str,Any]:
        if isinstance(row.get("premises"),list):premise_clauses=[str(x).strip() for x in row["premises"] if str(x).strip()]
        else:premise_clauses=[x.strip() for x in re.split(r"(?<=[.!?])\s+",row["context"].strip()) if x.strip()]
        self._prefetch_siblings(premise_clauses+[row["question"]]);premises=[self.parse_sentence(text,row["example_id"],"premise",f"p{i}") for i,text in enumerate(premise_clauses)];hypothesis=self.parse_sentence(row["question"],row["example_id"],"hypothesis","h")
        theory={"premises":premises,"hypothesis":hypothesis};validate_theory(theory);return theory
