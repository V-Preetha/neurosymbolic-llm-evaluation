"""Convert dataset gold annotations to the exact bridge JSON AST used at inference."""
import re

TOKEN_SYMBOLS={"∀":"FORALL","∃":"EXISTS","¬":"NOT","∧":"AND","∨":"OR","→":"IMPLIES","↔":"IFF","≡":"IFF","⟷":"IFF","⊕":"XOR","≠":"NEQ","=":"EQ","(":"LPAREN",")":"RPAREN",",":"COMMA"}

def _normalize_folio_formula(text):
    value=str(text).strip().replace("—>","→").replace("->","→")
    value=re.sub(r"\)\s*\.\s*$",")",value)
    value=re.sub(r"\)\s+[A-Za-z][A-Za-z ]{0,40}:\s*.*$",")",value)
    value=re.sub(r"([\w.\-]+)\(\(",r"\1(",value,flags=re.UNICODE)
    if value.startswith(("∀","∃")) and value.count(")")>value.count("("):
        value=re.sub(r"^([∀∃]\s*[^\s()]+\s*)\((.*?)\)\s*([→↔≡⟷])\s*(.*)\)$",r"\1(\2 \3 \4)",value)
    cleaned=[];balance=0
    for char in value:
        if char=="(":balance+=1;cleaned.append(char)
        elif char==")":
            if balance:balance-=1;cleaned.append(char)
        else:cleaned.append(char)
    cleaned.extend(")" for _ in range(balance))
    return "".join(cleaned)

def tokenize(text):
    text=_normalize_folio_formula(text);out=[];pos=0;delimiters=set(TOKEN_SYMBOLS)
    while pos<len(text):
        if text[pos].isspace():pos+=1;continue
        if text[pos] in delimiters:out.append((TOKEN_SYMBOLS[text[pos]],text[pos]));pos+=1;continue
        start=pos
        while pos<len(text) and not text[pos].isspace() and text[pos] not in delimiters:pos+=1
        if start==pos:raise ValueError(f"unsupported FOL token at {pos}: {text[pos:pos+20]!r}")
        raw=text[start:pos]
        if raw=="v" and (start==0 or text[start-1].isspace()) and (pos==len(text) or text[pos].isspace()):out.append(("OR",raw))
        else:out.append(("IDENT",raw))
    return out
class FolParser:
    def __init__(self,t):self.t=t;self.i=0
    def peek(self):return self.t[self.i] if self.i<len(self.t) else(None,None)
    def pop(self,k=None):
        x=self.t[self.i];self.i+=1
        if k and x[0]!=k:raise ValueError(f"expected {k}, got {x}")
        return x
    def parse(self):
        x=self.iff()
        while self.peek()[0]=="RPAREN":self.pop()
        if self.i!=len(self.t):raise ValueError(f"trailing tokens {self.t[self.i:]}")
        return x
    def iff(self):
        x=self.implies()
        while self.peek()[0] in("IFF","XOR"):
            op=self.pop()[0];y=self.implies();base={"type":"iff","left":x,"right":y};x=base if op=="IFF" else{"type":"not","body":base}
        return x
    def implies(self):
        x=self.or_()
        if self.peek()[0]=="IMPLIES":self.pop();return{"type":"implies","antecedent":x,"consequent":self.implies()}
        return x
    def or_(self):
        a=[self.and_()]
        while self.peek()[0]=="OR":self.pop();a.append(self.and_())
        return a[0] if len(a)==1 else{"type":"or","args":a}
    def and_(self):
        a=[self.not_()]
        while self.peek()[0]=="AND":self.pop();a.append(self.not_())
        return a[0] if len(a)==1 else{"type":"and","args":a}
    def not_(self):
        if self.peek()[0]=="NOT":self.pop();return{"type":"not","body":self.not_()}
        return self.atom()
    def atom(self):
        kind,text=self.peek()
        if kind in("FORALL","EXISTS"):
            self.pop()
            if self.peek()[0]=="IDENT":var=self.pop()[1].lower()
            else:
                var=next((value.lower() for token,value in self.t[self.i:] if token=="IDENT" and re.fullmatch(r"[tuvwxyz](?:[0-9]+)?",value,re.I)),None)
                if var is None:raise ValueError("quantifier is missing a recoverable variable")
            return{"type":"forall" if kind=="FORALL" else"exists","variables":[var],"body":self.not_()}
        if kind=="LPAREN":
            self.pop();x=self.iff()
            if self.peek()[0]=="RPAREN":self.pop()
            return x
        left=self._term()
        if self.peek()[0] in("EQ","NEQ"):
            operator=self.pop()[0];right=self._term();node={"type":"equals","left":left,"right":right}
            return{"type":"not","body":node} if operator=="NEQ" else node
        name=left
        if self.peek()[0]!="LPAREN":return{"type":"predicate","name":name,"args":[]}
        self.pop();args=[]
        if self.peek()[0]!="RPAREN":
            while True:
                if self.peek()[0] in("RPAREN","AND","OR","IMPLIES","IFF","XOR",None):break
                args.append(self._term())
                if self.peek()[0]!="COMMA":break
                self.pop()
                if self.peek()[0]=="RPAREN":break
        if self.peek()[0]=="RPAREN":self.pop()
        return{"type":"predicate","name":name,"args":args}
    def _term(self):
        while self.peek()[0]=="LPAREN":self.pop()
        parts=[]
        while self.peek()[0]=="IDENT":parts.append(self.pop()[1])
        if not parts:raise ValueError(f"expected term, got {self.peek()}")
        return"_".join(parts).lower()
def parse_fol(text):return FolParser(tokenize(text)).parse()

VERBS={"chase":"chase","chases":"chase","eat":"eat","eats":"eat","like":"like","likes":"like","need":"need","needs":"need","see":"see","sees":"see","visit":"visit","visits":"visit"};VP="|".join(sorted(VERBS,key=len,reverse=True))
def entity(x):
    v=x.strip().lower()
    if v in{"someone","something","they","it","them"}:return"x"
    if v.startswith("the "):v=v[4:]
    return re.sub(r"\W+","_",v).strip("_")
def pred(name,args,neg=False):
    x={"type":"predicate","name":name.lower(),"args":args};return{"type":"not","body":x} if neg else x
def adjectives(text,subject):
    out=[]
    for p in re.split(r",\s*|\s+and\s+",text.strip(),flags=re.I):
        p=p.strip();neg=p.lower().startswith("not ");name=p[4:] if neg else p
        if not re.fullmatch(r"[A-Za-z]+",name):raise ValueError(f"unsupported adjective: {p}")
        out.append(pred(name,[subject],neg))
    return out
def clause(text,default=None):
    text=text.strip().strip(".,")
    if default and re.fullmatch(r"(?:not\s+)?[A-Za-z]+",text,re.I):return adjectives(text,default),default
    m=re.fullmatch(r"(.+?)\s+(?:is|are)\s+(.+)",text,re.I)
    if m:s=entity(m.group(1));return adjectives(m.group(2),s),s
    m=re.fullmatch(rf"(.+?)\s+(?:do|does)\s+not\s+({VP})\s+(.+)",text,re.I)
    if m:s=entity(m.group(1));return[pred(VERBS[m.group(2).lower()],[s,entity(m.group(3))],True)],s
    m=re.fullmatch(rf"(.+?)\s+({VP})\s+(.+)",text,re.I)
    if m:s=entity(m.group(1));return[pred(VERBS[m.group(2).lower()],[s,entity(m.group(3))])],s
    raise ValueError(f"unsupported clause: {text}")
def pw_sentence(sentence):
    text=sentence.strip().rstrip(".")
    m=re.fullmatch(r"If\s+(.+?)\s+then\s+(.+)",text,re.I)
    if m:
        ants=[];last=None
        for part in re.split(r"\s+and\s+",m.group(1),flags=re.I):a,last=clause(part,last);ants+=a
        cons,_=clause(m.group(2),last);ant=ants[0] if len(ants)==1 else{"type":"and","args":ants};rules=[{"type":"implies","antecedent":ant,"consequent":c} for c in cons]
        def uses_x(node):
            if node["type"]=="predicate":return "x" in node["args"]
            if node["type"]=="not":return uses_x(node["body"])
            if node["type"] in{"and","or"}:return any(uses_x(x) for x in node["args"])
            if node["type"]=="implies":return uses_x(node["antecedent"]) or uses_x(node["consequent"])
            return False
        rules=[{"type":"forall","variables":["x"],"body":rule} if uses_x(rule) else rule for rule in rules]
        return rules
    m=re.fullmatch(r"(?:All\s+)?(.+?)\s+(?:people|things)\s+are\s+(.+)",text,re.I)
    if m:
        ants=adjectives(m.group(1),"x");cons=adjectives(m.group(2),"x");ant=ants[0] if len(ants)==1 else{"type":"and","args":ants};return[{"type":"forall","variables":["x"],"body":{"type":"implies","antecedent":ant,"consequent":c}} for c in cons]
    a,_=clause(text);return a
def proofwriter_target(row):
    sentences=[x.strip()+"." for x in re.split(r"\.\s*",row["context"]) if x.strip()];premises=[]
    for s in sentences:premises+=pw_sentence(s)
    q=pw_sentence(row["question"])
    if len(q)!=1:raise ValueError("ProofWriter hypothesis must be atomic")
    return{"premises":premises,"hypothesis":q[0]}
def _explicit_free_constants(node,bound=None):
    bound=set() if bound is None else set(bound);kind=node["type"]
    if kind=="predicate":return node|{"args":[f"free_{x}" if re.fullmatch(r"[tuvwxyz](?:[0-9]+)?",x) and x not in bound else x for x in node["args"]]}
    if kind=="equals":return node|{key:(f"free_{node[key]}" if re.fullmatch(r"[tuvwxyz](?:[0-9]+)?",node[key]) and node[key] not in bound else node[key]) for key in ("left","right")}
    if kind=="not":return node|{"body":_explicit_free_constants(node["body"],bound)}
    if kind in{"and","or"}:return node|{"args":[_explicit_free_constants(x,bound) for x in node["args"]]}
    if kind=="implies":return node|{"antecedent":_explicit_free_constants(node["antecedent"],bound),"consequent":_explicit_free_constants(node["consequent"],bound)}
    if kind=="iff":return node|{"left":_explicit_free_constants(node["left"],bound),"right":_explicit_free_constants(node["right"],bound)}
    names=set(node["variables"]);return node|{"body":_explicit_free_constants(node["body"],bound|names)}
def folio_target(row):return{"premises":[_explicit_free_constants(parse_fol(x)) for x in row["gold_premises_fol"] if str(x).strip()],"hypothesis":_explicit_free_constants(parse_fol(row["gold_conclusion_fol"]))}
