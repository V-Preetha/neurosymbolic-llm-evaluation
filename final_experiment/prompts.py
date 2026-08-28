COMPONENT_SCHEMAS={
    "parser_selector":'{"parser":"atomic|quantified|binary|unary"}',
    "relation_selector":'{"relation":"proposition|adjective|intransitive|transitive|ditransitive|equality"}',
    "proposition":'{"predicate":"base_proposition"}',
    "equality":'{"left":"entity","right":"entity"}',
    "adjective":'{"predicate":"base_property","subject":"entity"}',
    "intransitive":'{"predicate":"base_verb","subject":"entity"}',
    "transitive":'{"predicate":"base_verb","subject":"entity","object":"entity"}',
    "ditransitive":'{"predicate":"base_verb","subject":"entity","indirect_object":"entity","direct_object":"entity"}',
    "quantified":'{"quantifier":"forall|exists","variable":"x|y|z","child_clause":"sentence without outer quantifier"}',
    "binary":'{"operator":"and|or|implies|iff","left_clause":"standalone clause","right_clause":"standalone clause"}',
    "unary":'{"operator":"not","child_clause":"sentence without outer negation"}',
}
# Prototype variant. The six atomic schemas above put *filler noun values* in the value slots
# ("base_property", "entity"), which Qwen3-1.7B copies verbatim into its answer. The quantified/
# binary/unary schemas put *field descriptions* in the same slots and are never echoed, so this
# variant rewrites only the six atomic schemas in that already-working style. Keys, component
# names, guidance text and the surrounding template are unchanged.
COMPONENT_SCHEMAS_DESCRIPTIVE=COMPONENT_SCHEMAS|{
    "proposition":'{"predicate":"the whole clause as one lowercase proposition name"}',
    "equality":'{"left":"first entity named in the clause","right":"second entity named in the clause"}',
    "adjective":'{"predicate":"the property word in the clause","subject":"the entity that property describes"}',
    "intransitive":'{"predicate":"the verb in the clause","subject":"the entity performing the verb"}',
    "transitive":'{"predicate":"the verb in the clause","subject":"the entity performing the verb","object":"the entity the verb acts on"}',
    "ditransitive":'{"predicate":"the verb in the clause","subject":"the entity performing the verb","indirect_object":"the recipient entity","direct_object":"the entity being given"}',
}
# Second prototype iteration. On the frozen 20 the "descriptive" variant cleared the echo but left
# two defects, each attributable to one component: every over-long predicate name came from
# `proposition` (chosen for 60% of atoms, including clauses that clearly name an entity), and every
# `is_`/`is_not_` prefixed name came from `adjective`. This variant sharpens exactly those two
# slots plus the shared atomic guidance. Keys, component names and the template are unchanged.
COMPONENT_SCHEMAS_NORMALIZED=COMPONENT_SCHEMAS_DESCRIPTIVE|{
    "proposition":'{"predicate":"one short lowercase name for the clause, used only when the clause names no entity"}',
    "adjective":'{"predicate":"the single property word, without any copula or negation","subject":"the entity that property describes"}',
    "intransitive":'{"predicate":"the single verb word, without any copula or negation","subject":"the entity performing the verb"}',
    "transitive":'{"predicate":"the single verb word, without any copula or negation","subject":"the entity performing the verb","object":"the entity the verb acts on"}',
}

GUIDANCE={
    "parser_selector":"Classify only the outermost structure. Atomic has no outer quantifier, connective, or negation; binary covers and/or/if/iff; unary is outer negation.",
    "relation_selector":"Classify this atomic predicate by valency; use equality only for identity between two terms.",
    "quantified":"Remove only the outermost quantifier, replace its entity consistently by one variable, and return a complete child clause.",
    "binary":"Extract only the outermost connective and two complete child clauses. Resolve pronouns without adding facts.",
    "unary":"Remove only the outermost negation and return a complete child clause.",
}
DEFAULT_GUIDANCE="Extract a normalized lowercase predicate and its ordered arguments without adding facts."
GUIDANCE_NORMALIZED=GUIDANCE|{
    "relation_selector":"Classify this atomic predicate by valency; use equality only for identity between two terms. Use proposition only when the clause names no entity at all; a clause with a named entity or a variable subject is adjective, intransitive, transitive, or ditransitive.",
}
DEFAULT_GUIDANCE_NORMALIZED="Extract a normalized lowercase predicate and its ordered arguments without adding facts. The predicate is one bare word taken from the clause: drop copulas (is, are, was), articles and adverbs, drop any negation, and never use a whole sentence as the predicate."

SCHEMA_VARIANTS={"v1":COMPONENT_SCHEMAS,"descriptive":COMPONENT_SCHEMAS_DESCRIPTIVE,"normalized":COMPONENT_SCHEMAS_NORMALIZED}
GUIDANCE_VARIANTS={
    "v1":(GUIDANCE,DEFAULT_GUIDANCE),
    "descriptive":(GUIDANCE,DEFAULT_GUIDANCE),
    "normalized":(GUIDANCE_NORMALIZED,DEFAULT_GUIDANCE_NORMALIZED),
}
# The frozen bridge keeps "v1" until a prompt change is explicitly accepted.
ACTIVE_SCHEMA_VARIANT="v1"
def component_prompt(component,clause,variant=None):
    active=variant or ACTIVE_SCHEMA_VARIANT
    table,default=GUIDANCE_VARIANTS[active];guidance=table.get(component,default)
    schemas=SCHEMA_VARIANTS[active]
    return f"NL2LOGIC COMPONENT: {component}\n{guidance}\nReturn JSON only matching {schemas[component]}.\nCLAUSE: {clause}"
def direct_prompt(r): return "Classify using only the premises under open-world reasoning. Answer exactly True, False, or Unknown.\n\nPremises:\n"+r["context"]+"\n\nConclusion:\n"+r["question"]
def direct_training_text(r): return direct_prompt(r)+"\n\nAnswer:\n"+r["label"]
# Answer-only direct prompt. Used by the direct-classification paths (C1/C3) only. The original
# `direct_prompt` above is kept verbatim because `direct_training_text` derives the LoRA training
# text from it; changing it would change the training data.
DIRECT_ANSWER_ONLY_INSTRUCTION=(
    "Determine whether the conclusion follows from the premises under open-world reasoning.\n\n"
    "Output exactly one word:\nTrue\nFalse\nUnknown\n\n"
    "Do not explain your answer.\nDo not output anything else."
)
def direct_prompt_answer_only(r): return DIRECT_ANSWER_ONLY_INSTRUCTION+"\n\nPremises:\n"+r["context"]+"\n\nConclusion:\n"+r["question"]
