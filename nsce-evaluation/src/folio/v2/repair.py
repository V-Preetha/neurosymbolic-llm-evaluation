"""One constrained, sentence-local repair attempt. Step 10 of the phase spec.

Only fires for structural error classes that are (a) attributable to a
single sentence's own decomposition and (b) fixable without rewriting what
the sentence asserts -- a missing/misplaced quantifier, a broken id
reference, an unknown operator, malformed JSON shape. Cross-theory arity
conflicts are deliberately NOT repairable here (see validator.py's
docstring): fixing one would require guessing which of two sentences is
"wrong", which is exactly the kind of silent repair this module must not do.

Hard rules, enforced by the caller (pipeline.py) and this module together:
  - at most one repair attempt per sentence
  - the repair prompt shows the previous JSON + the specific error, and asks
    for the corrected JSON in the same schema -- never gold FOL, never the
    gold label
  - pre-repair and post-repair decompositions are both preserved by the
    caller for before/after metrics
  - if the repaired output still fails validation, that is a hard
    TRANSLATION_FAILURE (reason "repair_failed"), never a second attempt
"""

from __future__ import annotations

import json

from src.folio.v2.decomposer import DecompositionAttempt, _parse_generation


REPAIRABLE_REASONS = frozenset({
    "invalid_json",
    "missing_field",
    "malformed_node",
    "unknown_operator",
    "dangling_reference",
    "duplicate_id",
    "cyclic_reference",
    "unbound_variable",
    "variable_shadowing",
    "entity_shaped_variable",
    "entity_variable_collision",
})


REPAIR_PROMPT = """You previously decomposed one English sentence into structured JSON (entities / atoms / nodes / root, nodes referencing atoms or other nodes by id). That JSON has a specific structural error: a variable is used inside an atom but is not bound by any enclosing "forall"/"exists" node that actually contains that atom.

Sentence: "{sentence}"

Previous JSON:
{previous_json}

Error: {reason} -- {detail}

How to fix an unbound-variable error (the usual case): find the atom that uses the unbound variable, then make sure a forall/exists node whose "variables" list includes that variable actually sits ABOVE that atom in the tree reachable from "root". Two ways to do this -- pick whichever changes less:
  (a) If there is already a forall/exists node above that atom, just add the missing variable name to that node's existing "variables" list (e.g. "variables": ["x"] -> "variables": ["x", "y"]). This is usually all that's needed.
  (b) If no quantifier is above that atom at all, wrap the smallest node that contains it in a new forall/exists node, and update whatever pointed to the old node (a parent node's "args"/"antecedent"/"consequent"/"body", or "root" itself) to point to the new quantifier node instead.

CRITICAL: after your fix, "root" must still reach every atom that was reachable before -- do not drop, replace, or orphan any existing atom or node. Do not add, remove, or rename any predicate or atom. Do not change what the sentence asserts.

Worked example. Sentence: "No mass product design is timeless." Broken JSON (x used in atom A1, which sits in the "and" node N3, but the only forall is INSIDE N3's second branch, not above N3 itself, so A1's x is unbound):
{"atoms": [{"id": "A1", "predicate": "mass_product_design", "args": ["x"]}, {"id": "A2", "predicate": "timeless", "args": ["x"]}], "nodes": [{"id": "N1", "op": "not", "body": "A2"}, {"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}, {"id": "N3", "op": "and", "args": ["A1", "N2"]}], "root": "N3"}
Fixed JSON (move the forall to wrap the whole "and", using implies instead so A1 is the antecedent -- this is option (b): the quantifier now sits above the atom that needed it, and root still reaches both A1 and A2):
{"atoms": [{"id": "A1", "predicate": "mass_product_design", "args": ["x"]}, {"id": "A2", "predicate": "timeless", "args": ["x"]}], "nodes": [{"id": "N1", "op": "not", "body": "A2"}, {"id": "N2", "op": "implies", "antecedent": "A1", "consequent": "N1"}, {"id": "N3", "op": "forall", "variables": ["x"], "body": "N2"}], "root": "N3"}

Now output the complete corrected JSON object for the sentence above, in the exact same schema. Output ONLY the JSON object -- no explanation, no markdown fences.
JSON:"""


def is_repairable(reason: str | None) -> bool:
    return reason is not None and reason in REPAIRABLE_REASONS


def render_repair_prompt(sentence: str, previous_json: dict | None, reason: str, detail: str) -> str:
    previous_text = json.dumps(previous_json, ensure_ascii=False) if previous_json is not None else "(not valid JSON)"
    return (
        REPAIR_PROMPT.replace("{sentence}", sentence.strip())
        .replace("{previous_json}", previous_text)
        .replace("{reason}", reason)
        .replace("{detail}", detail)
    )


def attempt_repair(
    runner,
    sentence: str,
    previous_attempt: DecompositionAttempt,
    reason: str,
    detail: str,
    max_new_tokens: int = 500,
) -> DecompositionAttempt:
    """Issue exactly one repair generation call and parse it. Caller is
    responsible for enforcing the "at most once per sentence" rule and for
    logging pre-repair vs. post-repair state.
    """
    prompt = render_repair_prompt(sentence, previous_attempt.decomposition, reason, detail)
    generation = runner.generate_text(prompt, max_new_tokens=max_new_tokens)
    return _parse_generation(sentence, generation.raw_text, generation.latency_ms, repaired=True)
