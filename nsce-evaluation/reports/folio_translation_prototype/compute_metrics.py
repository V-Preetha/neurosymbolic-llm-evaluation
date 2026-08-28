"""Compute the quantitative metrics for the 30-item translation prototype.

Structural failures (schema/compilation/scope/arity) are auto-tagged from the
recorded error_reason. Semantic failures (wrong verdict despite valid,
compilable IR) are NOT auto-classified here -- those need the manual
translation audit (item 9E of the phase spec), done separately by reading
translations.jsonl against the raw premises and recorded in error_analysis.json.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
translations = [json.loads(l) for l in (HERE / "translations.jsonl").open(encoding="utf-8")]
gold_oracle = {o["idx"]: o for o in json.loads((HERE / "gold_oracle.json").read_text(encoding="utf-8"))["items"]}

n = len(translations)

# A. Schema validity rate -- per-sentence and per-item (all sentences in the theory valid)
total_sentences = sum(len(t["premise_attempts"]) + 1 for t in translations)
valid_sentences = sum(
    sum(1 for a in t["premise_attempts"] if a["schema_valid"]) + (1 if t["hypothesis_attempt"]["schema_valid"] else 0)
    for t in translations
)
items_all_schema_valid = sum(1 for t in translations if t["all_schema_valid"])

# B. Compilation validity rate (only meaningful for items that were all schema-valid)
items_compiled = sum(1 for t in translations if t["compiled_and_solved"])

# C. Verdict preservation: generated-logic verdict vs gold-FOL oracle verdict
#    (restricted to items where both produced a verdict)
verdict_preservation_n = 0
verdict_preservation_match = 0
for t in translations:
    oracle = gold_oracle.get(t["idx"])
    if t["generated_label"] is not None and oracle and oracle["oracle_label"] is not None:
        verdict_preservation_n += 1
        if t["generated_label"] == oracle["oracle_label"]:
            verdict_preservation_match += 1

# D. Official-label accuracy: generated-logic verdict vs official FOLIO label
official_n = sum(1 for t in translations if t["generated_label"] is not None)
official_correct = sum(1 for t in translations if t["generated_label"] == t["gold_label"])
official_correct_among_generated = sum(
    1 for t in translations if t["generated_label"] is not None and t["generated_label"] == t["gold_label"]
)

# Structural error taxonomy (auto-tagged from error_reason strings)
REASON_TO_CATEGORY = {
    "invalid_json": "SCHEMA_ERROR",
    "unknown_node_type": "SCHEMA_ERROR",
    "missing_field": "SCHEMA_ERROR",
    "malformed_node": "SCHEMA_ERROR",
    "unbound_variable": "QUANTIFIER_SCOPE_ERROR",
    "variable_shadowing": "QUANTIFIER_SCOPE_ERROR",
    "entity_shaped_variable": "QUANTIFIER_ERROR",
    "arity_conflict": "PREDICATE_ERROR",
    "compilation_error": "COMPILATION_ERROR",
}
error_tags = Counter()
per_item_errors = defaultdict(list)
for t in translations:
    for a in t["premise_attempts"] + [t["hypothesis_attempt"]]:
        if not a["schema_valid"] and a["error_reason"]:
            tag = REASON_TO_CATEGORY.get(a["error_reason"], "OTHER")
            error_tags[tag] += 1
            per_item_errors[t["idx"]].append({"sentence": a["sentence"], "reason": a["error_reason"], "tag": tag})
    if t["all_schema_valid"] and t["solver_failure_reason"]:
        tag = REASON_TO_CATEGORY.get(t["solver_failure_reason"], "OTHER")
        error_tags[tag] += 1
        per_item_errors[t["idx"]].append({"sentence": "(theory-level)", "reason": t["solver_failure_reason"], "tag": tag})

# Accuracy by construct tag (gold_constructs from the sample manifest)
by_construct = defaultdict(lambda: {"n": 0, "correct": 0})
for t in translations:
    for c in t["gold_constructs"]:
        by_construct[c]["n"] += 1
        if t["generated_label"] == t["gold_label"]:
            by_construct[c]["correct"] += 1

metrics = {
    "n_items": n,
    "A_schema_valid_rate_per_sentence": valid_sentences / total_sentences,
    "A_schema_valid_rate_per_item_all_sentences": items_all_schema_valid / n,
    "B_compilable_rate_per_item": items_compiled / n,
    "C_verdict_preservation_vs_gold_fol_oracle": {
        "n_comparable": verdict_preservation_n,
        "matches": verdict_preservation_match,
        "rate": (verdict_preservation_match / verdict_preservation_n) if verdict_preservation_n else None,
    },
    "D_official_label_accuracy": {
        "over_all_30_items": official_correct / n,
        "over_items_that_produced_a_verdict": (
            (official_correct_among_generated / official_n) if official_n else None
        ),
        "n_items_that_produced_a_verdict": official_n,
    },
    "structural_error_taxonomy_counts": dict(error_tags),
    "accuracy_by_construct": {
        k: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"]} for k, v in by_construct.items()
    },
}

(HERE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(HERE / "error_analysis.json").write_text(
    json.dumps({"structural_errors_by_item": dict(per_item_errors)}, indent=2), encoding="utf-8"
)
print(json.dumps(metrics, indent=2))
