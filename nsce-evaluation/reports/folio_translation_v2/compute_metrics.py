"""Metrics for a frozen-30 Translation-v2 run. Mirrors
reports/folio_translation_prototype/compute_metrics.py's metric definitions
(A/B/C/D) exactly, so V1 and V2 numbers are directly comparable, and adds
the v2-specific metrics the phase spec asks for (component metrics, error
taxonomy, pre/post-repair).

Two categories of metric here:
  - Automated (schema validity, compilability, oracle/label agreement,
    structural error taxonomy, pre/post-repair, latency): computed exactly,
    no manual step.
  - Component-metric PROXIES (quantifier accuracy, negation preservation,
    predicate/entity "consistency"): approximated automatically by counting
    quantifier/negation symbols in gold FOL strings vs. generated node
    counts, and by comparing raw-name vs. canonical-name counts. These are
    NOT gold-validated semantic-equivalence checks -- the phase spec itself
    says gold FOL makes some of these hard to automate and calls for a
    manual audit of the frozen 30 (Section 16). Proxies here are reported
    as such, not presented as ground truth.

Usage: python reports/folio_translation_v2/compute_metrics.py --variant C
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
PROTOTYPE_DIR = HERE.parent / "folio_translation_prototype"

REASON_TO_CATEGORY = {
    "invalid_json": "SCHEMA_ERROR",
    "missing_field": "SCHEMA_ERROR",
    "malformed_node": "SCHEMA_ERROR",
    "unknown_operator": "SCHEMA_ERROR",
    "dangling_reference": "SCHEMA_ERROR",
    "duplicate_id": "SCHEMA_ERROR",
    "cyclic_reference": "SCHEMA_ERROR",
    "unbound_variable": "QUANTIFIER_SCOPE_ERROR",
    "variable_shadowing": "QUANTIFIER_SCOPE_ERROR",
    "entity_shaped_variable": "QUANTIFIER_ERROR",
    "entity_variable_collision": "ENTITY_ERROR",
    "arity_conflict": "PREDICATE_ARITY_ERROR",
    "unknown_predicate_reference": "PREDICATE_ERROR",
    "repair_failed": "REPAIR_FAILED",
    "compilation_error": "COMPILATION_ERROR",
}


def count_quantifier_nodes(node_dict, op):
    if node_dict is None:
        return 0
    if node_dict.get("type") == op:
        return 1 + sum(count_quantifier_nodes(c, op) for c in _children(node_dict))
    return sum(count_quantifier_nodes(c, op) for c in _children(node_dict))


def count_negations(node_dict):
    if node_dict is None:
        return 0
    n = 1 if node_dict.get("type") == "not" else 0
    return n + sum(count_negations(c) for c in _children(node_dict))


def _children(node_dict):
    t = node_dict.get("type")
    if t == "not":
        return [node_dict["body"]]
    if t in ("and", "or"):
        return node_dict["args"]
    if t == "implies":
        return [node_dict["antecedent"], node_dict["consequent"]]
    if t == "iff":
        return [node_dict["left"], node_dict["right"]]
    if t in ("forall", "exists"):
        return [node_dict["body"]]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    runs = [json.loads(l) for l in (HERE / f"frozen30_variant_{args.variant}.jsonl").open(encoding="utf-8")]
    gold_oracle = {o["idx"]: o for o in json.loads((PROTOTYPE_DIR / "gold_oracle.json").read_text(encoding="utf-8"))["items"]}
    manifest = {it["idx"]: it for it in json.loads((PROTOTYPE_DIR / "sample_manifest.json").read_text(encoding="utf-8"))["items"]}

    n = len(runs)

    total_sentences = sum(len(r["premise_results"]) + (1 if r["hypothesis_result"] else 0) for r in runs)
    valid_sentences = sum(
        sum(1 for p in r["premise_results"] if p["valid"]) + (1 if r["hypothesis_result"] and r["hypothesis_result"]["valid"] else 0)
        for r in runs
    )
    items_all_valid = sum(1 for r in runs if r["all_valid"])
    items_compiled = sum(1 for r in runs if r["compiled_and_solved"])

    verdict_preservation_n = 0
    verdict_preservation_match = 0
    for r in runs:
        oracle = gold_oracle.get(r["idx"])
        if r["generated_label"] is not None and oracle and oracle["oracle_label"] is not None:
            verdict_preservation_n += 1
            if r["generated_label"] == oracle["oracle_label"]:
                verdict_preservation_match += 1

    official_n = sum(1 for r in runs if r["generated_label"] is not None)
    official_correct = sum(1 for r in runs if r["generated_label"] == r["gold_label"])
    official_correct_among_generated = sum(1 for r in runs if r["generated_label"] is not None and r["generated_label"] == r["gold_label"])

    error_tags = Counter()
    per_item_errors = defaultdict(list)
    for r in runs:
        all_sentence_results = r["premise_results"] + ([r["hypothesis_result"]] if r["hypothesis_result"] else [])
        for s in all_sentence_results:
            if not s["valid"] and s["reason"]:
                tag = REASON_TO_CATEGORY.get(s["reason"], "OTHER")
                error_tags[tag] += 1
                per_item_errors[r["idx"]].append({"sentence": s["sentence"], "reason": s["reason"], "tag": tag})
        if r["theory_level_reason"]:
            tag = REASON_TO_CATEGORY.get(r["theory_level_reason"], "OTHER")
            error_tags[tag] += 1
            per_item_errors[r["idx"]].append({"sentence": "(theory-level)", "reason": r["theory_level_reason"], "tag": tag})

    # Pre/post-repair (only meaningful for variant D; harmless no-op counts for others)
    repair_attempted = 0
    repair_fixed = 0
    pre_repair_valid_sentences = 0
    for r in runs:
        all_sentence_results = r["premise_results"] + ([r["hypothesis_result"]] if r["hypothesis_result"] else [])
        for s in all_sentence_results:
            if s["pre_repair_valid"]:
                pre_repair_valid_sentences += 1
            if s["repaired"]:
                repair_attempted += 1
                if s["valid"]:
                    repair_fixed += 1

    by_construct = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in runs:
        for c in r["gold_constructs"]:
            by_construct[c]["n"] += 1
            if r["generated_label"] == r["gold_label"]:
                by_construct[c]["correct"] += 1

    # Component-metric proxies (see module docstring)
    quantifier_matches = 0
    quantifier_comparable = 0
    negation_matches = 0
    negation_comparable = 0
    for r in runs:
        item = manifest.get(r["idx"])
        if not item or not r["premise_dicts"]:
            continue
        gold_fol_list = item["premises_fol_gold"] + [item["conclusion_fol_gold"]]
        generated_list = list(r["premise_dicts"]) + [r["hypothesis_dict"]]
        for gold_fol, generated in zip(gold_fol_list, generated_list):
            gold_q = gold_fol.count("∀") + gold_fol.count("∃")
            gen_q = count_quantifier_nodes(generated, "forall") + count_quantifier_nodes(generated, "exists")
            quantifier_comparable += 1
            if gold_q == gen_q:
                quantifier_matches += 1
            gold_neg = "¬" in gold_fol
            gen_neg = count_negations(generated) > 0
            negation_comparable += 1
            if gold_neg == gen_neg:
                negation_matches += 1

    predicate_compression = []
    entity_compression = []
    for r in runs:
        pt = r.get("predicate_table") or {}
        raw_predicate_mentions = set()
        for p in r["premise_results"] + ([r["hypothesis_result"]] if r["hypothesis_result"] else []):
            raw = re.findall(r'"predicate":\s*"([^"]+)"', p["initial_raw"] or "")
            raw_predicate_mentions.update(raw)
        if raw_predicate_mentions:
            predicate_compression.append({"idx": r["idx"], "raw_distinct": len(raw_predicate_mentions), "canonical_distinct": len(pt)})
        et = r.get("entity_table") or {}
        if et:
            entity_compression.append({
                "idx": r["idx"], "canonical_entities": len(et),
                "total_surface_forms": sum(len(v["surface_forms"]) for v in et.values()),
            })

    metrics = {
        "variant": args.variant,
        "n_items": n,
        "A_schema_valid_rate_per_sentence": valid_sentences / total_sentences if total_sentences else None,
        "A_schema_valid_rate_per_item_all_sentences": items_all_valid / n,
        "B_compilable_rate_per_item": items_compiled / n,
        "C_verdict_preservation_vs_gold_fol_oracle": {
            "n_comparable": verdict_preservation_n,
            "matches": verdict_preservation_match,
            "rate": (verdict_preservation_match / verdict_preservation_n) if verdict_preservation_n else None,
        },
        "D_official_label_accuracy": {
            "over_all_30_items": official_correct / n,
            "over_items_that_produced_a_verdict": (official_correct_among_generated / official_n) if official_n else None,
            "n_items_that_produced_a_verdict": official_n,
        },
        "structural_error_taxonomy_counts": dict(error_tags),
        "repair": {
            "sentences_repair_attempted": repair_attempted,
            "sentences_repair_fixed": repair_fixed,
            "repair_success_rate": (repair_fixed / repair_attempted) if repair_attempted else None,
            "pre_repair_valid_sentence_rate": pre_repair_valid_sentences / total_sentences if total_sentences else None,
            "post_repair_valid_sentence_rate": valid_sentences / total_sentences if total_sentences else None,
        },
        "component_metric_proxies": {
            "note": "automated proxies, NOT gold-validated -- see module docstring / phase spec Section 16",
            "quantifier_count_match_rate": (quantifier_matches / quantifier_comparable) if quantifier_comparable else None,
            "quantifier_comparable_n": quantifier_comparable,
            "negation_presence_match_rate": (negation_matches / negation_comparable) if negation_comparable else None,
            "negation_comparable_n": negation_comparable,
        },
        "predicate_compression_by_item": predicate_compression,
        "entity_compression_by_item": entity_compression,
        "accuracy_by_construct": {k: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"]} for k, v in by_construct.items()},
        "latency": {
            "mean_item_elapsed_s": sum(r["elapsed_s"] for r in runs) / n if n else None,
            "total_elapsed_s": sum(r["elapsed_s"] for r in runs),
        },
    }

    (HERE / f"metrics_variant_{args.variant}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (HERE / f"error_analysis_variant_{args.variant}.json").write_text(
        json.dumps({"structural_errors_by_item": dict(per_item_errors)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
