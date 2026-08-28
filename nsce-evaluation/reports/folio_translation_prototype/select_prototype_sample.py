import hashlib
import json
import re
from collections import Counter

VAL_PATH = r"C:\Users\Preetha\AppData\Local\Temp\claude\E--SRM-Projects-and-Internships-Univeristy-of-Hull\48077d5e-a72c-4756-8afe-9d239f44a1e3\scratchpad\folio_val.jsonl"

FEASIBILITY_SAMPLE_IDX = {12, 13, 14, 24, 25, 26, 58, 59, 69, 70, 135, 136, 137, 138, 139, 143, 144, 154}

rows = [json.loads(l) for l in open(VAL_PATH, encoding="utf-8")]
for i, r in enumerate(rows):
    r["_idx"] = i
    r["_h"] = hashlib.sha256(json.dumps(r["premises"], sort_keys=True).encode()).hexdigest()

pool = [r for r in rows if r["_idx"] not in FEASIBILITY_SAMPLE_IDX]
print(len(rows), "total /", len(pool), "after excluding the 18 feasibility-audit items")
print(Counter(r["label"] for r in pool))

by_label = {"True": [], "False": [], "Uncertain": []}
for r in pool:
    by_label[r["label"]].append(r)

sample = []
for lab in ["True", "False", "Uncertain"]:
    lst = sorted(by_label[lab], key=lambda r: r["_h"])
    sample.extend(lst[:10])
sample.sort(key=lambda r: r["_idx"])


def tags(premises_fol, concl_fol):
    all_fol = " ".join(premises_fol) + " " + concl_fol
    t = set()
    if "¬" in all_fol:
        t.add("negation")
    if "∃" in all_fol:
        t.add("existential")
    if "∀" in all_fol:
        t.add("universal")
    if "∨" in all_fol:
        t.add("disjunction")
    if "∧" in all_fol:
        t.add("conjunction")
    if "⊕" in all_fol:
        t.add("xor")
    if "↔" in all_fol or "≡" in all_fol:
        t.add("iff")
    if re.search(r"∀x\s*∀y|∃x\s*∃y|∀x∀y|∃x∃y", all_fol):
        t.add("multi_variable")
    if len(premises_fol) >= 5:
        t.add("multi_step")
    binary_arity = any(re.search(r"\w+\([^,()]+,\s*[^,()]+\)", p) for p in premises_fol)
    if binary_arity:
        t.add("binary_relation")
    return sorted(t)


coverage = Counter()
manifest = []
for r in sample:
    construct_tags = tags(r["premises-FOL"], r["conclusion-FOL"])
    for t in construct_tags:
        coverage[t] += 1
    manifest.append({
        "idx": r["_idx"],
        "label": r["label"],
        "premises": r["premises"],
        "conclusion": r["conclusion"],
        "premises_fol_gold": r["premises-FOL"],
        "conclusion_fol_gold": r["conclusion-FOL"],
        "constructs": construct_tags,
    })

print("\nSample size:", len(sample))
print("Label balance:", Counter(r["label"] for r in sample))
print("\nConstruct coverage across the 30-item sample:")
for k, v in sorted(coverage.items()):
    print(f"  {k}: {v}/30")

checksum = hashlib.sha256(json.dumps([m["idx"] for m in manifest]).encode()).hexdigest()
print("\nSample IDs:", [m["idx"] for m in manifest])
print("Manifest checksum (sha256 of sorted idx list):", checksum)
overlap = set(m["idx"] for m in manifest) & FEASIBILITY_SAMPLE_IDX
print("Overlap with feasibility-audit sample (must be empty):", overlap)

with open(r"C:\Users\Preetha\AppData\Local\Temp\claude\E--SRM-Projects-and-Internships-Univeristy-of-Hull\48077d5e-a72c-4756-8afe-9d239f44a1e3\scratchpad\prototype_manifest_v2.json", "w", encoding="utf-8") as f:
    json.dump({
        "source": "Yale-LILY/FOLIO data/v0.0/folio-validation.jsonl (MIT license) -- chosen over the train split "
                   "because only this split carries gold conclusion-FOL, needed for the verdict-preservation-vs-oracle "
                   "metric (item 9 of the prototype deliverable); the train split has premises-FOL only, no conclusion-FOL.",
        "note_on_split_semantics": "This is FOLIO's public 'validation' file, used here purely as a frozen development/"
                                    "prototype sample, per the phase's own instruction to use train/dev examples and "
                                    "not the final evaluation split. FOLIO's official held-out test annotations are not "
                                    "public. A separate, disjoint slice of this same validation file (or a fresh split) "
                                    "must be frozen as the actual held-out set before any headline end-to-end number is "
                                    "reported -- this 30-item set is prototype-only, exactly like the existing "
                                    "72-example ProofWriter dev pilot vs. its 360-example frozen test.",
        "excluded_indices_reason": "The 18 items already hand-audited in FOLIO_REPRESENTATION_FEASIBILITY.md are "
                                    "excluded from this pool so the two artifacts stay independent.",
        "excluded_indices": sorted(FEASIBILITY_SAMPLE_IDX),
        "selection_method": "sha256(json(premises)) ascending over the remaining pool, top 10 per gold label "
                             "(True/False/Uncertain), deterministic hash sort, no manual cherry-picking.",
        "sample_size": len(sample),
        "label_balance": dict(Counter(r["label"] for r in sample)),
        "construct_coverage": dict(coverage),
        "checksum_sha256_of_idx_list": checksum,
        "items": manifest,
    }, f, ensure_ascii=False, indent=2)
print("\nWrote prototype_manifest_v2.json")
