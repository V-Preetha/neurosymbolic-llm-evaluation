"""Select Translation-v2's development subset: disjoint from both the 18
FOLIO_REPRESENTATION_FEASIBILITY.md audit items and the frozen 30-item
Translation-v1 evaluation sample (reports/folio_translation_prototype/
sample_manifest.json). Used ONLY for debugging the v2 pipeline (prompt
clarity, schema defects, deterministic-code bugs) -- never for a reported
metric. v2 is evaluated exactly once, after freezing, on the same frozen 30
used for v1 (see run_v2_frozen30.py), for a clean paired comparison.

Same selection method as select_prototype_sample.py: deterministic
sha256(json(premises)) ascending sort, top N per gold label, no manual
cherry-picking. FOLIO's public validation split (Yale-LILY/FOLIO, MIT
license), used purely as a source of dev/prototype examples per the phase
instruction to keep dev and frozen-eval data disjoint.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
VAL_PATH = r"C:\Users\Preetha\AppData\Local\Temp\folio_val.jsonl"
DEV_SAMPLE_SIZE_PER_LABEL = 6

manifest = json.loads((HERE.parent / "folio_translation_prototype" / "sample_manifest.json").read_text(encoding="utf-8"))
frozen30_idx = {item["idx"] for item in manifest["items"]}
feasibility18_idx = set(manifest["excluded_indices"])
excluded = frozen30_idx | feasibility18_idx

rows = [json.loads(l) for l in open(VAL_PATH, encoding="utf-8")]
for i, r in enumerate(rows):
    r["_idx"] = i
    r["_h"] = hashlib.sha256(json.dumps(r["premises"], sort_keys=True).encode()).hexdigest()

pool = [r for r in rows if r["_idx"] not in excluded]
print(len(rows), "total /", len(pool), "after excluding frozen-30 + feasibility-18")
print("Pool label balance:", Counter(r["label"] for r in pool))

by_label = {"True": [], "False": [], "Uncertain": []}
for r in pool:
    by_label[r["label"]].append(r)

sample = []
for lab in ["True", "False", "Uncertain"]:
    lst = sorted(by_label[lab], key=lambda r: r["_h"])
    sample.extend(lst[:DEV_SAMPLE_SIZE_PER_LABEL])
sample.sort(key=lambda r: r["_idx"])

items = [
    {"idx": r["_idx"], "label": r["label"], "premises": r["premises"], "conclusion": r["conclusion"]}
    for r in sample
]

checksum = hashlib.sha256(json.dumps([m["idx"] for m in items]).encode()).hexdigest()
overlap = set(m["idx"] for m in items) & excluded
print("\nDev sample size:", len(items))
print("Dev label balance:", Counter(m["label"] for m in items))
print("Overlap with frozen-30 + feasibility-18 (must be empty):", overlap)
assert not overlap

out = {
    "purpose": "Translation-v2 development/debugging subset ONLY -- never used for a reported metric. "
               "Disjoint from the frozen 30 (Translation-v1 evaluation set) and the 18 feasibility-audit items.",
    "source": VAL_PATH,
    "selection_method": "sha256(json(premises)) ascending over the pool (frozen-30 and feasibility-18 excluded), "
                         f"top {DEV_SAMPLE_SIZE_PER_LABEL} per gold label, deterministic hash sort, no cherry-picking.",
    "sample_size": len(items),
    "label_balance": dict(Counter(m["label"] for m in items)),
    "checksum_sha256_of_idx_list": checksum,
    "excluded_frozen30_idx": sorted(frozen30_idx),
    "excluded_feasibility18_idx": sorted(feasibility18_idx),
    "items": items,
}
(HERE / "dev_manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nWrote", HERE / "dev_manifest.json")
