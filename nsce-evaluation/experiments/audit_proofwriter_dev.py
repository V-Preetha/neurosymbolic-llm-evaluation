"""Phase C1: audit the ProofWriter development split before building a fixed pilot.

Reads only structural/metadata fields (facts, rules, question, labels, depth,
context) -- never runs a model prediction, and only reads the test split's
structure (not predictions) to check for dev/test world overlap.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.proofwriter import iter_jsonl

DEV_PATH = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
TEST_PATH = dataset_package_path() / "normalized" / "proofwriter_test.jsonl"
OUT_MD = PROJECT_ROOT / "reports" / "revised_study" / "dataset_audit.md"
OUT_JSON = PROJECT_ROOT / "reports" / "revised_study" / "dataset_audit.json"


def theory_key(record: dict) -> str:
    """Group records sharing the same underlying facts+rules ('world')."""
    facts = tuple(sorted(record["input"]["facts"]))
    rules = tuple(sorted(record["input"]["rules"]))
    payload = json.dumps([facts, rules], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def structure_tags(record: dict) -> set[str]:
    tags: set[str] = set()
    facts = record["input"]["facts"]
    rules = record["input"]["rules"]
    question = record["input"]["question"]
    all_text = " ".join(facts + rules + [question]).lower()
    if "not " in all_text:
        tags.add("negation")
    if any(" and " in rule.lower() for rule in rules):
        tags.add("conjunction")
    if rules:
        tags.add("has_rules")
    if not rules:
        tags.add("facts_only")
    if any(rule.lower().startswith("if ") for rule in rules):
        tags.add("conditional_rule")
    if any("people" in rule.lower() or "things" in rule.lower() for rule in rules):
        tags.add("universal_quantifier")
    return tags


def audit_split(path: Path) -> dict:
    records = list(iter_jsonl(path))
    label_counts = Counter(record["ground_truth"]["label"] for record in records)
    depth_counts = Counter(record["metadata"]["reasoning_depth"] for record in records)
    label_depth_counts: Counter[tuple[str, int]] = Counter(
        (record["ground_truth"]["label"], record["metadata"]["reasoning_depth"])
        for record in records
    )

    structure_counts: Counter[str] = Counter()
    for record in records:
        for tag in structure_tags(record):
            structure_counts[tag] += 1

    exact_dup_key = Counter(
        json.dumps(
            [record["input"]["question"], sorted(record["input"]["facts"]), sorted(record["input"]["rules"])],
            sort_keys=True,
        )
        for record in records
    )
    exact_duplicates = sum(count - 1 for count in exact_dup_key.values() if count > 1)

    question_counts = Counter(record["input"]["question"] for record in records)
    duplicate_questions = {q: c for q, c in question_counts.items() if c > 1}

    world_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        world_groups[theory_key(record)].append(record["sample_id"])
    worlds_with_multiple_questions = {
        key: ids for key, ids in world_groups.items() if len(ids) > 1
    }

    fact_sets = [frozenset(record["input"]["facts"]) for record in records]
    pairwise_overlap_samples = []
    for i in range(min(len(fact_sets), 50)):
        for j in range(i + 1, min(len(fact_sets), 50)):
            a, b = fact_sets[i], fact_sets[j]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard > 0.5:
                pairwise_overlap_samples.append(
                    {"a": records[i]["sample_id"], "b": records[j]["sample_id"], "jaccard": round(jaccard, 3)}
                )

    return {
        "path": str(path),
        "count": len(records),
        "label_counts": dict(label_counts),
        "depth_counts": {str(k): v for k, v in sorted(depth_counts.items())},
        "label_depth_grid": {f"{label}|depth={depth}": count for (label, depth), count in sorted(label_depth_counts.items())},
        "structure_tag_counts": dict(structure_counts),
        "exact_duplicate_count": exact_duplicates,
        "duplicate_question_count": len(duplicate_questions),
        "duplicate_questions_sample": dict(list(duplicate_questions.items())[:10]),
        "distinct_worlds": len(world_groups),
        "worlds_with_multiple_questions": len(worlds_with_multiple_questions),
        "worlds_with_multiple_questions_sample": {
            k: v for k, v in list(worlds_with_multiple_questions.items())[:5]
        },
        "high_fact_overlap_pairs_first50 (jaccard>0.5)": pairwise_overlap_samples[:20],
        "records": records,
        "world_keys": {record["sample_id"]: theory_key(record) for record in records},
    }


def main() -> None:
    dev = audit_split(DEV_PATH)
    test = audit_split(TEST_PATH)

    dev_worlds = set(dev["world_keys"].values())
    test_worlds = set(test["world_keys"].values())
    shared_worlds = dev_worlds & test_worlds
    dev_questions = {r["input"]["question"] for r in dev["records"]}
    test_questions = {r["input"]["question"] for r in test["records"]}
    shared_questions = dev_questions & test_questions

    if shared_worlds:
        verdict = "WORLD-LEVEL LEAKAGE DETECTED -- INVESTIGATE BEFORE FREEZING TEST SPLIT"
    elif shared_questions:
        verdict = (
            "NO WORLD-LEVEL LEAKAGE. Query surface text is reused across distinct worlds "
            "(expected in ProofWriter's small closed entity/verb vocabulary) -- this is NOT "
            "sample-level leakage since the underlying facts/rules differ, but is recorded "
            "for transparency."
        )
    else:
        verdict = "NO WORLD-LEVEL OR QUESTION-LEVEL OVERLAP DETECTED BETWEEN DEV AND TEST"

    leakage = {
        "dev_distinct_worlds": len(dev_worlds),
        "test_distinct_worlds": len(test_worlds),
        "worlds_shared_between_dev_and_test": len(shared_worlds),
        "questions_shared_between_dev_and_test": len(shared_questions),
        "verdict": verdict,
    }

    for split in (dev, test):
        del split["records"]
        del split["world_keys"]

    payload = {"dev": dev, "test": test, "dev_test_leakage_check": leakage}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = []
    md.append("# ProofWriter Dataset Audit (Phase C1)\n")
    md.append(f"Source dev split: `{DEV_PATH}`\n")
    md.append(f"Source test split (structure-only, not touched for predictions): `{TEST_PATH}`\n")

    md.append("## Development split\n")
    md.append(f"- Total examples: **{dev['count']}**")
    md.append(f"- Label distribution: {dev['label_counts']}")
    md.append(f"- Reasoning-depth distribution: {dev['depth_counts']}")
    md.append(f"- Structure tag counts (a record may carry multiple tags): {dev['structure_tag_counts']}")
    md.append(f"- Exact duplicate (question+facts+rules) records: {dev['exact_duplicate_count']}")
    md.append(f"- Duplicate questions (same question text, different worlds): {dev['duplicate_question_count']}")
    md.append(f"- Distinct underlying worlds (unique facts+rules sets): {dev['distinct_worlds']}")
    md.append(
        f"- Worlds queried by more than one question: {dev['worlds_with_multiple_questions']} "
        f"(sample: {dev['worlds_with_multiple_questions_sample']})"
    )
    md.append(f"- High fact-overlap pairs among first 50 records (Jaccard>0.5): {len(dev['high_fact_overlap_pairs_first50 (jaccard>0.5)'])} found\n")

    md.append("### Label x depth grid (development)\n")
    md.append("| Label | Depth | Count |")
    md.append("|---|---|---|")
    for key, count in sorted(dev["label_depth_grid"].items()):
        label, depth = key.split("|depth=")
        md.append(f"| {label} | {depth} | {count} |")
    md.append("")

    md.append("## Test split (structure-only audit)\n")
    md.append(f"- Total examples: **{test['count']}**")
    md.append(f"- Label distribution: {test['label_counts']}")
    md.append(f"- Reasoning-depth distribution: {test['depth_counts']}")
    md.append("- **No predictions were run on this split.** This audit only reads facts/rules/question/label/depth metadata to check for leakage against the development split.\n")

    md.append("## Dev/test leakage check\n")
    md.append(f"- Distinct worlds in dev: {leakage['dev_distinct_worlds']}")
    md.append(f"- Distinct worlds in test: {leakage['test_distinct_worlds']}")
    md.append(f"- Worlds shared between dev and test: {leakage['worlds_shared_between_dev_and_test']}")
    md.append(f"- Questions shared between dev and test: {leakage['questions_shared_between_dev_and_test']}")
    md.append(f"- **Verdict: {leakage['verdict']}**\n")

    md.append("## Notes\n")
    md.append(
        "- \"Structure tags\" are a coarse heuristic (substring/keyword based), not a formal classification; "
        "see `dataset_audit.json` for per-tag counts. They approximate: negation, conjunction, presence of "
        "conditional (`If ... then`) rules, universal-quantifier rules (`All ... are ...`), and facts-only records."
    )
    md.append(
        "- Full per-record data (including world-key hashes) is in `dataset_audit.json`'s intermediate "
        "computation but is stripped from the final file to keep it compact; rerun this script for full detail."
    )

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
