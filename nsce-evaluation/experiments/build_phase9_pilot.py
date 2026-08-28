"""Build the fixed Phase 9 eligible pool, exclusion ledger, and stratified pilot."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import PROJECT_ROOT, dataset_package_path
from src.datasets.felm import iter_felm
from src.evaluation.felm_gold import construct_three_way_gold


TARGETS = {"SUPPORTED": 17, "CONTRADICTED": 17, "UNKNOWN": 16}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def meaningful_atomic_claim(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text)
    sentence_ends = len(re.findall(r"[.!?](?:\s|$)", text))
    return len(words) >= 3 and sentence_ends <= 1


def stratified_pilot(rows: list[dict], seed: int = 42) -> list[dict]:
    by_label_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_label_domain[(row["gold_label"], row["metadata"]["domain"])].append(row)
    for values in by_label_domain.values():
        values.sort(key=lambda row: hashlib.sha256(
            f"{seed}|{row['claim_id']}".encode("utf-8")
        ).hexdigest())
    selected: list[dict] = []
    domains = sorted({row["metadata"]["domain"] for row in rows})
    for label, target in TARGETS.items():
        positions = {domain: 0 for domain in domains}
        label_selected = 0
        while label_selected < target:
            progressed = False
            for domain in domains:
                values = by_label_domain[(label, domain)]
                position = positions[domain]
                if position < len(values) and label_selected < target:
                    selected.append(values[position])
                    positions[domain] += 1
                    label_selected += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Insufficient eligible {label} claims for pilot")
    return sorted(selected, key=lambda row: row["claim_id"])


def main() -> None:
    full_rows = [
        json.loads(line) for line in (PROJECT_ROOT / "data" / "processed" / "felm_full_claims.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    source_rows = list(iter_felm(dataset_package_path() / "normalized" / "felm_test.jsonl"))
    source_by_id = {row["sample_id"]: row for row in source_rows}
    full_by_id = {row["claim_id"]: row for row in full_rows}
    annotation_by_id = {
        item["claim_id"]: item
        for record in source_rows for item in record["ground_truth"]["claim_labels"]
    }
    ledger: list[dict] = []
    eligible: list[dict] = []
    for claim_id, audit in sorted(full_by_id.items()):
        source = source_by_id[audit["sample_id"]]
        annotation = annotation_by_id[claim_id]
        decision = construct_three_way_gold(
            audit["claim_text"], annotation["label"], annotation.get("annotator_comment"),
            annotation.get("error_type"),
        )
        exclusion = None
        if not audit["claim_extraction_valid"]:
            exclusion = "CLAIM_EXTRACTION_FAILURE"
        elif not audit["claim_normalization_valid"]:
            exclusion = "CLAIM_NORMALIZATION_FAILURE"
        elif not meaningful_atomic_claim(audit["claim_text"]):
            exclusion = "NON_MEANINGFUL_OR_NON_ATOMIC_CLAIM"
        elif decision.label == "GOLD_AMBIGUOUS":
            exclusion = "GOLD_AMBIGUOUS"
        record = {
            "sample_id": audit["sample_id"], "claim_id": claim_id,
            "question": source["input"]["question"],
            "response_text": source["candidate_response"]["text"] if isinstance(source["candidate_response"]["text"], str) else None,
            "claim_text": audit["claim_text"], "gold_label": decision.label,
            "gold_basis": decision.basis,
            "gold_annotation_comment": annotation.get("annotator_comment"),
            "corpus_query": f"{source['input']['question']}\nClaim: {audit['claim_text']}",
            "evidence_source_ids": [],
            "metadata": {
                "domain": audit["domain"], "original_felm_id": audit["sample_id"],
                "exclusion_reason": exclusion,
            },
        }
        ledger.append({**record, "eligible": exclusion is None})
        if exclusion is None:
            eligible.append(record)
    pilot = stratified_pilot(eligible)
    processed = PROJECT_ROOT / "data" / "processed"
    write_jsonl(processed / "felm_phase9_eligibility.jsonl", ledger)
    write_jsonl(processed / "felm_controlled_eligible.jsonl", eligible)
    write_jsonl(processed / "felm_phase9_pilot.jsonl", pilot)
    (processed / "felm_phase9_pilot_ids.json").write_text(
        json.dumps({
            "selection": "sha256(seed=42, claim_id), round-robin by label and domain",
            "claim_ids": [row["claim_id"] for row in pilot],
        }, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "total_claims": len(ledger), "eligible_claims": len(eligible),
        "gold_ambiguous": sum(row["metadata"]["exclusion_reason"] == "GOLD_AMBIGUOUS" for row in ledger),
        "exclusions": dict(Counter(row["metadata"]["exclusion_reason"] or "eligible" for row in ledger)),
        "eligible_labels": dict(Counter(row["gold_label"] for row in eligible)),
        "eligible_domains": dict(Counter(row["metadata"]["domain"] for row in eligible)),
        "pilot_labels": dict(Counter(row["gold_label"] for row in pilot)),
        "pilot_domains": dict(Counter(row["metadata"]["domain"] for row in pilot)),
    }
    (processed / "felm_phase9_pool_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
