"""Conservative deterministic checks used by Phase 9 NSCE."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.nsce.factual_verifier import extract_structured_relation


_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")


@dataclass(frozen=True)
class Finding:
    status: str | None
    kind: str | None
    justification: str | None
    representation: dict | None


def _normalized_frame(text: str) -> str:
    text = _NUMBER.sub("<number>", text.lower())
    return " ".join(re.findall(r"[a-z]+|<number>", text))


def check(claim: str, documents: list[dict]) -> Finding:
    arithmetic = extract_structured_relation(claim)
    if arithmetic.status:
        return Finding(arithmetic.status.upper(), "deterministic_arithmetic",
                       arithmetic.evidence[0]["text"], arithmetic.representation)
    claim_numbers = [x.replace(",", "") for x in _NUMBER.findall(claim)]
    if not claim_numbers:
        return Finding(None, None, None, None)
    claim_frame = _normalized_frame(claim)
    if len(claim_frame.split()) < 4:
        return Finding(None, None, None, None)
    for doc in documents:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.get("text", "")):
            evidence_numbers = [x.replace(",", "") for x in _NUMBER.findall(sentence)]
            if not evidence_numbers or _normalized_frame(sentence) != claim_frame:
                continue
            status = "SUPPORTED" if evidence_numbers == claim_numbers else "CONTRADICTED"
            return Finding(status, "exact_numeric_relation_frame",
                           f"Claim values {claim_numbers}; evidence values {evidence_numbers} in {doc['doc_id']}",
                           {"claim_frame": claim_frame, "claim_values": claim_numbers,
                            "evidence_values": evidence_numbers, "doc_id": doc["doc_id"]})
    return Finding(None, None, None, None)
