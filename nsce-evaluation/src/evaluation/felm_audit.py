"""Label-blind evidence and claim-integrity audits for FELM."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any


EVIDENCE_STATUSES = (
    "SUFFICIENT_EVIDENCE", "PARTIAL_EVIDENCE", "IRRELEVANT_EVIDENCE", "NO_EVIDENCE",
)
SEMANTIC_RELEVANCE_THRESHOLD = 0.25
LEXICAL_RELEVANCE_THRESHOLD = 0.15
SEMANTIC_SUFFICIENCY_THRESHOLD = 0.50
LEXICAL_SUFFICIENCY_THRESHOLD = 0.50
_TOKENS = re.compile(r"[a-z0-9]+")
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with",
}


def content_tokens(text: str) -> set[str]:
    return {token for token in _TOKENS.findall(text.lower()) if len(token) > 1 and token not in _STOPWORDS}


def lexical_coverage(claim: str, evidence: str) -> float:
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & content_tokens(evidence)) / len(claim_tokens)


def numeric_coverage(claim: str, evidence: str) -> float:
    values = {value.replace(",", "") for value in _NUMBERS.findall(claim)}
    if not values:
        return 1.0
    evidence_values = {value.replace(",", "") for value in _NUMBERS.findall(evidence)}
    return len(values & evidence_values) / len(values)


def classify_evidence(
    evidence_present: bool,
    semantic_similarity: float | None,
    lexical_score: float,
    number_score: float,
) -> str:
    """Classify evidence without accepting a gold label or verifier prediction."""
    if not evidence_present:
        return "NO_EVIDENCE"
    semantic = semantic_similarity if semantic_similarity is not None else -1.0
    relevant = (
        semantic >= SEMANTIC_RELEVANCE_THRESHOLD
        or lexical_score >= LEXICAL_RELEVANCE_THRESHOLD
    )
    if not relevant:
        return "IRRELEVANT_EVIDENCE"
    if (
        semantic >= SEMANTIC_SUFFICIENCY_THRESHOLD
        and lexical_score >= LEXICAL_SUFFICIENCY_THRESHOLD
        and number_score == 1.0
    ):
        return "SUFFICIENT_EVIDENCE"
    return "PARTIAL_EVIDENCE"


def audit_evidence(claim: str, documents: list[dict], semantic_similarity: float | None) -> dict:
    usable = [
        document for document in documents
        if isinstance(document, dict) and isinstance(document.get("text"), str)
        and document["text"].strip()
    ]
    combined = "\n".join(document["text"] for document in usable)
    lexical = lexical_coverage(claim, combined) if usable else 0.0
    numbers = numeric_coverage(claim, combined) if usable else 0.0
    status = classify_evidence(bool(usable), semantic_similarity, lexical, numbers)
    return {
        "evidence_present": bool(usable),
        "evidence_relevant": status in {"SUFFICIENT_EVIDENCE", "PARTIAL_EVIDENCE"},
        "evidence_sufficient": status == "SUFFICIENT_EVIDENCE",
        "reference_evidence_count": len(usable),
        "audit_status": status,
        "audit_notes": (
            f"label-blind thresholds; semantic={semantic_similarity if semantic_similarity is not None else 'n/a'}; "
            f"lexical_coverage={lexical:.4f}; numeric_coverage={numbers:.4f}"
        ),
        "semantic_similarity": round(semantic_similarity, 6) if semantic_similarity is not None else None,
        "lexical_coverage": round(lexical, 6),
        "numeric_coverage": round(numbers, 6),
    }


def _normalized_tokens(text: Any) -> list[str]:
    return _TOKENS.findall(str(text).lower()) if isinstance(text, str) else []


def audit_claim_extraction(source_record: dict, normalized_record: dict) -> dict:
    """Check source linkage and decomposition integrity with reproducible heuristics."""
    source_response = source_record.get("candidate_response", {}).get("text")
    claims = normalized_record["candidate_response"]["claims"]
    reasons: list[str] = []
    if not claims:
        reasons.append("no_claims_extracted")
    empty = [claim["claim_id"] for claim in claims if not str(claim.get("text", "")).strip()]
    if empty:
        reasons.append("empty_claims:" + ",".join(empty))
    response_tokens = _normalized_tokens(source_response)
    claim_tokens = _normalized_tokens(" ".join(str(claim.get("text", "")) for claim in claims))
    if not response_tokens:
        reasons.append("source_response_not_comparable")
        linkage_ratio = None
    else:
        linkage_ratio = SequenceMatcher(None, response_tokens, claim_tokens, autojunk=False).ratio()
        response_counts, claim_counts = Counter(response_tokens), Counter(claim_tokens)
        preserved = sum(min(count, claim_counts[token]) for token, count in response_counts.items())
        content_recall = preserved / sum(response_counts.values())
        invented = sum(max(0, count - response_counts[token]) for token, count in claim_counts.items())
        invention_rate = invented / max(1, sum(claim_counts.values()))
        if content_recall < 0.80:
            reasons.append(f"source_content_recall_below_0.80:{content_recall:.4f}")
        if invention_rate > 0.20:
            reasons.append(f"claim_invention_rate_above_0.20:{invention_rate:.4f}")
    return {
        "claim_extraction_valid": not reasons,
        "claim_extraction_failure_reasons": reasons,
        "claim_source_linkage_ratio": round(linkage_ratio, 6) if linkage_ratio is not None else None,
        "claim_extraction_scope": (
            "Automated source-linkage, empty-claim, omission, and invention checks; semantic "
            "atomicity/coreference requires review of the deterministic manual subset."
        ),
    }


def audit_normalization(source_record: dict, normalized_record: dict) -> dict:
    source_response = source_record.get("candidate_response", {}).get("text")
    source_claims = source_record.get("candidate_response", {}).get("claims")
    reasons: list[str] = []
    if not isinstance(source_response, str):
        reasons.append("non_finite_or_non_string_source_response_reconstructed")
    if source_claims != normalized_record["candidate_response"]["claims"]:
        reasons.append("claim_payload_changed_during_normalization")
    return {
        "claim_normalization_valid": not reasons,
        "claim_normalization_failure_reasons": reasons,
    }


def reconcile_evidence_ids(full_rows: list[dict], evidence_rows: list[dict]) -> bool:
    full_ids = {row["claim_id"] for row in full_rows}
    evidence_ids = [row["claim_id"] for row in evidence_rows]
    return len(evidence_ids) == len(set(evidence_ids)) and set(evidence_ids).issubset(full_ids)
