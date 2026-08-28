"""Evidence-grounded, claim-level verification for FELM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from src.models.qwen import GenerationResult


VERIFICATION_STATUSES = ("supported", "contradicted", "unknown")
_MULTIPLICATION = re.compile(
    r"(?P<a>\d[\d,]*)\s*(?:\*|×|x|multiplied\s+by)\s*"
    r"(?P<b>\d[\d,]*)\s*(?:=|equals?|is)\s*(?P<c>\d[\d,]*)",
    re.IGNORECASE,
)


class ChoiceRunner(Protocol):
    def generate_choice(self, prompt: str, choices: tuple[str, ...]) -> GenerationResult: ...


@dataclass(frozen=True)
class StructuredCheck:
    representation: dict | None
    status: str | None
    evidence: list[dict]


def _integer(text: str) -> int:
    return int(text.replace(",", ""))


def extract_structured_relation(claim: str) -> StructuredCheck:
    """Verify only narrow arithmetic relations; do not pretend to parse all prose."""
    matches = list(_MULTIPLICATION.finditer(claim))
    if not matches:
        return StructuredCheck(None, None, [])
    match = matches[-1]
    left, right, stated = (_integer(match.group(name)) for name in ("a", "b", "c"))
    computed = left * right
    status = "supported" if computed == stated else "contradicted"
    representation = {
        "relation": "multiplication_equals", "left": left, "right": right,
        "stated_result": stated, "computed_result": computed,
    }
    evidence = [{
        "evidence_id": "deterministic_arithmetic", "type": "computed_relation",
        "text": f"{left} * {right} = {computed}",
    }]
    return StructuredCheck(representation, status, evidence)


def _usable_documents(record: dict) -> list[dict]:
    documents = record.get("evidence", {}).get("documents", [])
    return [
        {"doc_id": str(document.get("doc_id", "")), "type": document.get("type"),
         "text": document["text"].strip()}
        for document in documents
        if isinstance(document, dict) and isinstance(document.get("text"), str)
        and document["text"].strip()
    ]


def verify_claim(runner: ChoiceRunner, record: dict, claim: dict, prompt_template: str) -> dict:
    """Verify one claim without exposing benchmark labels or atomic-fact annotations."""
    claim_text = claim["text"]
    structured = extract_structured_relation(claim_text)
    if structured.status is not None:
        return {
            "claim_id": claim["claim_id"], "claim_text": claim_text,
            "evidence": structured.evidence, "structured_relation": structured.representation,
            "verification_status": structured.status, "verification_method": "arithmetic",
            "raw_verifier_output": None, "latency_ms": 0.0,
            "input_tokens": 0, "output_tokens": 0,
        }

    documents = _usable_documents(record)
    if not documents:
        return {
            "claim_id": claim["claim_id"], "claim_text": claim_text, "evidence": [],
            "structured_relation": None, "verification_status": "unknown",
            "verification_method": "no_evidence_abstention", "raw_verifier_output": None,
            "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
        }

    evidence_text = "\n\n".join(
        f"[{document['doc_id']}]\n{document['text']}" for document in documents
    )
    prompt = prompt_template.format(
        question=record["input"]["question"], claim=claim_text, evidence=evidence_text,
    )
    result = runner.generate_choice(prompt, ("Supported", "Contradicted", "Unknown"))
    status = result.prediction if result.prediction in VERIFICATION_STATUSES else "unknown"
    return {
        "claim_id": claim["claim_id"], "claim_text": claim_text,
        "evidence": documents, "structured_relation": None,
        "verification_status": status, "verification_method": "qwen_evidence_nli",
        "raw_verifier_output": result.raw_text, "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
    }


def verify_record(runner: ChoiceRunner, record: dict, prompt_template: str) -> dict:
    label_by_claim = {
        item["claim_id"]: item["label"] for item in record["ground_truth"]["claim_labels"]
    }
    claims = [
        verify_claim(runner, record, claim, prompt_template)
        for claim in record["candidate_response"]["claims"]
    ]
    for claim in claims:
        truth = label_by_claim.get(claim["claim_id"])
        claim["ground_truth"] = truth
        claim["correct"] = (
            claim["verification_status"] == "supported" if truth == "supported"
            else claim["verification_status"] == "contradicted"
        )
    return {
        "sample_id": record["sample_id"], "dataset": "felm",
        "method": "evidence_claim_verification", "domain": record["metadata"]["domain"],
        "claims": claims,
    }
