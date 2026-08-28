from __future__ import annotations

import unittest

from src.evaluation.felm_audit import (
    audit_claim_extraction, classify_evidence, reconcile_evidence_ids,
)
from src.evaluation.felm_metrics import verification_metrics
from src.models.qwen import GenerationResult
from src.nsce.factual_verifier import verify_claim


class CapturingRunner:
    def __init__(self) -> None:
        self.prompt = ""

    def generate_choice(self, prompt: str, choices: tuple[str, ...]) -> GenerationResult:
        self.prompt = prompt
        return GenerationResult("Unknown", "unknown", 1.0, 10, 1)


class Phase8Tests(unittest.TestCase):
    def test_evidence_status_classification(self) -> None:
        self.assertEqual(classify_evidence(False, None, 0.0, 0.0), "NO_EVIDENCE")
        self.assertEqual(classify_evidence(True, 0.1, 0.1, 1.0), "IRRELEVANT_EVIDENCE")
        self.assertEqual(classify_evidence(True, 0.4, 0.4, 1.0), "PARTIAL_EVIDENCE")
        self.assertEqual(classify_evidence(True, 0.6, 0.7, 1.0), "SUFFICIENT_EVIDENCE")
        self.assertEqual(classify_evidence(True, 0.6, 0.7, 0.5), "PARTIAL_EVIDENCE")

    def test_verifier_prompt_has_no_gold_leakage(self) -> None:
        runner = CapturingRunner()
        record = {
            "input": {"question": "What color is the sky?"},
            "candidate_response": {"claims": []},
            "evidence": {"documents": [{"doc_id": "d1", "type": "reference", "text": "The sky is blue."}]},
            "ground_truth": {"label": "SECRET_GOLD", "claim_labels": [{"label": "SECRET_CLAIM_LABEL"}]},
        }
        result = verify_claim(
            runner, record, {"claim_id": "c1", "text": "The sky is blue."},
            "Question: {question}\nClaim: {claim}\nEvidence: {evidence}",
        )
        self.assertEqual(result["verification_status"], "unknown")
        self.assertNotIn("SECRET_GOLD", runner.prompt)
        self.assertNotIn("SECRET_CLAIM_LABEL", runner.prompt)

    def test_abstention_without_evidence(self) -> None:
        runner = CapturingRunner()
        record = {"input": {"question": "Q"}, "evidence": {"documents": []}}
        result = verify_claim(runner, record, {"claim_id": "c", "text": "A claim"}, "unused")
        self.assertEqual(result["verification_status"], "unknown")
        self.assertEqual(runner.prompt, "")

    def test_claim_to_source_linkage(self) -> None:
        source = {"candidate_response": {"text": "Alpha is blue. Beta is red."}}
        normalized = {"candidate_response": {"claims": [
            {"claim_id": "c1", "text": "Alpha is blue."},
            {"claim_id": "c2", "text": "Beta is red."},
        ]}}
        audit = audit_claim_extraction(source, normalized)
        self.assertTrue(audit["claim_extraction_valid"])
        self.assertEqual(audit["claim_source_linkage_ratio"], 1.0)

    def test_full_evidence_id_reconciliation(self) -> None:
        full = [{"claim_id": "a"}, {"claim_id": "b"}]
        self.assertTrue(reconcile_evidence_ids(full, [{"claim_id": "b"}]))
        self.assertFalse(reconcile_evidence_ids(full, [{"claim_id": "c"}]))
        self.assertFalse(reconcile_evidence_ids(full, [{"claim_id": "b"}, {"claim_id": "b"}]))

    def test_three_way_metric_calculation(self) -> None:
        rows = [
            {"gold_label": "SUPPORTED", "prediction": "SUPPORTED", "correct": True},
            {"gold_label": "CONTRADICTED", "prediction": "CONTRADICTED", "correct": True},
            {"gold_label": "SUPPORTED", "prediction": "UNKNOWN", "correct": False},
            {"gold_label": "CONTRADICTED", "prediction": "SUPPORTED", "correct": False},
        ]
        metrics = verification_metrics(rows)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["accuracy_including_unknown"], 0.5)
        self.assertAlmostEqual(metrics["supported"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["supported"]["recall"], 0.5)
        self.assertEqual(metrics["contradicted"]["precision"], 1.0)
        self.assertEqual(metrics["contradicted"]["recall"], 0.5)
        self.assertEqual(metrics["abstention_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
