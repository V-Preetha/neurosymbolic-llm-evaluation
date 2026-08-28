from __future__ import annotations

import unittest

from experiments.build_phase9_pilot import meaningful_atomic_claim, stratified_pilot
from src.evaluation.felm_gold import construct_three_way_gold


class Phase9GoldTests(unittest.TestCase):
    def test_native_supported_is_supported(self) -> None:
        result = construct_three_way_gold("Paris is in France.", "supported", "", None)
        self.assertEqual(result.label, "SUPPORTED")

    def test_bare_binary_error_is_ambiguous(self) -> None:
        result = construct_three_way_gold("A claim.", "factual_error", "", "knowledge_error")
        self.assertEqual(result.label, "GOLD_AMBIGUOUS")

    def test_explicit_numeric_mismatch_is_contradicted(self) -> None:
        result = construct_three_way_gold(
            "The answer is 12.", "factual_error", "The correct answer is 14.", "reasoning_error",
        )
        self.assertEqual(result.label, "CONTRADICTED")

    def test_no_evidence_and_irrelevance_are_unknown(self) -> None:
        self.assertEqual(
            construct_three_way_gold("A claim.", "factual_error", "No evidence found", None).label,
            "UNKNOWN",
        )
        self.assertEqual(
            construct_three_way_gold("A claim.", "factual_error", "", "irrelevant_with_qst").label,
            "UNKNOWN",
        )

    def test_atomic_meaningful_gate(self) -> None:
        self.assertTrue(meaningful_atomic_claim("Paris is in France."))
        self.assertFalse(meaningful_atomic_claim(""))
        self.assertFalse(meaningful_atomic_claim("One fact. Another fact."))

    def test_pilot_selection_is_reproducible(self) -> None:
        rows = []
        for label in ("SUPPORTED", "CONTRADICTED", "UNKNOWN"):
            for index in range(25):
                rows.append({
                    "claim_id": f"{label}_{index}", "gold_label": label,
                    "metadata": {"domain": f"d{index % 5}"},
                })
        self.assertEqual(stratified_pilot(rows), stratified_pilot(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
