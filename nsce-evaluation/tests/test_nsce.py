from __future__ import annotations

import unittest

from src.models.qwen import GenerationResult
from src.nsce.pipeline import evaluate_proofwriter_record


class SequenceRunner:
    def __init__(self, predictions: list[str]) -> None:
        self.predictions = iter(predictions)

    def generate_label(self, prompt: str) -> GenerationResult:
        prediction = next(self.predictions)
        return GenerationResult(prediction.title(), prediction, 2.0, 10, 1)


class NSCETests(unittest.TestCase):
    def test_multihop_conflict_is_corrected_with_trace(self) -> None:
        record = {
            "sample_id": "example", "dataset": "proofwriter",
            "input": {
                "question": "Anne is successful.", "context": None,
                "facts": ["Anne is young."],
                "rules": [
                    "If someone is young then they are smart.",
                    "If someone is smart then they are successful.",
                ],
            },
            "ground_truth": {"label": "true"},
            "metadata": {"reasoning_depth": 2},
        }
        initial_template = "Facts:\n{facts}\nRules:\n{rules}\nQuery: {question}"
        feedback_template = (
            "{facts}\n{rules}\n{question}\n{previous_answer}\n"
            "{symbolic_explanation}\n{proof_trace}"
        )
        result = evaluate_proofwriter_record(
            SequenceRunner(["false", "true"]), record, initial_template, feedback_template
        )
        self.assertEqual(result["verification_status"], "conflict")
        self.assertEqual(result["symbolic_prediction"], "true")
        self.assertEqual(len(result["proof_trace"]), 3)
        self.assertTrue(result["correction_successful"])
        self.assertTrue(result["feedback_accepted"])


if __name__ == "__main__":
    unittest.main()

