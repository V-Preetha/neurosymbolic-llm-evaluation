from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import dataset_package_path
from src.baselines.llm_only import evaluate_proofwriter_record
from src.baselines.self_check import evaluate_proofwriter_record as evaluate_self_check
from src.datasets.proofwriter import iter_jsonl, write_phase1_sample
from src.datasets.schema import TOP_LEVEL_KEYS, validate_record
from src.models.qwen import GenerationResult, extract_logical_label
from src.prompts import (
    load_template,
    PROOFWRITER_LLM_ONLY_TEMPLATE,
    PROOFWRITER_SELF_CHECK_TEMPLATE,
)


class FakeRunner:
    def __init__(self, prediction: str | None, raw_text: str) -> None:
        self.prediction = prediction
        self.raw_text = raw_text

    def generate_label(self, prompt: str) -> GenerationResult:
        assert "ground_truth" not in prompt
        return GenerationResult(self.raw_text, self.prediction, 1.5, 10, 1)


class SequenceRunner:
    def __init__(self, predictions: list[str]) -> None:
        self.predictions = iter(predictions)

    def generate_label(self, prompt: str) -> GenerationResult:
        prediction = next(self.predictions)
        return GenerationResult(prediction.title(), prediction, 2.0, 10, 1)


class Phase1Tests(unittest.TestCase):
    def test_label_extraction_is_strict(self) -> None:
        self.assertEqual(extract_logical_label("True"), "true")
        self.assertEqual(extract_logical_label(" unknown \n"), "unknown")
        self.assertIsNone(extract_logical_label("The answer is True"))

    def test_twenty_records_normalize_to_canonical_schema(self) -> None:
        source = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proofwriter.jsonl"
            self.assertEqual(write_phase1_sample(source, output, 20), 20)
            rows = list(iter_jsonl(output))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(set(row) == TOP_LEVEL_KEYS for row in rows))
        self.assertTrue(all(not validate_record(row) for row in rows))
        self.assertEqual(len({row["sample_id"] for row in rows}), 20)
        json.dumps(rows, allow_nan=False)

    def test_llm_only_result_contract(self) -> None:
        source = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
        record = next(iter(iter_jsonl(source)))
        record.pop("provenance")
        template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
        result = evaluate_proofwriter_record(FakeRunner("unknown", "Unknown"), record, template)
        self.assertEqual(result["method"], "llm_only")
        self.assertTrue(result["correct_finally"])
        self.assertIsNone(result["error_category"])

    def test_self_check_successful_correction(self) -> None:
        source = dataset_package_path() / "normalized" / "proofwriter_dev.jsonl"
        record = next(iter(iter_jsonl(source)))
        record.pop("provenance")
        result = evaluate_self_check(
            SequenceRunner(["false", "unknown"]),
            record,
            load_template(PROOFWRITER_LLM_ONLY_TEMPLATE),
            load_template(PROOFWRITER_SELF_CHECK_TEMPLATE),
        )
        self.assertTrue(result["answer_changed"])
        self.assertTrue(result["correction_successful"])
        self.assertTrue(result["correct_finally"])


if __name__ == "__main__":
    unittest.main()
