from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import dataset_package_path
from src.datasets.felm import iter_felm, normalize_felm_record, write_felm_dataset
from src.datasets.schema import TOP_LEVEL_KEYS, validate_record
from src.models.qwen import GenerationResult
from src.nsce.factual_verifier import extract_structured_relation, verify_claim


class FailingRunner:
    def generate_choice(self, prompt: str, choices: tuple[str, ...]) -> GenerationResult:
        raise AssertionError("model must not be called without usable evidence")


class FELMTests(unittest.TestCase):
    def test_all_records_normalize_to_strict_json(self) -> None:
        source = dataset_package_path() / "normalized" / "felm_test.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "felm.jsonl"
            self.assertEqual(write_felm_dataset(source, output), 847)
            rows = list(iter_felm(output))
        self.assertTrue(all(set(row) == TOP_LEVEL_KEYS for row in rows))
        self.assertTrue(all(not validate_record(row) for row in rows))
        json.dumps(rows, allow_nan=False)

    def test_arithmetic_relation_is_checked_deterministically(self) -> None:
        result = extract_structured_relation("143287 multiplied by 534156 equals 76529372")
        self.assertEqual(result.status, "contradicted")
        self.assertEqual(result.representation["computed_result"], 76537610772)

    def test_missing_evidence_abstains(self) -> None:
        source = dataset_package_path() / "normalized" / "felm_test.jsonl"
        record = normalize_felm_record(next(iter(iter_felm(source))))
        claim = {"claim_id": "x", "text": "The Moon is made of cheese."}
        result = verify_claim(FailingRunner(), record, claim, "unused")
        self.assertEqual(result["verification_status"], "unknown")
        self.assertEqual(result["verification_method"], "no_evidence_abstention")


if __name__ == "__main__":
    unittest.main()
