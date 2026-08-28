from __future__ import annotations

import json
import unittest

from src.config import dataset_package_path
from src.datasets.proofwriter import iter_jsonl
from src.nsce.parser import Atom, Rule, parse_query, parse_sentence, parse_theory
from src.nsce.symbolic_reasoner import classify


class SymbolicTests(unittest.TestCase):
    def test_universal_conjunction(self) -> None:
        parsed = parse_sentence("All green, smart people are quiet.")
        self.assertIsInstance(parsed, tuple)
        self.assertEqual(len(parsed[0].premises), 2)

    def test_plural_pronoun_explicit_negation(self) -> None:
        parsed = parse_sentence("If someone is rough then they do not like the tiger.")
        self.assertIsInstance(parsed[0], Rule)
        self.assertEqual(parsed[0].conclusion, Atom("like", ("?x", "tiger"), True))

    def test_multihop_forward_chain_and_trace(self) -> None:
        theory = parse_theory([
            "Anne is young.",
            "If someone is young then they are smart.",
            "If someone is smart then they are successful.",
        ])
        result = classify(parse_query("Anne is successful."), theory.facts, theory.rules)
        self.assertEqual(result.prediction, "true")
        self.assertEqual(result.status, "supported")
        self.assertEqual(len(result.proof_trace), 3)

    def test_all_available_proofwriter_ground_truth(self) -> None:
        root = dataset_package_path() / "normalized"
        rows = [
            record
            for filename in ("proofwriter_dev.jsonl", "proofwriter_test.jsonl")
            for record in iter_jsonl(root / filename)
        ]
        predictions = []
        for record in rows:
            theory = parse_theory(record["input"]["facts"] + record["input"]["rules"])
            result = classify(parse_query(record["input"]["question"]), theory.facts, theory.rules)
            predictions.append(result.prediction)
        self.assertEqual(len(rows), 720)
        self.assertEqual(predictions, [record["ground_truth"]["label"] for record in rows])
        json.dumps(predictions, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

