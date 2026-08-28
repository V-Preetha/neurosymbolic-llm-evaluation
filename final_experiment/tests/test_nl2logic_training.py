import json
import unittest

from final_experiment.data import normalize_row
from final_experiment.nl2logic_training import component_records, expanded_record_counts, training_examples


PW = {
    "id": "train-world",
    "maxD": 1,
    "NFact": 1,
    "NRule": 1,
    "theory": "Bob is red. If Bob is red then Bob is nice.",
    "question": "Bob is nice.",
    "answer": "True",
    "QDep": 1,
    "allProofs": "proof",
}


class NL2LogicTrainingTests(unittest.TestCase):
    def setUp(self):
        self.row = normalize_row("proofwriter", PW, "train", 0)

    def test_implication_expands_to_component_json_supervision(self):
        records = component_records(self.row)
        components = {record["component"] for record in records}
        self.assertTrue({"parser_selector", "binary", "relation_selector", "adjective"}.issubset(components))
        binary = next(record for record in records if record["component"] == "binary")
        self.assertEqual(json.loads(binary["target"])["operator"], "implies")
        selector_targets = {json.loads(record["target"])["parser"] for record in records if record["component"] == "parser_selector"}
        self.assertIn("binary", selector_targets)
        self.assertIn("atomic", selector_targets)

    def test_bridge_examples_supervise_only_component_output(self):
        examples = training_examples([self.row], "bridge")
        self.assertGreater(len(examples), 1)
        for example in examples:
            self.assertTrue(example["text"].startswith(example["prefix"]))
            self.assertEqual(example["text"], example["prefix"] + example["target"])
            json.loads(example["target"])

    def test_direct_training_remains_one_example_per_manifest_row(self):
        examples = training_examples([self.row], "direct")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["target"], "True")

    def test_expanded_record_counts_form_complete_stage_partition(self):
        examples = training_examples([self.row], "bridge")
        counts = expanded_record_counts(examples)
        self.assertEqual(counts["total_expanded_records"], len(examples))
        self.assertEqual(
            counts["selector_records"] + counts["atomic_records"]
            + counts["quantified_records"] + counts["logical_records"],
            len(examples),
        )
        self.assertGreater(counts["selector_records"], 0)
        self.assertGreater(counts["atomic_records"], 0)
        self.assertGreater(counts["logical_records"], 0)

    def test_equality_is_supervised_as_an_atomic_component(self):
        row = {
            "dataset": "folio",
            "premises": ["Alice is Bob."],
            "gold_premises_fol": ["alice=bob"],
            "question": "Alice is not Bob.",
            "gold_conclusion_fol": "¬(alice=bob)",
            "example_id": "folio-equality",
        }
        records = component_records(row)
        equality = [record for record in records if record["component"] == "equality"]
        self.assertEqual(len(equality), 2)
        self.assertEqual(json.loads(equality[0]["target"]), {"left": "alice", "right": "bob"})
        counts = expanded_record_counts(training_examples([row], "bridge"))
        self.assertEqual(counts["atomic_records"], 2)

    def test_zero_arity_predicate_is_supervised_as_a_proposition(self):
        row = {
            "dataset": "folio",
            "premises": ["It is available."],
            "gold_premises_fol": ["Available()"],
            "question": "It is ready.",
            "gold_conclusion_fol": "Ready()",
            "example_id": "folio-proposition",
        }
        records = component_records(row)
        propositions = [record for record in records if record["component"] == "proposition"]
        self.assertEqual(len(propositions), 2)
        self.assertEqual(json.loads(propositions[0]["target"]), {"predicate": "available"})


if __name__ == "__main__":
    unittest.main()
