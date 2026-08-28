import unittest

from final_experiment.training_data import (
    DynamicCausalCollator,
    audit_record_groups,
    choose_safe_max_sequence_length,
    summarize_token_lengths,
    tokenize_training_record,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        if isinstance(text, list):
            return {"input_ids": [self(item, add_special_tokens=add_special_tokens)["input_ids"] for item in text]}
        tokens = [len(word) for word in text.split()]
        return {"input_ids": ([1] if add_special_tokens else []) + tokens}


class TrainingDataTests(unittest.TestCase):
    def test_direct_labels_exclude_prompt_and_include_only_answer_and_eos(self):
        encoded = tokenize_training_record(
            {"prefix": "Question Answer:", "target": "True"},
            FakeTokenizer(),
            max_sequence_length=8,
        )
        self.assertEqual(encoded["input_ids"], [1, 8, 7, 4, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 99])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1, 1])

    def test_record_longer_than_audited_limit_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds audited max_sequence_length"):
            tokenize_training_record(
                {"prefix": "one two three", "target": "True"},
                FakeTokenizer(),
                max_sequence_length=4,
            )

    def test_dynamic_collator_pads_only_to_longest_record_in_batch(self):
        collator = DynamicCausalCollator(pad_token_id=0)
        batch = collator([
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [4, 5, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 3], [-100, 5, -100]])

    def test_audit_uses_observed_maximum_as_shortest_non_truncating_limit(self):
        summary = summarize_token_lengths([12, 20, 20, 31])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50"], 20)
        self.assertEqual(summary["p95"], 31)
        self.assertEqual(summary["max"], 31)
        self.assertEqual(choose_safe_max_sequence_length([summary]), 31)

    def test_empty_length_audit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No token lengths"):
            choose_safe_max_sequence_length([])

    def test_group_audit_reports_each_distribution_and_global_safe_limit(self):
        report = audit_record_groups({
            "direct_train": [
                {"prefix": "short", "target": "True"},
                {"prefix": "two words", "target": "False"},
            ],
            "bridge_validation": [
                {"prefix": "one two three", "target": "json target"},
            ],
        }, FakeTokenizer())
        self.assertEqual(report["groups"]["direct_train"]["count"], 2)
        self.assertEqual(report["groups"]["bridge_validation"]["max"], 7)
        self.assertEqual(report["chosen_max_sequence_length"], 7)
        self.assertEqual(report["selection_policy"], "maximum observed train/development record length")


if __name__ == "__main__":
    unittest.main()
