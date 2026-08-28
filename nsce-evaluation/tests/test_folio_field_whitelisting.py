"""Prove gold FOLIO fields never reach the translation prompt -- same pattern
as tests/test_field_whitelisting.py for the ProofWriter path.
"""
from __future__ import annotations

import unittest

from src.folio.io_guard import FORBIDDEN_FOLIO_KEYS, FolioAllowedInput, extract_allowed_folio_input
from src.folio.translator import render_sentence_translation_prompt


class LeakGuardRecord(dict):
    def __getitem__(self, key):
        if key in FORBIDDEN_FOLIO_KEYS:
            raise AssertionError(f"evaluation-only field {key!r} was read while building the translation prompt")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in FORBIDDEN_FOLIO_KEYS:
            raise AssertionError(f"evaluation-only field {key!r} was read while building the translation prompt")
        return super().get(key, default)


def make_record(gold_marker: str = "SHOULD_NEVER_APPEAR_IN_A_PROMPT") -> LeakGuardRecord:
    return LeakGuardRecord({
        "premises": ["All dogs bark.", "Fluffy is a dog."],
        "conclusion": "Fluffy barks.",
        "label": gold_marker,
        "premises-FOL": [gold_marker],
        "conclusion-FOL": gold_marker,
    })


class FolioFieldWhitelistingTests(unittest.TestCase):
    def test_extract_allowed_folio_input_exposes_only_premises_and_hypothesis(self) -> None:
        allowed = extract_allowed_folio_input(make_record())
        self.assertIsInstance(allowed, FolioAllowedInput)
        self.assertEqual({f.name for f in allowed.__dataclass_fields__.values()}, {"premises", "hypothesis"})
        self.assertEqual(allowed.hypothesis, "Fluffy barks.")

    def test_translation_prompt_never_contains_gold_marker(self) -> None:
        allowed = extract_allowed_folio_input(make_record())
        for sentence in list(allowed.premises) + [allowed.hypothesis]:
            prompt = render_sentence_translation_prompt(sentence)
            self.assertNotIn("SHOULD_NEVER_APPEAR_IN_A_PROMPT", prompt)

    def test_translation_prompt_never_asks_for_a_truth_label(self) -> None:
        allowed = extract_allowed_folio_input(make_record())
        prompt = render_sentence_translation_prompt(allowed.hypothesis)
        for banned in ("true or false", "is the hypothesis true", "entailed?", "supported?"):
            self.assertNotIn(banned, prompt.lower())


if __name__ == "__main__":
    unittest.main()
