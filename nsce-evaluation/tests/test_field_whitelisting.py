"""Phase C3: prove evaluation-only fields never reach prompt/verifier code.

Uses a dict subclass that raises if a forbidden top-level key is ever read.
Wrapping a real record in this guard and calling each prompt/verifier function
with it -- and getting no exception -- is a direct, executable proof that those
functions never look at `ground_truth`, `evidence`, `metadata`, or
`candidate_response`, covering the LLM-only, self-check, and NS bridge paths.
"""

from __future__ import annotations

import unittest

from src.io_guard import AllowedInput, FORBIDDEN_TOP_LEVEL_KEYS, extract_allowed_input
from src.nsce.feedback import render_feedback
from src.nsce.symbolic_reasoner import classify
from src.nsce.parser import parse_query, parse_theory
from src.nsce.verifier import verify_record
from src.prompts import render_proofwriter_prompt, render_self_check_prompt


class LeakGuardRecord(dict):
    """A record wrapper that raises if an evaluation-only key is ever accessed."""

    def __getitem__(self, key):
        if key in FORBIDDEN_TOP_LEVEL_KEYS:
            raise AssertionError(f"evaluation-only field {key!r} was read by prompt/verifier code")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in FORBIDDEN_TOP_LEVEL_KEYS:
            raise AssertionError(f"evaluation-only field {key!r} was read by prompt/verifier code")
        return super().get(key, default)


def make_record(gold_marker: str = "SHOULD_NEVER_APPEAR_IN_A_PROMPT") -> LeakGuardRecord:
    return LeakGuardRecord({
        "sample_id": "guard_test_0001",
        "dataset": "proofwriter",
        "task_type": "logical_reasoning",
        "input": {
            "question": "Anne is successful.",
            "context": None,
            "facts": ["Anne is young."],
            "rules": [
                "If someone is young then they are smart.",
                "If someone is smart then they are successful.",
            ],
        },
        "candidate_response": {"text": gold_marker, "claims": []},
        "ground_truth": {"label": gold_marker, "reference_answer": gold_marker},
        "evidence": {"documents": [{"text": gold_marker}]},
        "metadata": {"reasoning_depth": gold_marker},
    })


PROOFWRITER_TEMPLATE = "Facts:\n{facts}\nRules:\n{rules}\nQuery: {question}"
SELF_CHECK_TEMPLATE = "{facts}\n{rules}\n{question}\n{previous_answer}"
FEEDBACK_TEMPLATE = (
    "{facts}\n{rules}\n{question}\n{previous_answer}\n{symbolic_explanation}\n{proof_trace}"
)


class FieldWhitelistingTests(unittest.TestCase):
    def test_extract_allowed_input_exposes_only_facts_rules_query(self) -> None:
        record = make_record()
        allowed = extract_allowed_input(record)
        self.assertIsInstance(allowed, AllowedInput)
        self.assertEqual({f.name for f in allowed.__dataclass_fields__.values()}, {"facts", "rules", "query"})
        self.assertEqual(allowed.query, "Anne is successful.")

    def test_llm_only_prompt_never_reads_forbidden_fields(self) -> None:
        record = make_record()
        prompt = render_proofwriter_prompt(record, PROOFWRITER_TEMPLATE)
        self.assertNotIn("SHOULD_NEVER_APPEAR_IN_A_PROMPT", prompt)

    def test_self_check_prompt_never_reads_forbidden_fields(self) -> None:
        record = make_record()
        initial_prompt = render_proofwriter_prompt(record, PROOFWRITER_TEMPLATE)
        self.assertNotIn("SHOULD_NEVER_APPEAR_IN_A_PROMPT", initial_prompt)
        verification_prompt = render_self_check_prompt(record, "true", SELF_CHECK_TEMPLATE)
        self.assertNotIn("SHOULD_NEVER_APPEAR_IN_A_PROMPT", verification_prompt)

    def test_ns_bridge_verifier_never_reads_forbidden_fields(self) -> None:
        record = make_record()
        result = verify_record(record)  # must not raise despite ground_truth/evidence/metadata being guarded
        self.assertIsNotNone(result.reasoning)
        self.assertIsNone(result.parser_error)

    def test_ns_bridge_feedback_never_reads_forbidden_fields(self) -> None:
        record = make_record()
        verification = verify_record(record)
        feedback_prompt = render_feedback(record, "false", verification.reasoning, FEEDBACK_TEMPLATE)
        self.assertNotIn("SHOULD_NEVER_APPEAR_IN_A_PROMPT", feedback_prompt)

    def test_symbolic_engine_never_receives_gold_label(self) -> None:
        # Direct-call equivalent: parse_theory/classify only ever take AllowedInput-derived
        # strings, never the record itself, so gold leakage is structurally impossible here.
        allowed = extract_allowed_input(make_record())
        theory = parse_theory(list(allowed.facts) + list(allowed.rules))
        result = classify(parse_query(allowed.query), theory.facts, theory.rules)
        self.assertEqual(result.prediction, "true")


if __name__ == "__main__":
    unittest.main()
