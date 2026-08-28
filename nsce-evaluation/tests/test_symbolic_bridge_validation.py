"""Phase D2/D3: independent validation of the symbolic bridge -- NO Qwen calls.

Exercises `parser.py` and `symbolic_reasoner.py` directly (unit-testing the
translation and reasoning stages in isolation), and `verifier.py` (to prove the
translation-failure vs reasoning-outcome separation holds at the integration
point used by the pipeline). Every reasoning test also asserts on the proof
trace, confirming it originates from the reasoner's own derivation state and
not from any external explanation.
"""

from __future__ import annotations

import unittest

from src.nsce.parser import Atom, ParserError, parse_query, parse_theory
from src.nsce.symbolic_reasoner import classify
from src.nsce.verifier import verify_record


def run(facts: list[str], rules: list[str], query: str):
    theory = parse_theory(facts + rules)
    return classify(parse_query(query), theory.facts, theory.rules)


class DirectSupportTests(unittest.TestCase):
    def test_direct_support(self) -> None:
        result = run(["The cat is red."], [], "The cat is red.")
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.prediction, "true")
        self.assertFalse(result.inconsistent)
        self.assertEqual(result.proof_trace, ("Fact: red(cat)",))


class DirectContradictionTests(unittest.TestCase):
    def test_direct_contradiction_from_negated_fact(self) -> None:
        result = run(["The cat is not red."], [], "The cat is red.")
        self.assertEqual(result.status, "contradicted")
        self.assertEqual(result.prediction, "false")
        self.assertEqual(result.proof_trace, ("Fact: Not red(cat)",))

    def test_negated_query_contradicted_by_positive_fact(self) -> None:
        result = run(["The cat is red."], [], "The cat is not red.")
        self.assertEqual(result.status, "contradicted")
        self.assertEqual(result.prediction, "false")
        self.assertEqual(result.proof_trace, ("Fact: red(cat)",))


class UnknownTests(unittest.TestCase):
    def test_unrelated_fact_leaves_query_unknown(self) -> None:
        result = run(["The cat is blue."], [], "The cat is red.")
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.prediction, "unknown")
        self.assertEqual(result.proof_trace, ())

    def test_missing_premise_leaves_query_unknown(self) -> None:
        # The rule's premise ("Anne is smart") is never asserted or derivable,
        # so the rule can never fire -- this must resolve to unknown, not an error.
        result = run(
            ["Anne is young."],
            ["If someone is smart then they are successful."],
            "Anne is successful.",
        )
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.proof_trace, ())


class OneHopRuleTests(unittest.TestCase):
    def test_one_hop_rule(self) -> None:
        result = run(
            ["The cat is red."],
            ["If something is red then it is warm."],
            "The cat is warm.",
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.prediction, "true")
        self.assertEqual(
            result.proof_trace,
            ("Fact: red(cat)", "Rule: red(cat) -> warm(cat)"),
        )


class MultiHopRuleTests(unittest.TestCase):
    def test_two_hop_chain(self) -> None:
        result = run(
            ["Anne is young."],
            [
                "If someone is young then they are smart.",
                "If someone is smart then they are successful.",
            ],
            "Anne is successful.",
        )
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.prediction, "true")
        self.assertEqual(
            result.proof_trace,
            (
                "Fact: young(anne)",
                "Rule: young(anne) -> smart(anne)",
                "Rule: smart(anne) -> successful(anne)",
            ),
        )


class ContradictoryFactsTests(unittest.TestCase):
    def test_contradictory_facts_flagged_inconsistent(self) -> None:
        result = run(["The cat is red.", "The cat is not red."], [], "The cat is red.")
        self.assertTrue(result.inconsistent)
        # supported is checked before contradicted in classify(); both derivations
        # exist, but the returned status reflects the priority order -- the
        # `inconsistent` flag is what a caller must check to detect this case.
        self.assertEqual(result.status, "supported")


class MalformedAndUnsupportedInputTests(unittest.TestCase):
    def test_unsupported_verb_raises_parser_error(self) -> None:
        with self.assertRaises(ParserError):
            parse_theory(["The cat jumps the fence."])

    def test_disjunction_raises_parser_error(self) -> None:
        # Disjunction is not part of the supported grammar (see SYMBOLIC_BRIDGE_SPEC.md);
        # it must fail loudly at parse time, not be silently misparsed.
        with self.assertRaises(ParserError):
            parse_theory(["The cat is red or blue."])

    def test_malformed_conditional_raises_parser_error(self) -> None:
        with self.assertRaises(ParserError):
            parse_theory(["If the cat is red the cat is warm."])  # missing "then"


class TranslationVsReasoningSeparationTests(unittest.TestCase):
    """verify_record() is the integration point the pipeline actually calls; prove it
    keeps translation failure (parser_error set, reasoning=None) and reasoning outcomes
    (reasoning set, including status='unknown') structurally distinct."""

    def _record(self, facts: list[str], rules: list[str], question: str) -> dict:
        return {
            "sample_id": "validation_record",
            "input": {"question": question, "context": None, "facts": facts, "rules": rules},
        }

    def test_valid_translation_with_unknown_reasoning_outcome(self) -> None:
        record = self._record(["The cat is blue."], [], "The cat is red.")
        result = verify_record(record)
        self.assertIsNone(result.parser_error)
        self.assertIsNotNone(result.reasoning)
        self.assertEqual(result.reasoning.status, "unknown")

    def test_translation_failure_leaves_reasoning_none(self) -> None:
        record = self._record(["The cat jumps the fence."], [], "The cat is red.")
        result = verify_record(record)
        self.assertIsNone(result.reasoning)
        self.assertIsNotNone(result.parser_error)


if __name__ == "__main__":
    unittest.main()
