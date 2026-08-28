import unittest

from final_experiment.fol_ast import (
    BridgeError,
    extract_json,
    normalized_logic,
    validate_theory,
)
from final_experiment.gold_targets import folio_target, parse_fol, proofwriter_target


class FolAstTests(unittest.TestCase):
    def test_valid_quantified_theory_is_accepted_and_normalized(self):
        theory = {
            "premises": [{
                "type": "forall",
                "variables": ["x"],
                "body": {
                    "type": "implies",
                    "antecedent": {"type": "predicate", "name": "red", "args": ["x"]},
                    "consequent": {"type": "predicate", "name": "warm", "args": ["x"]},
                },
            }],
            "hypothesis": {"type": "predicate", "name": "warm", "args": ["alice"]},
        }
        validate_theory(theory)
        self.assertEqual(
            normalized_logic(theory),
            '{"hypothesis":{"args":["alice"],"name":"warm","type":"predicate"},"premises":[{"body":{"antecedent":{"args":["x"],"name":"red","type":"predicate"},"consequent":{"args":["x"],"name":"warm","type":"predicate"},"type":"implies"},"type":"forall","variables":["x"]}]}',
        )

    def test_free_variable_is_rejected(self):
        theory = {
            "premises": [{"type": "predicate", "name": "red", "args": ["x"]}],
            "hypothesis": {"type": "predicate", "name": "red", "args": ["alice"]},
        }
        with self.assertRaisesRegex(BridgeError, "free variable") as caught:
            validate_theory(theory)
        self.assertEqual(caught.exception.status, "ast_invalid")

    def test_official_predicate_arity_overloading_is_accepted(self):
        theory = {
            "premises": [{"type": "predicate", "name": "likes", "args": ["alice", "bob"]}],
            "hypothesis": {"type": "predicate", "name": "likes", "args": ["alice"]},
        }
        validate_theory(theory)

    def test_malformed_json_has_parser_status(self):
        with self.assertRaises(BridgeError) as caught:
            extract_json("```json\nnot-json\n```")
        self.assertEqual(caught.exception.status, "malformed_output")

    def test_single_json_object_surrounded_by_prose_is_accepted(self):
        self.assertEqual(
            extract_json('Parser result: {"parser":"atomic"}\nDone.'),
            {"parser": "atomic"},
        )

    def test_multiple_json_objects_are_rejected_as_ambiguous(self):
        with self.assertRaises(BridgeError) as caught:
            extract_json('{"parser":"atomic"}\n{"parser":"binary"}')
        self.assertEqual(caught.exception.status, "malformed_output")
        self.assertIn("multiple", caught.exception.detail)

    def test_proofwriter_someone_rule_is_explicitly_universally_bound(self):
        row={"dataset":"proofwriter","context":"If someone is red then they are warm.","question":"Alice is warm."}
        theory=proofwriter_target(row)
        self.assertEqual(theory["premises"][0]["type"],"forall")
        validate_theory(theory)

    def test_folio_equality_and_inequality_have_typed_ast_nodes(self):
        equality = parse_fol("x=y")
        inequality = parse_fol("¬(x=y) ∧ x≠z")
        self.assertEqual(equality, {"type": "equals", "left": "x", "right": "y"})
        self.assertEqual(inequality["args"][0], {"type": "not", "body": equality})
        self.assertEqual(inequality["args"][1]["type"], "not")
        self.assertEqual(inequality["args"][1]["body"]["type"], "equals")

    def test_folio_numeric_hyphenated_and_unicode_symbols_are_preserved(self):
        expression = parse_fol("Five-Story(pSOJ318.5-22) ∧ Won(Świątek, 2024UnitedStatesElection)")
        self.assertEqual(expression["args"][0]["name"], "five-story")
        self.assertEqual(expression["args"][0]["args"], ["psoj318.5-22"])
        self.assertEqual(expression["args"][1]["args"], ["świątek", "2024unitedstateselection"])

    def test_quantifiers_scope_over_the_complete_following_expression(self):
        expression = parse_fol("∀x ∀y (∃z (¬(x=y) ∧ Know(x,z)) → CanCommunicateWith(x,y) ∧ CanCommunicateWith(y,x))")
        self.assertEqual(expression["type"], "forall")
        self.assertEqual(expression["body"]["type"], "forall")
        self.assertEqual(expression["body"]["body"]["type"], "implies")

    def test_official_formatting_defects_are_recovered_deterministically(self):
        expression = parse_fol("∀x (Starred(x)) → Hard(x))")
        self.assertEqual(expression["type"], "forall")
        self.assertEqual(expression["body"]["type"], "implies")
        zero_arity = parse_fol("Available()")
        self.assertEqual(zero_arity, {"type": "predicate", "name": "available", "args": []})

    def test_official_ascii_disjunction_apostrophe_and_terminal_period(self):
        expression = parse_fol("SoldAt(avocado, nica'sMarket) v Giant(gpt-3).")
        self.assertEqual(expression["type"], "or")
        self.assertEqual(expression["args"][0]["args"][1], "nica'smarket")
        self.assertEqual(expression["args"][1]["args"], ["gpt-3"])

    def test_quantifier_with_omitted_variable_uses_its_written_free_term(self):
        expression = parse_fol("∃(CarBody(x) ∧ Made(x, carrozzeriaColli))")
        self.assertEqual(expression["type"], "exists")
        self.assertEqual(expression["variables"], ["x"])

    def test_folio_gold_free_single_letter_terms_become_explicit_constants(self):
        row = {
            "gold_premises_fol": ["Depressing(v)"],
            "gold_conclusion_fol": "Invigorating(v)",
        }
        theory = folio_target(row)
        self.assertEqual(theory["premises"][0]["args"], ["free_v"])
        self.assertEqual(theory["hypothesis"]["args"], ["free_v"])
        validate_theory(theory)


if __name__ == "__main__":
    unittest.main()
