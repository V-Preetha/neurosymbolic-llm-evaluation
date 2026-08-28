import unittest

from final_experiment.fol_ast import BridgeError
from final_experiment.z3_compiler import collect_declarations, compile_theory
from final_experiment.z3_solver import semantic_agreement, solve, solve_compiled


def atom(name, *args):
    return {"type": "predicate", "name": name, "args": list(args)}


class Z3PipelineTests(unittest.TestCase):
    def test_first_pass_collects_constants_predicates_and_arities(self):
        theory = {
            "premises": [atom("likes", "alice", "bob"), atom("red", "alice")],
            "hypothesis": atom("likes", "bob", "alice"),
        }
        declarations = collect_declarations(theory)
        self.assertEqual(set(declarations.constants), {"alice", "bob"})
        self.assertEqual(declarations.predicate_arities, {"likes": 2, "red": 1})

    def test_compile_supports_official_predicate_arity_overloading(self):
        theory = {"premises": [atom("likes", "alice", "bob")], "hypothesis": atom("likes", "alice")}
        compiled = compile_theory(theory)
        self.assertEqual(set(compiled.declarations.predicates), {"likes/1", "likes/2"})

    def test_solver_returns_true_false_and_unknown(self):
        true_theory = {"premises": [atom("red", "alice")], "hypothesis": atom("red", "alice")}
        false_theory = {
            "premises": [{"type": "not", "body": atom("red", "alice")}],
            "hypothesis": atom("red", "alice"),
        }
        unknown_theory = {"premises": [atom("warm", "alice")], "hypothesis": atom("red", "alice")}
        self.assertEqual(solve(true_theory)["label"], "True")
        self.assertEqual(solve(false_theory)["label"], "False")
        self.assertEqual(solve_compiled(compile_theory(unknown_theory))["label"], "Unknown")

    def test_inconsistent_premises_are_system_failure(self):
        theory = {
            "premises": [atom("red", "alice"), {"type": "not", "body": atom("red", "alice")}],
            "hypothesis": atom("red", "alice"),
        }
        with self.assertRaises(BridgeError) as caught:
            solve(theory)
        self.assertEqual(caught.exception.status, "inconsistent_premises")

    def test_semantic_agreement_distinguishes_equivalent_and_different(self):
        gold = {"premises": [atom("red", "alice")], "hypothesis": atom("warm", "alice")}
        equivalent = {"premises": [atom("red", "alice")], "hypothesis": atom("warm", "alice")}
        different = {"premises": [atom("blue", "alice")], "hypothesis": atom("warm", "alice")}
        self.assertEqual(semantic_agreement(equivalent, gold)["status"], "equivalent")
        self.assertEqual(semantic_agreement(different, gold)["status"], "different")

    def test_term_equality_compiles_to_native_z3_equality(self):
        equality = {"type": "equals", "left": "alice", "right": "bob"}
        theory = {
            "premises": [equality, atom("red", "alice")],
            "hypothesis": atom("red", "bob"),
        }
        declarations = collect_declarations(theory)
        self.assertEqual(set(declarations.constants), {"alice", "bob"})
        self.assertEqual(solve(theory)["label"], "True")

    def test_term_inequality_is_logical_negation_of_equality(self):
        equality = {"type": "equals", "left": "alice", "right": "bob"}
        theory = {"premises": [{"type": "not", "body": equality}], "hypothesis": equality}
        self.assertEqual(solve(theory)["label"], "False")


if __name__ == "__main__":
    unittest.main()
