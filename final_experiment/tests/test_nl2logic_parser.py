import json
import unittest

from final_experiment.fol_ast import BridgeError
from final_experiment.nl2logic_parser import NL2LogicParser, _apply_name_negation, _predicate_symbol, _symbol, _uncovered_content, canonical_clause


class ScriptedGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.call_count = 0

    def __call__(self, prompt, bridge=True, component=None):
        self.call_count += 1
        raw = next(self.outputs)
        return {"raw": raw, "latency_ms": 1.5, "input_tokens": 10, "output_tokens": 4, "label": None, "generation_cap": 64, "hit_generation_cap": False}


class RoutedBatchGenerator:
    def __init__(self):
        self.invocations = 0
        self.batch_sizes = []

    @staticmethod
    def _result(prompt, component):
        clause = prompt.split("CLAUSE: ", 1)[1]
        if component == "parser_selector":
            raw = '{"parser":"binary"}' if clause.startswith("If ") else '{"parser":"atomic"}'
        elif component == "binary":
            raw = '{"operator":"implies","left_clause":"Alice is red","right_clause":"Alice is warm"}'
        elif component == "relation_selector":
            raw = '{"relation":"adjective"}'
        elif component == "adjective":
            predicate = "red" if "red" in clause else "warm"
            raw = json.dumps({"predicate": predicate, "subject": "alice"})
        else:
            raise AssertionError(f"unexpected component {component}")
        return {"raw":raw,"label":None,"latency_ms":2.0,"input_tokens":10,"output_tokens":4,"generation_cap":64,"hit_generation_cap":False,"batch_size":1}

    def __call__(self, prompt, bridge=True, component=None):
        self.invocations += 1
        return self._result(prompt, component)

    def generate_batch(self, prompts, bridge=True, components=None):
        self.invocations += 1
        self.batch_sizes.append(len(prompts))
        results = [self._result(prompt, component) for prompt, component in zip(prompts, components)]
        for result in results:
            result["latency_ms"] = 1.0
            result["batch_size"] = len(prompts)
        return results


class MixedSiblingBatchGenerator(RoutedBatchGenerator):
    @staticmethod
    def _result(prompt, component):
        clause = prompt.split("CLAUSE: ", 1)[1]
        if component == "parser_selector":
            raw = '{"parser":"binary"}' if clause.startswith("If ") else '{"parser":"unary"}' if "not" in clause else '{"parser":"atomic"}'
        elif component == "binary":
            raw = '{"operator":"implies","left_clause":"Alice is red","right_clause":"Alice is not warm"}'
        elif component == "unary":
            raw = '{"operator":"not","child_clause":"Alice is warm"}'
        elif component == "relation_selector":
            raw = '{"relation":"adjective"}'
        elif component == "adjective":
            predicate = "red" if "red" in clause else "warm"
            raw = json.dumps({"predicate": predicate, "subject": "alice"})
        else:
            raise AssertionError(f"unexpected component {component}")
        return {"raw":raw,"label":None,"latency_ms":2.0,"input_tokens":10,"output_tokens":4,"generation_cap":64,"hit_generation_cap":False,"batch_size":1}


class NL2LogicParserTests(unittest.TestCase):
    def test_canonical_clause_normalizes_only_cache_key_formatting(self):
        self.assertEqual(canonical_clause("  Alice   is red .  "), "alice is red")
        self.assertEqual(canonical_clause("Alice is red!"), "alice is red")
        self.assertNotEqual(canonical_clause("Alice is not red."), "alice is red")

    def test_per_example_ast_memo_avoids_reparsing_equivalent_clause(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}', '{"relation":"adjective"}',
            '{"predicate":"red","subject":"alice"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "memo-run", trace.append)
        theory = parser.parse_theory({
            "example_id":"memo-example",
            "premises":["Alice is red.", "  alice   is red !"],
            "question":"ALICE IS RED",
        })
        self.assertEqual(theory["premises"][0], theory["premises"][1])
        self.assertEqual(theory["premises"][0], theory["hypothesis"])
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(parser.usage["model_calls"], 3)
        self.assertEqual(parser.usage["ast_cache_hits"], 2)
        self.assertEqual(parser.usage["unique_normalized_clauses"], 1)

    def test_selector_and_subparser_caches_are_reused_separately(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}', '{"relation":"adjective"}',
            '{"predicate":"red","subject":"alice"}',
        ])
        parser = NL2LogicParser(generator, "proofwriter", 4, "cache-run", lambda _event: None)
        first = parser.parse_sentence("Alice is red.", "cache-example", "premise", "p0", bound=set())
        second = parser.parse_sentence(" alice  is red! ", "cache-example", "premise", "p1", bound={"x"})
        self.assertEqual(first, second)
        self.assertEqual(generator.call_count, 3)
        self.assertEqual(parser.usage["selector_cache_hits"], 2)
        self.assertEqual(parser.usage["subparser_cache_hits"], 1)

    def test_self_referential_unary_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"unary"}',
            '{"operator":"not","child_clause":"Alice is not red ."}',
            '{"relation":"adjective"}',
            '{"predicate":"not_red","subject":"alice"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "cycle-run", trace.append)
        node = parser.parse_sentence("Alice is not red!", "cycle-example", "premise", "p0")
        self.assertEqual(node, {"type": "not", "body": {"type": "predicate", "name": "red", "args": ["alice"]}})
        self.assertEqual(generator.call_count, 4)
        self.assertEqual(trace[-1]["fallback_from"], "unary")
        self.assertIn("self-referential", trace[-1]["fallback_reason"])
        self.assertEqual(trace[-1]["chosen_subparser"], "atomic.adjective")
        self.assertEqual(trace[-1]["status"], "success")

    def test_invented_unary_negation_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"unary"}',
            '{"operator":"not","child_clause":"Alice is not red"}',
            '{"relation":"adjective"}',
            '{"predicate":"red","subject":"alice"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "invented-run", trace.append)
        node = parser.parse_sentence("Alice is red", "invented-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "red", "args": ["alice"]})
        self.assertEqual(trace[-1]["fallback_from"], "unary")
        self.assertIn("invented negation", trace[-1]["fallback_reason"])

    def test_empty_binary_operand_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"binary"}',
            '{"operator":"and","left_clause":"Green, big and red are qualities","right_clause":""}',
            '{"relation":"adjective"}',
            '{"predicate":"green","subject":"alan"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "empty-binary-run", trace.append)
        node = parser.parse_sentence("Green, big and red are qualities which all describe Alan.", "binary-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "green", "args": ["alan"]})
        self.assertEqual(trace[-1]["fallback_from"], "binary")
        self.assertIn("empty right operand", trace[-1]["fallback_reason"])
        self.assertEqual(trace[-1]["child_clauses"], [])

    def test_invalid_binary_operator_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"binary"}',
            '{"operator":"because","left_clause":"Alice is red","right_clause":"Alice is warm"}',
            '{"relation":"adjective"}',
            '{"predicate":"red","subject":"alice"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "bad-operator-run", trace.append)
        node = parser.parse_sentence("Alice is red because Alice is warm", "operator-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "red", "args": ["alice"]})
        self.assertIn("no valid operator", trace[-1]["fallback_reason"])

    def test_unchanged_quantified_scope_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"quantified"}',
            '{"quantifier":"forall","variable":"x","child_clause":"All dogs bark"}',
            '{"relation":"intransitive"}',
            '{"predicate":"bark","subject":"dogs"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "scope-run", trace.append)
        node = parser.parse_sentence("All dogs bark", "scope-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "bark", "args": ["dogs"]})
        self.assertEqual(trace[-1]["fallback_from"], "quantified")
        self.assertIn("unchanged scope", trace[-1]["fallback_reason"])

    def test_invalid_quantified_variable_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"quantified"}',
            '{"quantifier":"forall","variable":"student","child_clause":"student studies"}',
            '{"relation":"intransitive"}',
            '{"predicate":"study","subject":"someone"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "variable-run", trace.append)
        node = parser.parse_sentence("Every student studies", "variable-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "study", "args": ["someone"]})
        self.assertIn("invalid variable", trace[-1]["fallback_reason"])

    def test_empty_quantified_variable_defaults_to_x_without_fallback(self):
        generator = ScriptedGenerator([
            '{"parser":"quantified"}',
            '{"quantifier":"forall","variable":"","child_clause":"x barks"}',
            '{"parser":"atomic"}',
            '{"relation":"intransitive"}',
            '{"predicate":"bark","subject":"x"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "default-variable-run", trace.append)
        node = parser.parse_sentence("All dogs bark", "default-variable-example", "premise", "p0")
        self.assertEqual(node, {
            "type": "forall",
            "variables": ["x"],
            "body": {"type": "predicate", "name": "bark", "args": ["x"]},
        })
        self.assertIsNone(trace[-1]["fallback_from"])

    def test_independent_binary_sibling_stages_use_batches(self):
        generator = RoutedBatchGenerator()
        parser = NL2LogicParser(generator, "proofwriter", 4, "batch-run", lambda _event: None)
        node = parser.parse_sentence("If Alice is red then Alice is warm", "batch-example", "premise", "p0")
        self.assertEqual(node["antecedent"], {"type":"predicate","name":"red","args":["alice"]})
        self.assertEqual(node["consequent"], {"type":"predicate","name":"warm","args":["alice"]})
        self.assertEqual(parser.usage["model_calls"], 8)
        self.assertEqual(parser.usage["model_invocations"], 5)
        self.assertEqual(generator.batch_sizes, [2, 2, 2])

    def test_mixed_sibling_types_fall_back_without_missing_cache_exception(self):
        generator = MixedSiblingBatchGenerator()
        parser = NL2LogicParser(generator, "proofwriter", 4, "mixed-batch-run", lambda _event: None)
        node = parser.parse_sentence("If Alice is red then Alice is not warm", "mixed-example", "premise", "p0")
        self.assertEqual(node["antecedent"], {"type":"predicate","name":"red","args":["alice"]})
        self.assertEqual(node["consequent"], {"type":"not","body":{"type":"predicate","name":"warm","args":["alice"]}})
        self.assertEqual(generator.batch_sizes, [2])
    def test_atomic_equality_returns_typed_equality_node(self):
        trace = []
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"equality"}',
                '{"left":"alice","right":"bob"}',
            ]),
            "folio", 2, "run-equality", trace.append,
        )
        node = parser.parse_sentence("Alice is Bob", "example-eq", "premise", "p0")
        self.assertEqual(node, {"type": "equals", "left": "alice", "right": "bob"})
        self.assertEqual(trace[-1]["chosen_subparser"], "atomic.equality")

    def test_invalid_equality_node_falls_back_through_relation_ladder(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}',
            '{"relation":"equality"}',
            '{"left":"round","right":null}',
            '{"predicate":"round","subject":"alan"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "relation-ladder-run", trace.append)
        node = parser.parse_sentence("Alan is round", "ladder-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "round", "args": ["alan"]})
        self.assertEqual(generator.call_count, 4)
        self.assertEqual(trace[-1]["relation_fallback_from"], "equality")
        self.assertIn("right must be non-empty string", trace[-1]["relation_fallback_reason"])
        self.assertEqual(trace[-1]["chosen_subparser"], "atomic.adjective")
        self.assertEqual(trace[-1]["status"], "success")

    def test_relation_ladder_skips_the_relation_that_already_failed(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}',
            '{"relation":"adjective"}',
            '{"predicate":"sing","subject":""}',
            '{"predicate":"sing","subject":"alice"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "skip-run", trace.append)
        node = parser.parse_sentence("Alice sings", "skip-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "sing", "args": ["alice"]})
        self.assertEqual(generator.call_count, 4)
        self.assertEqual(trace[-1]["relation_fallback_from"], "adjective")
        self.assertEqual(trace[-1]["chosen_subparser"], "atomic.intransitive")

    def test_invalid_proposition_node_falls_back_through_relation_ladder(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}',
            '{"relation":"proposition"}',
            '{"predicate":""}',
            '{"predicate":"cold","subject":"alan"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "proposition-ladder-run", trace.append)
        node = parser.parse_sentence("Alan is cold", "proposition-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "cold", "args": ["alan"]})
        self.assertEqual(trace[-1]["relation_fallback_from"], "proposition")
        self.assertEqual(trace[-1]["chosen_subparser"], "atomic.adjective")

    def test_exhausted_relation_ladder_reraises_the_original_atomic_error(self):
        generator = ScriptedGenerator([
            '{"parser":"atomic"}',
            '{"relation":"equality"}',
            '{"left":"round","right":null}',
            '{"predicate":"","subject":""}',
            '{"predicate":"","subject":""}',
            '{"predicate":"","subject":""}',
            '{"predicate":"","subject":""}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "exhausted-run", trace.append)
        with self.assertRaises(BridgeError) as caught:
            parser.parse_sentence("Alan is round", "exhausted-example", "premise", "p0")
        self.assertEqual(caught.exception.status, "schema_invalid")
        self.assertIn("right must be non-empty string", caught.exception.detail)
        self.assertEqual(generator.call_count, 7)
        rejected = trace[-1]["subparser_raw_output"]["relation_fallback"]["rejected"]
        self.assertEqual([entry["relation"] for entry in rejected], ["adjective", "intransitive", "transitive", "ditransitive"])
        self.assertIsNone(trace[-1]["subparser_raw_output"]["relation_fallback"]["to"])

    def test_predicate_symbol_strips_only_leading_copulas(self):
        for raw, expected in (
            ("is_green", "green"), ("Is Green", "green"), ("are_kind", "kind"),
            ("was_big", "big"), ("were_big", "big"), ("be_nice", "nice"), ("been_nice", "nice"),
        ):
            self.assertEqual(_predicate_symbol(raw), expected)

    def test_predicate_symbol_leaves_non_copula_names_untouched(self):
        for raw in ("green", "isolated", "island", "issue", "not_round", "is", "is_2fast"):
            self.assertEqual(_predicate_symbol(raw), _symbol(raw))

    def test_copula_predicate_is_normalized_while_subject_term_is_not(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"adjective"}',
                '{"predicate":"is green","subject":"Is Anne"}',
            ]),
            "proofwriter", 4, "copula-run", lambda _event: None,
        )
        node = parser.parse_sentence("Is Anne is green", "copula-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "green", "args": ["is_anne"]})

    def test_copula_variants_collapse_onto_one_predicate_symbol(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}', '{"relation":"adjective"}',
                '{"predicate":"is furry","subject":"gary"}',
                '{"parser":"atomic"}', '{"relation":"adjective"}',
                '{"predicate":"furry","subject":"anne"}',
            ]),
            "proofwriter", 4, "collapse-run", lambda _event: None,
        )
        first = parser.parse_sentence("Gary is furry", "collapse-example", "premise", "p0")
        second = parser.parse_sentence("Anne is furry", "collapse-example", "premise", "p1")
        self.assertEqual(first["name"], second["name"])
        self.assertEqual(first["name"], "furry")

    def test_equality_operands_are_not_copula_stripped(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"equality"}',
                '{"left":"is_anne","right":"is_bob"}',
            ]),
            "folio", 2, "equality-term-run", lambda _event: None,
        )
        node = parser.parse_sentence("Is Anne is Is Bob", "equality-term-example", "premise", "p0")
        self.assertEqual(node, {"type": "equals", "left": "is_anne", "right": "is_bob"})

    def test_apply_name_negation_lifts_only_valid_not_prefixed_predicates(self):
        lifted, name = _apply_name_negation({"type": "predicate", "name": "not_blue", "args": ["lion"]})
        self.assertEqual(lifted, {"type": "not", "body": {"type": "predicate", "name": "blue", "args": ["lion"]}})
        self.assertEqual(name, "not_blue")
        for node in (
            {"type": "predicate", "name": "blue", "args": ["lion"]},
            {"type": "predicate", "name": "not", "args": []},
            {"type": "predicate", "name": "nothing", "args": []},
            {"type": "predicate", "name": "cannot_fly", "args": ["bat"]},
            {"type": "predicate", "name": "not_2fast", "args": ["x"]},
            {"type": "equals", "left": "not_a", "right": "b"},
        ):
            self.assertEqual(_apply_name_negation(node), (node, None))

    def test_negated_predicate_name_becomes_a_not_node(self):
        trace = []
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"adjective"}',
                '{"predicate":"not blue","subject":"lion"}',
            ]),
            "proofwriter", 4, "negation-run", trace.append,
        )
        node = parser.parse_sentence("The lion is not blue", "negation-example", "premise", "p0")
        self.assertEqual(node, {"type": "not", "body": {"type": "predicate", "name": "blue", "args": ["lion"]}})
        self.assertEqual(trace[-1]["negation_lifted_from_name"], ["not_blue"])

    def test_copula_and_negation_are_lifted_together(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"adjective"}',
                '{"predicate":"is not round","subject":"alan"}',
            ]),
            "proofwriter", 4, "copula-negation-run", lambda _event: None,
        )
        node = parser.parse_sentence("Alan is not round", "copula-negation-example", "premise", "p0")
        self.assertEqual(node, {"type": "not", "body": {"type": "predicate", "name": "round", "args": ["alan"]}})

    def test_negated_and_plain_predicates_share_one_symbol(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}', '{"relation":"adjective"}',
                '{"predicate":"is not blue","subject":"lion"}',
                '{"parser":"atomic"}', '{"relation":"adjective"}',
                '{"predicate":"blue","subject":"bob"}',
            ]),
            "proofwriter", 4, "shared-symbol-run", lambda _event: None,
        )
        negated = parser.parse_sentence("The lion is not blue", "shared-example", "premise", "p0")
        plain = parser.parse_sentence("Bob is blue", "shared-example", "premise", "p1")
        self.assertEqual(negated["body"]["name"], plain["name"])

    def test_negation_lifting_leaves_untraced_when_not_applicable(self):
        trace = []
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"adjective"}',
                '{"predicate":"blue","subject":"lion"}',
            ]),
            "proofwriter", 4, "no-negation-run", trace.append,
        )
        parser.parse_sentence("The lion is blue", "no-negation-example", "premise", "p0")
        self.assertEqual(trace[-1]["negation_lifted_from_name"], [])

    def test_uncovered_content_flags_only_a_dropped_consequent(self):
        clause = "If Bob is young and Bob is kind then Bob is furry."
        self.assertEqual(_uncovered_content(clause, ["Bob is young", "Bob is kind"]), ["furry"])
        self.assertEqual(_uncovered_content(clause, ["Bob is young and Bob is kind", "Bob is furry"]), [])

    def test_uncovered_content_tolerates_spelling_and_pronoun_resolution(self):
        self.assertEqual(_uncovered_content(
            "Rough people who are red in color and big are usally round in shape.",
            ["Rough people who are red in color and big", "are usually round in shape"]), [])
        self.assertEqual(_uncovered_content(
            "If someone visits the mouse and they see it then they eat the squirrel.",
            ["someone visits the mouse and someone sees the mouse", "someone eats the squirrel"]), [])

    def test_binary_dropping_the_consequent_falls_back_to_relation_parsing(self):
        generator = ScriptedGenerator([
            '{"parser":"binary"}',
            '{"operator":"and","left_clause":"Bob is young","right_clause":"Bob is kind"}',
            '{"relation":"proposition"}',
            '{"predicate":"bob_young_kind_implies_furry"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "coverage-run", trace.append)
        node = parser.parse_sentence("If Bob is young and Bob is kind then Bob is furry.", "coverage-example", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "bob_young_kind_implies_furry", "args": []})
        self.assertEqual(trace[-1]["fallback_from"], "binary")
        self.assertIn("dropped clause content", trace[-1]["fallback_reason"])
        self.assertIn("furry", trace[-1]["fallback_reason"])
        self.assertEqual(trace[-1]["child_clauses"], [])

    def test_binary_covering_the_whole_clause_still_recurses(self):
        generator = ScriptedGenerator([
            '{"parser":"binary"}',
            '{"operator":"implies","left_clause":"Bob is young","right_clause":"Bob is furry"}',
            '{"parser":"atomic"}', '{"relation":"adjective"}', '{"predicate":"young","subject":"bob"}',
            '{"parser":"atomic"}', '{"relation":"adjective"}', '{"predicate":"furry","subject":"bob"}',
        ])
        trace = []
        parser = NL2LogicParser(generator, "proofwriter", 4, "covered-run", trace.append)
        node = parser.parse_sentence("If Bob is young then Bob is furry.", "covered-example", "premise", "p0")
        self.assertEqual(node, {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "young", "args": ["bob"]},
            "consequent": {"type": "predicate", "name": "furry", "args": ["bob"]},
        })
        self.assertIsNone(trace[-1]["fallback_from"])

    def test_zero_arity_atomic_returns_proposition_node(self):
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"proposition"}',
                '{"predicate":"available"}',
            ]),
            "folio", 2, "run-proposition", lambda _event: None,
        )
        node = parser.parse_sentence("It is available", "example-prop", "premise", "p0")
        self.assertEqual(node, {"type": "predicate", "name": "available", "args": []})

    def test_binary_parse_builds_ast_and_complete_recursive_trace(self):
        outputs = [
            '{"parser":"binary"}',
            '{"operator":"implies","left_clause":"Alice is red","right_clause":"Alice is warm"}',
            '{"parser":"atomic"}', '{"relation":"adjective"}', '{"predicate":"red","subject":"alice"}',
            '{"parser":"atomic"}', '{"relation":"adjective"}', '{"predicate":"warm","subject":"alice"}',
        ]
        trace = []
        parser = NL2LogicParser(ScriptedGenerator(outputs), "proofwriter", 2, "run-1", trace.append)
        node = parser.parse_sentence("If Alice is red then Alice is warm", "example-1", "premise", "p0")
        self.assertEqual(node, {
            "type": "implies",
            "antecedent": {"type": "predicate", "name": "red", "args": ["alice"]},
            "consequent": {"type": "predicate", "name": "warm", "args": ["alice"]},
        })
        self.assertEqual([event["recursive_path"] for event in trace], ["0.0", "0.1", "0"])
        root = trace[-1]
        self.assertEqual(root["selector_decision"], "binary")
        self.assertEqual(root["chosen_subparser"], "binary")
        self.assertEqual([child["clause"] for child in root["child_clauses"]], ["Alice is red", "Alice is warm"])
        self.assertEqual(root["returned_ast_node"]["type"], "implies")
        self.assertEqual(root["validation_status"], "valid")
        self.assertEqual(root["status"], "success")
        for required in ("step_id", "parent_step_id", "selector_raw_output", "failure_reason", "latency_ms", "token_count"):
            self.assertIn(required, root)

    def test_trace_preserves_component_level_generation_profile(self):
        trace = []
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"atomic"}',
                '{"relation":"adjective"}',
                '{"predicate":"red","subject":"alice"}',
            ]),
            "proofwriter", 4, "profile-run", trace.append,
        )
        parser.parse_sentence("Alice is red", "profile-example", "premise", "p0")
        calls = trace[-1]["model_calls"]
        self.assertEqual([call["component"] for call in calls], ["parser_selector", "relation_selector", "adjective"])
        self.assertEqual([call["component_group"] for call in calls], ["selector", "selector", "atomic"])
        self.assertTrue(all(call["generated_token_count"] == 4 for call in calls))
        self.assertTrue(all(call["generation_cap"] == 64 for call in calls))
        self.assertTrue(all(call["hit_generation_cap"] is False for call in calls))
        self.assertEqual(parser.usage["model_calls"], 3)
        self.assertEqual(parser.usage["selector_ms"], 3.0)
        self.assertEqual(parser.usage["subparser_ms"], 1.5)
        self.assertEqual(parser.usage["generated_token_count"], 12)

    def test_malformed_selector_emits_failed_trace(self):
        trace = []
        parser = NL2LogicParser(ScriptedGenerator(["not-json"]), "folio", 4, "run-2", trace.append)
        with self.assertRaises(BridgeError):
            parser.parse_sentence("Alice is red", "example-2", "hypothesis", "h")
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["status"], "failed")
        self.assertEqual(trace[0]["validation_status"], "invalid")
        self.assertIn("malformed", trace[0]["failure_reason"])

    def test_malformed_selector_trace_preserves_exact_raw_output(self):
        raw = '{"parser":"atomic"}\n{"parser":"binary"}'
        trace = []
        parser = NL2LogicParser(ScriptedGenerator([raw]), "folio", 4, "raw-run", trace.append)
        with self.assertRaises(BridgeError):
            parser.parse_sentence("Alice is red", "raw-example", "hypothesis", "h")
        self.assertEqual(
            trace[0]["failed_raw_output"],
            {"component": "parser_selector", "raw": raw},
        )

    def test_ancestor_cycle_still_emits_partial_trace_and_stops_recursion(self):
        trace = []
        parser = NL2LogicParser(
            ScriptedGenerator([
                '{"parser":"binary"}',
                '{"operator":"and","left_clause":"Alice is red","right_clause":"Alice is blue"}',
                '{"parser":"binary"}',
                '{"operator":"and","left_clause":"Alice is red and Alice is blue","right_clause":"Bob is tall"}',
            ]),
            "proofwriter", 2, "run-3", trace.append,
        )
        with self.assertRaises(BridgeError) as caught:
            parser.parse_sentence("Alice is red and Alice is blue", "example-3", "premise", "p0")
        self.assertEqual(caught.exception.status, "parser_cycle")
        self.assertEqual(trace[-1]["status"], "partial")
        self.assertIn("recursive clause cycle", trace[0]["failure_reason"])


if __name__ == "__main__":
    unittest.main()
