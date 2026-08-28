import json
import unittest
from pathlib import Path

from final_experiment import core, token_audit, training, training_data


class FrozenBridgeSingleAdapterTests(unittest.TestCase):
    def test_conditions_route_two_model_states_to_two_inference_paths(self):
        route = getattr(core, "adapter_name_for_condition", None)
        uses_bridge = getattr(core, "uses_frozen_bridge", None)
        self.assertTrue(callable(route), "condition adapter routing API is missing")
        self.assertTrue(callable(uses_bridge), "condition bridge routing API is missing")
        for dataset in ("proofwriter", "folio"):
            self.assertIsNone(route(dataset, 1))
            self.assertIsNone(route(dataset, 2))
            self.assertEqual(route(dataset, 3), f"{dataset}_direct_qwen3_1_7b_lora")
            self.assertEqual(route(dataset, 4), f"{dataset}_direct_qwen3_1_7b_lora")
        self.assertEqual([uses_bridge(c) for c in (1, 2, 3, 4)], [False, True, False, True])

    def test_only_dataset_direct_adapters_are_active(self):
        self.assertEqual(core.ADAPTER_NAMES, {
            "proofwriter": "proofwriter_direct_qwen3_1_7b_lora",
            "folio": "folio_direct_qwen3_1_7b_lora",
        })

    def test_resolved_configs_declare_the_bridge_frozen_and_untrainable(self):
        for dataset in ("proofwriter", "folio"):
            bridge = core.resolved_config(dataset).get("bridge")
            self.assertIsNotNone(bridge, "resolved config is missing the frozen bridge contract")
            self.assertEqual(bridge["id"], core.FROZEN_BRIDGE_ID)
            self.assertTrue(bridge["frozen"])
            self.assertFalse(bridge["adapter_training"])
            self.assertEqual(bridge["pipeline"], [
                "nl2logic_parser", "fol_ast", "two_pass_compiler", "z3",
            ])

    def test_programmatic_bridge_training_is_rejected_before_model_loading(self):
        validate = getattr(training, "validate_training_task", None)
        self.assertTrue(callable(validate), "training task guard is missing")
        self.assertEqual(validate("direct"), "direct")
        with self.assertRaisesRegex(ValueError, "frozen.*never fine-tuned"):
            validate("bridge")

    def test_training_cli_rejects_bridge_task(self):
        with self.assertRaises(SystemExit):
            training.build_parser().parse_args([
                "--dataset", "proofwriter", "--task", "bridge", "--mode", "main",
            ])

    def test_active_audit_records_are_one_direct_record_per_row(self):
        build = getattr(token_audit, "active_training_records", None)
        self.assertTrue(callable(build), "active direct-record audit builder is missing")
        rows = [
            {"context": "A.", "question": "Q?", "label": "True"},
            {"context": "B.", "question": "R?", "label": "False"},
        ]
        records = build(rows)
        self.assertEqual(len(records), 2)
        self.assertEqual([record["target"] for record in records], ["True", "False"])

    def test_smallest_lossless_standard_limit(self):
        choose = getattr(training_data, "smallest_lossless_standard_limit", None)
        self.assertTrue(callable(choose), "standard lossless limit selector is missing")
        self.assertEqual(choose(292), 292)
        self.assertEqual(choose(293), 320)
        self.assertEqual(choose(321), 384)

    def test_configured_limit_must_equal_smallest_lossless_limit(self):
        validate = getattr(training_data, "validate_configured_standard_limit", None)
        self.assertTrue(callable(validate), "configured sequence-limit audit guard is missing")
        self.assertEqual(validate(292, 292), 0)
        self.assertEqual(validate(320, 293), 0)
        with self.assertRaisesRegex(ValueError, "truncate"):
            validate(291, 292)
        with self.assertRaisesRegex(ValueError, "smallest lossless"):
            validate(320, 292)

    def test_existing_direct_audit_proves_292_is_lossless(self):
        report_path = Path(core.ROOT) / "data" / "processed" / "token_length_audit_qwen3_1_7b_main.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        maxima = [
            report["datasets"][dataset]["groups"][group]["max"]
            for dataset in ("proofwriter", "folio")
            for group in ("direct_train", "direct_validation")
        ]
        self.assertEqual(max(maxima), 292)
        self.assertTrue(all(maximum <= 292 for maximum in maxima))


if __name__ == "__main__":
    unittest.main()
