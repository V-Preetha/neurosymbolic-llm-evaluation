import json
import contextlib
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from final_experiment.data import (
    create_mode_manifests,
    load_manifest_split,
    load_training_validation,
    materialize_manifest_rows,
    mode_sample_sizes,
    training_pool_policy,
    training_validation_size,
    save_mode_manifests,
    validate_splits,
    write_frozen_manifest,
    normalize_row,
)
from final_experiment.core import ADAPTER_NAMES,write_jsonl
from final_experiment.core import resolved_config
from final_experiment.training import build_parser, training_schedule_kwargs


def rows(split: str, count_per_stratum: int = 10):
    output = []
    index = 0
    for label in ("True", "False", "Unknown"):
        for depth in (0, 1):
            for _ in range(count_per_stratum):
                output.append({
                    "example_id": f"{split}-{index}",
                    "world_id": f"{split}-world-{index}",
                    "split": split,
                    "label": label,
                    "QDep": depth,
                })
                index += 1
    return output


class SamplingManifestTests(unittest.TestCase):
    def setUp(self):
        self.parts = {
            "train": rows("train"),
            "validation": rows("validation", 2),
            "test": rows("test", 3),
        }

    def test_proofwriter_mode_manifest_has_requested_sizes_and_strata(self):
        manifests = create_mode_manifests(
            self.parts, "proofwriter", "smoke", train_size=24,
            evaluation_size=12, seed=42,
        )
        self.assertEqual(manifests["train"]["size"], 24)
        self.assertEqual(manifests["validation"]["size"], 12)
        self.assertEqual(manifests["test"]["size"], 12)
        selected = materialize_manifest_rows(self.parts["train"], manifests["train"])
        strata = Counter((row["label"], row["QDep"]) for row in selected)
        self.assertEqual(set(strata.values()), {4})

    def test_manifest_generation_is_reproducible(self):
        first = create_mode_manifests(self.parts, "proofwriter", "pilot", 30, 15, 17)
        second = create_mode_manifests(self.parts, "proofwriter", "pilot", 30, 15, 17)
        self.assertEqual(first, second)

    def test_main_training_validation_is_frozen_stratified_and_auditable(self):
        manifests = create_mode_manifests(
            self.parts, "proofwriter", "main", 24, 12, 42,
            validation_size=6,
        )
        subset = manifests["training_validation"]
        self.assertEqual(subset["size"], 6)
        self.assertEqual(subset["seed"], 42)
        self.assertEqual(subset["stratify_by"], ["label", "QDep"])
        self.assertEqual(set(subset["strata"].values()), {1})
        self.assertEqual(manifests["validation"]["size"], len(self.parts["validation"]))
        repeated = create_mode_manifests(
            self.parts, "proofwriter", "main", 24, 12, 42,
            validation_size=6,
        )
        self.assertEqual(subset, repeated["training_validation"])

    def test_controlled_language_pool_applies_only_to_train_and_development(self):
        parts = {key: [dict(row) for row in value] for key, value in self.parts.items()}
        for split in ("train", "validation"):
            by_stratum = {}
            for row in parts[split]:
                key = (row["label"], row["QDep"])
                index = by_stratum.get(key, 0)
                row["config"] = "NatLang" if index == 0 else "depth-3"
                by_stratum[key] = index + 1
        for row in parts["test"]:
            row["config"] = "NatLang"
        manifests = create_mode_manifests(
            parts, "proofwriter", "main", 24, len(parts["test"]), 42,
            validation_size=6, training_pool="controlled_language",
        )
        selected_train = materialize_manifest_rows(parts["train"], manifests["train"])
        selected_validation = materialize_manifest_rows(parts["validation"], manifests["training_validation"])
        selected_test = materialize_manifest_rows(parts["test"], manifests["test"])
        self.assertFalse(any("NatLang" in row["config"] for row in selected_train))
        self.assertFalse(any("NatLang" in row["config"] for row in selected_validation))
        self.assertTrue(all("NatLang" in row["config"] for row in selected_test))
        self.assertEqual(manifests["train"]["sampling_pool"], "controlled_language")
        self.assertEqual(manifests["training_validation"]["sampling_pool"], "controlled_language")
        self.assertNotIn("sampling_pool", manifests["test"])

    def test_full_training_manifest_uses_entire_training_partition(self):
        manifests = create_mode_manifests(self.parts, "proofwriter", "full", None, None, 42)
        self.assertEqual(manifests["train"]["size"], len(self.parts["train"]))
        self.assertEqual(set(manifests["train"]["ids"]), {r["example_id"] for r in self.parts["train"]})

    def test_manifest_materialization_rejects_duplicate_or_missing_ids(self):
        duplicate = {"ids": ["train-0", "train-0"]}
        missing = {"ids": ["train-0", "not-present"]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            materialize_manifest_rows(self.parts["train"], duplicate)
        with self.assertRaisesRegex(ValueError, "missing"):
            materialize_manifest_rows(self.parts["train"], missing)

    def test_frozen_manifest_is_reused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_smoke.json"
            original = {"dataset": "proofwriter", "split": "train", "mode": "smoke", "seed": 42, "ids": ["train-0"], "size": 1}
            changed = original | {"ids": ["train-1"], "size": 1}
            write_frozen_manifest(path, original, {"train-0", "train-1"})
            reused = write_frozen_manifest(path, changed, {"train-0", "train-1"})
            self.assertEqual(reused["ids"], ["train-0"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["ids"], ["train-0"])

    def test_frozen_manifest_rejects_configured_size_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_main.json"
            original = {"dataset":"proofwriter","split":"train","mode":"main","seed":42,"ids":["train-0","train-1"],"size":2}
            requested = original | {"ids":["train-0"],"size":1}
            write_frozen_manifest(path, original, {"train-0","train-1"})
            with self.assertRaisesRegex(ValueError, "requested size"):
                write_frozen_manifest(path, requested, {"train-0","train-1"})

    def test_manifest_ids_are_disjoint_across_splits(self):
        manifests = create_mode_manifests(self.parts, "proofwriter", "smoke", 24, 12, 42)
        train_ids = set(manifests["train"]["ids"])
        validation_ids = set(manifests["validation"]["ids"])
        test_ids = set(manifests["test"]["ids"])
        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)

    def test_split_validation_rejects_world_leakage(self):
        parts = {
            "train": [{"example_id": "a", "world_id": "shared", "label": "True"}],
            "validation": [{"example_id": "b", "world_id": "shared", "label": "False"}],
            "test": [{"example_id": "c", "world_id": "held-out", "label": "Unknown"}],
        }
        with self.assertRaisesRegex(ValueError, "World leakage"):
            validate_splits(parts)

    def test_proofwriter_world_identity_uses_theory_not_reused_display_id(self):
        base = {
            "id": "reused-1", "maxD": 1, "NFact": 1, "NRule": 0,
            "question": "Alice is red.", "answer": "True", "QDep": 0,
            "allProofs": "fact",
        }
        train = normalize_row("proofwriter", base | {"theory": "Alice is red."}, "train", 0)
        validation = normalize_row("proofwriter", base | {"theory": "Bob is blue."}, "validation", 0)
        duplicate = normalize_row("proofwriter", base | {"theory": "Alice is red."}, "test", 0)
        self.assertNotEqual(train["world_id"], validation["world_id"])
        self.assertEqual(train["world_id"], duplicate["world_id"])

    def test_saved_mode_manifests_drive_exact_training_and_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split, split_rows in self.parts.items():
                write_jsonl(base / f"{split}.jsonl", split_rows)
            saved = save_mode_manifests(base, self.parts, "proofwriter", "smoke", 24, 12, 42)
            train_rows, train_manifest = load_manifest_split(base, "train", "smoke")
            validation_rows, validation_manifest = load_manifest_split(base, "validation", "smoke")
            test_rows, test_manifest = load_manifest_split(base, "test", "smoke")
            self.assertEqual([r["example_id"] for r in train_rows], saved["train"]["ids"])
            self.assertEqual([r["example_id"] for r in test_rows], saved["test"]["ids"])
            self.assertEqual(len(validation_rows), len(self.parts["validation"]))
            self.assertEqual(train_manifest["size"], 24)
            self.assertEqual(validation_manifest["mode"], "fixed")
            self.assertEqual(test_manifest["size"], 12)
            self.assertTrue((base / "manifests" / "train_smoke.json").exists())
            self.assertTrue((base / "manifests" / "validation.json").exists())
            self.assertTrue((base / "manifests" / "test_smoke.json").exists())

    def test_training_validation_loader_uses_subset_and_preserves_full_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split, split_rows in self.parts.items():
                write_jsonl(base / f"{split}.jsonl", split_rows)
            save_mode_manifests(base, self.parts, "proofwriter", "main", 24, 12, 42, validation_size=6)
            subset_rows, subset_manifest, full_manifest = load_training_validation(base, "main")
            full_rows, loaded_full = load_manifest_split(base, "validation", "main")
            self.assertEqual(len(subset_rows), 6)
            self.assertEqual(subset_manifest["size"], 6)
            self.assertEqual(len(full_rows), len(self.parts["validation"]))
            self.assertEqual(full_manifest, loaded_full)
            self.assertTrue((base / "manifests" / "validation_train_main.json").exists())

    def test_loading_unprepared_mode_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Prepare the dataset"):
                load_manifest_split(Path(directory), "train", "main")

    def test_configured_mode_sizes_control_training_and_evaluation(self):
        proofwriter = resolved_config("proofwriter")
        folio = resolved_config("folio")
        self.assertEqual(mode_sample_sizes(proofwriter, "smoke"), (200, 30))
        self.assertEqual(mode_sample_sizes(proofwriter, "pilot"), (2000, 200))
        self.assertEqual(mode_sample_sizes(proofwriter, "main"), (2000, 2000))
        self.assertEqual(mode_sample_sizes(proofwriter, "full"), (None, None))
        self.assertEqual(mode_sample_sizes(folio, "smoke"), (200, 30))
        self.assertEqual(mode_sample_sizes(folio, "pilot"), (500, 200))
        self.assertEqual(mode_sample_sizes(folio, "main"), (None, 2000))
        self.assertEqual(mode_sample_sizes(folio, "full"), (None, None))
        self.assertEqual(training_validation_size(proofwriter, "main"), 2000)
        self.assertEqual(training_pool_policy(proofwriter, "main"), "controlled_language")
        self.assertIsNone(training_validation_size(folio, "main"))
        self.assertIsNone(training_pool_policy(folio, "main"))

    def test_final_profile_freezes_qwen3_1_7b_and_proofwriter_three_epochs(self):
        proofwriter = resolved_config("proofwriter")
        folio = resolved_config("folio")
        for config in (proofwriter, folio):
            self.assertEqual(config["model"]["id"], "Qwen/Qwen3-1.7B")
            self.assertEqual(config["model"]["revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
            self.assertEqual(config["model"]["max_sequence_length"], 292)
            self.assertTrue(config["training"]["dynamic_padding"])
            self.assertTrue(config["training"]["answer_only_loss"])
        self.assertEqual(proofwriter["training"]["epochs"], 3)
        self.assertTrue(all("qwen3_1_7b" in name for name in ADAPTER_NAMES.values()))

    def test_training_cli_requires_mode(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--dataset", "proofwriter", "--task", "direct"])
        args = parser.parse_args(["--dataset", "proofwriter", "--task", "direct", "--mode", "main"])
        self.assertEqual(args.mode, "main")

    def test_training_uses_epoch_evaluation_and_checkpointing(self):
        schedule = training_schedule_kwargs()
        self.assertEqual(schedule["eval_strategy"], "epoch")
        self.assertEqual(schedule["save_strategy"], "epoch")
        self.assertTrue(schedule["load_best_model_at_end"])
        self.assertEqual(schedule["metric_for_best_model"], "eval_loss")

    def test_training_cli_optionally_requests_one_full_validation(self):
        args = build_parser().parse_args([
            "--dataset", "proofwriter", "--task", "direct", "--mode", "main",
            "--final-full-validation",
        ])
        self.assertTrue(args.final_full_validation)


if __name__ == "__main__":
    unittest.main()
