import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

from final_experiment.core import resolved_config
from final_experiment.modeling import ModelRunner, bridge_component_group, bridge_generation_cap, extract_direct_label, validate_model_artifact, write_model_identity


class FakeBatch(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, _device):
        return self


class FakeTokenizer:
    eos_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        self.padding_requested = kwargs.get("padding")
        return FakeBatch(input_ids=torch.tensor([[9,10],[8,7]]),attention_mask=torch.tensor([[1,1],[1,1]]))

    def decode(self, tokens, **_kwargs):
        values=[int(value) for value in tokens]
        return "True" if 1 in values else "False"


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        self.kwargs=kwargs
        return torch.tensor([[9,10,1,0,0],[8,7,2,3,0]])


class ModelCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = resolved_config("proofwriter")

    def test_matching_1_7b_identity_is_accepted_from_checkpoint_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "latest" / "checkpoint-12"
            checkpoint.mkdir(parents=True)
            write_model_identity(root, self.config)
            identity = validate_model_artifact(checkpoint, self.config)
            self.assertEqual(identity["model_id"], "Qwen/Qwen3-1.7B")

    def test_qwen3_4b_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_model_identity(root, {
                "model": {"id": "Qwen/Qwen3-4B", "revision": "old-4b-revision"}
            })
            with self.assertRaisesRegex(ValueError, "Incompatible model artifact"):
                validate_model_artifact(root, self.config)

    def test_qwen3_8b_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_model_identity(root, {
                "model": {"id": "Qwen/Qwen3-8B", "revision": "old-8b-revision"}
            })
            with self.assertRaisesRegex(ValueError, "Incompatible model artifact"):
                validate_model_artifact(root, self.config)

    def test_artifact_without_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing base-model identity"):
                validate_model_artifact(Path(directory), self.config)

    def test_bridge_components_route_to_four_generation_cap_groups(self):
        expected = {
            "parser_selector": "selector", "relation_selector": "selector",
            "proposition": "atomic", "equality": "atomic", "adjective": "atomic",
            "intransitive": "atomic", "transitive": "atomic", "ditransitive": "atomic",
            "quantified": "quantified", "binary": "logical", "unary": "logical",
        }
        self.assertEqual({name: bridge_component_group(name) for name in expected}, expected)

    def test_bridge_generation_cap_uses_component_specific_config(self):
        config = {"model": {"max_new_tokens_bridge": 1024}, "bridge": {"generation_caps": {
            "selector": 48, "atomic": 72, "quantified": 96, "logical": 128,
        }}}
        self.assertEqual(bridge_generation_cap(config, "parser_selector"), 48)
        self.assertEqual(bridge_generation_cap(config, "adjective"), 72)
        self.assertEqual(bridge_generation_cap(config, "quantified"), 96)
        self.assertEqual(bridge_generation_cap(config, "binary"), 128)
        self.assertEqual(bridge_generation_cap(config, None), 1024)

    def test_batched_generation_preserves_greedy_cached_inference_contract(self):
        runner=ModelRunner.__new__(ModelRunner);runner.config={"model":{"max_new_tokens_bridge":1024,"max_new_tokens_direct":8,"max_sequence_length":292},"bridge":{"generation_caps":{"selector":64}}};runner.tokenizer=FakeTokenizer();runner.model=FakeModel();runner.torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),inference_mode=torch.inference_mode)
        results=runner.generate_batch(["first","second"],True,["parser_selector","parser_selector"])
        self.assertEqual([result["output_tokens"] for result in results],[2,3])
        self.assertEqual([result["raw"] for result in results],["True","False"])
        self.assertEqual([result["batch_size"] for result in results],[2,2])
        self.assertTrue(runner.tokenizer.padding_requested)
        self.assertEqual(runner.tokenizer.padding_side,"right")
        self.assertFalse(runner.model.kwargs["do_sample"])
        self.assertTrue(runner.model.kwargs["use_cache"])
        self.assertEqual(runner.model.kwargs["max_new_tokens"],64)

    def test_verbose_direct_output_with_one_standalone_label_is_accepted(self):
        self.assertEqual(
            extract_direct_label("The conclusion is not established. Answer: Unknown"),
            "Unknown",
        )

    def test_verbose_direct_output_with_conflicting_labels_is_rejected(self):
        self.assertIsNone(
            extract_direct_label("It could be True or False. Answer: Unknown")
        )

    def test_resolved_resource_caps_reach_model_generation(self):
        runner=ModelRunner.__new__(ModelRunner);runner.config=resolved_config("proofwriter");runner.tokenizer=FakeTokenizer();runner.model=FakeModel();runner.torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),inference_mode=torch.inference_mode)
        runner.generate_batch(["first","second"],False)
        self.assertEqual(runner.model.kwargs["max_new_tokens"],256)
        runner.generate_batch(["first","second"],True,["parser_selector","parser_selector"])
        self.assertEqual(runner.model.kwargs["max_new_tokens"],256)


if __name__ == "__main__":
    unittest.main()
