"""Tests for src/folio/v2/decomposer.py using a stub runner (no GPU/model load)."""
from __future__ import annotations

from dataclasses import dataclass

from src.folio.v2.decomposer import decompose_sentence


@dataclass
class StubGeneration:
    raw_text: str
    latency_ms: float = 1.0


class StubRunner:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    def generate_text(self, prompt: str, max_new_tokens: int = 500) -> StubGeneration:
        self.calls.append(prompt)
        return StubGeneration(raw_text=self._outputs.pop(0))


VALID_DECOMPOSITION = (
    '{"entities": [], "atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}, '
    '{"id": "A2", "predicate": "bark", "args": ["x"]}], '
    '"nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}, '
    '{"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}], "root": "N2"}'
)


def test_decompose_sentence_valid_json_is_shape_valid():
    runner = StubRunner([VALID_DECOMPOSITION])
    attempt = decompose_sentence(runner, "All dogs bark.")
    assert attempt.shape_valid is True
    assert attempt.decomposition["root"] == "N2"
    assert attempt.repaired is False


def test_decompose_sentence_strips_markdown_fences():
    runner = StubRunner([f"```json\n{VALID_DECOMPOSITION}\n```"])
    attempt = decompose_sentence(runner, "All dogs bark.")
    assert attempt.shape_valid is True


def test_decompose_sentence_not_json_fails_as_invalid_json():
    runner = StubRunner(["Sure! Here's an explanation instead of JSON."])
    attempt = decompose_sentence(runner, "All dogs bark.")
    assert attempt.shape_valid is False
    assert attempt.error_reason == "invalid_json"


def test_decompose_sentence_missing_root_fails_shape_check():
    runner = StubRunner(['{"atoms": [], "nodes": []}'])
    attempt = decompose_sentence(runner, "All dogs bark.")
    assert attempt.shape_valid is False
    assert attempt.error_reason == "missing_field"


def test_decompose_sentence_atoms_not_a_list_fails_shape_check():
    runner = StubRunner(['{"atoms": "not-a-list", "nodes": [], "root": "A1"}'])
    attempt = decompose_sentence(runner, "All dogs bark.")
    assert attempt.shape_valid is False
    assert attempt.error_reason == "malformed_node"


def test_decompose_sentence_no_auto_repair_second_call():
    runner = StubRunner(["not json at all"])
    decompose_sentence(runner, "All dogs bark.")
    assert len(runner.calls) == 1


def test_decompose_sentence_prompt_never_contains_gold_fol_markers():
    runner = StubRunner([VALID_DECOMPOSITION])
    decompose_sentence(runner, "All dogs bark.")
    prompt = runner.calls[0]
    assert "gold" not in prompt.lower()
    assert "label" not in prompt.lower()
