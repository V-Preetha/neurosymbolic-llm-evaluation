"""Tests for src/folio/translator.py using a stub runner (no GPU/model load).

These test the translator's parsing/validation/aggregation logic in
isolation from Qwen3-8B itself -- the actual model's translation quality is
measured separately by the frozen 30-example prototype run (see
reports/folio_translation_prototype/).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.folio.io_guard import FolioAllowedInput
from src.folio.translator import translate_sentence, translate_theory


@dataclass
class StubGeneration:
    raw_text: str
    latency_ms: float = 1.23


class StubRunner:
    """Returns a fixed, pre-scripted output per call, in call order."""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    def generate_text(self, prompt: str, max_new_tokens: int = 400) -> StubGeneration:
        self.calls.append(prompt)
        return StubGeneration(raw_text=self._outputs.pop(0))


VALID_FORALL_JSON = (
    '{"type": "forall", "variables": ["x"], '
    '"body": {"type": "implies", "antecedent": {"type": "predicate", "name": "dog", "args": ["x"]}, '
    '"consequent": {"type": "predicate", "name": "bark", "args": ["x"]}}}'
)


def test_translate_sentence_valid_json_is_schema_valid():
    runner = StubRunner([VALID_FORALL_JSON])
    attempt = translate_sentence(runner, "All dogs bark.")
    assert attempt.schema_valid is True
    assert attempt.json_ast["type"] == "forall"
    assert attempt.error_reason is None


def test_translate_sentence_strips_markdown_fences():
    runner = StubRunner([f"```json\n{VALID_FORALL_JSON}\n```"])
    attempt = translate_sentence(runner, "All dogs bark.")
    assert attempt.schema_valid is True


def test_translate_sentence_not_json_fails_as_invalid_json():
    runner = StubRunner(["Sure! Dogs are animals that bark a lot."])
    attempt = translate_sentence(runner, "All dogs bark.")
    assert attempt.schema_valid is False
    assert attempt.error_reason == "invalid_json"


def test_translate_sentence_unknown_node_type_fails_schema():
    runner = StubRunner(['{"type": "xor", "left": {}, "right": {}}'])
    attempt = translate_sentence(runner, "Exactly one of A or B.")
    assert attempt.schema_valid is False
    assert attempt.error_reason == "unknown_node_type"


def test_translate_sentence_no_auto_repair_second_call():
    """A malformed output must not trigger a second, corrective model call."""
    runner = StubRunner(["not json at all"])
    translate_sentence(runner, "All dogs bark.")
    assert len(runner.calls) == 1


def test_translate_theory_all_valid_populates_dicts():
    runner = StubRunner([VALID_FORALL_JSON, '{"type": "predicate", "name": "dog", "args": ["fluffy"]}'])
    allowed = FolioAllowedInput(premises=("All dogs bark.",), hypothesis="Fluffy is a dog.")
    result = translate_theory(runner, allowed)
    assert result.all_schema_valid is True
    assert result.premise_dicts is not None
    assert result.hypothesis_dict is not None


def test_translate_theory_one_failure_blocks_dicts():
    runner = StubRunner([VALID_FORALL_JSON, "garbage output"])
    allowed = FolioAllowedInput(premises=("All dogs bark.",), hypothesis="Fluffy is a dog.")
    result = translate_theory(runner, allowed)
    assert result.all_schema_valid is False
    assert result.premise_dicts is None
    assert result.hypothesis_dict is None
    assert result.hypothesis_attempt.schema_valid is False


def test_translate_theory_translates_premises_and_hypothesis_independently():
    """Confirms each sentence gets its own isolated call -- translating the
    hypothesis is not conditioned on a shared 'here are the premises' framing."""
    runner = StubRunner([VALID_FORALL_JSON, '{"type": "predicate", "name": "dog", "args": ["fluffy"]}'])
    allowed = FolioAllowedInput(premises=("All corgis herd sheep.",), hypothesis="Fluffy is a dog.")
    translate_theory(runner, allowed)
    assert len(runner.calls) == 2
    assert "All corgis herd sheep." in runner.calls[0]
    assert "Fluffy is a dog." in runner.calls[1]
    assert "All corgis herd sheep." not in runner.calls[1]
