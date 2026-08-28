"""Tests for src/folio/v2/repair.py -- one-shot constrained repair."""
from __future__ import annotations

from dataclasses import dataclass

from src.folio.v2.decomposer import DecompositionAttempt
from src.folio.v2.repair import attempt_repair, is_repairable, render_repair_prompt

VALID_DECOMPOSITION = (
    '{"entities": [], "atoms": [{"id": "A1", "predicate": "researcher", "args": ["x"]}, '
    '{"id": "A2", "predicate": "careful", "args": ["x"]}], '
    '"nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}, '
    '{"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}], "root": "N2"}'
)


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


def test_is_repairable_true_for_unbound_variable():
    assert is_repairable("unbound_variable") is True


def test_is_repairable_false_for_arity_conflict():
    """Cross-sentence arity conflicts are never repaired (not attributable to
    one sentence -- see repair.py's docstring)."""
    assert is_repairable("arity_conflict") is False


def test_is_repairable_false_for_none():
    assert is_repairable(None) is False


def test_render_repair_prompt_includes_error_and_previous_json_but_no_gold():
    previous = {"atoms": [{"id": "A1", "predicate": "researcher", "args": ["x"]}], "nodes": [], "root": "A1"}
    prompt = render_repair_prompt("Every researcher is careful.", previous, "unbound_variable", "variable 'x' unbound")
    assert "unbound_variable" in prompt
    assert '"researcher"' in prompt
    assert "gold" not in prompt.lower()


def test_attempt_repair_issues_exactly_one_call_and_marks_repaired():
    runner = StubRunner([VALID_DECOMPOSITION])
    previous = DecompositionAttempt(
        sentence="Every researcher is careful.", raw_output="{...broken...}", decomposition=None,
        shape_valid=False, error_reason="invalid_json", error_detail="no JSON object found", latency_ms=1.0,
    )
    result = attempt_repair(runner, "Every researcher is careful.", previous, "invalid_json", "no JSON object found")
    assert len(runner.calls) == 1
    assert result.repaired is True
    assert result.shape_valid is True
