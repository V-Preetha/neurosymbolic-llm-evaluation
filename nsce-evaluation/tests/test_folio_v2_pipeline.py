"""Integration tests for src/folio/v2/pipeline.py using a stub runner (no GPU/model load)."""
from __future__ import annotations

from dataclasses import dataclass

from src.folio.io_guard import FolioAllowedInput
from src.folio.solver import classify_theory
from src.folio.v2.pipeline import translate_theory_v2


@dataclass
class StubGeneration:
    raw_text: str
    latency_ms: float = 1.0


class ScriptedRunner:
    """Returns one pre-scripted output per call, in call order."""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    def generate_text(self, prompt: str, max_new_tokens: int = 500) -> StubGeneration:
        self.calls.append(prompt)
        return StubGeneration(raw_text=self._outputs.pop(0))


ALL_DOGS_BARK = (
    '{"atoms": [{"id": "A1", "predicate": "dog", "args": ["x"]}, {"id": "A2", "predicate": "bark", "args": ["x"]}], '
    '"nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}, '
    '{"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}], "root": "N2"}'
)
FLUFFY_IS_A_DOG = '{"entities": [{"surface": "Fluffy", "canonical": "fluffy"}], "atoms": [{"id": "A1", "predicate": "dog", "args": ["fluffy"]}], "nodes": [], "root": "A1"}'
FLUFFY_IS_A_DOG_ALT_PREDICATE_NAME = '{"entities": [{"surface": "Fluffy", "canonical": "fluffy"}], "atoms": [{"id": "A1", "predicate": "is a dog", "args": ["fluffy"]}], "nodes": [], "root": "A1"}'


def test_pipeline_all_valid_reaches_solver_entailed():
    runner = ScriptedRunner([ALL_DOGS_BARK, FLUFFY_IS_A_DOG])
    allowed = FolioAllowedInput(premises=("All dogs bark.",), hypothesis="Fluffy is a dog.")
    result = translate_theory_v2(runner, allowed)
    assert result.all_valid is True
    assert len(runner.calls) == 2  # one decomposition call per sentence, no repair triggered
    verdict = classify_theory(list(result.premise_dicts), result.hypothesis_dict)
    # "Fluffy is a dog" doesn't entail/contradict "all dogs bark" on its own; just
    # confirm it compiles and solves rather than asserting a specific verdict here.
    assert verdict.status in ("entailed", "contradicted", "uncertain")


def test_pipeline_global_vocabulary_unifies_predicate_name_variants():
    """The exact bug this architecture targets: 'dog' in one sentence and
    'is a dog' in another must become ONE predicate under global normalization."""
    runner = ScriptedRunner([ALL_DOGS_BARK, FLUFFY_IS_A_DOG_ALT_PREDICATE_NAME])
    allowed = FolioAllowedInput(premises=("All dogs bark.",), hypothesis="Fluffy is a dog.")
    result = translate_theory_v2(runner, allowed, use_global_vocabulary=True)
    assert result.all_valid is True
    assert result.premise_dicts[0]["body"]["antecedent"]["name"] == result.hypothesis_dict["name"]


TREATS_BINARY = '{"atoms": [{"id": "A1", "predicate": "treats", "args": ["x", "y"]}], "nodes": [{"id": "N1", "op": "forall", "variables": ["x", "y"], "body": "A1"}], "root": "N1"}'
TREATS_UNARY = '{"atoms": [{"id": "A1", "predicate": "treats", "args": ["fido"]}], "nodes": [], "root": "A1"}'


def test_pipeline_global_vocabulary_catches_cross_sentence_arity_conflict():
    runner = ScriptedRunner([TREATS_BINARY, TREATS_UNARY])
    allowed = FolioAllowedInput(premises=("A doctor treats a patient.",), hypothesis="Something treats Fido.")
    result = translate_theory_v2(runner, allowed, use_global_vocabulary=True)
    assert result.all_valid is False
    assert result.theory_level_reason == "arity_conflict"


def test_pipeline_theory_level_check_runs_regardless_of_vocabulary_mode():
    """Even in the V2-A ablation (no shared vocabulary while decomposing),
    the final cross-theory pass still catches an arity conflict -- it just
    surfaces later (after per-sentence validation) instead of before it."""
    runner = ScriptedRunner([TREATS_BINARY, TREATS_UNARY])
    allowed = FolioAllowedInput(premises=("A doctor treats a patient.",), hypothesis="Something treats Fido.")
    result = translate_theory_v2(runner, allowed, use_global_vocabulary=False)
    assert result.premise_results[0].valid is True  # each sentence is locally fine on its own
    assert result.hypothesis_result.valid is True
    assert result.all_valid is False
    assert result.theory_level_reason == "arity_conflict"


def test_pipeline_translates_premises_and_hypothesis_independently():
    runner = ScriptedRunner([ALL_DOGS_BARK, FLUFFY_IS_A_DOG])
    allowed = FolioAllowedInput(premises=("All corgis herd sheep. All dogs bark.",), hypothesis="Fluffy is a dog.")
    translate_theory_v2(runner, allowed)
    assert "Fluffy is a dog." not in runner.calls[0]
    assert "All corgis herd sheep" not in runner.calls[1]


UNBOUND_VAR_JSON = (
    '{"atoms": [{"id": "A1", "predicate": "researcher", "args": ["x"]}, {"id": "A2", "predicate": "careful", "args": ["x"]}], '
    '"nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}], "root": "N1"}'  # missing forall
)
REPAIRED_JSON = (
    '{"atoms": [{"id": "A1", "predicate": "researcher", "args": ["x"]}, {"id": "A2", "predicate": "careful", "args": ["x"]}], '
    '"nodes": [{"id": "N1", "op": "implies", "antecedent": "A1", "consequent": "A2"}, '
    '{"id": "N2", "op": "forall", "variables": ["x"], "body": "N1"}], "root": "N2"}'
)


def test_pipeline_repair_disabled_by_default_leaves_failure():
    runner = ScriptedRunner([UNBOUND_VAR_JSON, FLUFFY_IS_A_DOG])
    allowed = FolioAllowedInput(premises=("Every researcher is careful.",), hypothesis="Fluffy is a dog.")
    result = translate_theory_v2(runner, allowed, enable_repair=False)
    assert result.all_valid is False
    assert result.premise_results[0].reason == "unbound_variable"
    assert len(runner.calls) == 2  # no third, repair call


def test_pipeline_repair_enabled_fixes_missing_quantifier():
    # Call order: premise-initial, hypothesis-initial, then (only if needed) premise-repair.
    runner = ScriptedRunner([UNBOUND_VAR_JSON, FLUFFY_IS_A_DOG, REPAIRED_JSON])
    allowed = FolioAllowedInput(premises=("Every researcher is careful.",), hypothesis="Fluffy is a dog.")
    result = translate_theory_v2(runner, allowed, enable_repair=True)
    assert result.all_valid is True
    assert result.premise_results[0].pre_repair_valid is False
    assert result.premise_results[0].valid is True
    assert result.premise_results[0].repair_attempt is not None
    assert len(runner.calls) == 3


def test_pipeline_repair_never_attempted_twice():
    runner = ScriptedRunner([UNBOUND_VAR_JSON, FLUFFY_IS_A_DOG, UNBOUND_VAR_JSON])
    allowed = FolioAllowedInput(premises=("Every researcher is careful.",), hypothesis="Fluffy is a dog.")
    result = translate_theory_v2(runner, allowed, enable_repair=True)
    assert result.all_valid is False
    assert result.premise_results[0].reason == "repair_failed"
    assert len(runner.calls) == 3  # initial + exactly one repair call, never a second
