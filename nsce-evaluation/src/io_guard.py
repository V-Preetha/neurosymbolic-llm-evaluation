"""Explicit whitelisting of benchmark fields visible to the LLM/verifier execution path.

Only `input.facts`, `input.rules`, and `input.question` may reach a prompt template,
the symbolic verifier, or a feedback/correction prompt. Evaluation-only fields
(`ground_truth`, `evidence`, `metadata`, `candidate_response`) exist solely for
scoring, which happens elsewhere (e.g. src/nsce/pipeline.py), never inside a
prompt-rendering or verification function.

`extract_allowed_input` is the single narrowing step every prompt/verifier
function performs before touching a record; nothing downstream of it can see a
forbidden field because it was never given one. See tests/test_field_whitelisting.py
for automated proof that this holds for the LLM-only, self-check, and NS bridge
paths.
"""

from __future__ import annotations

from dataclasses import dataclass


FORBIDDEN_TOP_LEVEL_KEYS = ("ground_truth", "evidence", "metadata", "candidate_response")


@dataclass(frozen=True)
class AllowedInput:
    """The only fields a prompt or verifier is permitted to see for a ProofWriter record."""

    facts: tuple[str, ...]
    rules: tuple[str, ...]
    query: str


def extract_allowed_input(record: dict) -> AllowedInput:
    """Narrow a raw benchmark record down to the fields a model/verifier may see."""
    input_block = record["input"]
    return AllowedInput(
        facts=tuple(input_block["facts"]),
        rules=tuple(input_block["rules"]),
        query=input_block["question"],
    )
