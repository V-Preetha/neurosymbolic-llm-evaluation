"""Fixed proof-feedback rendering for Qwen correction calls."""

from __future__ import annotations

from src.io_guard import extract_allowed_input
from src.nsce.symbolic_reasoner import ReasoningResult


def render_feedback(
    record: dict,
    previous_answer: str,
    reasoning: ReasoningResult,
    template: str,
) -> str:
    """Render the correction prompt using only whitelisted facts/rules/query.

    `record` is narrowed to an `AllowedInput` immediately; `reasoning` (the
    symbolic verifier's own result) supplies the proof trace -- it is never
    fabricated here.
    """
    allowed = extract_allowed_input(record)
    facts = "\n".join(f"- {item}" for item in allowed.facts)
    rules = "\n".join(f"- {item}" for item in allowed.rules)
    if reasoning.status == "unknown":
        explanation = (
            "Exhaustive forward chaining derived neither the query nor its explicit negation. "
            "The symbolic result is Unknown."
        )
        proof = "No supporting or contradicting derivation exists."
    else:
        explanation = (
            f"The symbolic result is {reasoning.prediction.title()} "
            f"({reasoning.status.upper()})."
        )
        proof = "\n".join(
            f"{index}. {step}" for index, step in enumerate(reasoning.proof_trace, start=1)
        )
    return template.format(
        facts=facts,
        rules=rules,
        question=allowed.query,
        previous_answer=previous_answer.title(),
        symbolic_explanation=explanation,
        proof_trace=proof,
    )

