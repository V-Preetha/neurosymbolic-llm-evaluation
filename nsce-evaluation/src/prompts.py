"""Shared prompt rendering; templates are fixed files, never sample-tuned."""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT
from src.io_guard import extract_allowed_input


PROOFWRITER_LLM_ONLY_TEMPLATE = PROJECT_ROOT / "prompts" / "proofwriter_llm_only.txt"
PROOFWRITER_SELF_CHECK_TEMPLATE = PROJECT_ROOT / "prompts" / "proofwriter_self_check.txt"


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_proofwriter_prompt(record: dict, template: str) -> str:
    """Render input fields only; ground-truth fields never enter the prompt.

    `record` is narrowed to an `AllowedInput` immediately -- everything below this
    line operates on that narrowed object, never on `record` again.
    """
    allowed = extract_allowed_input(record)
    facts = "\n".join(f"- {item}" for item in allowed.facts)
    rules = "\n".join(f"- {item}" for item in allowed.rules)
    return template.format(facts=facts, rules=rules, question=allowed.query)


def render_self_check_prompt(record: dict, previous_answer: str, template: str) -> str:
    allowed = extract_allowed_input(record)
    facts = "\n".join(f"- {item}" for item in allowed.facts)
    rules = "\n".join(f"- {item}" for item in allowed.rules)
    return template.format(
        facts=facts,
        rules=rules,
        question=allowed.query,
        previous_answer=previous_answer.title(),
    )

