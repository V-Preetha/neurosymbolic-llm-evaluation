"""Explicit whitelisting for the FOLIO translation bridge -- the same pattern
as src/io_guard.py's AllowedInput for ProofWriter, applied to FOLIO records.

Only `premises` (raw natural-language strings) and `hypothesis` (raw
natural-language string) may reach the translator prompt. `label`,
`premises-FOL`, `conclusion-FOL`, and any other evaluation-only field are
structurally unreachable from `extract_allowed_folio_input` onward, because
downstream code (translator.py) only ever receives a FolioAllowedInput, not
the raw record. See tests/test_folio_field_whitelisting.py for the
executable proof.
"""

from __future__ import annotations

from dataclasses import dataclass


FORBIDDEN_FOLIO_KEYS = ("label", "premises-FOL", "conclusion-FOL", "conclusion_fol", "premises_fol")


@dataclass(frozen=True)
class FolioAllowedInput:
    """The only fields a FOLIO translation prompt is permitted to see."""

    premises: tuple[str, ...]
    hypothesis: str


def extract_allowed_folio_input(record: dict) -> FolioAllowedInput:
    """Narrow a raw FOLIO record down to the fields the translator may see."""
    return FolioAllowedInput(
        premises=tuple(record["premises"]),
        hypothesis=str(record["conclusion"]),
    )
