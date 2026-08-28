"""Conservative, reproducible three-way FELM gold construction."""

from __future__ import annotations

import re
from dataclasses import dataclass


_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?![A-Za-z])")
_WORD = re.compile(r"[a-z]+")
_UNKNOWN_MARKERS = (
    "no evidence", "can't find", "cannot find", "couldn't find", "not on web",
    "no support material", "no such information", "can't be found", "unclear",
    "outdated", "asking for current", "maybe ", "i don't think",
)
_CORRECTION_MARKERS = (
    " should be ", " should not be ", " instead", "not ", "does not ",
    "did not ", "cannot ", "isn't ", "aren't ", " is false", " are false",
)


@dataclass(frozen=True)
class GoldDecision:
    label: str
    basis: str


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in _NUMBER.finditer(text)}


def construct_three_way_gold(
    claim_text: str,
    native_label: str,
    annotator_comment: str | None,
    error_type: str | None,
) -> GoldDecision:
    """Never map a bare factual_error annotation directly to CONTRADICTED."""
    if native_label == "supported":
        return GoldDecision("SUPPORTED", "native_positive_factual_annotation")
    if native_label != "factual_error":
        return GoldDecision("GOLD_AMBIGUOUS", "unrecognized_native_label")

    comment = " ".join(str(annotator_comment or "").split())
    lowered = f" {comment.lower()} "
    if error_type == "irrelevant_with_qst":
        return GoldDecision("UNKNOWN", "annotated_irrelevant_to_question_not_factually_resolved")
    if not comment:
        return GoldDecision("GOLD_AMBIGUOUS", "binary_error_without_explanatory_annotation")
    if any(marker in lowered for marker in _UNKNOWN_MARKERS):
        return GoldDecision("UNKNOWN", "annotator_reports_missing_or_uncertain_evidence")

    claim_numbers = _numbers(claim_text)
    comment_numbers = _numbers(comment)
    if claim_numbers and comment_numbers and (claim_numbers - comment_numbers) and (comment_numbers - claim_numbers):
        return GoldDecision("CONTRADICTED", "explicit_incompatible_numeric_or_date_annotation")

    words = _WORD.findall(comment.lower())
    explicit_correction = any(marker in lowered for marker in _CORRECTION_MARKERS)
    if explicit_correction and len(words) >= 5:
        return GoldDecision("CONTRADICTED", "explicit_corrective_or_negated_fact_annotation")
    return GoldDecision("GOLD_AMBIGUOUS", "binary_error_does_not_establish_explicit_incompatibility")
