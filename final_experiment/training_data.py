"""Tokenization, dynamic padding, and sequence-length audit helpers."""
from __future__ import annotations

import math
from typing import Any, Iterable


def _record_token_ids(record: dict[str, str], tokenizer: Any) -> tuple[list[int], list[int]]:
    prefix_ids = list(tokenizer(record["prefix"], add_special_tokens=True)["input_ids"])
    target_ids = list(tokenizer(record["target"], add_special_tokens=False)["input_ids"])
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and (not target_ids or target_ids[-1] != eos_token_id):
        target_ids.append(eos_token_id)
    return prefix_ids, target_ids


def training_record_length(record: dict[str, str], tokenizer: Any) -> int:
    prefix_ids, target_ids = _record_token_ids(record, tokenizer)
    return len(prefix_ids) + len(target_ids)


def tokenize_training_record(
    record: dict[str, str], tokenizer: Any, max_sequence_length: int,
) -> dict[str, list[int]]:
    prefix_ids, target_ids = _record_token_ids(record, tokenizer)
    input_ids = prefix_ids + target_ids
    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Training record length {len(input_ids)} exceeds audited "
            f"max_sequence_length={max_sequence_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prefix_ids) + target_ids,
    }


class DynamicCausalCollator:
    """Pad each batch to its longest record; prompt/padding labels stay ignored."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]):
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = longest - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def summarize_token_lengths(lengths: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(length) for length in lengths)
    if not ordered:
        raise ValueError("No token lengths were measured")

    def percentile(value: float) -> int:
        return ordered[max(0, math.ceil(value * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def choose_safe_max_sequence_length(summaries: Iterable[dict[str, int]]) -> int:
    maxima = [int(summary["max"]) for summary in summaries if summary.get("count", 0)]
    if not maxima:
        raise ValueError("No token lengths were measured")
    return max(maxima)


def smallest_lossless_standard_limit(observed_max: int) -> int:
    """Select the smallest approved training cap that cannot truncate a record."""
    if observed_max < 1:
        raise ValueError("Observed maximum token length must be positive")
    for limit in (292, 320, 384, 512, 640, 768, 1024, 1536, 2048):
        if observed_max <= limit:
            return limit
    raise ValueError(f"Observed token length {observed_max} exceeds supported standard limits")


def validate_configured_standard_limit(configured_limit: int, observed_max: int) -> int:
    if configured_limit < observed_max:
        raise ValueError(
            f"Configured max_sequence_length={configured_limit} would truncate "
            f"active records with observed maximum {observed_max}"
        )
    expected = smallest_lossless_standard_limit(observed_max)
    if configured_limit != expected:
        raise ValueError(
            f"Configured max_sequence_length={configured_limit} is not the smallest lossless "
            f"standard limit {expected} for observed maximum {observed_max}"
        )
    return 0


def training_record_lengths(
    records: list[dict[str, str]], tokenizer: Any, batch_size: int = 256,
) -> list[int]:
    lengths: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prefixes = tokenizer(
            [record["prefix"] for record in batch],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        targets = tokenizer(
            [record["target"] for record in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for prefix_ids, target_ids in zip(prefixes, targets):
            needs_eos = eos_token_id is not None and (
                not target_ids or target_ids[-1] != eos_token_id
            )
            lengths.append(len(prefix_ids) + len(target_ids) + int(needs_eos))
    return lengths


def audit_record_groups(
    groups: dict[str, list[dict[str, str]]], tokenizer: Any,
) -> dict[str, Any]:
    summaries = {
        name: summarize_token_lengths(training_record_lengths(records, tokenizer))
        for name, records in groups.items()
    }
    return {
        "selection_policy": "maximum observed train/development record length",
        "heldout_test_used_for_selection": False,
        "groups": summaries,
        "chosen_max_sequence_length": choose_safe_max_sequence_length(summaries.values()),
    }
