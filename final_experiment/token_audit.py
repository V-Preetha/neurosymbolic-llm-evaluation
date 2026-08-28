"""Tokenizer-only sequence-length audit for frozen experiment manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import ROOT, read_jsonl, resolved_config, write_json
from .data import (
    materialize_manifest_rows,
    mode_sample_sizes,
    save_mode_manifests,
    training_pool_policy,
    training_validation_size,
)
from .modeling import model_identity
from .nl2logic_training import direct_training_examples
from .training_data import (
    choose_safe_max_sequence_length,
    smallest_lossless_standard_limit,
    summarize_token_lengths,
    training_record_lengths,
    validate_configured_standard_limit,
)


def active_training_records(rows: list[dict]) -> list[dict[str, str]]:
    """Build only records consumed by the final experiment's direct trainer."""
    return direct_training_examples(rows)


def _freeze_selected_rows(dataset: str, mode: str):
    cfg = resolved_config(dataset)
    base = ROOT / "data" / "processed" / dataset
    parts = {split: read_jsonl(base / f"{split}.jsonl") for split in ("train", "validation", "test")}
    train_size, evaluation_size = mode_sample_sizes(cfg, mode)
    manifests = save_mode_manifests(
        base, parts, dataset, mode, train_size, evaluation_size,
        cfg["experiment"]["seed"], training_validation_size(cfg, mode),
        training_pool_policy(cfg, mode),
    )
    train_rows = materialize_manifest_rows(parts["train"], manifests["train"])
    validation_manifest = manifests.get("training_validation", manifests["validation"])
    validation_rows = materialize_manifest_rows(parts["validation"], validation_manifest)
    return cfg, train_rows, validation_rows, manifests


def _manifest_summary(manifest: dict, path: Path) -> dict:
    summary = {
        "path": str(path),
        "size": manifest["size"],
        "seed": manifest["seed"],
        "stratify_by": manifest["stratify_by"],
        "strata": manifest["strata"],
        "ids_sha256": manifest["ids_sha256"],
    }
    for key in ("sampling_pool", "eligibility", "eligible_pool_size", "excluded_pool_size"):
        if key in manifest:
            summary[key] = manifest[key]
    return summary


def run(datasets: list[str], mode: str) -> Path:
    from transformers import AutoTokenizer

    configs = {dataset: resolved_config(dataset) for dataset in datasets}
    identities = {tuple(model_identity(config).items()) for config in configs.values()}
    if len(identities) != 1:
        raise ValueError("All audited datasets must use the same frozen base model")
    model = next(iter(configs.values()))["model"]
    print(f"[Tokenizer audit] Loading tokenizer only: {model['id']} revision={model['revision']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"],
        trust_remote_code=model.get("trust_remote_code", False),
    )
    report = {
        "model": model_identity(next(iter(configs.values()))),
        "mode": mode,
        "selection_policy": "smallest approved lossless limit above active direct train/development maximum",
        "active_training_scope": "direct_only",
        "bridge_training_records_included": False,
        "heldout_test_used_for_selection": False,
        "datasets": {},
    }
    all_summaries = []
    for dataset in datasets:
        cfg, train_rows, validation_rows, manifests = _freeze_selected_rows(dataset, mode)
        dataset_report = {"manifests": {}, "groups": {}}
        manifest_dir = ROOT / "data" / "processed" / dataset / "manifests"
        dataset_report["manifests"]["train"] = _manifest_summary(
            manifests["train"], manifest_dir / f"train_{mode}.json",
        )
        validation_key = "training_validation" if "training_validation" in manifests else "validation"
        validation_name = f"validation_train_{mode}.json" if validation_key == "training_validation" else "validation.json"
        dataset_report["manifests"]["training_validation"] = _manifest_summary(
            manifests[validation_key], manifest_dir / validation_name,
        )
        dataset_report["manifests"]["heldout_evaluation"] = _manifest_summary(
            manifests["test"], manifest_dir / f"test_{mode}.json",
        )
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            records = active_training_records(rows)
            summary = summarize_token_lengths(training_record_lengths(records, tokenizer))
            key = f"direct_{split}"
            dataset_report["groups"][key] = summary | {"selected_rows": len(rows)}
            all_summaries.append(summary)
            print(
                f"[Tokenizer audit] {dataset} {key}: records={summary['count']} "
                f"p50={summary['p50']} p95={summary['p95']} p99={summary['p99']} max={summary['max']}",
                flush=True,
            )
        report["datasets"][dataset] = dataset_report
    report["observed_max_sequence_length"] = choose_safe_max_sequence_length(all_summaries)
    report["chosen_max_sequence_length"] = smallest_lossless_standard_limit(report["observed_max_sequence_length"])
    report["truncated_training_records_at_chosen_length"] = validate_configured_standard_limit(
        model["max_sequence_length"], report["observed_max_sequence_length"],
    )
    slug = model["id"].split("/")[-1].lower().replace(".", "_").replace("-", "_")
    target = ROOT / "data" / "processed" / f"token_length_audit_{slug}_{mode}.json"
    write_json(target, report)
    print(f"[Tokenizer audit] chosen_max_sequence_length={report['chosen_max_sequence_length']}", flush=True)
    print(f"[Tokenizer audit] saved={target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=["proofwriter", "folio"])
    parser.add_argument("--mode", choices=["smoke", "pilot", "main", "full"], default="main")
    args = parser.parse_args()
    run(args.dataset or ["proofwriter", "folio"], args.mode)


if __name__ == "__main__":
    main()
