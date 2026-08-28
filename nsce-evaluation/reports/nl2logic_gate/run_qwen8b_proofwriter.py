"""Qwen3-8B-only baseline (Branch A) on the frozen 20-item ProofWriter sample.

experiments/run_llm.py hardcodes configs/experiment.json (Qwen3-4B-Instruct-2507).
The NSCE spec calls for frozen Qwen3-8B, so this reuses the same evaluation
logic (src/baselines/llm_only.py, unmodified) against configs/experiment_qwen8b.json
instead. Does not modify experiments/run_llm.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.llm_only import evaluate_proofwriter_record  # noqa: E402
from src.config import ExperimentConfig  # noqa: E402
from src.datasets.proofwriter import iter_jsonl  # noqa: E402
from src.models.qwen import QwenRunner  # noqa: E402
from src.prompts import PROOFWRITER_LLM_ONLY_TEMPLATE, load_template  # noqa: E402
from experiments.run_llm import summarize  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_qwen8b.json"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "raw" / "llm_only_qwen8b_proofwriter.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "results" / "raw" / "llm_only_qwen8b_proofwriter_summary.json"


def main() -> None:
    config = ExperimentConfig.load(CONFIG_PATH)
    template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
    records = list(iter_jsonl(INPUT_PATH))
    if len(records) != config.proofwriter_phase1_samples:
        raise ValueError(f"Expected {config.proofwriter_phase1_samples} records, found {len(records)}")
    runner = QwenRunner(config.model_id, config.seed, config.max_new_tokens, config.quantization)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            result = evaluate_proofwriter_record(runner, record, template)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            results.append(result)
            print(json.dumps({"completed": index, "sample_id": result["sample_id"],
                               "prediction": result["final_prediction"], "correct": result["correct_finally"]}))

    summary = summarize(results, runner.runtime_info, template, config)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
