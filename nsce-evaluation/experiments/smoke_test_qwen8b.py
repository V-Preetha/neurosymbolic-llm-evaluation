"""Phase B smoke test: load Qwen3-8B under the existing NF4 runtime pattern,
run a trivial instruction and one frozen ProofWriter dev-pilot example, and
record load/inference metrics for reproducibility.

Does not touch the held-out 360-example ProofWriter test split.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig, PROJECT_ROOT
from src.models.qwen import QwenRunner
from src.prompts import PROOFWRITER_LLM_ONLY_TEMPLATE, load_template, render_proofwriter_prompt

DEV_PILOT_PATH = PROJECT_ROOT / "data" / "processed" / "proofwriter.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "revised_study" / "qwen8_smoke_test.json"


def load_first_dev_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    return json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_qwen8b.json"))
    args = parser.parse_args()

    import torch

    caught_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as collected:
        warnings.simplefilter("always")

        config = ExperimentConfig.load(Path(args.config))

        gpu_available = torch.cuda.is_available()
        vram_before_mb = None
        if gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            vram_before_mb = round((total_bytes - free_bytes) / (1024 ** 2), 1)
            vram_free_before_mb = round(free_bytes / (1024 ** 2), 1)
        else:
            vram_free_before_mb = None

        load_started = time.perf_counter()
        runner = QwenRunner(
            model_id=config.model_id,
            seed=config.seed,
            max_new_tokens=config.max_new_tokens,
            quantization=config.quantization,
        )
        load_time_s = round(time.perf_counter() - load_started, 3)

        peak_vram_after_load_mb = None
        if gpu_available:
            peak_vram_after_load_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

        device_map_summary = runner.runtime_info["device_map_summary"]
        cpu_offload = any(
            device.lower() in ("cpu", "disk") for device in device_map_summary
        )

        # --- Test A: trivial deterministic instruction, unconstrained decoding ---
        test_a_prompt = "Reply with exactly one word: Ready."
        messages = [{"role": "user", "content": test_a_prompt}]
        try:
            text_a = runner.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text_a = runner.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs_a = runner.tokenizer([text_a], return_tensors="pt")
        device = next(runner.model.parameters()).device
        inputs_a = {key: value.to(device) for key, value in inputs_a.items()}

        started_a = time.perf_counter()
        with torch.inference_mode():
            generated_a = runner.model.generate(
                **inputs_a,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=runner.tokenizer.eos_token_id,
            )
        if gpu_available:
            torch.cuda.synchronize()
        latency_a_ms = round((time.perf_counter() - started_a) * 1000, 3)
        new_tokens_a = generated_a[0, inputs_a["input_ids"].shape[1]:]
        raw_text_a = runner.tokenizer.decode(new_tokens_a, skip_special_tokens=True).strip()

        test_a_result = {
            "prompt": test_a_prompt,
            "raw_output": raw_text_a,
            "prompt_tokens": int(inputs_a["input_ids"].shape[1]),
            "generated_tokens": int(new_tokens_a.shape[0]),
            "latency_ms": latency_a_ms,
            "deterministic_nonempty": bool(raw_text_a),
        }

        # --- Test B: one frozen ProofWriter dev-pilot example ---
        dev_record = load_first_dev_record(DEV_PILOT_PATH)
        template = load_template(PROOFWRITER_LLM_ONLY_TEMPLATE)
        prompt_b = render_proofwriter_prompt(dev_record, template)

        assert "ground_truth" not in prompt_b, "gold label leaked into prompt"

        generation_b = runner.generate_label(prompt_b)
        permitted_labels = {label.lower() for label in config.logical_labels}
        parse_success = generation_b.prediction is not None and generation_b.prediction in permitted_labels

        peak_vram_after_inference_mb = None
        if gpu_available:
            peak_vram_after_inference_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

        test_b_result = {
            "sample_id": dev_record["sample_id"],
            "gold_label": dev_record["ground_truth"]["label"],
            "raw_output": generation_b.raw_text,
            "parsed_prediction": generation_b.prediction,
            "parse_success": parse_success,
            "matches_gold": generation_b.prediction == dev_record["ground_truth"]["label"],
            "prompt_tokens": generation_b.input_tokens,
            "generated_tokens": generation_b.output_tokens,
            "latency_ms": generation_b.latency_ms,
        }

        caught_warnings = [str(item.message) for item in collected]

    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except Exception:  # noqa: BLE001
        bnb_version = None

    import transformers

    report = {
        "phase": "B",
        "purpose": "Qwen3-8B NF4 runtime smoke test prior to ProofWriter dev/pilot bridge development",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(Path(args.config).relative_to(PROJECT_ROOT)),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "quantization": runner.runtime_info["quantization"],
        "compute_dtype": "float16",
        "torch_version": runner.runtime_info["torch_version"],
        "cuda_runtime": runner.runtime_info["cuda_runtime"],
        "transformers_version": transformers.__version__,
        "bitsandbytes_version": bnb_version,
        "python_version": platform.python_version(),
        "gpu_name": runner.runtime_info["gpu_name"],
        "vram_free_before_load_mb": vram_free_before_mb,
        "peak_vram_after_load_mb": peak_vram_after_load_mb,
        "peak_vram_after_inference_mb": peak_vram_after_inference_mb,
        "device_map_summary": device_map_summary,
        "cpu_offload_detected": cpu_offload,
        "model_load_time_s": load_time_s,
        "seed": config.seed,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "test_a_trivial_instruction": test_a_result,
        "test_b_proofwriter_dev_example": test_b_result,
        "warnings": caught_warnings,
        "held_out_test_split_touched": False,
    }

    overall_pass = (
        not cpu_offload
        and test_a_result["deterministic_nonempty"]
        and test_b_result["parse_success"]
    )
    report["smoke_test_passed"] = overall_pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
