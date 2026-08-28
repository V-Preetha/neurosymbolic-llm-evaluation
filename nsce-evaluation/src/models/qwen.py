"""Single shared Qwen loader and deterministic generation interface."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


LABEL_PATTERN = re.compile(r"^\s*(True|False|Unknown)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    prediction: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int


def extract_logical_label(text: str) -> str | None:
    match = LABEL_PATTERN.fullmatch(text)
    return match.group(1).lower() if match else None


class QwenRunner:
    """Load Qwen once and reuse it identically across all future branches."""

    def __init__(self, model_id: str, seed: int, max_new_tokens: int, quantization: str = "nf4") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        except ImportError as exc:
            raise RuntimeError("Install requirements-phase1.txt before loading Qwen") from exc

        self.torch = torch
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        load_args: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if quantization not in {"none", "int8", "nf4"}:
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        if quantization != "none":
            if not torch.cuda.is_available():
                raise RuntimeError("Quantized GPU inference requested but CUDA is unavailable")
            load_args["torch_dtype"] = torch.float16
        if quantization == "int8":
            load_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "nf4":
            load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if torch.cuda.is_available():
            load_args["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_args)
        self.model.eval()
        # The checkpoint carries sampling defaults, but this experiment uses greedy
        # decoding. Clearing unused values prevents misleading generation warnings.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        placement = Counter(str(device) for device in getattr(self.model, "hf_device_map", {}).values())
        self.runtime_info = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "quantization": quantization,
            "device_map_summary": dict(placement),
        }

    def _model_inputs(self, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer([text], return_tensors="pt")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def _choice_constraint(self, input_length: int, choices: tuple[str, ...]):
        """Restrict generation to one benchmark label followed by EOS."""
        label_sequences = [
            self.tokenizer.encode(label, add_special_tokens=False)
            for label in choices
        ]
        eos_token_id = self.tokenizer.eos_token_id

        def allowed_tokens(batch_id: int, input_ids: Any) -> list[int]:
            del batch_id
            generated = input_ids[input_length:].tolist()
            if generated in label_sequences:
                return [eos_token_id]
            candidates = [
                sequence for sequence in label_sequences
                if sequence[: len(generated)] == generated and len(sequence) > len(generated)
            ]
            if not candidates:
                return [eos_token_id]
            return sorted({sequence[len(generated)] for sequence in candidates})

        return allowed_tokens

    def generate_choice(self, prompt: str, choices: tuple[str, ...]) -> GenerationResult:
        inputs = self._model_inputs(prompt)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=self._choice_constraint(
                    inputs["input_ids"].shape[1], choices
                ),
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            raw_text=raw_text,
            prediction=next(
                (choice.lower() for choice in choices if raw_text.lower() == choice.lower()), None
            ),
            latency_ms=round(latency_ms, 3),
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(new_tokens.shape[0]),
        )

    def generate_label(self, prompt: str) -> GenerationResult:
        return self.generate_choice(prompt, ("True", "False", "Unknown"))

    def generate_text(self, prompt: str, max_new_tokens: int | None = None) -> GenerationResult:
        """Unconstrained greedy generation, for tasks with no fixed label set
        (e.g. the FOLIO JSON-IR translator). `prediction` is always None --
        callers that need a structured result must parse `raw_text`
        themselves (see src/folio/translator.py), and any parse failure is
        their responsibility to report, not this method's.
        """
        inputs = self._model_inputs(prompt)
        budget = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=budget,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            raw_text=raw_text,
            prediction=None,
            latency_ms=round(latency_ms, 3),
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(new_tokens.shape[0]),
        )
