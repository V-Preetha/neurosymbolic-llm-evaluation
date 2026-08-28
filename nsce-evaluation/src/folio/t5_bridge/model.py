"""Loader + deterministic generation interface for fvossel/t5-base-nl-to-fol.

Mirrors src/models/qwen.py's shape (load once, reuse, report runtime_info,
record latency per call) but for a T5 seq2seq checkpoint instead of a causal
LM. No LLM calls anywhere else in the t5_bridge package go around this class.

The checkpoint is ~0.2B params in float32 (~0.9GB resident on GPU per the
smoke-test measurement) -- it fits on an 8GB RTX 5050 with no quantization,
so this loader always requests full precision and never touches
BitsAndBytesConfig.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

PROMPT_PREFIX = "translate English natural language statements into first-order logic (FOL): "

DEFAULT_MAX_LENGTH = 256
DEFAULT_NUM_BEAMS = 5


@dataclass(frozen=True)
class T5GenerationResult:
    raw_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class T5FolBridge:
    """Load fvossel/t5-base-nl-to-fol once and reuse it for every sentence."""

    def __init__(
        self,
        model_id: str = "fvossel/t5-base-nl-to-fol",
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> None:
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers + sentencepiece before loading the T5 bridge") from exc

        self.torch = torch
        self.max_length = max_length
        self.num_beams = num_beams

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
        self.model.eval()

        vram_mb = None
        if device == "cuda":
            vram_mb = round(torch.cuda.memory_allocated() / 1e6, 1)

        self.runtime_info: dict[str, Any] = {
            "model_id": model_id,
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_allocated_mb": vram_mb,
            "num_beams": num_beams,
            "max_length": max_length,
            "quantization": "none",
        }

    def translate(self, nl_text: str) -> T5GenerationResult:
        """Deterministic beam-search decode (do_sample defaults to False for
        model.generate whenever num_beams > 1, so this is reproducible)."""
        input_text = PROMPT_PREFIX + nl_text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.max_length,
                num_beams=self.num_beams,
                do_sample=False,
            )
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return T5GenerationResult(
            raw_text=raw_text,
            latency_ms=round(latency_ms, 3),
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(outputs.shape[1]),
        )
