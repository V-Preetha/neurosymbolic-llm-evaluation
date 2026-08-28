"""Pinned, local-only SentenceTransformers embedding interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingModel:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        if not model_path.is_dir():
            raise FileNotFoundError(f"Embedding model is not cached at {model_path}")
        self.device = device
        self.model = SentenceTransformer(str(model_path), device=device, local_files_only=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode_document(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode_query(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)
