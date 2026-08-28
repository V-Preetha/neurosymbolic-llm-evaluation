"""Exact cosine retrieval over declared candidate pools using FAISS."""

from __future__ import annotations

import numpy as np


def build_global_index(document_embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    index.add(np.ascontiguousarray(document_embeddings, dtype="float32"))
    return index


def search_candidate_pool(
    document_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    candidate_indices: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    import faiss

    if not candidate_indices:
        return []
    candidates = document_embeddings[np.asarray(candidate_indices, dtype="int64")]
    index = faiss.IndexFlatIP(candidates.shape[1])
    index.add(np.ascontiguousarray(candidates, dtype="float32"))
    scores, local_indices = index.search(
        np.ascontiguousarray(query_embedding.reshape(1, -1), dtype="float32"),
        min(top_k, len(candidate_indices)),
    )
    return [
        (candidate_indices[int(local_index)], float(score))
        for local_index, score in zip(local_indices[0], scores[0])
        if local_index >= 0
    ]
