"""Phase 9A CUDA-only, resumable embedding and FAISS indexing for the frozen corpus."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import PROJECT_ROOT
from src.retrieval.factual_corpus import (
    build_faiss_index, embedding_text, file_hash, read_corpus, validate_manifest,
)


EXPECTED_COUNT = 162_696
CHECKPOINT_ROWS = 128
INITIAL_BATCH_SIZE = 128
MINIMUM_BATCH_SIZE = 32


def batch_path(directory: Path, batch_index: int) -> Path:
    return directory / f"batch_{batch_index:05d}.npy"


def expected_batch_rows(batch_index: int, total: int = EXPECTED_COUNT) -> int:
    start = batch_index * CHECKPOINT_ROWS
    return min(CHECKPOINT_ROWS, total - start)


def validate_batch(path: Path, rows: int, dimension: int) -> bool:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return values.shape == (rows, dimension) and values.dtype == np.float32
    except (OSError, ValueError):
        return False


def save_progress(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def encode_with_oom_backoff(model, texts: list[str], batch_size: int, torch):
    current = batch_size
    while True:
        try:
            values = model.encode_document(
                texts, batch_size=current, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype("float32", copy=False)
            return values, current
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current <= MINIMUM_BATCH_SIZE:
                raise RuntimeError("CUDA OOM persisted at batch size 32; CPU fallback is forbidden")
            current = max(MINIMUM_BATCH_SIZE, current // 2)
            print(json.dumps({"cuda_oom": True, "retry_batch_size": current}), flush=True)


def main() -> None:
    import faiss
    import mwparserfromhell
    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available(), "CUDA is not available"
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 5050" not in gpu_name:
        raise RuntimeError(f"Expected RTX 5050, found {gpu_name}")

    corpus_root = PROJECT_ROOT / "data" / "factual_corpus"
    corpus_path = corpus_root / "corpus.jsonl"
    config = json.loads((PROJECT_ROOT / "configs" / "factual_corpus.json").read_text(encoding="utf-8"))
    records = read_corpus(corpus_path)
    if len(records) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} frozen chunks, found {len(records)}")
    corpus_sha256 = file_hash(corpus_path)

    model_path = PROJECT_ROOT / ".model_cache" / "all-MiniLM-L6-v2"
    model = SentenceTransformer(str(model_path), device="cuda", local_files_only=True)
    if getattr(model.device, "type", str(model.device).split(":", 1)[0]) != "cuda":
        raise RuntimeError(f"Embedding model is not on CUDA: {model.device}")
    dimension = int(model.get_sentence_embedding_dimension())
    checkpoint_root = corpus_root / "embedding_batches"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    progress_path = corpus_root / "embedding_progress.json"
    progress = {
        "accumulated_embedding_seconds": 0.0, "peak_gpu_memory_bytes": 0,
        "final_batch_size": INITIAL_BATCH_SIZE, "completed_chunks": 0,
    }
    if progress_path.exists():
        progress.update(json.loads(progress_path.read_text(encoding="utf-8")))

    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({
        "cuda_available": torch.cuda.is_available(), "gpu_name": gpu_name,
        "model_device": str(model.device), "batch_size": progress["final_batch_size"],
        "total_chunks": EXPECTED_COUNT,
    }), flush=True)

    run_started = time.perf_counter()
    completed = 0
    batch_count = (EXPECTED_COUNT + CHECKPOINT_ROWS - 1) // CHECKPOINT_ROWS
    for batch_index in range(batch_count):
        start = batch_index * CHECKPOINT_ROWS
        rows = expected_batch_rows(batch_index)
        path = batch_path(checkpoint_root, batch_index)
        if path.exists():
            if not validate_batch(path, rows, dimension):
                raise RuntimeError(f"Invalid completed checkpoint: {path}")
            completed += rows
            continue
        texts = [embedding_text(record) for record in records[start : start + rows]]
        values, used_batch_size = encode_with_oom_backoff(
            model, texts, int(progress["final_batch_size"]), torch
        )
        if values.shape != (rows, dimension):
            raise RuntimeError(f"Embedding shape mismatch at batch {batch_index}: {values.shape}")
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
        completed += rows
        progress["final_batch_size"] = used_batch_size
        progress["completed_chunks"] = completed
        progress["peak_gpu_memory_bytes"] = max(
            int(progress["peak_gpu_memory_bytes"]), int(torch.cuda.max_memory_allocated())
        )
        progress["accumulated_embedding_seconds"] += time.perf_counter() - run_started
        run_started = time.perf_counter()
        save_progress(progress_path, progress)
        if (batch_index + 1) % 25 == 0 or completed == EXPECTED_COUNT:
            elapsed = float(progress["accumulated_embedding_seconds"])
            print(json.dumps({
                "chunks_completed": completed, "total_chunks": EXPECTED_COUNT,
                "chunks_per_second": round(completed / elapsed, 3) if elapsed else None,
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            }), flush=True)

    if completed != EXPECTED_COUNT:
        raise RuntimeError(f"Completed checkpoint rows differ: {completed}")

    index = faiss.IndexFlatIP(dimension)
    for batch_index in range(batch_count):
        values = np.load(batch_path(checkpoint_root, batch_index), allow_pickle=False)
        index.add(values)
    if index.ntotal != EXPECTED_COUNT:
        raise RuntimeError(f"FAISS count mismatch: {index.ntotal}")
    faiss.write_index(index, str(corpus_root / "corpus.faiss"))

    total_seconds = float(progress["accumulated_embedding_seconds"])
    metadata = {
        "device": "cuda", "gpu_name": gpu_name, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "embedding_model": config["embedding_model_id"],
        "embedding_revision": config["embedding_model_revision"],
        "batch_size": int(progress["final_batch_size"]), "checkpoint_rows": CHECKPOINT_ROWS,
        "normalized": True, "embedding_count": EXPECTED_COUNT,
        "embedding_dimension": dimension, "faiss_index_type": type(index).__name__,
        "faiss_ntotal": int(index.ntotal), "corpus_sha256": corpus_sha256,
        "total_embedding_seconds": total_seconds,
        "average_chunks_per_second": EXPECTED_COUNT / total_seconds,
        "peak_gpu_memory_bytes": int(progress["peak_gpu_memory_bytes"]),
        "dependencies": {
            "python": platform.python_version(), "numpy": np.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "mwparserfromhell": mwparserfromhell.__version__,
        },
    }
    (corpus_root / "embeddings_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    stats = json.loads((corpus_root / "build_stats.json").read_text(encoding="utf-8"))
    source_path = corpus_root / "source" / config["source_filename"]
    manifest = {
        "corpus_name": config["corpus_name"], "source": config["source"],
        "source_version": config["source_version"], "source_date": config["source_date"],
        "download_or_origin": config["source_url"],
        "preparation_timestamp": datetime.now(timezone.utc).isoformat(),
        "chunking_policy": config["chunking_policy"],
        "selection_policy": config["selection_policy"],
        "deduplication_policy": config["deduplication_policy"],
        "document_count": stats["document_count"], "chunk_count": stats["chunk_count"],
        "exact_duplicates_removed": stats["exact_duplicates_removed"],
        "deduplicated_chunk_count": stats["deduplicated_chunk_count"],
        "corpus_size_bytes": corpus_path.stat().st_size, "sha256": corpus_sha256,
        "source_size_bytes": source_path.stat().st_size,
        "source_sha1": file_hash(source_path, "sha1"), "embedding": metadata,
        "notes": (
            "Fixed prefix of one official Wikimedia shard, independent of FELM. Low page-ID/"
            "source-order coverage bias is a known limitation. Frozen for Phase 9 by SHA-256."
        ),
    }
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    (corpus_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **metadata}), flush=True)


if __name__ == "__main__":
    main()
