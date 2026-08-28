"""Phase 9A: build and freeze an offline Wikipedia factual corpus and FAISS index."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import PROJECT_ROOT
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.factual_corpus import (
    build_corpus, build_faiss_index, embedding_text, file_hash, read_corpus,
    validate_manifest,
)


GENERIC_QUERIES = [
    "Who developed the theory of general relativity?",
    "What discovery is Alexander Fleming known for?",
    "What is the capital and largest city of France?",
    "What is the approximate speed of light in a vacuum?",
    "When was the United Nations founded?",
]


def download(url: str, destination: Path, expected_size: int) -> None:
    if destination.exists() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NSCE-Research-Corpus-Preparation/1.0 (offline academic evaluation)"},
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        copied = 0
        while block := response.read(1024 * 1024):
            handle.write(block)
            copied += len(block)
            if copied % (64 * 1024 * 1024) < len(block):
                print(json.dumps({"downloaded_bytes": copied, "expected_bytes": expected_size}), flush=True)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded {temporary.stat().st_size} bytes; expected {expected_size}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--index-existing", action="store_true")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "configs" / "factual_corpus.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_root = PROJECT_ROOT / "data" / "factual_corpus"
    source_path = corpus_root / "source" / config["source_filename"]
    if args.download:
        download(config["source_url"], source_path, config["source_size_bytes"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source is absent; rerun with --download: {source_path}")
    actual_sha1 = file_hash(source_path, "sha1")
    if actual_sha1 != config["source_sha1"]:
        raise RuntimeError(f"Source SHA-1 mismatch: {actual_sha1}")

    corpus_path = corpus_root / "corpus.jsonl"
    stats_path = corpus_root / "build_stats.json"
    if args.index_existing:
        if not corpus_path.exists() or not stats_path.exists():
            raise FileNotFoundError("--index-existing requires corpus.jsonl and build_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = build_corpus(source_path, corpus_path, config)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    corpus_sha256 = file_hash(corpus_path)
    records = read_corpus(corpus_path)
    if len(records) != stats["deduplicated_chunk_count"]:
        raise RuntimeError("Corpus record count mismatch")

    embedding_model = EmbeddingModel(
        PROJECT_ROOT / ".model_cache" / "all-MiniLM-L6-v2", config["embedding_device"]
    )
    embeddings = embedding_model.encode_documents([embedding_text(record) for record in records])
    index = build_faiss_index(embeddings)
    if index.ntotal != len(records) or embeddings.shape[0] != len(records):
        raise RuntimeError("Corpus/index/embedding counts differ")
    import faiss
    import mwparserfromhell
    import numpy
    import sentence_transformers

    index_path = corpus_root / "corpus.faiss"
    faiss.write_index(index, str(index_path))
    metadata = {
        "embedding_model_id": config["embedding_model_id"],
        "embedding_model_revision": config["embedding_model_revision"],
        "embedding_dimension": int(embeddings.shape[1]), "normalized": True,
        "embedding_device": config["embedding_device"],
        "faiss_index_type": type(index).__name__, "vector_count": int(index.ntotal),
        "corpus_sha256": corpus_sha256,
        "dependencies": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "mwparserfromhell": mwparserfromhell.__version__,
        },
    }
    (corpus_root / "embeddings_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    rng = random.Random(config["random_seed"])
    inspected = [records[position] for position in rng.sample(range(len(records)), 20)]
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "factual_corpus_chunk_inspection.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in inspected:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    query_vectors = embedding_model.encode_queries(GENERIC_QUERIES)
    scores, positions = index.search(query_vectors, 5)
    sanity = []
    for query, query_scores, query_positions in zip(GENERIC_QUERIES, scores, positions):
        sanity.append({
            "query": query,
            "hits": [
                {"rank": rank, "score": round(float(score), 8),
                 "doc_id": records[int(position)]["doc_id"],
                 "title": records[int(position)]["title"],
                 "section": records[int(position)]["section"],
                 "text_excerpt": records[int(position)]["text"][:300]}
                for rank, (score, position) in enumerate(zip(query_scores, query_positions), start=1)
            ],
        })
    (reports / "factual_corpus_retrieval_sanity.json").write_text(
        json.dumps(sanity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

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
        "source_size_bytes": source_path.stat().st_size, "source_sha1": actual_sha1,
        "embedding": metadata,
        "notes": (
            "Fixed prefix of one official Wikimedia shard. Selection is independent of FELM. "
            "Low page-ID/source-order coverage bias is a known limitation. Corpus is immutable "
            "for Phase 9 under the recorded SHA-256."
        ),
    }
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    (corpus_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
