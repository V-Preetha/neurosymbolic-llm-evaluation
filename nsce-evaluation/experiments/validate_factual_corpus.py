"""Phase 9A inspection and generic retrieval sanity checks; never reads FELM."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from src.config import PROJECT_ROOT
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.factual_corpus import read_corpus


GENERIC_QUERIES = [
    "Who developed the theory of general relativity?",
    "What discovery is Alexander Fleming known for?",
    "What is the capital and largest city of France?",
    "What is the approximate speed of light in a vacuum?",
    "When was the United Nations founded?",
]


def main() -> None:
    import faiss

    corpus_root = PROJECT_ROOT / "data" / "factual_corpus"
    records = read_corpus(corpus_root / "corpus.jsonl")
    if len(records) != 162_696:
        raise RuntimeError(f"Frozen corpus count mismatch: {len(records)}")
    index = faiss.read_index(str(corpus_root / "corpus.faiss"))
    if index.ntotal != len(records):
        raise RuntimeError("FAISS/corpus count mismatch")
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    inspected = [records[position] for position in rng.sample(range(len(records)), 20)]
    with (reports / "factual_corpus_chunk_inspection.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in inspected:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    model = EmbeddingModel(PROJECT_ROOT / ".model_cache" / "all-MiniLM-L6-v2", "cuda")
    query_vectors = model.encode_queries(GENERIC_QUERIES)
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
    print(json.dumps([
        {"query": item["query"], "top_title": item["hits"][0]["title"],
         "top_section": item["hits"][0]["section"], "score": item["hits"][0]["score"]}
        for item in sanity
    ], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
