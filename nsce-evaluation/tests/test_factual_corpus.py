from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT
from src.retrieval.factual_corpus import (
    CORPUS_FIELDS, build_faiss_index, chunk_words, file_hash, stable_doc_id,
    validate_manifest,
)


class FactualCorpusTests(unittest.TestCase):
    def test_stable_document_ids(self) -> None:
        self.assertEqual(stable_doc_id(123, 4, 2), "wiki_00000123_s0004_c0002")
        self.assertEqual(stable_doc_id(123, 4, 2), stable_doc_id(123, 4, 2))

    def test_chunking_is_deterministic_nonempty_and_overlapping(self) -> None:
        text = " ".join(f"word{index}" for index in range(900))
        first = chunk_words(text, 350, 50, 50)
        second = chunk_words(text, 350, 50, 50)
        self.assertEqual(first, second)
        self.assertTrue(all(chunk.strip() for chunk in first))
        self.assertEqual(first[0].split()[-50:], first[1].split()[:50])

    def test_short_tail_is_merged_and_short_section_removed(self) -> None:
        chunks = chunk_words(" ".join(f"w{i}" for i in range(630)), 350, 50, 50)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[-1].split()), 330)
        self.assertEqual(chunk_words("too short", 350, 50, 50), [])

    def test_exact_duplicate_policy(self) -> None:
        chunks = ["same factual text", "different factual text", "same factual text"]
        unique = list(dict.fromkeys(chunks))
        self.assertEqual(unique, ["same factual text", "different factual text"])

    def test_corpus_checksum_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_bytes(b'{"doc_id":"x"}\n')
            self.assertEqual(
                file_hash(path), "9c9ac5c1f734f455b2e8d98757ae10f2fc07b6a9b42d818c3a463ae302b6b4ae",
            )

    def test_corpus_schema_has_no_benchmark_label_fields(self) -> None:
        self.assertNotIn("gold_label", CORPUS_FIELDS)
        self.assertNotIn("factual_error", CORPUS_FIELDS)
        self.assertNotIn("verification_status", CORPUS_FIELDS)

    def test_reproducible_index_dimensions_and_counts(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3]], dtype="float32")
        first = build_faiss_index(embeddings)
        second = build_faiss_index(embeddings)
        self.assertEqual(first.d, embeddings.shape[1])
        self.assertEqual(first.ntotal, embeddings.shape[0])
        self.assertEqual(second.d, first.d)
        self.assertEqual(second.ntotal, first.ntotal)

    def test_manifest_validity(self) -> None:
        manifest = {
            "corpus_name": "test", "source": "wikipedia", "source_version": "v1",
            "source_date": "2026-01-01", "download_or_origin": "https://example.test",
            "preparation_timestamp": "2026-01-02T00:00:00Z",
            "chunking_policy": {"chunk_size": 350, "overlap": 50, "unit": "words"},
            "document_count": 1, "chunk_count": 1, "deduplicated_chunk_count": 1,
            "sha256": "a" * 64, "notes": "test",
        }
        self.assertEqual(validate_manifest(manifest), [])

    def test_frozen_artifact_count_checksum_and_index_equality(self) -> None:
        import faiss

        root = PROJECT_ROOT / "data" / "factual_corpus"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((root / "embeddings_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(file_hash(root / "corpus.jsonl"), manifest["sha256"])
        count = 0
        ids = set()
        with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.assertEqual(set(record), CORPUS_FIELDS)
                self.assertTrue(record["text"].strip())
                ids.add(record["doc_id"])
                count += 1
        self.assertEqual(count, 162_696)
        self.assertEqual(len(ids), count)
        self.assertEqual(metadata["embedding_count"], count)
        self.assertEqual(faiss.read_index(str(root / "corpus.faiss")).ntotal, count)


if __name__ == "__main__":
    unittest.main()
