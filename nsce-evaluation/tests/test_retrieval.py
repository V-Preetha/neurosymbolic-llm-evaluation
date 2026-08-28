from __future__ import annotations

import unittest

import numpy as np

from src.config import dataset_package_path
from src.datasets.proofwriter import iter_jsonl
from src.retrieval.corpus import build_documents, build_queries
from src.retrieval.index import search_candidate_pool


class RetrievalTests(unittest.TestCase):
    def test_candidate_pools_are_deterministic_and_scoped(self) -> None:
        rows = list(
            iter_jsonl(dataset_package_path() / "normalized" / "proofwriter_dev.jsonl")
        )[:5]
        documents = build_documents(rows)
        first = build_queries(rows, documents, seed=42, distractor_count=4)
        second = build_queries(rows, documents, seed=42, distractor_count=4)
        self.assertEqual(first, second)
        for query in first:
            self.assertTrue(set(query["relevant_doc_ids"]).issubset(query["candidate_doc_ids"]))
            self.assertEqual(
                len(query["candidate_doc_ids"]), len(query["relevant_doc_ids"]) + 4
            )

    def test_faiss_subset_search_returns_global_positions(self) -> None:
        documents = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
        hits = search_candidate_pool(documents, np.asarray([1.0, 0.0], dtype="float32"), [1, 2], 2)
        self.assertEqual([position for position, _ in hits], [2, 1])


if __name__ == "__main__":
    unittest.main()

