from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.embed_factual_corpus_gpu import (
    batch_path, expected_batch_rows, validate_batch,
)


class GPUEmbeddingBatchTests(unittest.TestCase):
    def test_batch_names_and_final_row_count_are_stable(self) -> None:
        self.assertEqual(batch_path(Path("x"), 7), Path("x") / "batch_00007.npy")
        self.assertEqual(expected_batch_rows(1271), 8)

    def test_completed_batch_shape_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch_00000.npy"
            np.save(path, np.zeros((128, 384), dtype="float32"), allow_pickle=False)
            self.assertTrue(validate_batch(path, 128, 384))
            self.assertFalse(validate_batch(path, 64, 384))


if __name__ == "__main__":
    unittest.main()
