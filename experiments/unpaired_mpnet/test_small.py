"""Offline smoke tests for split semantics and exact chunked similarities; no model weights."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from metrics import mean_selected, retrieval, topk_cosine_indices
from split_coco import make_splits


class ExperimentTests(unittest.TestCase):
    def test_disjoint_split_reproducible(self):
        train = [{"image_id": i, "captions": [str(i)]} for i in range(40)]
        val = [{"image_id": 100+i, "captions": [str(i)]} for i in range(20)]
        p = make_splits(train, val, 42, 10, 9)
        q = make_splits(train, val, 42, 10, 9)
        self.assertEqual(p, q)
        a = {r["image_id"] for r in p["vision_train"]}
        b = {r["image_id"] for r in p["text_train"]}
        v = {r["image_id"] for r in p["paired_val"]}
        self.assertFalse(a & b or a & v or b & v)
        self.assertEqual((len(a), len(b), len(v)), (10, 10, 9))

    def test_blockwise_topk_matches_dense(self):
        g = torch.Generator().manual_seed(2)
        a = torch.randn(65, 9, generator=g)
        b = torch.randn(47, 9, generator=g)
        exact = (F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T).topk(5, dim=-1).indices
        found = topk_cosine_indices(a, b, 5, block_size=7)
        self.assertTrue(torch.equal(exact, found))
        self.assertTrue(torch.allclose(mean_selected(b, found, batch_size=3), b[found].mean(dim=1)))

    def test_paired_metrics(self):
        vectors = torch.eye(12)
        best = retrieval(vectors, vectors, block_size=3)
        self.assertEqual(best["num_validation_samples"], 12)
        self.assertEqual(best["foscttm"], 0)
        self.assertEqual(best["mean_rank"], 1)
        self.assertEqual(best["top1"], 1)
        reversed_metrics = retrieval(vectors, vectors.flip(0), block_size=4)
        self.assertEqual(reversed_metrics["top1"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
