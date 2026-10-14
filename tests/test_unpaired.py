"""Protocol, leakage, exact metric and end-to-end CPU regression tests."""
import copy
import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import OptimizeResult
from threadpoolctl import threadpool_limits

from src.evaluation.retrieval import retrieval, topk_cosine_indices
from src.training.mini_vec2vec import aligned_centroids, fit_unpaired, preprocess
from src.unpaired.config import load_config
from src.unpaired.data import audit_manifests, make_splits, prepare, read_coco, read_spc
from src.unpaired.features import caption_means, extract, pool_vision
from src.unpaired.io import load_bank, manifest, save_bank, save_tensor, work_dir
from src.unpaired.pipeline import evaluate, train

torch.set_num_threads(1)


def small_config(root, setting="disjoint"):
    cfg = load_config("configs/unpaired_coco2014.yaml")
    cfg["paths"].update(coco_root=str(root / "coco"), work_dir=str(root / setting), result_dir=str(root / (setting + "_result")))
    cfg["split"].update(setting=setting, expected_train_images=13, expected_eval_images=8, val_size=3)
    cfg["mapping"].update(num_anchor_runs=2, num_clusters=2, qap_runs=2, anchor_subsample=5,
                          initial_k=2, refine_iters=2, refine_sample=4, refine_k=2,
                          refine2_clusters=2, similarity_batch_size=3)
    cfg["vision"].update(image_size=28, batch_size=2, num_workers=0)
    cfg["runtime"]["device"] = "cpu"
    cfg["paths"].update(spc_paragraphs=str(root / "paragraphs.json"), spc_image_metadata=str(root / "vg.json"))
    return cfg


def create_data(root):
    coco = root / "coco"
    (coco / "annotations").mkdir(parents=True)
    for split, ids in (("train2014", range(1, 14)), ("val2014", range(101, 109))):
        (coco / split).mkdir()
        images, annotations = [], []
        for iid in ids:
            filename = f"COCO_{split}_{iid:012d}.jpg"
            Image.new("RGB", (16, 16), (iid % 255, 40, 70)).save(coco / split / filename)
            images.append({"id": iid, "file_name": filename})
            for j in range(2 + iid % 2):
                annotations.append({"id": iid * 10 + j, "image_id": iid, "caption": f"caption {iid} {j}"})
        # Annotations are deliberately out of image order.
        (coco / "annotations" / f"captions_{split}.json").write_text(json.dumps({"images": images[::-1], "annotations": annotations[::-1]}))
    paragraphs = [{"image_id": i, "paragraph": f"paragraph {i}"} for i in range(1, 11)]
    metadata = [{"image_id": i, "coco_id": (1 if i == 1 else 101 if i == 2 else None)} for i in range(1, 11)]
    (root / "paragraphs.json").write_text(json.dumps(paragraphs))
    (root / "vg.json").write_text(json.dumps(metadata))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        create_data(self.root)
        self.cfg = small_config(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_odd_disjoint_coverage_and_reproducibility(self):
        meta = prepare(self.cfg)
        self.assertEqual(meta["counts"], {"vision_train": 6, "text_train": 7, "paired_val": 3, "paired_test": 5})
        vx, ty = manifest(self.cfg, "vision_train"), manifest(self.cfg, "text_train")
        self.assertEqual({r["image_id"] for r in vx + ty}, set(range(1, 14)))
        self.assertFalse(any("captions" in r for r in vx))
        self.assertEqual(prepare(self.cfg), meta)
        tr, ev = read_coco(self.root / "coco", "train2014"), read_coco(self.root / "coco", "val2014")
        other = make_splits(tr, ev, {**self.cfg["split"], "seed": 43})
        self.assertNotEqual(other["vision_train"], vx)

    def test_cross_dataset_full_vision_and_source_overlap_removal(self):
        cfg = small_config(self.root, "cross_dataset")
        meta = prepare(cfg)
        self.assertEqual(meta["counts"]["vision_train"], 13)
        self.assertEqual(meta["counts"]["text_train"], 8)
        self.assertEqual(meta["excluded_spc_vg_ids"], [1, 2])
        self.assertEqual({r["corpus"] for r in manifest(cfg, "text_train")}, {"spc"})

    def test_captions_join_by_id(self):
        rows = read_coco(self.root / "coco", "train2014")
        for row in rows:
            iid = row["image_id"]
            self.assertEqual(row["captions"], [f"caption {iid} {j}" for j in range(2 + iid % 2)])
            self.assertEqual(row["caption_ids"], [iid * 10 + j for j in range(2 + iid % 2)])

    def test_bad_annotations_fail_instead_of_silent_sample_drop(self):
        source = self.root / "coco/annotations/captions_train2014.json"
        original = json.loads(source.read_text())
        bad = copy.deepcopy(original)
        bad["annotations"][0]["image_id"] = 999
        source.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Orphan"):
            prepare(self.cfg)
        bad = copy.deepcopy(original)
        bad["images"].append(bad["images"][0])
        source.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare(self.cfg)
        source.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "expected"):
            read_coco(self.root / "coco", "train2014", 82783)

    def test_missing_provenance_fails(self):
        (self.root / "vg.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "provenance"):
            read_spc(self.root / "paragraphs.json", self.root / "vg.json", {1})

    def test_manifest_and_protocol_tampering_rejected(self):
        prepare(self.cfg)
        path = work_dir(self.cfg) / "text_train.jsonl"
        path.write_text(path.read_text().replace("caption", "altered"))
        with self.assertRaisesRegex(ValueError, "content changed"):
            audit_manifests(self.cfg)
        with self.assertRaises(ValueError):
            prepare({**self.cfg, "split": {**self.cfg["split"], "seed": 43}})

    def test_bank_order_encoder_and_caption_cache_guards(self):
        prepare(self.cfg)
        rows = manifest(self.cfg, "text_train")
        save_bank(self.cfg, "text_train", rows, torch.randn(len(rows), 4))
        path = work_dir(self.cfg) / "text_train.pt"
        obj = torch.load(path, weights_only=True)
        obj["sample_ids"] = obj["sample_ids"][::-1]
        save_tensor(path, obj)
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            load_bank(self.cfg, "text_train")
        save_bank(self.cfg, "text_train", rows, torch.randn(len(rows), 4))
        cfg = copy.deepcopy(self.cfg)
        cfg["text"]["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "cache"):
            load_bank(cfg, "text_train")
        with self.assertRaises(ValueError):
            save_bank(self.cfg, "text_train", rows, torch.zeros(len(rows), 4))

    def test_image_extraction_order_with_stub_encoder(self):
        prepare(self.cfg)
        class Encoder:
            def forward_features(self, x):
                pooled = x.mean(dim=(2, 3))
                return {"x_norm_patchtokens": pooled[:, None, :], "x_norm_clstoken": pooled}
        with patch("src.unpaired.features.load_vision", return_value=Encoder()):
            extract(self.cfg, "vision_train")
        obj = load_bank(self.cfg, "vision_train")
        self.assertEqual(obj["sample_ids"], [r["sample_id"] for r in manifest(self.cfg, "vision_train")])
        self.assertEqual(obj["features"].shape, (6, 3))

    def test_text_extraction_preserves_variable_caption_groups(self):
        prepare(self.cfg)
        rows = manifest(self.cfg, "text_train")
        class Encoder:
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return torch.tensor([[float(t.split()[1]), float(t.split()[2]) + 1] for t in texts])
        with patch.dict(sys.modules, {"sentence_transformers": SimpleNamespace(SentenceTransformer=Encoder)}):
            extract(self.cfg, "text_train")
        obj = load_bank(self.cfg, "text_train")
        expected = torch.cat([caption_means(Encoder().encode(row["captions"]), [len(row["captions"])]) for row in rows])
        self.assertTrue(torch.allclose(obj["features"], expected))
        self.assertEqual(obj["sample_ids"], [r["sample_id"] for r in rows])

    def test_train_eval_identity_overlap_rejected(self):
        tr, ev = read_coco(self.root / "coco", "train2014"), read_coco(self.root / "coco", "val2014")
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_splits(tr, ev + [tr[0]], self.cfg["split"])

    def test_unknown_config_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_config("configs/unpaired_coco2014.yaml", ["mapping.refine_itter=5"])

    def test_fit_and_evaluate_both_settings_without_eval_access(self):
        with threadpool_limits(1):
            for setting in ("disjoint", "cross_dataset"):
                cfg = small_config(self.root, setting)
                prepare(cfg)
                g = torch.Generator().manual_seed(42)
                for name in ("vision_train", "text_train", "vision_val", "text_val", "vision_test", "text_test"):
                    rows = manifest(cfg, name if name.endswith("train") else "paired_" + name.split("_")[1])
                    save_bank(cfg, name, rows, torch.randn(len(rows), 4, generator=g))
                actual_load = load_bank
                def only_train(c, name):
                    self.assertTrue(name.endswith("train"), "Training read evaluation bank")
                    return actual_load(c, name)
                with patch("src.unpaired.pipeline.load_bank", side_effect=only_train):
                    checkpoint = train(cfg)
                self.assertEqual(checkpoint["forward"]["mean_source"].shape, (4,))
                self.assertEqual(evaluate(cfg, "val")["num_pairs"], 3)
                self.assertEqual(evaluate(cfg, "test")["num_pairs"], 5)
                changed = copy.deepcopy(cfg)
                changed["vision"]["pooling"] = "cls"
                rows = manifest(changed, "paired_val")
                save_bank(changed, "vision_val", rows, torch.randn(len(rows), 4))
                with self.assertRaisesRegex(ValueError, "encoders differ"):
                    evaluate(changed)
                rows = manifest(cfg, "vision_train")
                save_bank(cfg, "vision_train", rows, torch.randn(len(rows), 4))
                with self.assertRaisesRegex(ValueError, "Checkpoint/training"):
                    evaluate(cfg)


class GeometryTests(unittest.TestCase):
    def test_block_topk_matches_dense(self):
        g = torch.Generator().manual_seed(7)
        a, b = torch.randn(17, 6, generator=g), torch.randn(23, 6, generator=g)
        expected = (F.normalize(a, dim=1) @ F.normalize(b, dim=1).T).topk(4).indices
        self.assertTrue(torch.equal(topk_cosine_indices(a, b, 4, 3), expected))

    def test_identity_retrieval_independent_orders_and_ties(self):
        a = torch.eye(7)
        ids = [f"i:{i}" for i in range(7)]
        order = torch.tensor([6, 0, 4, 3, 1, 5, 2])
        result = retrieval(a, a[order], ids, [ids[i] for i in order], block_size=2)
        self.assertEqual(result["r_at_1"], 1)
        self.assertEqual(result["foscttm"], 0)
        tied = retrieval(torch.ones(7, 2), torch.ones(7, 2), ids, ids)
        self.assertEqual(tied["mean_rank"], 4)
        self.assertEqual(tied["foscttm"], .5)
        with self.assertRaises(ValueError):
            retrieval(a, a, ids, ids[:-1] + ["wrong"])

    def test_metrics_agree_with_dense_reference(self):
        g = torch.Generator().manual_seed(9)
        a, b = torch.randn(12, 5, generator=g), torch.randn(12, 5, generator=g)
        scores = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
        ranks = (scores > scores.diag()[:, None]).sum(1).double() + 1
        ids = list(range(12))
        result = retrieval(a, b, ids, ids, block_size=3)
        self.assertAlmostEqual(result["mean_rank"], ranks.mean().item())

    def test_caption_normalization_and_token_pooling(self):
        emb = torch.tensor([[9., 0.], [0., 2.], [-3., 0.]])
        self.assertTrue(torch.allclose(caption_means(emb, [2, 1]),
                        torch.tensor([[2**-.5, 2**-.5], [-1., 0.]])))
        output = {"x_norm_patchtokens": emb[:2][None], "x_norm_clstoken": emb[2:3],
                  "x_norm_regtokens": torch.ones(1, 1, 2)}
        expected = F.normalize(torch.cat([emb, torch.ones(1, 2)]).mean(0), dim=0)
        self.assertTrue(torch.allclose(pool_vision(output, "all_token_mean")[0], expected))

    def test_qap_selects_maximum_and_uses_valid_random_starts(self):
        x = torch.randn(10, 3)
        results = [OptimizeResult(fun=10., col_ind=np.array([1, 0])), OptimizeResult(fun=1., col_ind=np.array([0, 1]))]
        centers = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
        with patch("src.training.mini_vec2vec.KMeans") as clusterer, \
                patch("src.training.mini_vec2vec.quadratic_assignment", side_effect=results) as solver:
            clusterer.return_value.fit.return_value = SimpleNamespace(cluster_centers_=centers)
            _, b = aligned_centroids(x, x, n_qap_runs=2, n_clusters=2)
            self.assertTrue(torch.equal(b, torch.from_numpy(centers[[1, 0]])))
            opts = solver.call_args.kwargs["options"]
            self.assertTrue(opts["maximize"])
            self.assertNotIn("P0", opts)

    def test_training_means_padding_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(1):
            cfg = small_config(Path(directory))
            x, y = torch.randn(9, 3), torch.randn(11, 4)
            a, h = fit_unpaired(x, y, cfg["mapping"])
            b, _ = fit_unpaired(x, y, cfg["mapping"])
            self.assertTrue(torch.equal(a["weight"], b["weight"]))
            self.assertTrue(torch.equal(a["mean_source"], x.mean(0)))
            self.assertEqual((h["train_source_rows"], h["train_target_rows"]), (9, 11))
            q = x[:2] + 5
            self.assertTrue(torch.allclose(preprocess(q, a["mean_source"], 4)[:, :3], F.normalize(q - x.mean(0), dim=1)))
            self.assertEqual(a["weight"].shape, (4, 4))
            with self.assertRaises(ValueError):
                fit_unpaired(torch.ones(9, 3), y, cfg["mapping"])


if __name__ == "__main__":
    unittest.main()
