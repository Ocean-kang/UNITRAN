# UNITRAN

mini-vec2vec alignment between frozen **DINOv2** image embeddings and **MPNet** text embeddings without supplied training pairs. The primary experiment covers **all 82,783 train2014 image IDs**. Each image gets one visual vector; each original caption is normalized, then captions of the same image are averaged and normalized again.

## Two verified unpaired settings

| Config | Vision training | Text training | Supplied pairs |
|---|---|---|---:|
| `configs/unpaired_coco2014.yaml` | 41,391 train2014 IDs | Captions of the other 41,392 IDs | 0 |
| `configs/unpaired_cross_dataset.yaml` | All 82,783 train2014 images | Stanford Paragraph Captions (SPC), independent corpus | 0 |

These follow Rosetta's [disjoint_split](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/unpaired_rosetta/evaluation.py) and [COCO-images × SPC-captions definition](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/experiments/cross_dataset.py), here using mini-vec2vec. Visual Genome metadata is required to exclude SPC images linked to any COCO train/evaluation image. Excluded IDs and resulting sizes are recorded. This safeguard is additional to the referenced implementation; the new cross-dataset results are not an exact reproduction of Rosetta's reported numbers.

One seeded permutation partitions COCO train IDs. The two feature banks are independently shuffled during fitting. There is no paired-row loss, shared-row sampling or second half-split. The mapper receives only two feature matrices and hyperparameters.

## Install and prepare data

Use Python 3.11:

```bash
python -m pip install -r requirements.txt
# Or:
conda env create -f conda.yaml
conda activate unitran
```

Install a matching CUDA build of PyTorch/torchvision for your GPU when needed. CPU runs use `--set runtime.device=cpu`. `OMP_NUM_THREADS=1` is useful for small Windows KMeans tests. Encoder revisions are pinned. Offline use: set `vision.repo` or `DINOV2_REPO` to a clean DINOv2 checkout at the configured revision; optionally set `vision.weights` to a local state_dict; set `text.model` to a local SentenceTransformer directory. Local source/model/weight contents enter cache fingerprints.

```text
data/mscoco14/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
data/spc/                    # cross-dataset only
├── paragraphs_v1.json
└── image_data.json          # Visual Genome image_id and coco_id provenance
```

Obtain SPC from the [authors](https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html) and source metadata from [Visual Genome](https://visualgenome.org/api/v0/api_home.html). No SPC image pixels or Llama weights are needed. Like the referenced protocol, SPC paragraphs form the external training corpus; evaluation is exclusively on held-out COCO images.

All **40,504 val2014 IDs** are partitioned into **5,000 validation** and **35,504 test** IDs by default. They retain true identities, are disjoint from training and each other, and use saved training statistics. These are internal holdouts of val2014, not the official COCO test2014 server split. Freeze model choices before evaluating test; tuning after inspecting test scores invalidates a held-out claim.

## Run

From the repository root:

```bash
python experiments/unpaired.py all --config configs/unpaired_coco2014.yaml
python experiments/unpaired.py all --config configs/unpaired_cross_dataset.yaml
```

Separate stages permit cache reuse and reserving test evaluation:

```bash
python experiments/unpaired.py prepare
python experiments/unpaired.py extract --bank vision_train
python experiments/unpaired.py extract --bank text_train
python experiments/unpaired.py train
python experiments/unpaired.py extract --bank vision_val
python experiments/unpaired.py extract --bank text_val
python experiments/unpaired.py evaluate --split val
# After freezing all choices:
python experiments/unpaired.py extract --bank vision_test
python experiments/unpaired.py extract --bank text_test
python experiments/unpaired.py evaluate --split test
python experiments/unpaired.py audit
```

Pass the same `--config` to every stage. Use `--set KEY=VALUE` for overrides. To subsample training set `split.train_size_per_side` and new work/result directories. `split.val_size=null` means half the evaluation pool; `split.test_size=null` uses the remainder. Counts are checked against full annotation sizes; missing captions or decode errors are never silently dropped.

The algorithm remains KMeans/QAP anchors → relative-neighbor Procrustes → NN refinement → cluster correction. QAP now selects the **maximum** objective and uses seeded valid random 2-opt starts. The UNITRAN preset retains `refine_sample=10000`; the upstream notebook used 1000. Saved train means and these fixes change old scores. Directions are fitted independently: an average of orthogonal maps need not be orthogonal, so reverse fitting does not assume `W.T`.

## Artifacts and validation

```text
feature/unpaired/<setting>/
  vision_train.jsonl, text_train.jsonl, paired_val.jsonl, paired_test.jsonl
  split_meta.json
  vision_train.pt, text_train.pt
  vision_val.pt, text_val.pt, vision_test.pt, text_test.pt
outputs/unpaired/<setting>/
  mapping.pt                  # both maps, train means, config, bank/code provenance
  resolved_config.json, training.json
  retrieval_val.json, retrieval_test.json, data_audit.json
```

Cache validation covers manifest content/order, caption content, encoder settings/revisions and finite/nonzero features. Checkpoints hash training banks and the split protocol. Changing a split requires a new `work_dir`; `--force` replaces explicit banks/checkpoints, not manifests. Safe tensor loading uses `weights_only=True`.

I2T/T2I report FOSCTTM, mean rank and Recall@1/5/10 for **image-level caption means**, with identity lookup after independent query/key shuffles and midrank ties. This is not the usual 5-caption COCO benchmark. Score blocks avoid full NxM matrices; exact search still costs O(N×M), so full-size timing and memory require real measurement.

```bash
python -m unittest discover -s tests -v
python tools/smoke_check.py
python tools/audit_repository.py   # ignored full baseline AST/call-site report
```

See [audit](docs/audit.md), [all 76 baseline files](docs/audit_inventory.md), [migration/deletions](docs/migration.md), and [validation limits](docs/validation.md). CPU regressions use synthetic data and stub encoders. Actual COCO/SPC data, pretrained feature extraction and CUDA execution have not been verified here; no successful full-data run or retrieval quality is claimed.

## Structure and optional experiments

The main entry is `experiments/unpaired.py`. Protocols/encoders/artifacts live in `src/unpaired/`; reusable mapping/metrics live in `src/training/mini_vec2vec.py` and `src/evaluation/retrieval.py`. Existing config helpers and DINOv2 preprocessing are reused.

Referenced patch/local/global clustering, Llama atoms, density, CLIP↔DINO, class prototypes, adversarial vec2vec and visualizations remain in [optional legacy workflows](docs/legacy_workflows.md), using `requirements-optional.txt`. Legacy row banks cannot certify the primary image-ID protocol. The duplicate `experiments/unpaired_mpnet/` subtree is merged into the core.

MIT license retained. References: [mini-vec2vec](https://arxiv.org/abs/2510.02348), [DINOv2](https://github.com/facebookresearch/dinov2), [MPNet encoder](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
