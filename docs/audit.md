# Experiment audit

Baseline: `6160deceb3ca457d4ac7b6e2e31ece497f1b38ad` (2026-10-14). The connected GitHub repository, recursive tree and commit history were inspected; the exact checkout was used for static audit/refactoring. `tools/audit_repository.py` reads every tracked file and parses every Python definition, import, dispatch string and call site without importing external code. [All 76 files](audit_inventory.md) have recorded dispositions; detailed JSON/call sites are reproducible under ignored `outputs/`.

## Experiment and original call chains

The original main task extracts 448px DINOv2 patches, builds global clusters or 20 local centers/image, rewrites captions into Llama atoms, encodes them with MPNet, compares distributions, and trains mini-vec2vec or adversarial vec2vec. The newest commit appended an independent image-level experiment: mean DINOv2 patches and averaged original captions. This refactor promotes the image-level task while retaining referenced exploratory functions.

| Entry/family | Chain and artifacts | Disposition |
|---|---|---|
| `extract_dinov2_features.py` | config helpers → `features.dinov2` → image rows/loader → forward features/hooks → sharded patches/pooled/head embeddings | Shared preprocessing and optional extraction retained |
| `run_global_patch_cluster.py` | `clustering.stream_kmeans` → shards → streaming assignment/centroids → counts/cluster IDs | Valid workflow retained |
| `run_local_patch_cluster.py` | `clustering.local_kmeans` → batched per-image KMeans → centers/weights/assignments/flat exports | Valid workflow retained |
| `run_text_atoms.py` | `features.text_atoms` → caption sample → HF/Meta backend → prompt → JSON parse/fallback → atoms JSONL | Optional; obsolete commented prompt removed |
| `external/llama/*` | `text_atoms.build_meta_generator` → `llama.Llama` → `generation_feat` → model/tokenizer | Referenced backend retained |
| `run_density_eval.py` | stage dispatch → `local_density`/`text_density` → vision shards/MPNet → MMD/SWD/coverage | Distribution diagnostics retained |
| `train_mapping.py` | config adapter → `training.unitran_mapping` → row-bank loader → mini/vec2vec | Mini delegates to shared core; row protocol warns |
| `extract_eval_prototypes.py` | masks → patch labels → class vision means and text prompts | Optional class evaluation retained |
| `extract_clip_dinov2_patch_bank.py` | CLIP/DINO square tokens → corresponding patch grids → dual vision bank → mapping | Distinct exploratory task retained |
| `run_visualize_cluster.py` | dispatch → top-k/extreme cluster patches, CLS/local attention, PCA/t-SNE | All dispatched modules retained |
| `visualize_disentangled_heads.py` | pooled/per-head embeddings → projections/plots/rank diagnostics | Latest standalone CLI retained |
| `datasets.coco`, `features.patch_store` | standalone caption/patch-assignment queries | Standalone CLIs retained despite no importer |
| `scripts/*.sh` | thin wrappers around config entries | Retained; smoke checking consolidated |
| `unpaired_mpnet/*` | local IO → ID split → encoders → duplicated fit → row-aligned validation | 11 files consolidated and removed |

## History and cleanup boundary

The baseline history has 42 commits. Cleanup commits `324a7bf`, `ad5678e`, `b9e6932`, `2c93ac1` already removed earlier top-level code. Later commits added templates (`92dacb3`), CLIP-DINO (`4007b5d`), pooled/head DINO (`8230950`, `d264d38`) and the ID-disjoint experiment (`6160dec`). No tracked checkpoints, feature arrays, images, `.pyc` files or result JSON existed at baseline. No result was deleted speculatively. Runtime artifacts are ignored; the duplicate MPNet subtree and commented prompt are removed; referenced optional code is retained rather than mislabeled garbage.

## Verified protocols

The baseline has one image-ID setting: two halves of a train2014 permutation. It contains no overlapping/shuffled same-corpus second setting. The user clarified that the second setting is cross-dataset. At [Rosetta commit `fdfad84`](https://github.com/dominik-schnaus/unpaired-rosetta/tree/fdfad84ea488f29702a61a455be71622c4ff5d49):

- [evaluation.py](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/unpaired_rosetta/evaluation.py): `disjoint_split` uses permutation halves; `independent_split` uses separate corpora.
- [cross_dataset.py](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/experiments/cross_dataset.py): X=`coco_train2014`, Y=`StanfordParagraphCaptioning`. Reported cross-dataset figures use Rosetta's aligner; mini-vec2vec under this protocol is an extension.
- [datasets.py](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/modalities/image_captions/datasets.py): one paragraph per Visual Genome image. Different corpus names do not prove distinct physical images; the new loader checks source `coco_id` and records exclusions.
- [mini baseline](https://github.com/dominik-schnaus/unpaired-rosetta/blob/fdfad84ea488f29702a61a455be71622c4ff5d49/unpaired_rosetta/baselines/mini_vec2vec.py): maximize QAP; use train-only means; smoothed maps are not reprojected to be orthogonal.

## Findings and fixes

| Finding | Consequence | Fix |
|---|---|---|
| Default 1k/5k sampling; filename mentioning 80k does not prove coverage | Full-data objective missing from defaults | Assert 82,783 source images; full disjoint union or full cross vision side |
| Feature-row splitting with multiple atoms/image | Image ID can straddle boundary | Split IDs before encoding; omit captions from vision training manifest |
| QAP `maximize=True` but chooses minimum `.fun`; invalid `P0` for 2opt; warnings suppressed | Worst restart selected, solver behavior obscured | Choose maximum; valid seeded explicit random starts |
| Evaluation-set centering; training opens paired val banks | Transductive statistics and mixed responsibilities | Save training means; fit only X/Y/params; no evaluation bank access in training |
| Legacy density lacks identities and treats feature rows as partners | Patch/atom ranks do not prove semantic retrieval | ID-aware independently shuffled primary evaluation; legacy limitation documented |
| Cache checks IDs/model label only | Stale captions/pooling/weights/revisions can survive | Manifest, source and encoder-content fingerprints; checkpoint hashes training banks |
| Only validation, no independent test | Test claims vulnerable to repeated inspection | Disjoint val/test partitions of val2014; freeze choices before test |
| Reverse ranking reuses forward similarity | Reverse mapping never fitted; EMA map not necessarily orthogonal | Fit each direction independently |
| Full score matrix and large neighbor gathers | Full-scale RAM cost | Blockwise exact queries/aggregation; still quadratic time |

Primary chain: `unpaired.main → prepare/audit → extract → train → fit_unpaired(X,Y,params) → evaluate → retrieval`. IDs are used only for protocol guards and evaluator truth, not fit objectives. Validation/test vectors never determine means, clusters, weights, early stopping or checkpoint selection.

## Scientific limits

ID/source metadata checks do not certify all perceptual duplicates/paraphrases or pretrained-encoder corpus overlap. KMeans/QAP can yield poor semantic alignment even with a correct protocol. Library versions, seeds and pooling affect results. Actual corpus overlap counts, convergence, timing/memory and retrieval quality were not inferred from synthetic fixtures; see [validation](validation.md).
