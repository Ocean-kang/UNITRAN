# Migration and deletion list

Use `experiments/unpaired.py` and `configs/unpaired_*.yaml` instead of the appended MPNet workflow. Raw datasets are unchanged. Old caches/checkpoints lack sufficient provenance for silent conversion: extract and train into new directories. Referenced patch, density, LLM, CLIP, head and vec2vec workflows remain in [legacy_workflows.md](legacy_workflows.md).

| Old action | New command |
|---|---|
| `split_coco.py` | `python experiments/unpaired.py prepare` |
| `extract_vision.py --split vision_train` | `python experiments/unpaired.py extract --bank vision_train` |
| `extract_text.py --split text_train` | `python experiments/unpaired.py extract --bank text_train` |
| Extract `paired_val` | `extract --bank vision_val` / `extract --bank text_val` |
| `train.py` | `python experiments/unpaired.py train` |
| `evaluate.py` | `python experiments/unpaired.py evaluate --split val` |
| `test_small.py` | `python -m unittest discover -s tests -v` |
| `run.sh [config]` | `python experiments/unpaired.py all --config <config>` |

Pass `--config configs/unpaired_cross_dataset.yaml` to every cross-dataset stage. That setting requires SPC paragraphs and Visual Genome provenance; it is not another COCO caption subset. Full coverage is 41,391+41,392 IDs for disjoint, or 82,783 visual IDs plus the filtered external corpus for cross-dataset.

## Schema and result changes

Banks contain `features`, namespaced `sample_ids`, and cache `metadata`. Checkpoints store both maps, train means, split digest, training-bank hashes, config and package/code provenance. Old W tensors cannot recreate this state. `mapping.pt` replaces `W_mini_vec2vec_vision_to_text_seed*.pt`; `training.json` records train-only refinement shifts instead of paired per-iteration cosine. Validation/test have separate retrieval reports.

QAP selection/random starts and train-only centering are corrected; rows are independently shuffled; reverse direction is fitted independently; encoder revisions are pinned. Old seeded trajectories/scores are not bitwise compatible. The same stages/objectives and UNITRAN hyperparameter preset remain. Treat corrected results as a new experiment version.

Legacy `train_mapping.py` delegates mini fitting to the shared core, preserves W/result filenames and adds `preprocessing_*.pt` for means. It warns that row banks cannot certify image-level disjointness; legacy scores must not be mixed with the primary benchmark. Adversarial vec2vec remains optional. Python changes to 3.11; dependencies are split into core/optional requirements. Install the optional file to use plots/Llama/mask prototypes.

## Removed files

All are recoverable from baseline Git history and have explicit replacements:

| File | Replacement |
|---|---|
| `experiments/unpaired_mpnet/README.md` | Root README/migration |
| `experiments/unpaired_mpnet/config.yaml` | Two primary presets |
| `experiments/unpaired_mpnet/common.py` | `src/unpaired/config.py`, `io.py`, `data.py` |
| `experiments/unpaired_mpnet/split_coco.py` | `src/unpaired/data.py` |
| `experiments/unpaired_mpnet/extract_vision.py` | `src/unpaired/features.py` |
| `experiments/unpaired_mpnet/extract_text.py` | `src/unpaired/features.py` |
| `experiments/unpaired_mpnet/train.py` | Shared mini core and pipeline |
| `experiments/unpaired_mpnet/metrics.py` | `src/evaluation/retrieval.py` |
| `experiments/unpaired_mpnet/evaluate.py` | `src/unpaired/pipeline.py` |
| `experiments/unpaired_mpnet/test_small.py` | `tests/test_unpaired.py` |
| `experiments/unpaired_mpnet/run.sh` | Unified `all` command / `scripts/run_unpaired.sh` |

Also removed in-place: duplicate mini math/training from `unitran_mapping.py` and obsolete commented prompt code from `text_atoms.py`. No tracked generated feature/checkpoint/result files existed. User datasets/runtime outputs are neither deleted nor moved. Full audit-call JSON is generated on demand under ignored `outputs/`; the concise requested audit inventory is tracked documentation.
