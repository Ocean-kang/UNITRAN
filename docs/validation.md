# Validation record and limits

Development environment (2026-10-14): Windows, Python 3.11.13, PyTorch 2.8.0 **CPU**, NumPy 1.26.4, SciPy 1.16.1, scikit-learn 1.7.1, Pillow 11.3.0, PyYAML 6.0.2, threadpoolctl 3.6.0. GPU hardware is detectable (RTX 2060 Max-Q, 6GB), but the available tested Torch environment has no CUDA backend. Actual repository data/features/model weights were unavailable.

## Executed checks

Local CPU regression result: **18 tests passed**.

The CPU unittest suite uses miniature COCO/SPC annotations, generated images, synthetic embeddings and stub encoders. It checks full odd-size disjoint coverage/reproducibility, full cross-dataset vision coverage, VG source exclusions, caption joins/aggregation, duplicate/orphan annotation rejection, bank order/revision/nonzero guards, manifest/config tampering, exact blockwise neighbors, shuffled ID lookup, midrank ties, QAP maximum selection, train means/padding and repeatable fitting. It runs both maps and both evaluation partitions in both settings, intercepts any training read of evaluation banks, and rejects altered training banks at evaluation.

`tools/smoke_check.py` compiles source/tests and checks every retained experiment argument parser. `tools/audit_repository.py` reproduces all baseline file/definition/call-site inventory. GitHub Actions is configured to run CPU checks on pushes/PRs. Clean installation of the complete pinned encoder/optional environment, actual model loading and real-data legacy execution are not certified by these synthetic checks.

## Unverified items

- Actual 82,783/40,504 source annotations, image completeness/decode success and real caption joins. Runtime guards exist, but the datasets were unavailable.
- Actual SPC/VG provenance coverage, exclusions and resulting training size.
- DINOv2/MPNet downloads, offline loading, encoder numerical outputs and CUDA/mixed-precision execution.
- Full-scale KMeans/QAP convergence, exact search/refinement runtime and peak RAM/VRAM.
- COCO val/test retrieval quality or comparison with papers. Synthetic scores are fixtures, not experiment results.
- Perceptual image/text duplication and overlap with pretrained-model corpora.
- Real optional workflow execution, Llama generation and multi-GPU model parallelism.

For a data-equipped run, freeze config/environment/revisions, prepare and inspect counts/exclusions, extract training banks, audit, fit, then encode/evaluate validation. Freeze choices before test encoding/evaluation. Archive source/split/model/bank digests, package versions, code commit and measured time/memory. Full data is the default; no successful 80k training run is claimed in this PR.
