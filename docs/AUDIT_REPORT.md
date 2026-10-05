# Clean-release audit

Audit date: 2026-10-05

The original research workspace was inspected read-only. No original file was
deleted or modified, and its Git history was not copied.

## Canonical implementations

| Concern | Research source | Clean-release destination | Decision |
|---|---|---|---|
| MM-Fi loading | Official `mmfi_lib/mmfi.py` | External toolbox used by `scripts/build_cache.py` | Do not vendor upstream code without a clear software license. |
| P3/S2 split | Validated cross-subject configuration | `scripts/build_cache.py`, `configs/baseline.yaml` | Preserves the official 32/8-subject split. |
| Cached dataset | `cached_mmfi_dataset.py` | `src/wifi_densepose/data/cache.py` | Current lossless mmap interface. |
| CNN model | `train_cnn_overfit.py` | `src/wifi_densepose/models/cnn.py` | CNN v0, exactly 95,827 parameters. |
| Training loop | Validated baseline trainer | `src/wifi_densepose/train.py` | Baseline recipe with portable CLI paths. |
| Evaluation loop | Baseline evaluator and metric validation | `src/wifi_densepose/evaluate.py` | Separate, corrected evaluation command. |
| Raw MPJPE | `mmfi_official_metrics.py` | `src/wifi_densepose/metrics/pose.py` | Retained. |
| Root-aligned MPJPE | `mmfi_official_metrics.py` | `src/wifi_densepose/metrics/pose.py` | Renamed explicitly. |
| PA-MPJPE | `artifacts_metric_validation/corrected_metrics.py` | `src/wifi_densepose/metrics/pose.py` | Corrected implementation; old routine excluded. |
| Baseline config/result | Validated configuration and reevaluation | `configs/baseline.yaml`, `results/baseline.json` | Sanitized, no checkpoints or local paths. |

## Keep

The clean release keeps only the current model, cached-data adapter, official
split declaration, portable training/evaluation commands, corrected metrics,
metric tests, dependency list, documentation, attribution, and a compact
validated result record.

## Exclude from GitHub

- Original `.git` history and untracked research history.
- `.venv`, `__pycache__`, editor state, logs, temporary and failure files.
- All `artifacts_*` directories except the manually transcribed validated
  result values and corrected metric logic.
- Multi-gigabyte NumPy caches, checkpoints, generated figures, presentations,
  environment captures, benchmark probes, and intermediate JSON reports.
- Historical experiment scripts/configs, LR-sweep variants, benchmarks, cache
  recovery/verification utilities, and duplicate training scripts.
- `mmfi_official_metrics.py` because its PA routine is buggy.
- Local absolute paths and machine-specific CUDA/GPU restrictions.

## Manual review

- Select an explicit license before public release; the present `LICENSE` is a
  restrictive placeholder.
- Confirm whether the upstream MM-Fi toolbox code may be redistributed. It is
  intentionally not copied into this release.
- Reconfirm MM-Fi CC BY-NC 4.0 obligations for the intended use.
- Decide whether any generated plots should be curated into a later release.

## Secret and privacy scan

No credential-named files or high-confidence API keys, passwords, private keys,
or access tokens were found in the publishable source set. Two excluded HTTP
header capture files matched generic credential-related terms and remain in the
private research artifact tree. Git logs contain an email address and are not
copied. Numerous artifacts contain absolute local paths; none are copied.

## Rename map

| Research name | Public name |
|---|---|
| Internal baseline training script | `src/wifi_densepose/train.py` |
| Internal baseline evaluation function | `src/wifi_densepose/evaluate.py` |
| `cached_mmfi_dataset.py` | `src/wifi_densepose/data/cache.py` |
| `train_cnn_overfit.py::SmallCSIPoseCNN` | `src/wifi_densepose/models/cnn.py` |
| `artifacts_metric_validation/corrected_metrics.py` | `src/wifi_densepose/metrics/pose.py` |
| Internal baseline configuration | `configs/baseline.yaml` |
| Selected corrected baseline metrics | `results/baseline.json` |

## Reproducibility status

Static import, metric, and model checks are expected to pass without data. Full
training requires the separately obtained MM-Fi dataset, official toolbox, and
approximately 4.5 GB for the generated cache. Exact reproduction can also vary
with PyTorch/CUDA/hardware versions; the validated run used one seed.
