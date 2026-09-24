# EEG Recording Repo Arrangement (External Source/Sink)

This repository contains processing and analysis code only.
Raw and processed/derivatives data live outside the repo and are accessed via configurable source/sink adapters.

## Core Principle
- Read from external source storage.
- Process and analyze in this repo.
- Write results to external sink storage.
- Keep cursor/checkpoint state for restart-safe recording processing.

## Top-level folders
- `configs/`: environment/data referencing and pipeline runtime settings.
- `notebooks/`: exploratory jupyter notebooks for development.
- `hypnose_eeg/`: the installable package (`pip install -e .`); every module
  imports from any directory.
- `src/`: pipeline entry points that chain the package's command-line steps,
  run from the repository root (`python -m src.run_pipeline`).
- `tests/`: fixtures and integration tests for quality control.
- `docs/`: contracts and operations runbooks.
- `cache/`: optional ephemeral local staging.

### `hypnose_eeg/` subpackages
- `analysis/`: reusable scientific calculations; these modules are not
  command-line entry points.
- `io/`: recording discovery, data-location configuration, and MNE file-loading
  helpers.
- `preprocessing/`: preprocessing workflows.
- `qc/`: quality-control reports and plots.
- `review/`: visual review of scored recordings (the interactive viewer).
- `sleep_scoring/`: automated Somnotate sleep scoring.
- `utils/`: small cross-cutting helpers such as configuration, artifact, and
  sleep-state handling.

Workflow modules are both command-line entry points (`python -m
hypnose_eeg.qc.spectra --help`) and Python APIs. `qc/spectra.py` and
`review/viewer.py` separate computing results, plotting them, and the CLI, so
their results can be used without the CLI. Configuration is read from
`configs/` on first use, not at import time.

## Sections for inclusion
- `io/`: source and sink connectors, contracts, checkpoint helpers.
- `analysis/`: calculations shared by multiple workflows.
- `processing/`: realtime transforms and analysis stages.
- `utils/`: small domain-independent helpers.
- `state/`: local runtime state (cursor/checkpoint snapshots).


## Runtime flow
1. `io/sources` reads complete EEG recordings from external storage.
2. `io/checkpoints` loads/saves the last completed recording cursor.
3. `processing` handles filtering/windowing/features for each recording.
4. `processing/analysis` computes metrics/summaries.
5. `io/sinks` writes results to external output storage.
6. `state/` updates cursor so recording processing can resume safely.
