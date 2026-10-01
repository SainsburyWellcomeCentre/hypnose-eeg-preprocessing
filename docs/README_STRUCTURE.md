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
- `src/hypnose_eeg/`: the installable package (`pip install -e .`), in the
  standard src layout; every module imports from any directory, and the
  pipeline commands (`hypnose-eeg-pipeline`, ...) are its console scripts.
- `tests/`: fixtures and integration tests for quality control.
- `docs/`: contracts and operations runbooks.
- `cache/`: optional ephemeral local staging.

### `src/hypnose_eeg/` subpackages
- `analysis/`: reusable scientific calculations; these modules are not
  command-line entry points.
- `io/`: recording discovery, data-location configuration, and MNE file-loading
  helpers.
- `pipeline/`: the pipeline stages (preprocessing, sleep scoring, QC), the full
  and batch runs that chain them, and the QC review. Each stage calls the
  package's command-line steps in-process.
- `preprocessing/`: preprocessing workflows.
- `qc/`: quality-control reports and plots.
- `sleep_scoring/`: automated Somnotate sleep scoring and the interactive
  viewer for scored recordings (`view_scoring.py`).
- `utils/`: small cross-cutting helpers such as configuration, artifact, and
  sleep-state handling.

Workflow modules are both command-line entry points (`python -m
hypnose_eeg.qc.spectra --help`) and Python APIs. The `qc/` modules and
`sleep_scoring/view_scoring.py` separate computing results, plotting or saving
them, and the CLI, so their results can be used without the CLI.
`hypnose_eeg/api.py` is the keyword-argument facade over the pipeline stages.
Configuration is read from `configs/` on first use, not at import time.

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
