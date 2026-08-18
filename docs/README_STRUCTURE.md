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
- `src/`: scripts containing major processing modules of repo.
- `scripts/`: developed scripts for easy run and execution points.
- `tests/`: fixtures and integration tests for quality control.
- `docs/`: contracts and operations runbooks.
- `cache/`: optional ephemeral local staging.

### `scripts/` folders
- `analysis/`: reusable scientific calculations imported by executable scripts;
  these modules are not command-line entry points.
- `io/`: recording discovery, data-location configuration, and MNE file-loading
  helpers.
- `preprocessing/`: executable preprocessing workflows.
- `qc/`: executable reporting and visualisation workflows.
- `sleep_scoring/`: executable automated and interactive sleep-scoring workflows.
- `utils/`: small cross-cutting helpers such as configuration, artifact, and
  sleep-state handling.

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
