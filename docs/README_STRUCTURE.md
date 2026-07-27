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

### src level folders
- `io/`: scripts for inputting and outputting data.
- `preprocessing/`: any processing which needs to be initially done to the raw data.
- `sleep_scoring/`: automated sleep scoring using somnotate analysis.
- `spectral`: scripts for spectral analysis.
- `visualisation`: any scripts required for visualising data.
- `utils`: small helper functions.

## Sections for inclusion
- `io/`: source and sink connectors, contracts, checkpoint helpers.
- `processing/`: realtime transforms and analysis stages.
- `utils/`: shared config, logging/metrics, quality/time helpers.
- `state/`: local runtime state (cursor/checkpoint snapshots).


## Runtime flow
1. `io/sources` reads complete EEG recordings from external storage.
2. `io/checkpoints` loads/saves the last completed recording cursor.
3. `processing` handles filtering/windowing/features for each recording.
4. `processing/analysis` computes metrics/summaries.
5. `io/sinks` writes results to external output storage.
6. `state/` updates cursor so recording processing can resume safely.
