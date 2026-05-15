# EEG Streaming Repo Arrangement (External Source/Sink)

This repository contains processing and analysis code only.
Raw and output data live outside the repo and are accessed via configurable source/sink adapters.

## Core Principle
- Read from external source storage.
- Process and analyze in this repo.
- Write results to external sink storage.
- Keep cursor/checkpoint state for restart-safe continuous processing.

## Top-level folders
- `io/`: source and sink connectors, contracts, checkpoint helpers.
- `processing/`: realtime transforms and analysis stages.
- `utils/`: shared config, logging/metrics, quality/time helpers.
- `configs/`: environment and pipeline runtime settings.
- `state/`: local runtime state (cursor/checkpoint snapshots).
- `cache/`: optional ephemeral local staging.
- `outputs/`: local debug artifacts only.
- `tests/`: fixtures and integration tests.
- `docs/`: contracts and operations runbooks.

## Runtime flow
1. `scripts/pvfs_to_edf.py` converts source `.pvfs` files to `.edf` in external sink storage.
2. `io/sources` reads new EEG chunks from external storage.
3. `io/checkpoints` loads/saves read cursor.
4. `processing/realtime` handles filtering/windowing/features.
5. `processing/analysis` computes metrics/summaries.
6. `io/sinks` writes results to external output storage.
7. `state/` updates cursor so the stream can resume safely.
