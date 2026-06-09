# EEG Streaming Repo Arrangement (External Source/Sink)

This repository contains processing and analysis code only.
Raw and processed/derivatives data live outside the repo and are accessed via configurable source/sink adapters.

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
1. `io/sources` reads new EEG chunks from external storage.
2. `io/checkpoints` loads/saves read cursor.
3. `processing/realtime` handles filtering/windowing/features.
4. `processing/analysis` computes metrics/summaries.
5. `io/sinks` writes results to external output storage.
6. `state/` updates cursor so the stream can resume safely.
