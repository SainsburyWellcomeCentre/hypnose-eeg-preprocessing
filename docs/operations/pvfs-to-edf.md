# PVFS to EDF Conversion (Stage 0)

This stage converts recorded `.pvfs` files from an external source location into `.edf` files at a different external destination.

## Why this exists
- Keeps raw data external to the repo.
- Produces a standard EDF format before realtime/analysis pipeline stages.
- Uses a state file to avoid reconverting unchanged files.

## Config
Set values in [configs/environments/dev.yaml](configs/environments/dev.yaml) under `pvfs_conversion`:
- `source_uri`: external path containing `.pvfs` files
- `sink_uri`: external path where `.edf` files should be written
- `command_template`: vendor converter command with `{input}` and `{output}` placeholders
- `state_file`: JSON state path used for dedupe/resume

## Run once
```bash
python scripts/pvfs_to_edf.py --config configs/environments/dev.yaml
```

## Run continuously
```bash
python scripts/pvfs_to_edf.py --config configs/environments/dev.yaml --watch
```

## Dry run
```bash
python scripts/pvfs_to_edf.py --config configs/environments/dev.yaml --dry-run
```

## Notes
- Conversion depends on your vendor `pvfs -> edf` tool.
- If converter command fails, file remains uncommitted in state and will be retried on next run.
