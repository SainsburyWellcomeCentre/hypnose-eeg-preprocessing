# Hypnose EEG Analysis

Tools and notebooks for inspecting, concatenating, and downsampling Hypnose EEG
recordings.

## Environment

```bash
conda env create -f environment.yml
conda activate hypnose-eeg-env
```

The environment expects the `hypnose-somnotate` repository in a sibling checkout
at `../hypnose-somnotate` and installs it in editable mode. Somnotate's legacy
`pomegranate` dependency builds from source, so a working C/C++ compiler is also
required.

## Data location

Data stays outside this repository and is referenced through a named,
machine-specific profile. No repository symlink is required.

The shared profiles live in `configs/environments/data_locations.yml`. Select a
profile once for each checkout:

```bash
python src/set_data_location.py --list
python src/set_data_location.py server-linux   # or server-mac / server-windows
python src/set_data_location.py --show
```

The selection is written to `configs/environments/data_locations.local.yml`,
which is ignored by git. Add another named profile to the shared config when a
machine uses a different mount or local copy.

For temporary overrides (for example CI or a one-off local run), set:

```bash
export HYPNOSE_EEG_RAWDATA_ROOT=/path/to/rawdata
export HYPNOSE_EEG_DERIVATIVES_ROOT=/path/to/derivatives
```

Environment variables take precedence over the active profile. CLI arguments
such as `--source-dir` and `--sink-dir` take precedence over both.

## Running preprocessing

Run processing stages in this order:

1. **Concatenate recordings** — `scripts/preprocessing/concatenate_recordings.py`
2. **Inspect and trim channels** — `scripts/preprocessing/inspect_and_trim_channels.py`
3. **Downsample recordings** — `scripts/preprocessing/downsample_recordings.py`
4. **Sleep scoring** — `scripts/sleep_scoring/score_recordings.py`
5. **Detect artifacts** — `scripts/preprocessing/detect_artifacts.py`

The development pipeline config leaves source and sink locations unset so they
are supplied by the active data-location profile:

```bash
python scripts/preprocessing/concatenate_recordings.py \
  --config configs/environments/dev.yaml --dry-run

python scripts/preprocessing/downsample_recordings.py \
  --config configs/environments/dev.yaml --dry-run
```

Configure a Somnotate model and subjects in
`configs/pipelines/sleep_scoring.yaml`, or override them from the command line:

```bash
python scripts/sleep_scoring/score_recordings.py \
  --model my-model --subject 66 --date 20260717
```

The model name resolves below
`derivatives/somnotate_training/<model>/model.pickle`; absolute paths are also
accepted. Raw-data and derivatives roots come from the active data-location
profile unless explicitly overridden.

Visually inspect the raw signals and predicted states for one scored session:

```bash
python scripts/sleep_scoring/view_scored_recording.py \
  --subject 66 --date 20260717
```

Viewer defaults live in the `sleep_scoring_view` section of the same pipeline
config. This is an interactive visual quality check, not a numerical accuracy
measurement; numerical performance requires matching manual ground-truth labels.

Use `--help` on a command for selection, output, and overwrite options.
