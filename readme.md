# Hypnose EEG Analysis

Tools and notebooks for inspecting, concatenating, and downsampling Hypnose EEG
recordings.

## Environment

```bash
conda env create -f environment.yml
conda activate hypnose-eeg-env
```

The environment expects the `hypnose-helpers` and `hypnose-somnotate` repositories
in sibling checkouts and installs them in editable mode. Somnotate's legacy
`pomegranate` dependency builds from source, so a working C/C++ compiler is also
required.

## Data location

Data stays outside this repository and is referenced through a named,
machine-specific profile. Profile resolution and selection are provided by
`hypnose_helpers`; this repository supplies the EEG-specific config directory
and `HYPNOSE_EEG` environment-variable prefix. No repository symlink is required.

The shared profiles live in `configs/data_locations.yml`. Select a
profile once for each checkout:

```bash
python scripts/utils/set_data_location.py --list
python scripts/utils/set_data_location.py server-linux   # or server-mac / server-windows
python scripts/utils/set_data_location.py --show
```

The selection is written to `configs/data_locations.local.yml`,
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
  --subject 66 --date 20260717 --hours 3 6
```

`--hours START END` limits EDF loading and the viewer to an elapsed-hour range
measured from the start of the recording. Omit it to load the complete recording.
Alternatively, select real timestamps from the EDF clock. When `--date` is
omitted, the viewer searches the subject's EDF headers and selects the recording
whose real-time span contains the requested start. This supports multi-day
recordings where the selected range begins after the recording's calendar date:

```bash
python scripts/sleep_scoring/view_scored_recording.py \
  --subject 66 \
  --time-range "20260718 03:00:00" "20260718 06:00:00"
```

`--hours` and `--time-range` are mutually exclusive.

For faster rendering, downsample the selected interval to 128 Hz. Artifact
regions from the matching `*_artifact_epochs.parquet` beside the scoring output
can also be shaded and labelled:

```bash
python scripts/sleep_scoring/view_scored_recording.py \
  --subject 66 \
  --time-range "20260718 03:00:00" "20260718 06:00:00" \
  --display-rate 128 --show-artifacts
```

Viewer defaults live in the `sleep_scoring_view` section of the same pipeline
config. This is an interactive visual quality check, not a numerical accuracy
measurement; numerical performance requires matching manual ground-truth labels.
For remote Linux execution with the window displayed on a local Mac through
XQuartz—including VS Code Remote SSH—follow
[`docs/remote_visualization.md`](docs/remote_visualization.md).

Use `--help` on a command for selection, output, and overwrite options.

## Recording integrity checks

With an active data-location profile, select a recording using its subject and
either session date or session number; rawdata and derivatives paths are resolved
automatically:

```bash
python scripts/quality_control/recording_integrity.py \
  --subject 66 --date 20260717

python scripts/quality_control/recording_integrity.py \
  --subject 66 --session 1
```

Compare one raw EDF with its derivative FIF, scanning only the EDF for gaps and
opening the FIF only for duration metadata:

```bash
python scripts/quality_control/recording_integrity.py \
  --edf /path/to/rawdata/session/recording.edf \
  --fif /path/to/derivatives/session/recording_resampled-128hz_raw.fif
```

Omit `--edf` and `--fif` to automatically pair recordings beneath the active
rawdata and derivatives roots. Use `--edf-pattern` and `--fif-pattern` to narrow
the batch selection:

```bash
python scripts/quality_control/recording_integrity.py \
  --edf-pattern "sub-066/**/*.edf" \
  --fif-pattern "sub-066/**/*_raw.fif"
```

The command prints the duration of every constant/non-finite interval and
gap-like EDF annotation in seconds. The default 30-minute chunks reduce reader
and network overhead while keeping memory bounded. It does not write files by default. Add
`--gaps /path/to/recording_integrity_gaps.csv` to save full gap details or
`--summary /path/to/recording_integrity.csv` to save duration comparisons and
pass/review status. Defaults are a one-second duration tolerance and a one-second
minimum gap; override them with `--duration-tolerance` and `--min-gap`. A
recording requiring review causes a non-zero command exit status.

## Sleep-state power spectra

Plot the mean EEG power spectral density for Wake, NREM, and REM for one
subject and either a session date or session number:

```bash
python scripts/quality_control/plot_spectra.py \
  --subject 66 --date 20260717

python scripts/quality_control/plot_spectra.py \
  --subject 66 --session 1
```

The script resolves the derivative FIF and matching Somnotate prediction file
from the active data-location profile. It reads the FIF in bounded chunks rather
than preloading the complete recording, uses four-second analysis epochs by
default, and plots one PSD panel per EEG channel. It also creates three separate
histograms of per-epoch EMG RMS amplitude in µV—one each for Wake, NREM, and
REM—when the FIF contains channels typed as EMG. Matching artifact epochs are excluded
automatically when an `*_artifact_epochs.parquet` file is available; use
`--include-artifacts` to retain them. Plots use the shared `hypnose_helpers`
figure style. When saved, the EEG and EMG figures are provenance-tagged PDFs
named with the `_sleep_state_power_spectra` and `_sleep_state_emg_rms` labels,
respectively.

Save plots without opening an interactive window with:

```bash
python scripts/quality_control/plot_spectra.py \
  --subject 66 --date 20260717 \
  --save-dir /path/to/plots --no-show
```

Use `--fmin`, `--fmax`, `--epoch-seconds`, `--welch-seconds`, and
`--chunk-epochs` to adjust the spectrum calculation and memory/runtime tradeoff.

## Sleep-state channel correlation

Plot the distribution of epoch-wise Pearson correlation for every EEG/EMG
channel pair, separated into Wake, NREM, and REM:

```bash
python scripts/quality_control/plot_channel_correlations.py \
  --subject 66 --date 20260717

python scripts/quality_control/plot_channel_correlations.py \
  --subject 66 --session 1
```

The command reads the derivative FIF in bounded chunks, uses four-second epochs
by default, and automatically excludes gaps, undefined sleep states, non-finite
or constant channel pairs, and epochs marked in the matching artifact parquet.
Each histogram shows the percentage of valid epochs for its sleep state on the
common Pearson range from −1 to +1; the dashed line marks the median. Use
`--include-artifacts` to retain flagged epochs, `--bins` to change histogram
resolution, or `--save-dir /path/to/plots --no-show` for non-interactive,
provenance-tagged PDF output using the shared figure style.
The terminal also reports counts above the default review thresholds of
`|r| > 0.90` for EEG–EEG pairs and `|r| > 0.50` for EEG–EMG pairs, including a
de-duplicated total of flagged recording epochs. Override these with
`--eeg-eeg-threshold` and `--eeg-emg-threshold`.

## Artifact burden report

Report artifact counts, duration, percentage per recording hour, sleep-state
breakdown, and longest contiguous artifact period separately by channel:

```bash
python scripts/quality_control/report_artifacts.py \
  --subject 66 --date 20260717

python scripts/quality_control/report_artifacts.py \
  --subject 66 --session 1
```

The report uses the matching `*_artifact_epochs.parquet`, infers its epoch
duration from `time_s`, and supports both `artifact_channels` and legacy
channel-prefixed `artifact_features`. EMG-supported artifact epochs are reported
as `EMG (combined)` when the stored reason identifies an EMG-supported EEG
outlier. Nothing is saved by default;
add `--save-dir /path/to/reports` to write separate overall, hourly, and
sleep-state CSV tables. Use `--epoch-seconds` only when the duration cannot be
reliably inferred from the artifact file.
