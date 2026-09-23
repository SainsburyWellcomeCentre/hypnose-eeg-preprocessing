# Hypnose EEG Preprocessing

Tools and notebooks for inspecting, concatenating, downsampling and qc Hypnose EEG
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
python scripts/io/repository_paths.py --list
python scripts/io/repository_paths.py server-linux   # or server-mac / server-windows
python scripts/io/repository_paths.py --show
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

### Output folders within a session

Every derivative lands below its recording's derivatives session directory,
`<derivatives>/sub-XXX/ses-YYY_date-.../`, inside a shared `eeg/` modality
folder, in one of five named folders -- `downsample`, `sleep_scoring`,
`sleep_scoring_qc`, `artifacts`, and `quality_control`:

```
derivatives/sub-066/ses-1_date-20250420/
└── eeg/
    ├── downsample/
    ├── sleep_scoring/
    ├── sleep_scoring_qc/
    ├── artifacts/
    └── quality_control/
```

The `eeg/` root and the five folder names come from
`configs/output_layout.yaml`, so by default the layout follows the data
location and nothing needs to be set.

When this repository is driven from elsewhere (another project, a shared
server job) and those folders should sit somewhere else *within* each session
directory, override them without editing the checkout:

```bash
# Whole layout: a copy of configs/output_layout.yaml with your own folder names
export HYPNOSE_EEG_OUTPUT_LAYOUT=/path/to/output_layout.yaml
# ...or the modality root alone (`.` puts the five folders in the session directory)
export HYPNOSE_EEG_OUTPUT_ROOT=eeg
# ...or one folder at a time, relative to that root
export HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS=analysis/artifacts

# The same three overrides as flags on the src/ entry points
python -m src.run_pipeline --subject 66 --session 1 --model my-model \
    --output-layout /path/to/output_layout.yaml --output-root eeg \
    --output-dir artifacts=analysis/artifacts --output-dir quality_control=reports/qc
```

Precedence, highest first: `HYPNOSE_EEG_OUTPUT_DIR_<GROUP>` (or `--output-dir`)
and `HYPNOSE_EEG_OUTPUT_ROOT` (or `--output-root`), then
`HYPNOSE_EEG_OUTPUT_LAYOUT` (or `--output-layout`), then the repository's
`configs/output_layout.yaml`. A group the override file does not mention keeps
its built-in name. The root applies to the group folders whichever way they
were set, so `HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS=analysis/artifacts` lands in
`eeg/analysis/artifacts`. Folders may be nested (`analysis/artifacts`) but must
stay relative to the session directory -- absolute paths and `..` are rejected
-- because readers (`scripts/qc/*`, the viewer) locate earlier outputs through
the same names. From Python,
`run_steps(..., output_layout=..., output_root=..., output_dirs={...})` on
`src.preprocessing`/`src.sleep_scoring`/`src.qc` takes the same overrides.

## Unified pipeline entry points

`src/run_pipeline.py` runs the full pipeline for one subject/session in the
order the stages actually require — trim, concatenate and downsample, then
sleep scoring, then artifact detection (which depends on the sleep-scoring
output), then the QC summary:

```bash
python -m src.run_pipeline --subject 66 --session 1 --model my-model
```

Each stage also has its own entry point that can be run on its own, with a
`--steps` selector for that stage's individual scripts:

```bash
python -m src.preprocessing --subject 66 --session 1 --steps trim concatenate downsample
python -m src.sleep_scoring --subject 66 --session 1 --model my-model
python -m src.qc --subject 66 --session 1
```

Pass `--stage preprocessing`/`sleep_scoring`/`qc` (one or more) to
`run_pipeline.py` to restrict a full run to those stages, each using its own
default step set.

`--view` ends the run in the interactive scoring viewer
(`scripts/sleep_scoring/view_scored_recording.py`), so a session can be
inspected as soon as it has been processed. It opens a plot window, so it needs
a display (see [docs/remote_visualization.md](docs/remote_visualization.md) for
working over SSH), and it runs after every selected stage — artifact detection
and the QC summary included. Because it only reads what is already on disk it
also works on its own:

```bash
# Process the session, then look at the result
python -m src.run_pipeline --subject 66 --session 1 --model my-model --view
# Just look at an already-processed session
python -m src.run_pipeline --subject 66 --session 1 --stage qc --view
# Viewer-only options (--hours, --eeg-channel, --show-artifacts, ...) go here
python -m src.sleep_scoring --subject 66 --session 1 --steps view --hours 3 6
```

Every step whose output already exists is skipped and the run continues with
the next one, so an interrupted or partially-completed session is resumed by
rerunning the same command: channel trimming, concatenation, downsampling,
sleep scoring, and artifact detection each leave their existing outputs in
place. Pass
`--overwrite` to recompute them regardless. The QC summary is the exception --
it is cheap and always refreshed.

### Batch: every session of one or more subjects

`--all-sessions` (or `--batch`) takes subject IDs alone and works through every
session each of them has, in order. `--subject all` covers every subject in the
rawdata tree:

```bash
python -m src.run_pipeline --subject 66 --all-sessions --model my-model
python -m src.run_pipeline --subject 66 67 68 --all-sessions --model my-model
python -m src.run_pipeline --subject all --all-sessions --model my-model
```

A session that fails does not stop the batch. Its derivative outputs are
erased, the failure is written to a per-run CSV report, and the next session
starts; the run ends with a summary and exits non-zero if anything failed:

```
Batch summary: 12/15 sessions completed across 3 subjects, 1 failed QC
  sub-066: 5/5 completed
    [    OK] sub-066/ses-001_date-20260711
    ...
  sub-067: 3/5 completed
    [FAILED] sub-067/ses-004_date-20260714 -- sleep_scoring:score (scripts.sleep_scoring.score_recordings) exited with status 1 (erased 1 path(s))
    [FAILED] sub-067/ses-005_date-20260715 -- QC FAIL (artifacts) (erased 1 path(s))
  sub-068: No sessions found for sub-068
Failed QC:
  sub-067/ses-005_date-20260715 (artifacts)
Total time: 5h 12m 40s
```

A QC summary that comes out FAIL fails its session like any other step (and
the batch exits non-zero), but it is marked `QC FAIL` and listed under
`Failed QC` with the sections that failed, so it is not mistaken for a crash.

A subject whose sessions cannot be resolved at all is recorded the same way as
a failed session (`missing`) rather than stopping the subjects after it; the
per-subject tallies appear once more than one subject runs.

Erasing matters because every step skips work whose output already exists: a
parquet or FIF half-written before the crash would otherwise be reused by the
next run as if it were complete. What is erased is only the output folders this
pipeline writes below that session's *derivatives* directory (`eeg/` by
default, following any `--output-root`/`--output-dir` override in force) —
never the rawdata, and never another modality's folder in the same session.
Add `--erase-derived-edf` to also remove the `_trimmed` and `_recording-concat`
EDFs the pipeline wrote beside the raw recordings, or `--keep-failed` to record
the failure and erase nothing.

The report has one row per session: its subject, status, QC status (`pass` or
`review` for a session that finished, `fail` for one that failed QC, blank when
the QC summary did not run) and failed QC sections, duration, the step that failed, the error, the paths erased, and the
whole batch's run time (`batch_duration_seconds`, the same on every row). A single-subject run leaves it at
`<derivatives>/sub-XXX/batch_report_<timestamp>.csv`; a run spanning several
subjects writes one combined report at `<derivatives>/batch_report_<timestamp>.csv`,
since no one subject owns it. `--report FILE` puts it anywhere else. It is
timestamped per run, so a rerun never overwrites the report that recorded why
the first run failed.

`--view` is refused in batch mode: the viewer waits for a window to be closed
and a batch run has nobody to close it. Review a session afterwards with
`--session N --stage qc --view`.

Run any of the four with `--help` for its complete option list; these wrap the `scripts/*` CLIs below as subprocesses rather than
reimplementing them, so step-specific flags such as `--config` are best
passed to the underlying script directly when a per-step entry point doesn't
already expose them.

## Output provenance

Artifact detection, sleep scoring, and the QC summary each write a
`<output stem>_provenance.json` sidecar beside their outputs recording the git
commit of this checkout (with a `dirty` flag when the working tree had
uncommitted edits), when the step ran, the command line, the inputs consumed,
and the parameters that determine the result. Sleep scoring additionally
records which model was used, identified by SHA-256 content hash as well as by
path, so a replaced `model.pickle` is still distinguishable:

```json
{
  "schema": "hypnose-eeg-provenance/1",
  "stage": "sleep_scoring",
  "generated_at": "2026-09-18T14:59:58+00:00",
  "git": {"commit": "93e8648...", "branch": "main", "dirty": false},
  "parameters": {
    "model": {"path": ".../somno_model_1/model.pickle", "sha256": "...", "size_bytes": 4194304},
    "model_name": "somno_model_1",
    "sampling_rate_hz": 512
  }
}
```

The sidecar is written only when the output itself is written, so a step
skipped because its output already exists keeps the provenance of the run that
actually produced it. Outside a git checkout (an installed copy, an exported
tarball, a machine without `git`) the `git` field is `null` rather than the run
failing.

## Running preprocessing

Run processing stages in this order:

1. **Trim duplicate channels** — `scripts/preprocessing/trim_duplicate_channels.py`
2. **Concatenate recordings** — `scripts/preprocessing/concatenate_recordings.py`
3. **Downsample recordings** — `scripts/preprocessing/downsample_recordings.py`
4. **Sleep scoring** — `scripts/sleep_scoring/score_recordings.py`
5. **Detect artifacts** — `scripts/preprocessing/detect_artifacts.py`

Trimming is automatic and usually a no-op. Some recordings list the same
channel label twice in their EDF header, which breaks concatenation (channel
layouts must match) and sleep scoring (MNE renames non-unique labels to
`EEG1A-B-0`, `EEG1A-B-1`, ...). For every source recording in the session the
step reads the header and, only when a label repeats, writes a
`<stem>_trimmed.edf` copy beside it that keeps the first occurrence of each
label; the source file is never modified. Every later step prefers a trimmed
copy over its source, so a session with duplicates flows through concatenation,
downsampling, scoring, and QC without further action. Use `--dry-run` to see
the channel list, the duplicates found, and whether each repeat carries the
same samples as the channel it duplicates:

```bash
python scripts/preprocessing/trim_duplicate_channels.py --subject 66 --session 1 --dry-run
```

For a recording whose surplus channels are *not* duplicates by name, the manual
`--keep-first N` mode keeps the leading N channels by position instead; check
the printed channel list with `--dry-run` first, since that rule cannot verify
itself. If a session was concatenated before its parts were trimmed, rerun
concatenation with `--overwrite` so the concatenated file is rebuilt from the
trimmed parts.

The preprocessing pipeline config leaves source and sink locations unset so
they are supplied by the active data-location profile:

```bash
python scripts/preprocessing/concatenate_recordings.py \
  --config configs/pipelines/preprocessing.yaml --dry-run

python scripts/preprocessing/downsample_recordings.py \
  --config configs/pipelines/preprocessing.yaml --dry-run
```

Both also accept `--subject`/`--date`/`--session` to restrict the sweep to one
session's folder instead of the whole tree:

```bash
python scripts/preprocessing/concatenate_recordings.py \
  --config configs/pipelines/preprocessing.yaml --subject 66 --date 20260717

python scripts/preprocessing/downsample_recordings.py \
  --config configs/pipelines/preprocessing.yaml --subject 66 --session 1
```

Configure the Somnotate model and channel settings in
`configs/pipelines/sleep_scoring.yaml`; subject and date/session selectors are
always passed on the command line:

```bash
python scripts/sleep_scoring/score_recordings.py \
  --model my-model --subject 66 --date 20260717

python scripts/sleep_scoring/score_recordings.py \
  --model my-model --subject 66 --session 1
```

The model name resolves below
`derivatives/somnotate_training/<model>/model.pickle`; absolute paths are also
accepted. Raw-data and derivatives roots come from the active data-location
profile unless explicitly overridden.

Inspect the stored Somnotate predictions and their per-state probabilities for
one session without loading the EDF or rerunning the model:

```bash
python scripts/qc/sleep_scoring.py \
  --subject 66 --session 1

python scripts/qc/sleep_scoring.py \
  --subject 66 --date 20260717 --confidence-threshold 0.80
```

The report prints duration and percentage in Wake, NREM, REM, and Undefined,
prediction-probability quantiles, and the number of signal epochs below the
review threshold. Nothing is saved by default. Use `--save` for the enhanced
epoch output or `--summary` for the state summary. Both are written beneath the
shared derivatives session's `eeg/sleep_scoring/` directory.

Run every session-level quality-control section and obtain one analysis-readiness
decision with a unified list of epochs requiring review:

```bash
python scripts/qc/summary_qc.py \
  --subject 66 --session 1
```

Both outputs are written to the session's `eeg/quality_control/` directory on every
run, whether the command is invoked directly or through `python -m src.qc`, and
both are named after the analyzed recording in the same
`sub-XXX_ses-YYY_recording-ZZZ_<output>` form as every other per-recording
derivative:

```
eeg/quality_control/
  sub-066_ses-001_recording-concat_qc_summary.csv
  sub-066_ses-001_recording-concat_qc_review_epochs.csv
  sub-066_ses-001_recording-concat_qc_review_epochs.parquet
  sub-066_ses-001_recording-concat_qc_summary_provenance.json
```

The summary holds the per-section results; the review-epoch parquet provides
typed, machine-readable review intervals for downstream processing. Pass a
filename to `--summary`/`--review-epochs` to rename either output (the
recording prefix is still applied), or `--no-summary`/`--no-review-epochs` to
skip writing it. `recording_integrity.py` and `scripts/qc/sleep_scoring.py`
name their outputs the same way.

The command checks EDF/FIF integrity and gaps, Somnotate confidence and undefined
epochs, artifact burden, EEG/EMG channel correlation, sleep-state power spectra,
and EMG RMS. Results are `PASS`, `REVIEW`, or `FAIL`. The power-spectra and EMG
sections use the frequency bands, state expectations, RMS ordering, and determining
EEG channel defined in `configs/pipelines/spectra.yaml`; select another definition
file with `--spectra-config`. Every review/pass threshold (duration tolerance, gap
percentage and longest-gap limits, confidence and correlation cutoffs, artifact and
undefined-epoch percentages) is defined in `configs/pipelines/quality_control.yaml`
and shared across `summary_qc.py`, `recording_integrity.py`,
`channel_correlations.py`, and `sleep_scoring.py`; select another definition file
with `--qc-config`, or override any single value from the command line. A duration
mismatch or invalid spectral result fails the recording. Review rows use start/end
seconds from recording onset so results from one-second Somnotate epochs and the
default four-second signal epochs can be combined safely. Run the command with
`--help` for the complete list. Reports are only saved when their output options
are supplied.
Outputs are separated beneath the shared derivatives session's `eeg/` folder:

- Somnotate predictions and scoring reports: `eeg/sleep_scoring/`
- Artifact detection and reports: `eeg/artifacts/`
- Combined QC, integrity, spectra, and correlation reports: `eeg/quality_control/`

A relative output name is placed below its corresponding directory; an absolute
path is an explicit override.

Visually inspect the raw signals and predicted states for one scored session:

```bash
python scripts/sleep_scoring/view_scored_recording.py \
  --subject 66 --date 20260717 --hours 3 6

python scripts/sleep_scoring/view_scored_recording.py \
  --subject 66 --session 1 --hours 3 6
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
regions from the matching `*_artifact_epochs.parquet` in the session's
`eeg/artifacts/` directory can also be shaded and labelled:

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
python scripts/qc/recording_integrity.py \
  --subject 66 --date 20260717

python scripts/qc/recording_integrity.py \
  --subject 66 --session 1
```

Compare one raw EDF with its derivative FIF, scanning only the EDF for gaps and
opening the FIF only for duration metadata:

```bash
python scripts/qc/recording_integrity.py \
  --edf /path/to/rawdata/session/recording.edf \
  --fif /path/to/derivatives/session/recording_resampled-128hz_raw.fif
```

Omit `--edf` and `--fif` to automatically pair recordings beneath the active
rawdata and derivatives roots. Use `--edf-pattern` and `--fif-pattern` to narrow
the batch selection:

```bash
python scripts/qc/recording_integrity.py \
  --edf-pattern "sub-066/**/*.edf" \
  --fif-pattern "sub-066/**/*_raw.fif"
```

The command prints the duration of every constant/non-finite interval and
gap-like EDF annotation in seconds. The default 30-minute chunks reduce reader
and network overhead while keeping memory bounded. It does not write files by
default. Add `--gaps` to save full gap details or `--summary` to save duration
comparisons and pass/review status in the shared session QC directory. Defaults
are a one-second duration tolerance and a one-second
minimum gap; override them with `--duration-tolerance` and `--min-gap`. A
recording requiring review causes a non-zero command exit status.

## Sleep-state power spectra

Plot the mean EEG power spectral density for Wake, NREM, and REM for one
subject and either a session date or session number:

```bash
python scripts/qc/spectra.py \
  --subject 66 --date 20260717

python scripts/qc/spectra.py \
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
respectively. Saving also writes a `_sleep_state_spectral_quality.csv` report
with one row per sleep state and EEG channel. It includes analyzable EEG/EMG epoch counts,
relative delta (0.5–4 Hz), theta (4–10 Hz), alpha (8–12 Hz), and beta
(12–30 Hz) power, EMG RMS distribution statistics, state quality status, and
the overall recording status. Wake delta power must be lower than NREM, NREM
delta power must exceed Wake and REM, REM's theta/delta ratio must exceed NREM,
and EMG RMS should follow Wake ≥ NREM ≥ REM. Every EEG channel is reported, but
only the first EEG channel determines the overall recording status.
These definitions, analysis defaults, comparison ratios, sleep-state display
labels/colors, EMG ordering, and the determining channel are configured in
`configs/pipelines/spectra.yaml` and can be overridden with `--spectra-config`.

Save plots without opening an interactive window with:

```bash
python scripts/qc/spectra.py \
  --subject 66 --date 20260717 \
  --save-dir --no-show
```

Use `--fmin`, `--fmax`, `--epoch-seconds`, `--welch-seconds`, and
`--chunk-epochs` to adjust the spectrum calculation and memory/runtime tradeoff.

## Sleep-state channel correlation

Plot the distribution of epoch-wise Pearson correlation for every EEG/EMG
channel pair, separated into Wake, NREM, and REM:

```bash
python scripts/qc/channel_correlations.py \
  --subject 66 --date 20260717

python scripts/qc/channel_correlations.py \
  --subject 66 --session 1
```

The command reads the derivative FIF in bounded chunks, uses four-second epochs
by default, and automatically excludes gaps, undefined sleep states, non-finite
or constant channel pairs, and epochs marked in the matching artifact parquet.
Each histogram shows the percentage of valid epochs for its sleep state on the
common Pearson range from −1 to +1; the dashed line marks the median. Use
`--include-artifacts` to retain flagged epochs, `--bins` to change histogram
resolution, or `--save-dir --no-show` for non-interactive,
provenance-tagged PDF output using the shared figure style.
The terminal also reports counts above the default review thresholds of
`|r| > 0.90` for EEG–EEG pairs and `|r| > 0.50` for EEG–EMG pairs, including a
de-duplicated total of flagged recording epochs. Override these with
`--eeg-eeg-threshold` and `--eeg-emg-threshold`.

## Artifact burden report

Report artifact counts, duration, percentage per recording hour, sleep-state
breakdown, and longest contiguous artifact period separately by channel:

```bash
python scripts/qc/artifacts.py \
  --subject 66 --date 20260717

python scripts/qc/artifacts.py \
  --subject 66 --session 1
```

The report uses the matching `*_artifact_epochs.parquet`, infers its epoch
duration from `time_s`, and supports both `artifact_channels` and legacy
channel-prefixed `artifact_features`. EMG-supported artifact epochs are reported
as `EMG (combined)` when the stored reason identifies an EMG-supported EEG
outlier. Nothing is saved by default; add `--save-dir` to write separate overall,
hourly, and sleep-state CSV tables to the shared session `eeg/artifacts/` directory. Use
`--epoch-seconds` only when the duration cannot be
reliably inferred from the artifact file.
