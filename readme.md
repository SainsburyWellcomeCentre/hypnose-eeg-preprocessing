# Hypnose EEG Preprocessing

Tools and notebooks for inspecting, concatenating, downsampling and qc Hypnose EEG
recordings.

## Environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync
```

This creates `.venv/` with Python 3.12 (downloaded by uv if needed) and installs
the exact versions recorded in `uv.lock`, including the `dev` (pytest) and
`notebook` (JupyterLab, Qt browser) groups. Run commands inside it with
`uv run <command>`, or activate it with `source .venv/bin/activate`
(`.venv\Scripts\activate` on Windows).

The environment expects the `hypnose-helpers` and `hypnose-somnotate` repositories
in sibling checkouts and installs them in editable mode. Somnotate's legacy
`pomegranate` dependency builds from source, so a working C/C++ compiler is also
required.

It also installs this repository in editable mode as the `hypnose_eeg` package,
so its modules import from any directory and other projects or notebooks in
the environment can use them. After pulling a change, re-run `uv sync`.
Configuration is read from this checkout's `configs/`, so keep the install
editable. Add or change dependencies with `uv add` / `uv remove` so that
`pyproject.toml` and `uv.lock` stay in step.

## Data location

Data stays outside this repository and is referenced through a named,
machine-specific profile. Profile resolution and selection are provided by
`hypnose_helpers`; this repository supplies the EEG-specific config directory
and `HYPNOSE_EEG` environment-variable prefix. No repository symlink is required.

The shared profiles live in `configs/data_locations.yml`. Select a
profile once for each checkout:

```bash
python -m hypnose_eeg.io.repository_paths --list
python -m hypnose_eeg.io.repository_paths server-linux   # or server-mac / server-windows
python -m hypnose_eeg.io.repository_paths --show
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

# The same three overrides as flags on the pipeline commands
hypnose-eeg-pipeline --subject 66 --session 1 --model my-model \
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
-- because readers (`src/hypnose_eeg/qc/*`, the viewer) locate earlier outputs through
the same names. From Python, `hypnose_eeg.api.DataLocations(output_layout=...,
output_root=..., output_dirs={...})` carries the same overrides (see [Running
from Python](#running-from-python)).

## Unified pipeline entry points

`hypnose-eeg-pipeline` runs the full pipeline for one subject/session in the
order the stages actually require — trim, concatenate, downsample and the
artifact prescan, then sleep scoring, then artifact detection (which depends on
the sleep-scoring output), then the QC summary and the review figures (see
[Review figures](#review-figures-no-display-needed)):

```bash
hypnose-eeg-pipeline --subject 66 --session 1 --model my-model
```

Each stage also has its own entry point that can be run on its own, with a
`--steps` selector for that stage's individual scripts:

```bash
hypnose-eeg-preprocess --subject 66 --session 1 --steps trim concatenate downsample
hypnose-eeg-score --subject 66 --session 1 --model my-model
hypnose-eeg-qc --subject 66 --session 1
```

Pass `--stage preprocessing`/`sleep_scoring`/`qc` (one or more) to
`hypnose-eeg-pipeline` to restrict a full run to those stages, each using its own
default step set.

These commands are console scripts that the editable install
(`uv sync`) puts on the `PATH` of the activated environment, so they
run from any directory (or prefix them with `uv run`). Each is also a module, run with `python -m`:

| Console script | Module (`src/hypnose_eeg/...`) |
| --- | --- |
| `hypnose-eeg-pipeline` | `hypnose_eeg.pipeline.run` |
| `hypnose-eeg-preprocess` | `hypnose_eeg.pipeline.preprocessing` |
| `hypnose-eeg-score` | `hypnose_eeg.pipeline.sleep_scoring` |
| `hypnose-eeg-qc` | `hypnose_eeg.pipeline.qc` |
| `hypnose-eeg-qc-review` | `hypnose_eeg.pipeline.qc_review` |
| `hypnose-eeg-qc-spectra` | `hypnose_eeg.qc.spectra` |
| `hypnose-eeg-view` | `hypnose_eeg.sleep_scoring.view_scoring` |
| `hypnose-eeg-locations` | `hypnose_eeg.io.repository_paths` |

`--view` ends the run in the interactive scoring viewer
(`src/hypnose_eeg/sleep_scoring/view_scoring.py`), so a session can be
inspected as soon as it has been processed. It opens a plot window, so it needs
a display (see [docs/remote_visualization.md](docs/remote_visualization.md) for
working over SSH), and it runs after every selected stage — artifact detection
and the QC summary included. Because it only reads what is already on disk it
also works on its own:

```bash
# Process the session, then look at the result
hypnose-eeg-pipeline --subject 66 --session 1 --model my-model --view
# Just look at an already-processed session
hypnose-eeg-pipeline --subject 66 --session 1 --stage qc --view
# Viewer-only options (--hours, --eeg-channel, --show-artifacts, ...) go here
hypnose-eeg-score --subject 66 --session 1 --steps view --hours 3 6
```

Every step whose output already exists is skipped and the run continues with
the next one, so an interrupted or partially-completed session is resumed by
rerunning the same command: channel trimming, concatenation, downsampling,
sleep scoring, and artifact detection each leave their existing outputs in
place. Pass
`--overwrite` to recompute them regardless. The QC summary is the exception --
it is cheap and always refreshed.

### Batch: several sessions or subjects

One subject with one `--session` (or `--date`) is a single run. Selecting more
than that runs a batch, one session after the other. `--all-sessions` (or
`--batch`) works through every session each subject has, in order, and
`--subject all` covers every subject in the rawdata tree:

```bash
hypnose-eeg-pipeline --subject 66 --all-sessions --model my-model
hypnose-eeg-pipeline --subject 66 67 68 --all-sessions --model my-model
hypnose-eeg-pipeline --subject all --all-sessions --model my-model
```

To run only some sessions, give `--session` several numbers or an inclusive
range (the same sessions for every subject), or choose them per subject as
`SUBJECT:SESSIONS`:

```bash
hypnose-eeg-pipeline --subject 66 --session 1 3 --model my-model        # sessions 1 and 3
hypnose-eeg-pipeline --subject 66 67 --session 2-5 --model my-model     # sessions 2-5 of both
hypnose-eeg-pipeline --subject 66:1,3 67:2-4 --model my-model           # per subject
hypnose-eeg-pipeline --subject 66:1 67 68 --session 2 --model my-model  # 66 ses 1; 67, 68 ses 2
hypnose-eeg-pipeline --subject 66 --date 20260711-20260720 --model my-model
```

`--date` takes dates and date ranges the same way. A per-subject choice
takes precedence over `--session`/`--date`/`--all-sessions`, which apply to the
subjects listed without one. A range runs whichever sessions fall inside it,
but a single session or date that a subject does not have is recorded as
`missing` in the report (and the batch exits non-zero) rather than skipped
silently.

A session that fails does not stop the batch. Its derivative outputs are
erased, the failure is written to a per-run CSV report, and the next session
starts; the run ends with a summary and exits non-zero if anything failed:

```
Batch summary: 12/15 sessions completed across 3 subjects, 1 failed QC
  sub-066: 5/5 completed
    [    OK] sub-066/ses-001_date-20260711
    ...
  sub-067: 3/5 completed
    [FAILED] sub-067/ses-004_date-20260714 -- sleep_scoring:score (hypnose_eeg.sleep_scoring.score_recordings) exited with status 1 (erased 1 path(s))
    [FAILED] sub-067/ses-005_date-20260715 -- QC FAIL (artifacts) (erased 1 path(s))
  sub-068: No sessions found for sub-068
Failed QC:
  sub-067/ses-005_date-20260715 (artifacts)
Total time: 5h 12m 40s
```

A QC summary that comes out FAIL fails its session like any other step (and
the batch exits non-zero), but it is marked `QC FAIL` and listed under
`Failed QC` with the sections that failed, so it is not mistaken for a crash.

A subject whose sessions cannot be resolved at all, like a selected session it
does not have, is recorded the same way as a failed session (`missing`) rather
than stopping the subjects after it; the per-subject tallies appear once more
than one subject runs.

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

### On the SWC HPC (SLURM)

A batch can also run as a SLURM job array on CPU nodes, with one array task per
subject (its sessions in order) or, with `--task-unit session`, per session:

```bash
./slurm/submit.sh --subject 66 67 --all-sessions --model my-model
```

`--list-tasks` previews how a selection splits into tasks, and `--task-index N`
runs one of them as a batch run; with `--task-file FILE`, a saved
`--list-tasks` output, it runs line N of that file instead. Setup with uv, resources, and logs are
covered in [`slurm/README.md`](slurm/README.md).

### Which sessions need QC review

`hypnose-eeg-qc-review` reads the QC summaries already on disk and lists every
session of the given subjects whose QC came out REVIEW (or FAIL), with the
sections behind it -- metric, value, threshold, detail, and how many review
ranges each section added to the review epochs. It reads the `.parquet` copies
(`qc_summary.parquet`, `qc_review_epochs.parquet`), falling back to the CSVs for
sessions summarized before those copies were written. Sessions with no summary
are named per subject. Nothing is recomputed.

```
hypnose-eeg-qc-review --subject 65 66
hypnose-eeg-qc-review --subject all --report qc_review.csv
```

```
QC review: 19/55 checked sessions need review across 2 subjects
  ...
  sub-066: 3/39 checked sessions need review
    [REVIEW] sub-066/ses-027_date-20260825
        sleep_state_proportions: signal_percent_by_state=Wake=48.04%; NREM=10.32%; REM=41.64% (threshold Wake<=75%; NREM<=75%; REM<=20%) -- REM review
        channel_correlation: review_epoch_percent=41.73 (threshold 2) -- 8547/20480 epochs; ... [8547 review entries, 9h 29m 48s]
```

`--report FILE` also writes one CSV row per flagged section. Pass the same
`--output-root`/`--output-dir`/`--output-layout` overrides the QC summary ran
with, so its summaries are found.

### Review figures (no display needed)

After the QC summary, the QC stage's `figures` step saves PNGs into the
session's `eeg/quality_control/` folder for every session whose QC summary is
REVIEW or FAIL, so a flagged session can be looked at without the interactive
viewer -- on a remote machine, or after a batch run:

- `<recording>_scoring_hours-0-12.png`: the first 12 hours of the scored
  recording as the viewer draws it -- EEG/EMG traces, EMG RMS, delta and theta
  envelopes, theta:delta ratio, and Somnotate's states, with gaps, excluded
  periods and detected artifacts shaded -- on an hours axis.
- `<recording>_sleep_state_power_spectra.png`,
  `<recording>_sleep_state_emg_rms.png` and the
  `_sleep_state_spectral_quality.csv`: the
  [sleep-state spectra](#sleep-state-power-spectra) behind the power-spectra
  and EMG checks.

A session that passes QC gets no figures. The step runs by default in the full
pipeline and in `hypnose-eeg-qc`, and costs about a minute for a session under
review. `--scoring` and `--spectra` (`review`, `always`, `never`) choose when
each is drawn, for example to draw a passing session anyway:

```bash
hypnose-eeg-qc --subject 66 --session 1 --steps figures
hypnose-eeg-qc --subject 66 --session 1 --steps figures --scoring always --spectra always
hypnose-eeg-qc --subject 66 --session 1 --steps figures --figure-hours 12 24
```

A summary that FAILs stops the QC stage before `figures`, so for such a
session run `--steps figures` afterwards (a batch run erases failed sessions,
figures included). The defaults -- hours, when to draw each figure, artifact
shading, size and resolution -- are under
`review_figures` in `configs/pipelines/quality_control.yaml`; the EEG channel
and display rate follow the viewer's `sleep_scoring_view` settings. From
Python: `hypnose_eeg.qc.review_figures.render_session_figures()`, or
`hypnose_eeg.sleep_scoring.view_scoring.render_scoring()` for one PNG of any
viewer selection.

Run any of them with `--help` for its complete option list. Each step calls
the matching `src/hypnose_eeg/*` CLI below in the same Python process rather than
reimplementing it, so step-specific flags such as `--config` are best passed
to the underlying script directly when a per-step entry point doesn't already
expose them. Output-folder overrides apply only while their step runs.

### Running from Python

`hypnose_eeg.api` runs the same stages from a notebook or another project,
with keyword arguments instead of flags. A step that fails raises `StepFailed`
rather than returning an exit status -- including a QC summary that comes out
FAIL:

```python
from hypnose_eeg import api

locations = api.DataLocations(
    rawdata_root="/mnt/hypnose/rawdata",
    derivatives_root="/mnt/hypnose/derivatives",
    output_dirs={"quality_control": "reports/qc"},  # optional, as --output-dir
)
api.run_session(66, session=1, model="my-model", locations=locations)
api.preprocess(66, session=1, steps=["trim", "concatenate"], locations=locations)

result = api.run_batch([65, 66], model="my-model", locations=locations)
print(result.ok, result.report)
api.run_batch([65, 66], sessions=[1, "3-5"], locations=locations)  # chosen sessions
api.run_batch({65: [1, 3], 66: "2-4"}, locations=locations)       # per subject

qc = api.session_qc(66, session=1, locations=locations)  # computed in memory, nothing written
print(qc.status)
print(qc.sections)
reviews = api.review_qc("all", locations=locations)
```

`locations` defaults to the active data-location profile. Another project can
pass its own `hypnose_helpers.io.paths.DataLocations` instead, and its profile
supplies both roots.

The QC modules split computing from plotting and saving in the same way, so
their results are available without the CLI:

| Module | Compute | Plot / save |
| --- | --- | --- |
| `qc/summary_qc.py` | `compute_session_qc`, `summary_qc_settings` | `save_session_qc` |
| `qc/spectra.py` | `compute_session_spectra` | `plot_spectra`, `save_spectra` |
| `qc/channel_correlations.py` | `compute_session_correlations`, `correlation_review` | `plot_correlations`, `save_correlations` |
| `qc/sleep_scoring.py` | `compute_session_scoring_qc` | `save_scoring_qc` |
| `qc/artifacts.py` | `compute_session_artifacts` | `save_artifact_report` |
| `qc/recording_integrity.py` | `check_session` | `save_integrity_summary`, `save_integrity_gaps` |
| `qc/review_figures.py` | `review_figure_settings` | `render_session_figures` |

Thresholds default to `configs/pipelines/quality_control.yaml`. Override them
with `with_overrides(default_qc_thresholds(), max_wake_percent=60.0)` (from
`hypnose_eeg.utils.config` and `hypnose_eeg.qc.thresholds`), or for the
summary with `summary_qc_settings(max_artifact_percent=10.0)`. Out-of-range
values raise `ValueError`.

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

1. **Trim duplicate channels** — `src/hypnose_eeg/preprocessing/trim_duplicate_channels.py`
2. **Concatenate recordings** — `src/hypnose_eeg/preprocessing/concatenate_recordings.py`
3. **Downsample recordings** — `src/hypnose_eeg/preprocessing/downsample_recordings.py`
4. **Prescan artifacts** — `src/hypnose_eeg/preprocessing/prescan_artifacts.py`
5. **Sleep scoring** — `src/hypnose_eeg/sleep_scoring/score_recordings.py`
6. **Detect artifacts** — `src/hypnose_eeg/preprocessing/detect_artifacts.py`

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
python -m hypnose_eeg.preprocessing.trim_duplicate_channels --subject 66 --session 1 --dry-run
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
python -m hypnose_eeg.preprocessing.concatenate_recordings \
  --config configs/pipelines/preprocessing.yaml --dry-run

python -m hypnose_eeg.preprocessing.downsample_recordings \
  --config configs/pipelines/preprocessing.yaml --dry-run
```

Both also accept `--subject`/`--date`/`--session` to restrict the sweep to one
session's folder instead of the whole tree:

```bash
python -m hypnose_eeg.preprocessing.concatenate_recordings \
  --config configs/pipelines/preprocessing.yaml --subject 66 --date 20260717

python -m hypnose_eeg.preprocessing.downsample_recordings \
  --config configs/pipelines/preprocessing.yaml --subject 66 --session 1
```

Before scoring, the artifact prescan looks through the downsampled FIF for long
stretches of unusable signal, which scoring then leaves out. It needs no sleep
states. An EEG epoch is flagged when it is dead (almost no power in the
analysed band, e.g. a disconnected headstage that is drifting rather than
perfectly flat), fails hard (non-finite values or clipping), or is extreme
against the recording's live epochs. Flagged epochs become *periods*. Runs
separated by at most 30 s are joined, runs shorter than 60 s are dropped, and
periods separated by less than 5 min of clean signal are bridged into one.
All of these limits are set under `artifact_prescan` in
`configs/pipelines/artifact_detection.yaml`:

```bash
python -m hypnose_eeg.preprocessing.prescan_artifacts --subject 66 --session 1
python -m hypnose_eeg.preprocessing.prescan_artifacts --subject 66 --session 1 \
  --bridge-gap-s 600 --overwrite
```

The periods are written to `eeg/artifacts/<recording>_prescan_artifacts.{csv,parquet}`.
Sleep scoring passes them to Somnotate, which treats them like missing data:
they are trimmed, masked or split around, kept out of normalization, and
labelled Undefined with `kind == "artifact"` in the predictions parquet. The
scoring viewer shades them as "Excluded artifact". A recording with no prescan
output is scored in full, with a warning. Set
`sleep_scoring.use_artifact_prescan: false` or pass `--no-artifact-prescan` to
score everything. Scoring skips recordings that already have predictions, so
pass `--overwrite` to rescore after changing the prescan.

Configure the Somnotate model and channel settings in
`configs/pipelines/sleep_scoring.yaml`; subject and date/session selectors are
always passed on the command line:

```bash
python -m hypnose_eeg.sleep_scoring.score_recordings \
  --model my-model --subject 66 --date 20260717

python -m hypnose_eeg.sleep_scoring.score_recordings \
  --model my-model --subject 66 --session 1
```

The model name resolves below
`derivatives/somnotate_training/<model>/model.pickle`; absolute paths are also
accepted. Raw-data and derivatives roots come from the active data-location
profile unless explicitly overridden.

#### Short recordings: borrowing a baseline

Somnotate normalizes each frequency band against robust statistics pooled over
the recording. A recording of only a few hours is usually dominated by one state.
Its statistics shift towards that state, which is then scored as if it were
average. Every scored recording therefore saves its own statistics beside its
predictions (`eeg/sleep_scoring/<recording>_somnotate_normalization.npz`). A
recording with less than `min_signal_hours` (default 6 h) of scoreable signal,
counted after gaps and prescan exclusions, is normalized against the statistics
of a long recording of the same animal instead. By default this is the nearest
earlier session within 14 days. A session that was scored before these files
existed, or never scored, has its statistics computed from its EDF and cached.
Settings live under `sleep_scoring.reference_normalization`:

```bash
# Name the reference yourself (a session number or a date); ignores the age limit
python -m hypnose_eeg.sleep_scoring.score_recordings \
  --subject 66 --session 4 --reference-session 3 --overwrite

# Look in both directions, or switch the behaviour off
... --reference-prefer nearest
... --no-reference-normalization
```

A borrowed baseline is only valid while the gain and electrode impedance are
unchanged. For each candidate, scoring therefore measures how far the short
recording's own statistics sit from it (`offset_z`: median over frequency bins,
in reference SDs, per channel). A candidate more than `max_offset_z` (default
1.0) away on any channel is rejected and the next one is tried. If no candidate
qualifies, the recording keeps its own baseline, with a warning. A reference
named with `--reference-session` is used regardless, with a warning. In
sub-053, ses-007 (3 h) sat about 4 SD from ses-006 on the day before. Scored
against it anyway, most of its NREM became REM.

The scoring provenance records under `parameters.normalization` which baseline
was used, its offset, and any rejected candidates. The QC summary's
`normalization` section sends a recording to review when the offset of the
reference used exceeds `quality_control.reference_normalization.max_offset_z`,
or when a short recording found no usable reference and was normalized
against itself.

Inspect the stored Somnotate predictions and their per-state probabilities for
one session without loading the EDF or rerunning the model:

```bash
python -m hypnose_eeg.qc.sleep_scoring \
  --subject 66 --session 1

python -m hypnose_eeg.qc.sleep_scoring \
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
python -m hypnose_eeg.qc.summary_qc \
  --subject 66 --session 1
```

Both outputs are written to the session's `eeg/quality_control/` directory on every
run, whether the command is invoked directly or through `hypnose-eeg-qc`, and
both are named after the analyzed recording in the same
`sub-XXX_ses-YYY_recording-ZZZ_<output>` form as every other per-recording
derivative:

```
eeg/quality_control/
  sub-066_ses-001_recording-concat_qc_summary.csv
  sub-066_ses-001_recording-concat_qc_summary.parquet
  sub-066_ses-001_recording-concat_qc_review_epochs.csv
  sub-066_ses-001_recording-concat_qc_review_epochs.parquet
  sub-066_ses-001_recording-concat_qc_summary_provenance.json
```

The summary holds the per-section results; the review epochs hold the review
intervals. Each is written as a CSV for reading by eye and a parquet copy that
`qc_review` and the batch run read (in the summary parquet, `value` and
`threshold` are text, as in the CSV, since some sections report `3/4` or `n/a`). Pass a
filename to `--summary`/`--review-epochs` to rename either output (the
recording prefix is still applied), or `--no-summary`/`--no-review-epochs` to
skip writing it. `recording_integrity.py` and `src/hypnose_eeg/qc/sleep_scoring.py`
name their outputs the same way.

The command checks EDF/FIF integrity and gaps, Somnotate confidence and undefined
epochs, artifact burden, prescan exclusions, EEG/EMG channel correlation,
sleep-state power spectra, and EMG RMS. Results are `PASS`, `REVIEW`, or `FAIL`.
The artifact burden only covers scored signal, so a separate `artifact_prescan`
section sends a recording to review when the pre-scoring scan left more than
5% of it unscored, or when the scan has not been run. Each excluded period is
listed as a review range. The `normalization` section reviews short recordings
whose borrowed baseline looks unsafe, or that had none (see "Short recordings:
borrowing a baseline"). The power-spectra and EMG
sections use the frequency bands, state expectations, RMS ordering, and determining
EEG channel defined in `configs/pipelines/spectra.yaml`; select another definition
file with `--spectra-config`. Every review/pass threshold (duration tolerance, gap
percentage and longest-gap limits, confidence and correlation cutoffs, artifact,
prescan-exclusion and undefined-epoch percentages) is defined in `configs/pipelines/quality_control.yaml`
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
python -m hypnose_eeg.sleep_scoring.view_scoring \
  --subject 66 --date 20260717 --hours 3 6

python -m hypnose_eeg.sleep_scoring.view_scoring \
  --subject 66 --session 1 --hours 3 6
```

`--hours START END` limits EDF loading and the viewer to an elapsed-hour range
measured from the start of the recording. Omit it to load the complete recording.
Alternatively, select real timestamps from the EDF clock. When `--date` is
omitted, the viewer searches the subject's EDF headers and selects the recording
whose real-time span contains the requested start. This supports multi-day
recordings where the selected range begins after the recording's calendar date:

```bash
python -m hypnose_eeg.sleep_scoring.view_scoring \
  --subject 66 \
  --time-range "20260718 03:00:00" "20260718 06:00:00"
```

`--hours` and `--time-range` are mutually exclusive.

For faster rendering, downsample the selected interval to 128 Hz. Artifact
regions from the matching `*_artifact_epochs.parquet` in the session's
`eeg/artifacts/` directory can also be shaded and labelled:

```bash
python -m hypnose_eeg.sleep_scoring.view_scoring \
  --subject 66 \
  --time-range "20260718 03:00:00" "20260718 06:00:00" \
  --display-rate 128 --show-artifacts
```

From Python, `view_settings()` takes the same options as keywords.
`load_scored_window()` reads the selected signals, predictions, and shaded
regions without needing a display, `plot_scored_window()` draws them, and
`show_scored_recording()` opens the interactive window:

```python
from hypnose_eeg.sleep_scoring.view_scoring import (
    load_scored_window, plot_scored_window, view_settings,
)

settings = view_settings(66, session=1, hours=(3, 6), display_rate_hz=128)
window = load_scored_window(settings)
fig, viewer = plot_scored_window(window, view_length_s=settings.view_length_s)
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
python -m hypnose_eeg.qc.recording_integrity \
  --subject 66 --date 20260717

python -m hypnose_eeg.qc.recording_integrity \
  --subject 66 --session 1
```

Compare one raw EDF with its derivative FIF, scanning only the EDF for gaps and
opening the FIF only for duration metadata:

```bash
python -m hypnose_eeg.qc.recording_integrity \
  --edf /path/to/rawdata/session/recording.edf \
  --fif /path/to/derivatives/session/recording_resampled-128hz_raw.fif
```

Omit `--edf` and `--fif` to automatically pair recordings beneath the active
rawdata and derivatives roots. Use `--edf-pattern` and `--fif-pattern` to narrow
the batch selection:

```bash
python -m hypnose_eeg.qc.recording_integrity \
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
hypnose-eeg-qc-spectra --subject 66 --date 20260717
hypnose-eeg-qc-spectra --subject 66 --session 1
```

(`hypnose-eeg-qc-spectra` is the console script for `python -m hypnose_eeg.qc.spectra`.)

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
hypnose-eeg-qc-spectra --subject 66 --date 20260717 --save-dir --no-show
hypnose-eeg-qc-spectra --subject 66 --session 1 --save-dir --no-show --figure-format png
```

Use `--fmin`, `--fmax`, `--epoch-seconds`, `--welch-seconds`, and
`--chunk-epochs` to adjust the spectrum calculation and memory/runtime tradeoff.

The same steps are available from Python, returning the results instead of
printing them:

```python
from dataclasses import replace

from hypnose_eeg.qc.spectra import (
    compute_session_spectra, default_spectra_config, plot_spectra, save_spectra,
)

config = replace(default_spectra_config(), fmax_hz=40.0)  # optional overrides
for result in compute_session_spectra(66, session=1, config=config):
    print(result.recording, result.quality_status)
    result.quality_report       # the quality CSV's table, as a DataFrame
    figures = plot_spectra(result, config=config)  # {"spectra": ..., "emg_rms": ...}
    save_spectra(result, "spectra_out", figures)
```

`rawdata_root=`/`derivatives_root=` override the active data-location profile.

## Sleep-state channel correlation

Plot the distribution of epoch-wise Pearson correlation for every EEG/EMG
channel pair, separated into Wake, NREM, and REM:

```bash
python -m hypnose_eeg.qc.channel_correlations \
  --subject 66 --date 20260717

python -m hypnose_eeg.qc.channel_correlations \
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
python -m hypnose_eeg.qc.artifacts \
  --subject 66 --date 20260717

python -m hypnose_eeg.qc.artifacts \
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
