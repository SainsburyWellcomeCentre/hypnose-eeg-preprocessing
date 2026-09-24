# Open Tasks

Punch list of known cleanup and improvement work. Not scheduled; pick items up as
capacity allows.

## ~~1. Consolidate CSV-writing~~

~~CSV output is currently written three different ways, each reimplementing the same
`mkdir(parents=True, exist_ok=True)` + write + report pattern~~ — done:
`scripts/io/output_paths.py` now has `save_csv()` (DataFrame) and `save_csv_rows()`
(dict rows via `csv.DictWriter`), both doing mkdir + write + `print(f"Saved: {path}")`.
Every listed call site (`recording_integrity.py`, `concatenate_recordings.py`/
`downsample_recordings.py` manifests, `summary_qc.py`, `sleep_scoring.py`,
`artifacts.py`, `spectra.py`, `detect_artifacts.py`) now uses one of the two.

## ~~2. Improve `scripts/qc`/`scripts/utils` organization~~

- ~~`--edf-pattern`/`--fif-pattern` glob defaults in `recording_integrity.py` are the
  last hardcoded CLI defaults left in `scripts/qc/`; decide whether they're worth
  moving to config or are fine as structural constants.~~
- ~~`select_recordings` naming collision between `scripts/utils/recording_selection.py`
  and `concatenate_recordings.py` — done: renamed the latter's method to
  `EdfSessionConcatenator.select_session_recordings`.~~
- ~~Sleep-state name/color dicts (`SLEEP_STATE_NAMES` in `artifacts.py`, `STATE_NAMES`/
  `STATE_COLORS` in `channel_correlations.py`, `STATE_NAMES` in `sleep_scoring.py`)
  are intentionally independent per-script copies (not sourced from
  `spectra.yaml`) — revisit only if they drift out of sync in practice.~~
- ~~Audit `scripts/qc/*.py` for any other cross-script imports beyond the two already
  extracted (`select_recordings`, `QCThresholds`) — decide case by case whether it
  belongs in `scripts/utils/` or `scripts/qc/thresholds.py`-style shared modules
  before moving anything.~~
- ~~`downsample_recordings.py:_prefer_concatenated_recordings()` independently
  hardcodes the `"_recording-concat"` naming convention that
  `concatenate_recordings.py:_concatenated_base_name()` defines and produces —
  same coupling risk `_trimmed` had (now fixed: `concatenate_recordings.py` reads
  `preprocessing.trim_channels.output_suffix` instead of hardcoding `"_trimmed"`).
  Worth a shared constant/config value so both scripts agree on the concat suffix.~~

## ~~3. `detect_artifacts.py` performance~~

~~`scripts/preprocessing/detect_artifacts.py` computes epoch features with numpy
(vectorized per chunk) but then serializes them with Python-level loops, which is
the likely bottleneck on large recordings:~~

- ~~`extract_features()` (~line 306): nested
  `for offset in range(epoch_count): for channel_index, channel in enumerate(...):`
  appends one dict per (epoch, channel) pair to `eeg_rows`/`emg_rows` before calling
  `pd.DataFrame(...)`. Building the DataFrame directly from the already-vectorized
  numpy arrays (e.g. via flattened columns or `pd.DataFrame` from a dict of 1-D
  arrays) would skip the per-row Python object construction.~~
- ~~`classify_artifacts()` (~line 445): `for row in eeg_features.to_dict(orient="records"):`
  rebuilds an `artifact_reason` string per row in Python. Candidate for a vectorized
  `np.where`/string-join approach over the underlying boolean/z-score arrays instead
  of round-tripping through per-row dicts.~~
- ~~Profile against a real multi-hour recording first to confirm these are the actual
  hot spots before rewriting.~~

## ~~4. Match input CLI for file locations to hypnose-helpers~~

~~`scripts/utils/recording_selection.py` reimplemented its own `SESSION_DIR_RE`/
`_subject_label`/`_session_matches` instead of using the shared layout parser~~ —
done: `select_recordings()` and the new `find_session_dirs()` now resolve subject/
session selectors through `hypnose_helpers.io.layout.SessionLayout`, the same
mechanism `scripts/io/input_paths.py` (`resolve_session_dir()`) and
`scripts/io/output_paths.py` (`session_output_dir()`) already used, so ses-vs-date
matching, `_id-`/other subject-dir suffixes, and duplicate-session detection are
handled in one place instead of three.

## ~~5. Understand purpose of tests and see if it can be reorganised~~

~~This is used for creating test scripts for when changes are made to
`scripts/`, to catch regressions before they reach real recordings~~ — done:
a local `.git/hooks/pre-commit` hook runs the full suite before every commit
(bypassable with `--no-verify`), and `tests/` now mirrors the `scripts/`
subpackage layout (`tests/preprocessing/`, `tests/qc/`, `tests/sleep_scoring/`,
`tests/analysis/`) so each test file's location matches the script it
exercises.

~~## 6. Remove need for sys.path.insert(0, str(Path(__file__).resolve().parents[2]))~~

~~Resolve duplication by ensuring that the repo root is specified in each repo and thus
each scripts package is only referenced once.~~

~~ ## 7. inspect_and_trim_channels.py needs better method for running and provided info ~~

~~ - For the moment keep as it is with manual curating until issues arise with ~~

~~ ## 8. Check for any for loops that would limit speed on large recordings - vectorisation ~~

## ~~9. Artifact detection naming not based on somnotate_predictions and based on derivatives location~~

~~- Change all cases in which the referencing is done explicitly and instead folders should be created by the name of the script being run~~

— done: `artifact_output_paths()` in `scripts/io/output_paths.py` now names
outputs after the analyzed FIF recording's own stem and resolves the
destination through the shared derivatives session layout, instead of
stripping `somnotate_predictions` out of the sleep-scoring parquet's filename.

## ~~10. Check any places that require full paths and change them into requiring sub/date/session~~

~~One example is detect_artifacts.py~~ — done: `detect_artifacts.py` and
`inspect_and_trim_channels.py` now accept `--subject`/`--date`/`--session`
selectors (resolved via `select_recordings()`/`find_session_dirs()`), matching
the pattern already used by `scripts/qc/*.py`, `score_recordings.py`, and
`recording_integrity.py`; the raw-path positional arguments remain as an escape
hatch for ad hoc files, mutually exclusive with the selectors, same as
`recording_integrity.py`'s `--edf`/`--fif`.
`downsample_recordings.py` and `concatenate_recordings.py` were initially left
as-is, since they sweep a whole directory tree rather than pointing at one
recording — since fixed: both now also accept `--subject`/`--date`/`--session`,
resolved via `find_session_dirs()` to one session directory and translated into
an `edf_pattern` scoped to that directory, so the existing directory-sweep
logic (concat-part preference, multi-recording-session detection) runs
unchanged but only sees that session's files; mutually exclusive with
`--edf-pattern` (both) and `--session-dir`/`--recordings` (concatenate).

## ~~11. Change folder naming to downsample for downsampling rather than ephys~~

~~Downsampled FIF output mirrored the raw `<modality>` folder (`ephys`) verbatim
under derivatives~~ — done: `downsample_recordings.py`'s `output_path()` now
resolves each recording's `sub-XXX/ses-YYY_date-.../` folder and writes under
`.../downsample/` instead, matching the `artifacts`/`sleep_scoring`/
`quality_control` output-group convention from item 9.

~~## 12. Best method for sleep scoring, either before or after concatenation~~

~~## 13. Ensure concatenation maintains zeitgeber time~~

~~Instead of zeitgeber time, the real gap is considered when concatenating so that 
once all preprocessing is completed the real start time and length of recording can
be used to analyse considering ZT.~~

~~## 13. Use of downsampled data for view_scored_recordings to reduce computing costs~~

~~## 14. Insert an output of sleep score qc which is saved into own folder~~

#~~# 15. Run all steps individually for sub-63~~

~~## 16. Does qc use .edf or can .fif be used for quicker processing~~

~~- No they all use .fif and can't be further optimised~~

## ~~17. Insert info that inspect_and_trim_channels.py should only be necessary for errors in concatenation~~

~~Insert info that inspect_and_trim_channels.py should only be necessary for
errors in concatenation~~ — superseded: `inspect_and_trim_channels.py` (since renamed
`trim_duplicate_channels.py`) now detects duplicate channel labels from the EDF header itself and drops them
automatically, so it runs as the first default step of `src/preprocessing.py`
(before concatenation) and is a no-op on clean recordings. Downstream selection
(`scripts/utils/recording_selection.py:prefer_trimmed_recordings()`) prefers a
`_trimmed` copy over its source, so single-recording sessions with duplicates
are handled too. `--keep-first N` remains the manual, positional fallback.

~~## 16. Rename to hypnose-eeg-preprocessing~~

## 18. Update readme to make more concise

- Reflect the fact that sleep scoring needs to occur before artifact detection

## ~~18. Create executable scripts for each separate section with API~~

~~Create executable scripts for each separate section with API~~ — done:
`src/preprocessing.py`, `src/sleep_scoring.py`, and `src/qc.py` each wrap
their matching `scripts/*` CLIs as subprocesses and can be run standalone
with a `--steps` selector, and `src/run_pipeline.py` runs all three in
pipeline order by default (`--stage` restricts it to one or more). See the
"Unified pipeline entry points" section of `readme.md`.

## ~~19. Ensure it can be run externally with prefered file locations~~

~~Ensure it can be run externally with prefered file locations~~ — done: the
data-location roots were already overridable (`--rawdata-root`/
`--derivatives-root`, `HYPNOSE_EEG_*_ROOT`); the per-session output folders
now are too. `scripts/io/output_layout.py:output_dir_name()` resolves
`HYPNOSE_EEG_OUTPUT_DIR_<GROUP>` and `HYPNOSE_EEG_OUTPUT_ROOT` (the shared
`eeg/` modality folder the groups sit in), then `HYPNOSE_EEG_OUTPUT_LAYOUT` (an
alternative `output_layout.yaml`), then the repository's own file, and the
`src/` entry points expose the same as `--output-dir GROUP=FOLDER` /
`--output-root FOLDER` / `--output-layout FILE` (forwarded to the wrapped
scripts via the environment) and as
`run_steps(output_layout=..., output_root=..., output_dirs=...)`. Within a
session every output group now sits below `eeg/`; the rest of the defaults are
unchanged, so an in-repo run still follows the data location alone. See "Output
folders within a session" in `readme.md`.

~~## 20. Batch running of pipelines~~

## 21. Guidance for running qc scripts and options for external running when requiring review

## 22. Automate running of the preprocessing for hypnose dataset with reports of errors

## ~~Pipeline output also parquet for qc review for efficiency~~

~~QC review read only CSVs~~ — done: `summary_qc.py` writes `qc_summary.parquet`
beside `qc_summary.csv`, and `src/qc_review.py` and the batch QC verdict read the
parquet copies of the summary and review epochs, falling back to the CSVs.

## Short recordings are currently not being scored well because they don't have long periods of baseline - consider using a previous long session to set baseline (<6 hour recordings>)

## 23. Improve external running of API

In progress. Done: `scripts/` is now the installable `hypnose_eeg` package (no
`sys.path` inserts; configuration loads on first use rather than at import);
the viewer moved to `hypnose_eeg/review/viewer.py`; `qc/spectra.py` and the
viewer are split into compute/plot/CLI layers (`compute_session_spectra`,
`plot_spectra`, `save_spectra`; `view_settings`, `load_scored_window`,
`plot_scored_window`, `show_scored_recording`). Remaining:
- A small `hypnose_eeg/api.py` facade with a `DataLocations` argument and
  console scripts, and `src/qc.run_steps` calling functions in-process
  rather than as subprocesses (folding `src/` into the package).
- The same compute/plot/CLI split for the other `qc/` modules
  (`summary_qc.run_qc` still takes an `argparse.Namespace`).
- A headless `render_scoring()` that saves a PNG of each section `qc_review`
  flags, built on `load_scored_window`/`plot_scored_window`.

## 24. Reduce amount of text at top of scripts