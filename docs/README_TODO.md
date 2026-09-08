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

## 14. Insert an output of sleep score qc which is saved into own folder

## 15. Run all steps individually for sub-63

## 16. Rename to hypnose-eeg-preprocessing

## 17. Update readme to make more concise

## 18. Create executable scripts for each separate section with API

## 19. Automate running of the preprocessing for hypnose dataset with reports of errors