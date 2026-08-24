# Open Tasks

Punch list of known cleanup and improvement work. Not scheduled; pick items up as
capacity allows.

## 1. Consolidate CSV-writing

CSV output is currently written three different ways, each reimplementing the same
`mkdir(parents=True, exist_ok=True)` + write + report pattern:

- **Raw `csv.DictWriter`**, once per caller:
  - `scripts/qc/recording_integrity.py:_write_csv()` — writes `IntegrityResult`/`Gap`
    dataclass rows.
  - `scripts/preprocessing/concatenate_recordings.py:EdfSessionConcatenator.write_manifest()`
  - `scripts/preprocessing/downsample_recordings.py:EdfDownsampler.write_manifest()`
- **`DataFrame.to_csv()`**, each preceded by its own `path.parent.mkdir(...)` and
  followed by its own `print(f"Saved: {path}")`:
  - `scripts/qc/summary_qc.py` (section summary, review CSV + parquet)
  - `scripts/qc/sleep_scoring.py` (epoch output, state summary)
  - `scripts/qc/artifacts.py` (overall/hourly/sleep-state reports)
  - `scripts/qc/spectra.py` (spectral quality report)
  - `scripts/preprocessing/detect_artifacts.py`

Task: pull the shared `mkdir` + write + `"Saved: {path}"` sequence into one helper
(e.g. `scripts/utils/io.py`), with a thin wrapper for the dataclass-rows case so
`recording_integrity.py`'s `_write_csv` and the two `write_manifest()` methods can
share it too.

## 2. Improve `scripts/qc`/`scripts/utils` organization

- `--edf-pattern`/`--fif-pattern` glob defaults in `recording_integrity.py` are the
  last hardcoded CLI defaults left in `scripts/qc/`; decide whether they're worth
  moving to config or are fine as structural constants.
- ~~`select_recordings` naming collision between `scripts/utils/recording_selection.py`
  and `concatenate_recordings.py`~~ — done: renamed the latter's method to
  `EdfSessionConcatenator.select_session_recordings`.
- Sleep-state name/color dicts (`SLEEP_STATE_NAMES` in `artifacts.py`, `STATE_NAMES`/
  `STATE_COLORS` in `channel_correlations.py`, `STATE_NAMES` in `sleep_scoring.py`)
  are intentionally independent per-script copies (not sourced from
  `spectra.yaml`) — revisit only if they drift out of sync in practice.
- Audit `scripts/qc/*.py` for any other cross-script imports beyond the two already
  extracted (`select_recordings`, `QCThresholds`) — decide case by case whether it
  belongs in `scripts/utils/` or `scripts/qc/thresholds.py`-style shared modules
  before moving anything.
- `downsample_recordings.py:_prefer_concatenated_recordings()` independently
  hardcodes the `"_recording-concat"` naming convention that
  `concatenate_recordings.py:_concatenated_base_name()` defines and produces —
  same coupling risk `_trimmed` had (now fixed: `concatenate_recordings.py` reads
  `preprocessing.trim_channels.output_suffix` instead of hardcoding `"_trimmed"`).
  Worth a shared constant/config value so both scripts agree on the concat suffix.

## 3. `detect_artifacts.py` performance

`scripts/preprocessing/detect_artifacts.py` computes epoch features with numpy
(vectorized per chunk) but then serializes them with Python-level loops, which is
the likely bottleneck on large recordings:

- `extract_features()` (~line 306): nested
  `for offset in range(epoch_count): for channel_index, channel in enumerate(...):`
  appends one dict per (epoch, channel) pair to `eeg_rows`/`emg_rows` before calling
  `pd.DataFrame(...)`. Building the DataFrame directly from the already-vectorized
  numpy arrays (e.g. via flattened columns or `pd.DataFrame` from a dict of 1-D
  arrays) would skip the per-row Python object construction.
- `classify_artifacts()` (~line 445): `for row in eeg_features.to_dict(orient="records"):`
  rebuilds an `artifact_reason` string per row in Python. Candidate for a vectorized
  `np.where`/string-join approach over the underlying boolean/z-score arrays instead
  of round-tripping through per-row dicts.
- Profile against a real multi-hour recording first to confirm these are the actual
  hot spots before rewriting.

## 4. Match input CLI for file locations to hypnose-helpers

## 5. Understand purpose of tests and see if it can be reorganised

## 6. Remove need for sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

## 7. inspect_and_trim_channels.py needs better method for running and provided info