"""Find long artifact periods in a FIF recording before it is sleep scored.

Sleep scoring normalizes every recording against its own robust statistics and
smooths its predictions with an HMM, so hours of dead or wildly noisy signal
can shift how the rest of the recording is scored. This scan runs before
scoring, without sleep states, and looks only for *periods* of unusable
signal; the state-dependent `detect_artifacts.py` still runs after scoring for
epoch-level flags.

An EEG channel-epoch is flagged when it is

- dead: broadband (analysed-band) RMS below an absolute floor -- a disconnected
  headstage drifting slowly, which is not constant enough for somnotate's own
  gap detection to catch;
- a hard failure: non-finite samples, or clipping/repeated edge values;
- extreme: at least N features above a robust z threshold, where the z-scores
  are computed per channel against the recording's *live* epochs only, so a
  recording dominated by dead signal does not make real EEG look extreme.

An epoch is flagged when any EEG channel is. Flagged epochs are then turned
into periods: runs separated by at most `merge_within_s` of clean signal are
joined, runs shorter than `minimum_run_s` are dropped, and surviving periods
separated by less than `bridge_gap_s` of clean signal are bridged into one
continuous period. The periods are written as
`<recording>_prescan_artifacts.{csv,parquet}` in the session's artifacts
directory, where `score_recordings.py` passes them to somnotate to be left
unscored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypnose_eeg.io.mne_io import import_mne
from hypnose_eeg.io.output_paths import artifact_output_paths, save_csv
from hypnose_eeg.io.repository_paths import get_derivatives_root, get_rawdata_root
from hypnose_eeg.preprocessing.detect_artifacts import extract_features
from hypnose_eeg.utils.config import (
    DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH,
    load_config,
    nested_get,
    two_float_tuple,
)
from hypnose_eeg.utils.epochs import complete_epoch_count
from hypnose_eeg.utils.provenance import write_provenance
from hypnose_eeg.utils.recording_selection import select_recordings

PERIOD_COLUMNS = [
    "start_s", "end_s", "duration_s", "flagged_fraction",
    "dead_epochs", "hard_failure_epochs", "extreme_epochs",
]


@dataclass(frozen=True)
class PrescanSettings:
    """Pre-scoring artifact scan settings; feature extraction is shared with detection."""

    epoch_seconds: float
    chunk_memory_mb: float
    chunk_working_array_factor: int
    max_chunk_epochs: int
    welch_segment_seconds: float
    psd_minimum_frequency_hz: float
    psd_maximum_frequency_hz: float
    high_frequency_band_hz: tuple[float, float]
    line_noise_band_hz: tuple[float, float]
    minimum_broadband_rms_uv: float
    max_edge_fraction: float
    robust_mad_scale: float
    extreme_z_threshold: float
    extreme_features_required: int
    extreme_features: tuple[str, ...]
    merge_within_s: float
    minimum_run_s: float
    bridge_gap_s: float
    output_suffix: str
    overwrite: bool

    @classmethod
    def from_yaml(cls, config_path: str | Path, **overrides: object) -> "PrescanSettings":
        """Read `artifact_prescan`, borrowing epoching/Welch settings from `artifact_detection`."""
        config = load_config(config_path)

        def value(*path: str) -> Any:
            configured = nested_get(config, path)
            if configured is None:
                raise ValueError(f"Missing artifact prescan config value: {'.'.join(path)}")
            return configured

        detection = ("artifact_detection",)
        prescan = ("artifact_prescan",)
        settings: dict[str, object] = dict(
            epoch_seconds=float(value(*detection, "epoch_seconds")),
            chunk_memory_mb=float(value(*detection, "chunking", "target_memory_mb")),
            chunk_working_array_factor=int(
                value(*detection, "chunking", "working_array_factor")
            ),
            max_chunk_epochs=int(value(*detection, "chunking", "max_chunk_epochs")),
            welch_segment_seconds=float(value(*detection, "welch", "segment_seconds")),
            psd_minimum_frequency_hz=float(
                value(*detection, "welch", "minimum_frequency_hz")
            ),
            psd_maximum_frequency_hz=float(
                value(*detection, "welch", "maximum_frequency_hz")
            ),
            high_frequency_band_hz=two_float_tuple(
                value(*detection, "welch", "high_frequency_band_hz"), "high-frequency band"
            ),
            line_noise_band_hz=two_float_tuple(
                value(*detection, "welch", "line_noise_band_hz"), "line-noise band"
            ),
            minimum_broadband_rms_uv=float(
                value(*prescan, "dead_signal", "minimum_broadband_rms_uv")
            ),
            max_edge_fraction=float(value(*prescan, "hard_failure", "maximum_edge_fraction")),
            robust_mad_scale=float(value(*detection, "thresholds", "robust_mad_scale")),
            extreme_z_threshold=float(value(*prescan, "extreme", "robust_z")),
            extreme_features_required=int(value(*prescan, "extreme", "features_required")),
            extreme_features=tuple(value(*prescan, "extreme", "features")),
            merge_within_s=float(value(*prescan, "periods", "merge_within_s")),
            minimum_run_s=float(value(*prescan, "periods", "minimum_run_s")),
            bridge_gap_s=float(value(*prescan, "periods", "bridge_gap_s")),
            output_suffix=str(value(*prescan, "output", "suffix")),
            overwrite=bool(value(*prescan, "output", "overwrite")),
        )
        settings.update({key: value for key, value in overrides.items() if value is not None})
        result = cls(**settings)
        for name in ("merge_within_s", "minimum_run_s", "bridge_gap_s"):
            if getattr(result, name) < 0:
                raise ValueError(f"artifact_prescan {name} must not be negative")
        return result


def flag_prescan_epochs(
    eeg_features: pd.DataFrame, settings: PrescanSettings
) -> pd.DataFrame:
    """Return one row per epoch with its dead/hard-failure/extreme flags."""
    features = eeg_features.copy()
    broadband_rms = np.sqrt(features["broadband_power_uv2"].clip(lower=0))
    features["dead"] = broadband_rms < settings.minimum_broadband_rms_uv
    features["hard_failure"] = features["nonfinite"] | (
        features["edge_fraction"] > settings.max_edge_fraction
    )
    live = ~(features["dead"] | features["hard_failure"])

    # z-scores against live epochs only: fit each channel's median/MAD on its
    # live rows, then score every row against that fit.
    above = np.zeros(len(features), dtype=int)
    for feature in settings.extreme_features:
        z = np.full(len(features), np.nan)
        for _, index in features.groupby("channel").groups.items():
            rows = features.index.get_indexer(index)
            channel_live = live.to_numpy()[rows]
            z[rows] = _robust_z_against(
                features[feature].to_numpy()[rows], channel_live, settings.robust_mad_scale
            )
        above += np.nan_to_num(z, nan=0.0) > settings.extreme_z_threshold
    features["extreme"] = live & (above >= settings.extreme_features_required)

    epochs = features.groupby("epoch_id", as_index=False).agg(
        time_s=("time_s", "first"),
        dead=("dead", "any"),
        hard_failure=("hard_failure", "any"),
        extreme=("extreme", "any"),
    )
    epochs["flagged"] = epochs["dead"] | epochs["hard_failure"] | epochs["extreme"]
    return epochs


def _robust_z_against(values: np.ndarray, reference: np.ndarray, mad_scale: float) -> np.ndarray:
    """Like `robust_upper_z`, but the log-median/MAD are fitted on `reference` rows only.

    Every value is scored against that fit, so rows outside the reference
    (dead or failed epochs) cannot move the baseline they are compared with.
    """
    transformed = np.log10(np.clip(values.astype(float), np.finfo(float).tiny, None))
    reference_values = transformed[reference & np.isfinite(transformed)]
    if reference_values.size == 0:
        return np.full(values.shape, np.nan)
    median = np.median(reference_values)
    mad = np.median(np.abs(reference_values - median))
    if mad == 0:
        return np.zeros(values.shape)
    return (transformed - median) / (mad_scale * mad)


def artifact_periods(
    flagged: np.ndarray,
    epoch_seconds: float,
    *,
    merge_within_s: float,
    minimum_run_s: float,
    bridge_gap_s: float,
) -> list[tuple[int, int]]:
    """Turn per-epoch flags into `[start_epoch, stop_epoch)` artifact periods.

    1. Join flagged runs separated by at most `merge_within_s` of clean signal,
       so one noisy block with the odd clean-looking epoch stays one run.
    2. Drop runs shorter than `minimum_run_s`; brief artifacts are left to the
       post-scoring detector.
    3. Bridge surviving periods separated by less than `bridge_gap_s` of clean
       signal into one continuous period.
    """
    runs = _runs(np.asarray(flagged, dtype=bool))
    runs = _join(runs, max_gap_epochs=merge_within_s / epoch_seconds, inclusive=True)
    runs = [(a, b) for a, b in runs if (b - a) * epoch_seconds >= minimum_run_s]
    return _join(runs, max_gap_epochs=bridge_gap_s / epoch_seconds, inclusive=False)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _join(
    runs: list[tuple[int, int]], *, max_gap_epochs: float, inclusive: bool
) -> list[tuple[int, int]]:
    joined: list[tuple[int, int]] = []
    for start, stop in runs:
        if joined:
            gap = start - joined[-1][1]
            if gap < max_gap_epochs or (inclusive and gap <= max_gap_epochs):
                joined[-1] = (joined[-1][0], stop)
                continue
        joined.append((start, stop))
    return joined


def period_table(
    epochs: pd.DataFrame, periods: list[tuple[int, int]], epoch_seconds: float
) -> pd.DataFrame:
    """Summarise each period: its span and how many epochs inside were flagged, and why."""
    rows = []
    for start, stop in periods:
        inside = epochs.iloc[start:stop]
        rows.append(
            {
                "start_s": start * epoch_seconds,
                "end_s": stop * epoch_seconds,
                "duration_s": (stop - start) * epoch_seconds,
                "flagged_fraction": float(inside["flagged"].mean()),
                "dead_epochs": int(inside["dead"].sum()),
                "hard_failure_epochs": int(inside["hard_failure"].sum()),
                "extreme_epochs": int(inside["extreme"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=PERIOD_COLUMNS)


def prescan_fif(
    fif_path: str | Path,
    settings: PrescanSettings,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Scan one FIF and write its artifact periods; skip when the output exists."""
    fif_path = Path(fif_path)
    csv_path, parquet_path = artifact_output_paths(
        fif_path, output_suffix=settings.output_suffix, output_dir=output_dir
    )
    existing = [path for path in (csv_path, parquet_path) if path.exists()]
    if existing and not settings.overwrite:
        return {"fif_path": str(fif_path), "status": "skipped_exists",
                "existing_outputs": [str(path) for path in existing]}
    if not fif_path.is_file():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")

    mne = import_mne()
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_epochs = complete_epoch_count(raw.n_times, sfreq, settings.epoch_seconds)
    if n_epochs == 0:
        raise ValueError("The recording does not contain one complete analysis epoch.")

    print(f"FIF: {fif_path}")
    print(f"Scanning all {n_epochs:,} complete epochs")
    eeg_features, _, eeg_channels, _, _ = extract_features(
        raw,
        n_epochs,
        settings.epoch_seconds,
        settings.chunk_memory_mb,
        settings.chunk_working_array_factor,
        settings.max_chunk_epochs,
        settings.welch_segment_seconds,
        settings.psd_minimum_frequency_hz,
        settings.psd_maximum_frequency_hz,
        settings.high_frequency_band_hz,
        settings.line_noise_band_hz,
    )
    epochs = flag_prescan_epochs(eeg_features, settings)
    periods = artifact_periods(
        epochs["flagged"].to_numpy(),
        settings.epoch_seconds,
        merge_within_s=settings.merge_within_s,
        minimum_run_s=settings.minimum_run_s,
        bridge_gap_s=settings.bridge_gap_s,
    )
    table = period_table(epochs, periods, settings.epoch_seconds)

    excluded_s = float(table["duration_s"].sum())
    total_s = n_epochs * settings.epoch_seconds
    print(
        f"Flagged {int(epochs['flagged'].sum()):,} / {n_epochs:,} epochs "
        f"(dead {int(epochs['dead'].sum()):,}, hard failure "
        f"{int(epochs['hard_failure'].sum()):,}, extreme {int(epochs['extreme'].sum()):,})"
    )
    print(
        f"{len(table)} artifact period(s) to leave unscored: "
        f"{excluded_s / 3600:.2f} h of {total_s / 3600:.2f} h ({excluded_s / total_s:.1%})"
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(table, csv_path)
    table.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path}")
    write_provenance(
        "artifact_prescan",
        outputs=[parquet_path, csv_path],
        inputs={"fif_path": str(fif_path), "eeg_channels": list(eeg_channels)},
        parameters={
            "epoch_seconds": settings.epoch_seconds,
            "minimum_broadband_rms_uv": settings.minimum_broadband_rms_uv,
            "broadband_hz": [settings.psd_minimum_frequency_hz, settings.psd_maximum_frequency_hz],
            "max_edge_fraction": settings.max_edge_fraction,
            "robust_mad_scale": settings.robust_mad_scale,
            "extreme_z_threshold": settings.extreme_z_threshold,
            "extreme_features_required": settings.extreme_features_required,
            "extreme_features": list(settings.extreme_features),
            "merge_within_s": settings.merge_within_s,
            "minimum_run_s": settings.minimum_run_s,
            "bridge_gap_s": settings.bridge_gap_s,
            "n_epochs": int(n_epochs),
            "sampling_rate_hz": sfreq,
        },
    )
    return {"fif_path": str(fif_path), "status": "written", "parquet_path": str(parquet_path),
            "n_periods": len(table), "excluded_s": excluded_s}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fif_path", nargs="?", default=None, help="Input MNE FIF recording.")
    parser.add_argument(
        "--subject", "--subjid", dest="subject", default=None,
        help="Subject ID, for example 66 or sub-066. Alternative to fif_path.",
    )
    session_selector = parser.add_mutually_exclusive_group()
    session_selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    session_selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
    )
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH),
        help=f"Artifact detection YAML config (default: {DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH}).",
    )
    parser.add_argument("--minimum-broadband-rms-uv", type=float, default=None)
    parser.add_argument("--extreme-z-threshold", type=float, default=None)
    parser.add_argument("--merge-within-s", type=float, default=None)
    parser.add_argument("--minimum-run-s", type=float, default=None)
    parser.add_argument("--bridge-gap-s", type=float, default=None)
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory; defaults to the session's dedicated artifacts directory.",
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def _resolve_fif_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[Path]:
    if args.fif_path is not None:
        if args.subject is not None:
            parser.error("use either fif_path or --subject/--date/--session, not both")
        return [Path(args.fif_path)]
    if args.subject is None:
        parser.error("fif_path or --subject is required")
    if args.date is None and args.session is None:
        parser.error("--subject requires either --date or --session")
    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(args.derivatives_root or get_derivatives_root()).resolve(strict=False)
    pairs = select_recordings(
        rawdata_root, derivatives_root, subject=args.subject, date=args.date, session=args.session
    )
    return [fif_path for _, fif_path in pairs]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    fif_paths = _resolve_fif_paths(parser, args)
    settings = PrescanSettings.from_yaml(
        args.config,
        minimum_broadband_rms_uv=args.minimum_broadband_rms_uv,
        extreme_z_threshold=args.extreme_z_threshold,
        merge_within_s=args.merge_within_s,
        minimum_run_s=args.minimum_run_s,
        bridge_gap_s=args.bridge_gap_s,
        overwrite=args.overwrite,
    )
    for fif_path in fif_paths:
        result = prescan_fif(fif_path, settings, args.output_dir)
        if result["status"] == "skipped_exists":
            existing = ", ".join(result["existing_outputs"])
            print(f"Skipped, output exists: {existing}. Pass --overwrite to replace it.")


if __name__ == "__main__":
    main()
