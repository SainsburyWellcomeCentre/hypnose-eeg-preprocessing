"""Run all session quality-control checks and decide analysis readiness."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.correlation import compute_state_channel_correlations
from scripts.analysis.emg import compute_state_emg_rms
from scripts.analysis.power_spectra import compute_state_spectra
from scripts.io.input_paths import artifact_path, scoring_path
from scripts.io.output_paths import quality_control_output_path, save_csv
from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
from scripts.qc.recording_integrity import check_pair
from scripts.utils.recording_selection import select_recordings
from scripts.qc.artifacts import build_artifact_report, channels_from_row
from scripts.qc.sleep_scoring import prepare_scoring_output
from scripts.qc.spectra import (
    SpectraConfig,
    build_spectral_quality_report,
    load_spectra_config,
)
from scripts.qc.thresholds import load_performance_check, load_qc_thresholds
from scripts.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    DEFAULT_SPECTRA_CONFIG_PATH,
    coalesce,
)
from scripts.utils.epochs import infer_epoch_seconds


PERFORMANCE_CHECK = load_performance_check()
SECTION_COLUMNS = ["section", "status", "metric", "value", "threshold", "detail"]
REVIEW_COLUMNS = [
    "section",
    "start_s",
    "end_s",
    "duration_s",
    "sleep_state",
    "channels",
    "metric",
    "value",
    "threshold",
    "reason",
]


def overall_status(sections: pd.DataFrame) -> str:
    """Return the most severe status represented in a section table."""
    if sections.empty:
        return "fail"
    return max(
        sections["status"].astype(str),
        key=lambda status: PERFORMANCE_CHECK.get(status, PERFORMANCE_CHECK["fail"]),
    )


def spectral_quality_sections(
    quality_report: pd.DataFrame,
    config: SpectraConfig,
) -> list[dict[str, object]]:
    """Convert the configured determining-channel results into QC sections."""
    determining = quality_report.loc[quality_report["determines_recording_quality"]]
    if determining.empty:
        raise ValueError("Spectral quality report has no determining EEG channel")

    spectral_status = max(
        determining["spectral_quality_status"].str.lower(),
        key=lambda status: PERFORMANCE_CHECK[status],
    )
    emg_status = max(
        determining["emg_quality_status"].str.lower(),
        key=lambda status: PERFORMANCE_CHECK[status],
    )
    spectral_passes = int(determining["frequency_expectation_met"].sum())
    emg_passes = int(determining["emg_expectation_met"].sum())
    state_count = len(determining)
    channel_name = str(determining["eeg_channel"].iloc[0])
    thresholds = (
        f"wake_delta<nrem×{config.wake_delta_max_nrem_ratio:g}; "
        f"nrem_delta>wake×{config.nrem_delta_min_wake_ratio:g}, "
        f"rem×{config.nrem_delta_min_rem_ratio:g}; "
        f"rem_theta/delta>nrem×{config.rem_theta_delta_min_nrem_ratio:g}"
    )
    return [
        {
            "section": "power_spectra",
            "status": spectral_status,
            "metric": "configured_state_band_expectations_met",
            "value": f"{spectral_passes}/{state_count}",
            "threshold": thresholds,
            "detail": (
                f"determining EEG channel {config.determining_eeg_channel_number}="
                f"{channel_name}; "
                + "; ".join(
                    f"{row.sleep_state}={str(row.spectral_quality_status).lower()}"
                    for row in determining.itertuples(index=False)
                )
            ),
        },
        {
            "section": "emg_rms",
            "status": emg_status,
            "metric": "configured_state_emg_order_met",
            "value": f"{emg_passes}/{state_count}",
            "threshold": ">=".join(
                config.sleep_state_names[state] for state in config.emg_state_order
            ),
            "detail": "; ".join(
                f"{row.sleep_state} median={row.emg_rms_median_uv:g} µV, "
                f"status={str(row.emg_quality_status).lower()}"
                for row in determining.itertuples(index=False)
            ),
        },
    ]


def _section(
    rows: list[dict[str, object]],
    section: str,
    status: str,
    metric: str,
    value: object,
    threshold: object,
    detail: str,
) -> None:
    rows.append(
        {
            "section": section,
            "status": status,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def _review(
    rows: list[dict[str, object]],
    *,
    section: str,
    start_s: float,
    end_s: float,
    sleep_state: object = "",
    channels: str = "",
    metric: str,
    value: object,
    threshold: object,
    reason: str,
) -> None:
    rows.append(
        {
            "section": section,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "sleep_state": sleep_state,
            "channels": channels,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "reason": reason,
        }
    )


def _correlation_threshold(
    channel_1_type: str,
    channel_2_type: str,
    *,
    eeg_eeg_threshold: float,
    eeg_emg_threshold: float,
) -> float | None:
    pair_types = {channel_1_type.lower(), channel_2_type.lower()}
    if pair_types == {"eeg"}:
        return eeg_eeg_threshold
    if pair_types == {"eeg", "emg"}:
        return eeg_emg_threshold
    return None


def _validate_fraction(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 100:
        parser.error(f"{name} must be between 0 and 100")


def review_output_paths(requested: str | Path) -> tuple[Path, Path]:
    """Return paired CSV and parquet paths for a requested review output."""
    path = Path(requested)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path, path.with_suffix(".parquet")
    if suffix in {".parquet", ".pq"}:
        return path.with_suffix(".csv"), path.with_suffix(".parquet")
    return path.with_suffix(".csv"), path.with_suffix(".parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--spectra-config",
        default=str(DEFAULT_SPECTRA_CONFIG_PATH),
        help=f"Spectral quality YAML (default: {DEFAULT_SPECTRA_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--qc-config",
        default=str(DEFAULT_QUALITY_CONTROL_CONFIG_PATH),
        help=f"Quality-control threshold YAML (default: {DEFAULT_QUALITY_CONTROL_CONFIG_PATH}).",
    )
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument("--analysis-epoch-seconds", type=float, default=None)
    parser.add_argument("--chunk-epochs", type=int, default=None)
    parser.add_argument("--duration-tolerance", type=float, default=None)
    parser.add_argument("--min-gap", type=float, default=None)
    parser.add_argument("--gap-scan-chunk-seconds", type=float, default=None)
    parser.add_argument("--max-gap-percent", type=float, default=None)
    parser.add_argument("--max-longest-gap", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--max-low-confidence-percent", type=float, default=None)
    parser.add_argument("--max-undefined-percent", type=float, default=None)
    parser.add_argument("--max-artifact-percent", type=float, default=None)
    parser.add_argument("--eeg-eeg-threshold", type=float, default=None)
    parser.add_argument("--eeg-emg-threshold", type=float, default=None)
    parser.add_argument("--max-correlation-review-percent", type=float, default=None)
    parser.add_argument("--welch-seconds", type=float, default=None)
    parser.add_argument("--fmin", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument(
        "--summary",
        nargs="?",
        const="qc_summary.csv",
        default=None,
        help="Optionally save section results in the shared session QC directory.",
    )
    parser.add_argument(
        "--review-epochs",
        nargs="?",
        const="qc_review_epochs.csv",
        default=None,
        help="Optionally save review ranges as paired CSV and parquet files.",
    )
    return parser


def run_qc(
    edf_path: Path,
    fif_path: Path,
    scoring_file: Path,
    artifact_file: Path,
    args: argparse.Namespace,
    spectra_config: SpectraConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all QC sections for one recording and return summary/review tables."""
    sections: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []

    integrity, gaps = check_pair(
        edf_path,
        fif_path,
        duration_tolerance_s=args.duration_tolerance,
        min_gap_s=args.min_gap,
        chunk_duration_s=args.gap_scan_chunk_seconds,
        max_gap_percent=args.max_gap_percent,
        max_longest_gap_s=args.max_longest_gap,
    )
    duration_ok = abs(integrity.edf_fif_difference_s) <= args.duration_tolerance
    integrity_status = "fail" if not duration_ok else integrity.status
    _section(
        sections,
        "recording_integrity",
        integrity_status,
        "edf_fif_duration_difference_s",
        integrity.edf_fif_difference_s,
        args.duration_tolerance,
        f"{len(gaps)} gap(s) detected ({integrity.edf_gap_total_s:.1f}s total, "
        f"{integrity.edf_gap_percent:.2f}% of recording, "
        f"longest {integrity.edf_longest_gap_s:.1f}s)",
    )
    for gap in gaps:
        _review(
            reviews,
            section="recording_integrity",
            start_s=gap.start_s,
            end_s=gap.end_s,
            metric="gap_duration_s",
            value=gap.duration_s,
            threshold=args.min_gap,
            reason=f"{gap.kind}: {gap.detail}",
        )

    raw_scores = pd.read_parquet(scoring_file)
    scored = prepare_scoring_output(
        raw_scores, confidence_threshold=args.confidence_threshold
    )
    scoring_epoch_s = infer_epoch_seconds(scored)
    signal = (
        scored["kind"].astype(str).eq("signal")
        if "kind" in scored
        else scored["predicted_state"].ne("Undefined")
    )
    signal_count = int(signal.sum())
    low_confidence = scored["low_confidence"].astype(bool)
    low_percent = 100.0 * int(low_confidence.sum()) / signal_count if signal_count else 100.0
    undefined = scored["predicted_state"].eq("Undefined")
    undefined_percent = 100.0 * int(undefined.sum()) / len(scored) if len(scored) else 100.0
    scoring_status = (
        "review"
        if low_percent > args.max_low_confidence_percent
        or undefined_percent > args.max_undefined_percent
        else "pass"
    )
    _section(
        sections,
        "somnotate_scoring",
        scoring_status,
        "low_confidence_percent",
        low_percent,
        args.max_low_confidence_percent,
        f"undefined={undefined_percent:.2f}% (limit {args.max_undefined_percent:.2f}%)",
    )
    scoring_review = scored.loc[low_confidence | undefined]
    for _, row in scoring_review.iterrows():
        is_undefined = row["predicted_state"] == "Undefined"
        _review(
            reviews,
            section="somnotate_scoring",
            start_s=float(row["time_s"]),
            end_s=float(row["time_s"]) + scoring_epoch_s,
            sleep_state=row["predicted_state"],
            metric="predicted_probability",
            value=row["predicted_probability"],
            threshold=args.confidence_threshold,
            reason="undefined/unscored epoch" if is_undefined else "low prediction confidence",
        )

    artifact_epochs = pd.read_parquet(artifact_file)
    artifact_report = build_artifact_report(artifact_epochs)
    flagged = artifact_epochs["artifact"].astype(bool)
    artifact_percent = 100.0 * int(flagged.sum()) / len(artifact_epochs) if len(artifact_epochs) else 0.0
    max_channel_percent = (
        float(artifact_report.overall["recording_percent"].max())
        if not artifact_report.overall.empty
        else 0.0
    )
    artifact_status = "review" if max_channel_percent > args.max_artifact_percent else "pass"
    _section(
        sections,
        "artifacts",
        artifact_status,
        "maximum_channel_artifact_percent",
        max_channel_percent,
        args.max_artifact_percent,
        f"{int(flagged.sum())} flagged epochs ({artifact_percent:.2f}% overall)",
    )
    for _, row in artifact_epochs.loc[flagged].iterrows():
        channels = channels_from_row(row)
        _review(
            reviews,
            section="artifacts",
            start_s=float(row["time_s"]),
            end_s=float(row["time_s"]) + artifact_report.epoch_seconds,
            sleep_state=str(row.get("sleep_state", "")),
            channels="; ".join(channels),
            metric="artifact",
            value=1.0,
            threshold=1.0,
            reason=str(row.get("artifact_features", "artifact detector flag")),
        )

    correlations, correlation_channels = compute_state_channel_correlations(
        fif_path,
        scoring_file,
        artifact_path=artifact_file,
        epoch_seconds=args.analysis_epoch_seconds,
        chunk_epochs=args.chunk_epochs,
    )
    pair_types = list(
        zip(
            correlations["channel_1_type"].astype(str).str.lower(),
            correlations["channel_2_type"].astype(str).str.lower(),
        )
    )
    threshold_by_pair = {
        pair: _correlation_threshold(
            pair[0],
            pair[1],
            eeg_eeg_threshold=args.eeg_eeg_threshold,
            eeg_emg_threshold=args.eeg_emg_threshold,
        )
        for pair in set(pair_types)
    }
    thresholds = pd.Series(pair_types, index=correlations.index).map(threshold_by_pair).to_numpy(dtype=float)
    correlation_flags = np.abs(correlations["pearson_r"].to_numpy(dtype=float)) > thresholds
    flagged_correlations = correlations.loc[correlation_flags].copy()
    flagged_correlations["review_threshold"] = thresholds[correlation_flags]
    flagged_epoch_count = flagged_correlations["epoch_id"].nunique()
    valid_epoch_count = correlations["epoch_id"].nunique()
    correlation_percent = (
        100.0 * flagged_epoch_count / valid_epoch_count if valid_epoch_count else 100.0
    )
    correlation_status = (
        "review" if correlation_percent > args.max_correlation_review_percent else "pass"
    )
    _section(
        sections,
        "channel_correlation",
        correlation_status,
        "review_epoch_percent",
        correlation_percent,
        args.max_correlation_review_percent,
        f"{flagged_epoch_count}/{valid_epoch_count} epochs; channels={correlation_channels}",
    )
    for row in flagged_correlations.itertuples(index=False):
        _review(
            reviews,
            section="channel_correlation",
            start_s=float(row.time_s),
            end_s=float(row.time_s) + args.analysis_epoch_seconds,
            sleep_state=str(
                spectra_config.sleep_state_names.get(int(row.sleep_state), row.sleep_state)
            ),
            channels=f"{row.channel_1} ↔ {row.channel_2}",
            metric="absolute_pearson_r",
            value=abs(float(row.pearson_r)),
            threshold=float(row.review_threshold),
            reason="cross-channel correlation above review threshold",
        )

    frequencies, spectra, counts, eeg_channels = compute_state_spectra(
        fif_path,
        scoring_file,
        artifact_path=(
            None if spectra_config.include_artifacts else artifact_file
        ),
        epoch_seconds=args.analysis_epoch_seconds,
        welch_seconds=args.welch_seconds,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        chunk_epochs=args.chunk_epochs,
    )

    emg_rms, emg_channels = compute_state_emg_rms(
        fif_path,
        scoring_file,
        artifact_path=(
            None if spectra_config.include_artifacts else artifact_file
        ),
        epoch_seconds=args.analysis_epoch_seconds,
        chunk_epochs=args.chunk_epochs,
    )
    spectral_report = build_spectral_quality_report(
        frequencies,
        spectra,
        counts,
        eeg_channels,
        emg_rms,
        emg_channels,
        config=spectra_config,
    )
    sections.extend(spectral_quality_sections(spectral_report, spectra_config))

    return (
        pd.DataFrame(sections, columns=SECTION_COLUMNS),
        pd.DataFrame(reviews, columns=REVIEW_COLUMNS).sort_values(
            ["start_s", "section"], ignore_index=True
        ),
    )


def _print_results(sections: pd.DataFrame, reviews: pd.DataFrame) -> None:
    print("Quality-control sections:")
    for row in sections.itertuples(index=False):
        print(
            f"  {str(row.status).upper():6s} {row.section}: "
            f"{row.metric}={row.value} (threshold={row.threshold}) — {row.detail}"
        )
    status = overall_status(sections)
    print(f"Overall analysis readiness: {status.upper()}")
    print(f"Review entries: {len(reviews):,}")
    if len(reviews):
        print(f"Unique review start times: {reviews['start_s'].nunique():,}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spectra_config = load_spectra_config(args.spectra_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        qc_thresholds = load_qc_thresholds(args.qc_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.analysis_epoch_seconds is None:
        args.analysis_epoch_seconds = spectra_config.epoch_seconds
    if args.chunk_epochs is None:
        args.chunk_epochs = spectra_config.chunk_epochs
    if args.welch_seconds is None:
        args.welch_seconds = spectra_config.welch_seconds
    if args.fmin is None:
        args.fmin = spectra_config.fmin_hz
    if args.fmax is None:
        args.fmax = spectra_config.fmax_hz

    args.duration_tolerance = coalesce(
        args.duration_tolerance, qc_thresholds.duration_tolerance_s
    )
    args.min_gap = coalesce(args.min_gap, qc_thresholds.min_gap_s)
    args.gap_scan_chunk_seconds = coalesce(
        args.gap_scan_chunk_seconds, qc_thresholds.gap_scan_chunk_seconds
    )
    args.max_gap_percent = coalesce(
        args.max_gap_percent, qc_thresholds.max_gap_percent
    )
    args.max_longest_gap = coalesce(
        args.max_longest_gap, qc_thresholds.max_longest_gap_s
    )
    args.confidence_threshold = coalesce(
        args.confidence_threshold, qc_thresholds.confidence_threshold
    )
    args.max_low_confidence_percent = coalesce(
        args.max_low_confidence_percent, qc_thresholds.max_low_confidence_percent
    )
    args.max_undefined_percent = coalesce(
        args.max_undefined_percent, qc_thresholds.max_undefined_percent
    )
    args.max_artifact_percent = coalesce(
        args.max_artifact_percent, qc_thresholds.max_artifact_percent
    )
    args.eeg_eeg_threshold = coalesce(
        args.eeg_eeg_threshold, qc_thresholds.eeg_eeg_threshold
    )
    args.eeg_emg_threshold = coalesce(
        args.eeg_emg_threshold, qc_thresholds.eeg_emg_threshold
    )
    args.max_correlation_review_percent = coalesce(
        args.max_correlation_review_percent,
        qc_thresholds.max_correlation_review_percent,
    )

    for name in (
        "max_low_confidence_percent",
        "max_undefined_percent",
        "max_artifact_percent",
        "max_correlation_review_percent",
        "max_gap_percent",
    ):
        _validate_fraction(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("confidence_threshold", "eeg_eeg_threshold", "eeg_emg_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in (
        "analysis_epoch_seconds",
        "duration_tolerance",
        "min_gap",
        "gap_scan_chunk_seconds",
        "max_longest_gap",
        "welch_seconds",
        "fmin",
        "fmax",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    if args.analysis_epoch_seconds == 0 or args.min_gap == 0 or args.welch_seconds == 0:
        parser.error("epoch, gap, and Welch durations must be positive")
    if args.fmax <= args.fmin:
        parser.error("--fmax must be greater than --fmin")
    if args.chunk_epochs <= 0:
        parser.error("--chunk-epochs must be positive")

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(
        args.derivatives_root or get_derivatives_root()
    ).resolve(strict=False)
    try:
        pairs = select_recordings(
            rawdata_root,
            derivatives_root,
            subject=args.subject,
            date=args.date,
            session=args.session,
        )
        if len(pairs) != 1:
            raise ValueError(
                f"Selection matched {len(pairs)} recordings; choose one recording session"
            )
        edf_path, fif_path = pairs[0]
        scoring_file = scoring_path(edf_path, rawdata_root, derivatives_root)
        artifact_file = artifact_path(edf_path, rawdata_root, derivatives_root)
        if artifact_file is None:
            raise FileNotFoundError(f"No artifact parquet found for {edf_path.name}")
        print(f"EDF: {edf_path}")
        print(f"FIF: {fif_path}")
        print(f"Scoring: {scoring_file}")
        print(f"Artifacts: {artifact_file}")
        sections, reviews = run_qc(
            edf_path,
            fif_path,
            scoring_file,
            artifact_file,
            args,
            spectra_config,
        )
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))

    _print_results(sections, reviews)
    if args.summary is not None:
        summary_path = quality_control_output_path(
            args.summary, edf_path, rawdata_root, derivatives_root
        )
        save_csv(sections, summary_path)
    if args.review_epochs is not None:
        requested_review_path = quality_control_output_path(
            args.review_epochs, edf_path, rawdata_root, derivatives_root
        )
        review_csv_path, review_parquet_path = review_output_paths(
            requested_review_path
        )
        save_csv(reviews, review_csv_path)
        reviews.to_parquet(review_parquet_path, index=False)
        print(f"Saved: {review_parquet_path}")
    return 0 if overall_status(sections) != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
