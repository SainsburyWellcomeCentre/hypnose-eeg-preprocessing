"""Run all session quality-control checks and decide analysis readiness.

The section results (`qc_summary.csv`) and the unified review ranges
(`qc_review_epochs.csv`), each with a `.parquet` copy, are both written to
the shared session quality-control directory on every run, each prefixed with
the analyzed recording's stem -- for example
`sub-066_ses-001_recording-concat_qc_summary.csv`. Pass a filename to
`--summary`/`--review-epochs` to rename either one, or `--no-summary`/
`--no-review-epochs` to skip writing it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hypnose_eeg.analysis.correlation import compute_state_channel_correlations
from hypnose_eeg.analysis.emg import compute_state_emg_rms
from hypnose_eeg.analysis.power_spectra import compute_state_spectra
from hypnose_eeg.io.input_paths import artifact_path, prescan_artifact_path, scoring_path
from hypnose_eeg.io.output_paths import (
    quality_control_output_path,
    recording_output_name,
    save_csv,
)
from hypnose_eeg.io.repository_paths import resolve_data_roots
from hypnose_eeg.qc.recording_integrity import check_pair
from hypnose_eeg.utils.recording_selection import select_recordings
from hypnose_eeg.qc.artifacts import (
    build_artifact_report,
    channels_from_row,
    restrict_to_continuous_epochs,
)
from hypnose_eeg.qc.sleep_scoring import (
    prepare_scoring_output,
    sleep_state_proportion_report,
    sleep_state_proportions,
)
from hypnose_eeg.qc.spectra import (
    SpectraConfig,
    build_spectral_quality_report,
    load_spectra_config,
)
from hypnose_eeg.qc.thresholds import (
    QCThresholds,
    default_performance_check,
    load_qc_thresholds,
)
from hypnose_eeg.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    DEFAULT_SPECTRA_CONFIG_PATH,
    with_overrides,
)
from hypnose_eeg.utils.epochs import infer_epoch_seconds
from hypnose_eeg.utils.provenance import provenance_path, write_provenance


DEFAULT_SUMMARY_FILENAME = "qc_summary.csv"
DEFAULT_REVIEW_FILENAME = "qc_review_epochs.csv"
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
        key=lambda status: default_performance_check().get(
            status, default_performance_check()["fail"]
        ),
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
        key=lambda status: default_performance_check()[status],
    )
    emg_status = max(
        determining["emg_quality_status"].str.lower(),
        key=lambda status: default_performance_check()[status],
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


def sleep_state_proportion_section(proportion_report: pd.DataFrame) -> dict[str, object]:
    """Convert a `sleep_state_proportion_report` table into one QC section.

    Only rows with a configured threshold ("n/a" rows, i.e. Undefined, are
    excluded) count toward the section's pass/review status.
    """
    checked = proportion_report.loc[proportion_report["status"] != "n/a"]
    status = "review" if (checked["status"] == "review").any() else "pass"
    return {
        "section": "sleep_state_proportions",
        "status": status,
        "metric": "signal_percent_by_state",
        "value": "; ".join(
            f"{row.sleep_state}={row.signal_percent:.2f}%"
            for row in checked.itertuples(index=False)
        ),
        "threshold": "; ".join(
            f"{row.sleep_state}<={row.threshold_percent:g}%"
            for row in checked.itertuples(index=False)
        ),
        "detail": (
            "; ".join(
                f"{row.sleep_state} {row.status}"
                for row in checked.itertuples(index=False)
                if row.status == "review"
            )
            or "within thresholds"
        ),
    }


def artifact_prescan_section(
    periods: pd.DataFrame | None,
    scored: pd.DataFrame,
    *,
    scoring_epoch_s: float,
    max_excluded_percent: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """QC section and review ranges for the pre-scoring artifact scan's exclusions.

    `periods` is the recording's `*_prescan_artifacts.parquet` (None when the
    prescan has not run, which is itself a reason for review: dead or extreme
    signal could not have been kept out of scoring). The percentage is of the
    whole recording, taken from the prescan periods themselves; the detail also
    reports how much the predictions actually carry as `kind == "artifact"`, so
    predictions scored before the prescan (or without it) stand out.
    """
    if periods is None:
        section = {
            "section": "artifact_prescan",
            "status": "review",
            "metric": "excluded_percent",
            "value": "n/a",
            "threshold": max_excluded_percent,
            "detail": "no prescan output; run preprocessing:prescan_artifacts, then rescore",
        }
        return section, []

    duration_s = (
        float(scored["time_s"].max()) + scoring_epoch_s if len(scored) else 0.0
    )
    starts = periods["start_s"].astype(float).clip(lower=0, upper=duration_s)
    ends = periods["end_s"].astype(float).clip(lower=0, upper=duration_s)
    excluded_s = float((ends - starts).clip(lower=0).sum())
    excluded_percent = 100.0 * excluded_s / duration_s if duration_s else 0.0

    detail = (
        f"{len(periods)} period(s), {excluded_s / 3600:.2f} h of "
        f"{duration_s / 3600:.2f} h"
    )
    if "kind" in scored:
        applied_s = float(scored["kind"].astype(str).eq("artifact").sum()) * scoring_epoch_s
        detail += f"; {applied_s / 3600:.2f} h labelled artifact in the predictions"
        if excluded_s > 0 and applied_s == 0:
            detail += " (scored without the prescan; rescore with --overwrite)"

    section = {
        "section": "artifact_prescan",
        "status": "review" if excluded_percent > max_excluded_percent else "pass",
        "metric": "excluded_percent",
        "value": excluded_percent,
        "threshold": max_excluded_percent,
        "detail": detail,
    }
    reviews: list[dict[str, object]] = []
    for row in periods.itertuples(index=False):
        _review(
            reviews,
            section="artifact_prescan",
            start_s=float(row.start_s),
            end_s=float(row.end_s),
            metric="flagged_fraction",
            value=float(getattr(row, "flagged_fraction", 1.0)),
            # Periods have no per-range threshold; the limit applies to their total.
            threshold=float("nan"),
            reason=(
                "excluded from sleep scoring by the artifact prescan "
                f"(dead {int(getattr(row, 'dead_epochs', 0))}, hard failure "
                f"{int(getattr(row, 'hard_failure_epochs', 0))}, extreme "
                f"{int(getattr(row, 'extreme_epochs', 0))} epochs)"
            ),
        )
    return section, reviews


def normalization_section(
    provenance: dict[str, object] | None, *, max_offset_z: float
) -> dict[str, object]:
    """QC section for the baseline the recording was normalized against.

    Read from the scoring provenance's `normalization` entry. A short recording
    scored against a reference baseline is sent to review when any channel's
    offset from that reference exceeds `max_offset_z` (a gain or impedance
    change would make the borrowed baseline unsafe); a short recording for
    which no reference was found is sent to review because it was normalized
    against its own unrepresentative statistics. Predictions scored before
    this was recorded pass, with the value reported as n/a.
    """
    record = (provenance or {}).get("parameters", {}).get("normalization")
    if not isinstance(record, dict):
        return {
            "section": "normalization",
            "status": "pass",
            "metric": "max_abs_reference_offset_z",
            "value": "n/a",
            "threshold": max_offset_z,
            "detail": "no normalization record in the scoring provenance (scored before it was recorded)",
        }

    signal_hours = record.get("signal_hours")
    signal_text = "n/a" if signal_hours is None else f"{float(signal_hours):.2f} h"
    status_name = record.get("reference_status")
    reference = record.get("reference") or {}
    offsets = [value for value in (record.get("offset_z") or []) if value is not None]

    if status_name == "used" and offsets:
        largest = max(abs(float(value)) for value in offsets)
        return {
            "section": "normalization",
            "status": "review" if largest > max_offset_z else "pass",
            "metric": "max_abs_reference_offset_z",
            "value": largest,
            "threshold": max_offset_z,
            "detail": (
                f"short recording ({signal_text} of signal) normalized against "
                f"{reference.get('session')} (date {reference.get('date')}, "
                f"{reference.get('signal_hours')} h, {reference.get('days_from_recording')} days); "
                "offsets " + ", ".join(f"{float(value):+.2f}" for value in offsets)
            ),
        }
    if record.get("short_recording"):
        rejected = record.get("rejected_references") or []
        rejected_text = (
            "; rejected for their offset: "
            + ", ".join(str(candidate.get("session")) for candidate in rejected)
            if rejected
            else ""
        )
        return {
            "section": "normalization",
            "status": "review",
            "metric": "max_abs_reference_offset_z",
            "value": "n/a",
            "threshold": max_offset_z,
            "detail": (
                f"short recording ({signal_text} of signal, below "
                f"{record.get('min_signal_hours')} h) normalized against its own statistics "
                f"(reference {status_name}{rejected_text}); rescore with --reference-session"
            ),
        }
    return {
        "section": "normalization",
        "status": "pass",
        "metric": "max_abs_reference_offset_z",
        "value": "n/a",
        "threshold": max_offset_z,
        "detail": f"{record.get('source')} baseline; {signal_text} of signal",
    }


def _read_provenance(output: Path) -> dict[str, object] | None:
    path = provenance_path(output)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


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


def paired_output_paths(requested: str | Path) -> tuple[Path, Path]:
    """Return paired CSV and parquet paths for a requested summary or review output."""
    path = Path(requested)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path, path.with_suffix(".parquet")
    if suffix in {".parquet", ".pq"}:
        return path.with_suffix(".csv"), path.with_suffix(".parquet")
    return path.with_suffix(".csv"), path.with_suffix(".parquet")


def summary_parquet_table(sections: pd.DataFrame) -> pd.DataFrame:
    """The section table with `value`/`threshold` as their CSV text.

    Those columns hold numbers for some sections and text such as `3/4` or
    `n/a` for others, which a parquet column cannot mix; the text form matches
    what the CSV holds, so readers see the same values from either file.
    """
    table = sections.copy()
    for column in ("value", "threshold"):
        table[column] = ["" if pd.isna(value) else str(value) for value in table[column]]
    return table.astype({column: str for column in SECTION_COLUMNS})


@dataclass(frozen=True)
class SummaryQCSettings:
    """The thresholds and analysis settings every summary QC section uses.

    `spectra` supplies the analysis epoch, read chunking, Welch window, and
    frequency range shared by the correlation, spectra, and EMG sections, as
    well as the spectral quality expectations. `qc_config`/`spectra_config`
    name the files they were loaded from, for the provenance sidecar.
    """

    thresholds: QCThresholds
    spectra: SpectraConfig
    qc_config: str | None = None
    spectra_config: str | None = None


# Spectra-config fields `summary_qc_settings` accepts as overrides; every other
# override names a `QCThresholds` field.
SPECTRA_OVERRIDES = frozenset(
    {"epoch_seconds", "chunk_epochs", "welch_seconds", "fmin_hz", "fmax_hz"}
)


def summary_qc_settings(
    *,
    qc_config: str | Path = DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    spectra_config: str | Path = DEFAULT_SPECTRA_CONFIG_PATH,
    **overrides: float | int | None,
) -> SummaryQCSettings:
    """Load the QC and spectra YAMLs, then apply any overrides that are not None.

    Overrides are `QCThresholds` fields (`max_artifact_percent=10.0`) or the
    spectra analysis fields in `SPECTRA_OVERRIDES` (`epoch_seconds=4.0`).
    Raises ValueError for an unknown name or an out-of-range value.
    """
    spectra_names = SPECTRA_OVERRIDES & overrides.keys()
    threshold_names = overrides.keys() - spectra_names
    unknown = threshold_names - {field.name for field in fields(QCThresholds)}
    if unknown:
        raise ValueError(f"Unknown summary QC setting(s): {', '.join(sorted(unknown))}")
    return SummaryQCSettings(
        thresholds=with_overrides(
            load_qc_thresholds(qc_config),
            **{name: overrides[name] for name in threshold_names},
        ),
        spectra=with_overrides(
            load_spectra_config(spectra_config),
            **{name: overrides[name] for name in spectra_names},
        ),
        qc_config=str(qc_config),
        spectra_config=str(spectra_config),
    )


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
    parser.add_argument("--max-wake-percent", type=float, default=None)
    parser.add_argument("--max-nrem-percent", type=float, default=None)
    parser.add_argument("--max-rem-percent", type=float, default=None)
    parser.add_argument("--max-artifact-percent", type=float, default=None)
    parser.add_argument("--max-prescan-excluded-percent", type=float, default=None)
    parser.add_argument("--eeg-eeg-threshold", type=float, default=None)
    parser.add_argument("--eeg-emg-threshold", type=float, default=None)
    parser.add_argument("--max-correlation-review-percent", type=float, default=None)
    parser.add_argument("--max-reference-offset-z", type=float, default=None)
    parser.add_argument("--welch-seconds", type=float, default=None)
    parser.add_argument("--fmin", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument(
        "--summary",
        nargs="?",
        const=DEFAULT_SUMMARY_FILENAME,
        default=DEFAULT_SUMMARY_FILENAME,
        help="Filename for the section results, saved as paired CSV and parquet "
        "files in the shared session QC directory, prefixed with the recording "
        "stem (default: "
        f"<recording>_{DEFAULT_SUMMARY_FILENAME}).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_const",
        const=None,
        dest="summary",
        help="Do not save the section results.",
    )
    parser.add_argument(
        "--review-epochs",
        nargs="?",
        const=DEFAULT_REVIEW_FILENAME,
        default=DEFAULT_REVIEW_FILENAME,
        help="Filename for the review ranges, saved as paired CSV and parquet "
        f"files prefixed with the recording stem (default: "
        f"<recording>_{DEFAULT_REVIEW_FILENAME}).",
    )
    parser.add_argument(
        "--no-review-epochs",
        action="store_const",
        const=None,
        dest="review_epochs",
        help="Do not save the review ranges.",
    )
    return parser


def run_qc(
    edf_path: Path,
    fif_path: Path,
    scoring_file: Path,
    artifact_file: Path,
    settings: SummaryQCSettings | None = None,
    prescan_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all QC sections for one recording and return summary/review tables.

    `settings` defaults to the repository's QC and spectra configuration.
    `prescan_file` is the recording's prescan artifact periods, or None when
    the prescan has not run (which sends the artifact_prescan section to review).
    """
    settings = settings or summary_qc_settings()
    qc_thresholds = settings.thresholds
    spectra_config = settings.spectra
    sections: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []

    integrity, gaps = check_pair(
        edf_path,
        fif_path,
        duration_tolerance_s=qc_thresholds.duration_tolerance_s,
        min_gap_s=qc_thresholds.min_gap_s,
        chunk_duration_s=qc_thresholds.gap_scan_chunk_seconds,
        max_gap_percent=qc_thresholds.max_gap_percent,
        max_longest_gap_s=qc_thresholds.max_longest_gap_s,
    )
    duration_ok = abs(integrity.edf_fif_difference_s) <= qc_thresholds.duration_tolerance_s
    integrity_status = "pass" if duration_ok else "fail"
    _section(
        sections,
        "recording_integrity",
        integrity_status,
        "edf_fif_duration_difference_s",
        integrity.edf_fif_difference_s,
        qc_thresholds.duration_tolerance_s,
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
            threshold=qc_thresholds.min_gap_s,
            reason=f"{gap.kind}: {gap.detail}",
        )

    raw_scores = pd.read_parquet(scoring_file)
    scored = prepare_scoring_output(
        raw_scores, confidence_threshold=qc_thresholds.confidence_threshold
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
    scoring_status = "review" if low_percent > qc_thresholds.max_low_confidence_percent else "pass"
    _section(
        sections,
        "somnotate_scoring",
        scoring_status,
        "low_confidence_percent",
        low_percent,
        qc_thresholds.max_low_confidence_percent,
        f"undefined={undefined_percent:.2f}% (informational; not a review criterion; "
        f"configured expectation {qc_thresholds.max_undefined_percent:.2f}%)",
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
            threshold=qc_thresholds.confidence_threshold,
            reason="undefined/unscored epoch" if is_undefined else "low prediction confidence",
        )

    proportion_report = sleep_state_proportion_report(
        sleep_state_proportions(scored),
        max_wake_percent=qc_thresholds.max_wake_percent,
        max_nrem_percent=qc_thresholds.max_nrem_percent,
        max_rem_percent=qc_thresholds.max_rem_percent,
    )
    sections.append(sleep_state_proportion_section(proportion_report))
    sections.append(
        normalization_section(
            _read_provenance(scoring_file),
            max_offset_z=qc_thresholds.max_reference_offset_z,
        )
    )

    artifact_epochs = pd.read_parquet(artifact_file)
    artifact_epochs = restrict_to_continuous_epochs(artifact_epochs, scored)
    artifact_report = build_artifact_report(artifact_epochs)
    flagged = artifact_epochs["artifact"].astype(bool)
    artifact_percent = 100.0 * int(flagged.sum()) / len(artifact_epochs) if len(artifact_epochs) else 0.0
    max_channel_percent = (
        float(artifact_report.overall["recording_percent"].max())
        if not artifact_report.overall.empty
        else 0.0
    )
    artifact_status = "review" if max_channel_percent > qc_thresholds.max_artifact_percent else "pass"
    _section(
        sections,
        "artifacts",
        artifact_status,
        "maximum_channel_artifact_percent",
        max_channel_percent,
        qc_thresholds.max_artifact_percent,
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

    prescan_section, prescan_reviews = artifact_prescan_section(
        pd.read_parquet(prescan_file) if prescan_file is not None else None,
        scored,
        scoring_epoch_s=scoring_epoch_s,
        max_excluded_percent=qc_thresholds.max_prescan_excluded_percent,
    )
    sections.append(prescan_section)
    reviews.extend(prescan_reviews)

    correlations, correlation_channels = compute_state_channel_correlations(
        fif_path,
        scoring_file,
        artifact_path=artifact_file,
        epoch_seconds=spectra_config.epoch_seconds,
        chunk_epochs=spectra_config.chunk_epochs,
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
            eeg_eeg_threshold=qc_thresholds.eeg_eeg_threshold,
            eeg_emg_threshold=qc_thresholds.eeg_emg_threshold,
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
        "review" if correlation_percent > qc_thresholds.max_correlation_review_percent else "pass"
    )
    _section(
        sections,
        "channel_correlation",
        correlation_status,
        "review_epoch_percent",
        correlation_percent,
        qc_thresholds.max_correlation_review_percent,
        f"{flagged_epoch_count}/{valid_epoch_count} epochs; channels={correlation_channels}",
    )
    for row in flagged_correlations.itertuples(index=False):
        _review(
            reviews,
            section="channel_correlation",
            start_s=float(row.time_s),
            end_s=float(row.time_s) + spectra_config.epoch_seconds,
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
        epoch_seconds=spectra_config.epoch_seconds,
        welch_seconds=spectra_config.welch_seconds,
        fmin_hz=spectra_config.fmin_hz,
        fmax_hz=spectra_config.fmax_hz,
        chunk_epochs=spectra_config.chunk_epochs,
    )

    emg_rms, emg_channels = compute_state_emg_rms(
        fif_path,
        scoring_file,
        artifact_path=(
            None if spectra_config.include_artifacts else artifact_file
        ),
        epoch_seconds=spectra_config.epoch_seconds,
        chunk_epochs=spectra_config.chunk_epochs,
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


@dataclass(frozen=True)
class QCInputs:
    """The files one recording's summary QC reads, and the roots they resolve under."""

    edf_path: Path
    fif_path: Path
    scoring_file: Path
    artifact_file: Path
    prescan_file: Path | None  # None when the prescan has not run
    rawdata_root: Path
    derivatives_root: Path


@dataclass(frozen=True, eq=False)
class SessionQC:
    """One recording's section results and review ranges, with what produced them."""

    inputs: QCInputs
    settings: SummaryQCSettings
    sections: pd.DataFrame
    reviews: pd.DataFrame

    @property
    def status(self) -> str:
        """The most severe section status: pass, review, or fail."""
        return overall_status(self.sections)


def find_qc_inputs(
    subject: str | int,
    *,
    date: str | int | None = None,
    session: str | int | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
) -> QCInputs:
    """Resolve the one recording a session selects and the outputs QC reads for it.

    The roots default to the active data-location profile. Raises ValueError
    unless the selection matches exactly one recording, and FileNotFoundError
    when its artifact parquet is missing.
    """
    rawdata_root, derivatives_root = resolve_data_roots(rawdata_root, derivatives_root)
    pairs = select_recordings(
        rawdata_root, derivatives_root, subject=subject, date=date, session=session
    )
    if len(pairs) != 1:
        raise ValueError(
            f"Selection matched {len(pairs)} recordings; choose one recording session"
        )
    edf_path, fif_path = pairs[0]
    artifact_file = artifact_path(edf_path, rawdata_root, derivatives_root)
    if artifact_file is None:
        raise FileNotFoundError(f"No artifact parquet found for {edf_path.name}")
    return QCInputs(
        edf_path=edf_path,
        fif_path=fif_path,
        scoring_file=scoring_path(edf_path, rawdata_root, derivatives_root),
        artifact_file=artifact_file,
        prescan_file=prescan_artifact_path(edf_path, rawdata_root, derivatives_root),
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
    )


def compute_qc(inputs: QCInputs, settings: SummaryQCSettings | None = None) -> SessionQC:
    """Run every QC section for resolved inputs; nothing is written."""
    settings = settings or summary_qc_settings()
    sections, reviews = run_qc(
        inputs.edf_path,
        inputs.fif_path,
        inputs.scoring_file,
        inputs.artifact_file,
        settings,
        prescan_file=inputs.prescan_file,
    )
    return SessionQC(inputs=inputs, settings=settings, sections=sections, reviews=reviews)


def compute_session_qc(
    subject: str | int,
    *,
    date: str | int | None = None,
    session: str | int | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    settings: SummaryQCSettings | None = None,
) -> SessionQC:
    """Run every QC section for one subject's session; nothing is written.

    Override thresholds with `summary_qc_settings(max_artifact_percent=10.0)`;
    write the tables with `save_session_qc()`.
    """
    inputs = find_qc_inputs(
        subject,
        date=date,
        session=session,
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
    )
    return compute_qc(inputs, settings)


def save_session_qc(
    result: SessionQC,
    *,
    summary: str | Path | None = DEFAULT_SUMMARY_FILENAME,
    review_epochs: str | Path | None = DEFAULT_REVIEW_FILENAME,
) -> list[Path]:
    """Write the section results and review ranges, each as CSV and parquet.

    Relative names land in the session's quality-control directory, prefixed
    with the recording stem; None skips that table. A provenance sidecar is
    written beside whatever was saved.
    """
    inputs = result.inputs
    edf_path = inputs.edf_path
    written: list[Path] = []
    if summary is not None:
        summary_csv_path, summary_parquet_path = paired_output_paths(
            quality_control_output_path(
                recording_output_name(summary, edf_path),
                edf_path, inputs.rawdata_root, inputs.derivatives_root,
            )
        )
        save_csv(result.sections, summary_csv_path)
        summary_parquet_table(result.sections).to_parquet(summary_parquet_path, index=False)
        print(f"Saved: {summary_parquet_path}")
        written.extend([summary_csv_path, summary_parquet_path])
    if review_epochs is not None:
        requested_review_path = quality_control_output_path(
            recording_output_name(review_epochs, edf_path),
            edf_path, inputs.rawdata_root, inputs.derivatives_root,
        )
        review_csv_path, review_parquet_path = paired_output_paths(
            requested_review_path
        )
        save_csv(result.reviews, review_csv_path)
        result.reviews.to_parquet(review_parquet_path, index=False)
        print(f"Saved: {review_parquet_path}")
        written.extend([review_parquet_path, review_csv_path])

    if written:
        write_provenance(
            "quality_control",
            outputs=written,
            inputs={
                "edf_path": str(edf_path),
                "fif_path": str(inputs.fif_path),
                "scoring_file": str(inputs.scoring_file),
                "artifact_file": str(inputs.artifact_file),
                "prescan_file": str(inputs.prescan_file) if inputs.prescan_file else None,
            },
            parameters={
                "overall_status": result.status,
                "n_review_entries": int(len(result.reviews)),
                "qc_config": result.settings.qc_config,
                "spectra_config": result.settings.spectra_config,
            },
        )
    return written


def _print_inputs(inputs: QCInputs) -> None:
    print(f"EDF: {inputs.edf_path}")
    print(f"FIF: {inputs.fif_path}")
    print(f"Scoring: {inputs.scoring_file}")
    print(f"Artifacts: {inputs.artifact_file}")
    print(f"Prescan artifacts: {inputs.prescan_file or 'not found'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = summary_qc_settings(
            qc_config=args.qc_config,
            spectra_config=args.spectra_config,
            epoch_seconds=args.analysis_epoch_seconds,
            chunk_epochs=args.chunk_epochs,
            welch_seconds=args.welch_seconds,
            fmin_hz=args.fmin,
            fmax_hz=args.fmax,
            duration_tolerance_s=args.duration_tolerance,
            min_gap_s=args.min_gap,
            gap_scan_chunk_seconds=args.gap_scan_chunk_seconds,
            max_gap_percent=args.max_gap_percent,
            max_longest_gap_s=args.max_longest_gap,
            confidence_threshold=args.confidence_threshold,
            max_low_confidence_percent=args.max_low_confidence_percent,
            max_undefined_percent=args.max_undefined_percent,
            max_wake_percent=args.max_wake_percent,
            max_nrem_percent=args.max_nrem_percent,
            max_rem_percent=args.max_rem_percent,
            max_artifact_percent=args.max_artifact_percent,
            max_prescan_excluded_percent=args.max_prescan_excluded_percent,
            eeg_eeg_threshold=args.eeg_eeg_threshold,
            eeg_emg_threshold=args.eeg_emg_threshold,
            max_correlation_review_percent=args.max_correlation_review_percent,
            max_reference_offset_z=args.max_reference_offset_z,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        inputs = find_qc_inputs(
            args.subject,
            date=args.date,
            session=args.session,
            rawdata_root=args.rawdata_root,
            derivatives_root=args.derivatives_root,
        )
        _print_inputs(inputs)
        result = compute_qc(inputs, settings)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))

    _print_results(result.sections, result.reviews)
    save_session_qc(result, summary=args.summary, review_epochs=args.review_epochs)
    return 0 if result.status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
