"""Report artifact burden by channel, hour, and sleep state."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.io.output_paths import save_csv
from scripts.utils.epochs import continuous_epoch_ids, infer_epoch_seconds
from scripts.utils.sleep_states import load_sleep_states


SLEEP_STATE_NAMES = load_sleep_states().sleep_state_names


def sleep_state_name(value: object) -> str:
    """Return a display name for a numeric or textual sleep-state value."""
    if pd.isna(value):
        return "Undefined"
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "Undefined"
    return SLEEP_STATE_NAMES.get(numeric, str(numeric))


@dataclass(frozen=True)
class ArtifactReport:
    epoch_seconds: float
    overall: pd.DataFrame
    hourly: pd.DataFrame
    sleep_state: pd.DataFrame


def channels_from_row(row: pd.Series) -> list[str]:
    channels: list[str] = []
    explicit = row.get("artifact_channels")
    features = row.get("artifact_features")
    if pd.notna(explicit) and str(explicit).strip():
        channels.extend(
            item.strip()
            for item in re.split(r"\s*[;|]\s*", str(explicit))
            if item.strip()
        )
    elif pd.notna(features):
        for entry in str(features).split(" | "):
            channel, separator, _ = entry.partition(":")
            if separator and channel.strip():
                channels.append(channel.strip())
    if (
        bool(row.get("emg_extreme", False))
        and pd.notna(features)
        and "emg_outlier_with_eeg_outlier" in str(features)
    ):
        channels.append("EMG (combined)")
    return list(dict.fromkeys(channels))


def expand_artifact_channels(artifact_epochs: pd.DataFrame) -> pd.DataFrame:
    """Expand flagged epoch rows to one row per affected channel.

    Mirrors ``channels_from_row`` with vectorized pandas string operations
    instead of a Python loop per epoch, since a full recording can flag
    thousands of epochs.
    """
    required = {"epoch_id", "time_s", "sleep_state", "artifact"}
    missing = required - set(artifact_epochs.columns)
    if missing:
        raise ValueError(f"Artifact parquet is missing columns: {sorted(missing)}")

    empty_result = pd.DataFrame(
        columns=["epoch_id", "time_s", "hour", "sleep_state", "channel"]
    )
    flagged = artifact_epochs.loc[artifact_epochs["artifact"].astype(bool)]
    if flagged.empty:
        return empty_result

    features = (
        flagged["artifact_features"]
        if "artifact_features" in flagged.columns
        else pd.Series(pd.NA, index=flagged.index)
    )
    features_notna = features.notna()
    features_str = features.fillna("").astype(str)

    if "artifact_channels" in flagged.columns:
        explicit_str = flagged["artifact_channels"].fillna("").astype(str).str.strip()
        use_explicit = explicit_str.ne("")
    else:
        explicit_str = pd.Series("", index=flagged.index)
        use_explicit = pd.Series(False, index=flagged.index)

    explicit_channels = (
        explicit_str[use_explicit].str.split(r"\s*[;|]\s*").explode().str.strip()
    )
    explicit_channels = explicit_channels[explicit_channels.ne("")]

    fallback_index = flagged.index[~use_explicit & features_notna]
    fallback_entries = features_str.loc[fallback_index].str.split(r"\s*\|\s*").explode()
    partitioned = fallback_entries.str.partition(":")
    feature_channels = partitioned[0].str.strip().where(partitioned[1].ne(""))
    feature_channels = feature_channels.dropna()
    feature_channels = feature_channels[feature_channels.ne("")]

    emg_extreme = (
        flagged["emg_extreme"]
        if "emg_extreme" in flagged.columns
        else pd.Series(False, index=flagged.index)
    )
    emg_mask = (
        emg_extreme.fillna(False).astype(bool)
        & features_notna
        & features_str.str.contains("emg_outlier_with_eeg_outlier", regex=False)
    )
    emg_channels = pd.Series("EMG (combined)", index=flagged.index[emg_mask])

    channel_by_row = pd.concat([explicit_channels, feature_channels, emg_channels])
    if channel_by_row.empty:
        return empty_result

    row_metadata = pd.DataFrame(
        {
            "epoch_id": flagged["epoch_id"].astype(int),
            "time_s": flagged["time_s"].astype(float),
            "hour": np.floor(flagged["time_s"].astype(float) / 3600.0).astype(int),
            "sleep_state": flagged["sleep_state"].map(sleep_state_name),
        }
    )
    expanded = row_metadata.loc[channel_by_row.index].copy()
    expanded["channel"] = channel_by_row.to_numpy()
    return expanded.reset_index(drop=True).drop_duplicates()


def _period_lengths(times: pd.Series, epoch_seconds: float) -> list[float]:
    ordered = np.sort(times.astype(float).unique())
    if not len(ordered):
        return []
    breaks = np.diff(ordered) > epoch_seconds * 1.5
    run_ids = np.concatenate(([0], np.cumsum(breaks)))
    run_lengths = np.bincount(run_ids) * epoch_seconds
    return run_lengths.tolist()


def _channel_group_report(
    expanded: pd.DataFrame,
    *,
    channels: list[str],
    group_column: str,
    totals: pd.Series,
    duration: float,
    percent_column: str,
) -> pd.DataFrame:
    """Build one row per (channel, group) combination, including zero-artifact cells.

    Replaces a channel x group nested Python loop that re-filtered ``expanded``
    with a boolean mask on every iteration. A single grouped pass computes
    ``_period_lengths`` only for combinations that actually have flagged
    epochs; the full channel x group grid (needed for zero-artifact cells) is
    then filled in by reindexing.
    """
    full_index = pd.MultiIndex.from_product(
        [channels, totals.index], names=["channel", group_column]
    )
    grouped = expanded.groupby(["channel", group_column])
    artifact_epochs = grouped["epoch_id"].nunique().reindex(full_index, fill_value=0)
    periods_by_group = grouped["time_s"].apply(
        lambda values: _period_lengths(values, duration)
    ).reindex(full_index)

    rows = full_index.to_frame(index=False)
    rows["artifact_periods"] = [
        len(periods) if isinstance(periods, list) else 0
        for periods in periods_by_group.to_numpy()
    ]
    rows["artifact_epochs"] = artifact_epochs.to_numpy()
    rows["artifact_duration_s"] = rows["artifact_epochs"] * duration
    rows[percent_column] = (
        100.0 * rows["artifact_epochs"] / rows[group_column].map(totals)
    )
    rows["longest_artifact_s"] = [
        max(periods, default=0.0) if isinstance(periods, list) else 0.0
        for periods in periods_by_group.to_numpy()
    ]
    return rows


def restrict_to_continuous_epochs(
    artifact_epochs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    epoch_seconds: float | None = None,
) -> pd.DataFrame:
    """Drop artifact epochs outside sections deemed continuous by `score_recordings`.

    Concatenation gaps, dropouts, and too-short segments are stitched into the
    recording but aren't real signal, so artifact burden should only be
    reported over the continuous ("signal") sections, matching how
    `epoch_sleep_states` already restricts spectra/correlation/EMG checks.
    """
    duration = epoch_seconds or infer_epoch_seconds(artifact_epochs)
    continuous = continuous_epoch_ids(scores, epoch_seconds=duration)
    if continuous is None:
        return artifact_epochs
    return artifact_epochs.loc[artifact_epochs["epoch_id"].isin(continuous)]


def build_artifact_report(
    artifact_epochs: pd.DataFrame,
    *,
    epoch_seconds: float | None = None,
) -> ArtifactReport:
    """Build channel-level overall, hourly, and sleep-state artifact summaries."""
    duration = epoch_seconds or infer_epoch_seconds(artifact_epochs)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("epoch_seconds must be a positive finite number")
    expanded = expand_artifact_channels(artifact_epochs)
    if expanded.empty:
        empty = pd.DataFrame()
        return ArtifactReport(duration, empty, empty, empty)

    total_duration_s = artifact_epochs["epoch_id"].nunique() * duration
    overall_rows: list[dict[str, object]] = []
    for channel, rows in expanded.groupby("channel", sort=True):
        periods = _period_lengths(rows["time_s"], duration)
        artifact_duration_s = rows["epoch_id"].nunique() * duration
        overall_rows.append(
            {
                "channel": channel,
                "artifact_periods": len(periods),
                "artifact_epochs": rows["epoch_id"].nunique(),
                "artifact_duration_s": artifact_duration_s,
                "recording_percent": 100.0 * artifact_duration_s / total_duration_s,
                "longest_artifact_s": max(periods),
            }
        )
    overall = pd.DataFrame(overall_rows)

    channels = sorted(expanded["channel"].unique())

    epoch_hours = artifact_epochs.assign(
        hour=np.floor(artifact_epochs["time_s"].astype(float) / 3600.0).astype(int)
    )
    hour_totals = epoch_hours.groupby("hour")["epoch_id"].nunique()
    hourly = _channel_group_report(
        expanded,
        channels=channels,
        group_column="hour",
        totals=hour_totals,
        duration=duration,
        percent_column="hour_percent",
    )

    epoch_states = artifact_epochs.assign(
        sleep_state=artifact_epochs["sleep_state"].map(sleep_state_name)
    )
    state_totals = epoch_states.groupby("sleep_state")["epoch_id"].nunique()
    sleep_state = _channel_group_report(
        expanded,
        channels=channels,
        group_column="sleep_state",
        totals=state_totals,
        duration=duration,
        percent_column="state_percent",
    )
    return ArtifactReport(duration, overall, hourly, sleep_state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--epoch-seconds",
        type=float,
        default=None,
        help="Override epoch duration inferred from artifact time_s values.",
    )
    parser.add_argument(
        "--save-dir",
        nargs="?",
        const=".",
        default=None,
        help="Optionally save CSV reports in the session artifacts directory.",
    )
    return parser


def _print_report(report: ArtifactReport) -> None:
    print(f"Artifact epoch duration: {report.epoch_seconds:g} seconds")
    if report.overall.empty:
        print("No channel-separated artifacts were found.")
        return

    print("Overall by channel:")
    for row in report.overall.itertuples(index=False):
        print(
            f"  {row.channel}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.recording_percent:.2f}% of recording), "
            f"longest {row.longest_artifact_s:,.1f} s"
        )

    print("By recording hour:")
    for row in report.hourly.itertuples(index=False):
        print(
            f"  {row.channel} — hour {row.hour}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.hour_percent:.2f}%), longest {row.longest_artifact_s:,.1f} s"
        )

    print("By sleep state:")
    for row in report.sleep_state.itertuples(index=False):
        print(
            f"  {row.channel} — {row.sleep_state}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.state_percent:.2f}%), longest {row.longest_artifact_s:,.1f} s"
        )


def _save_report(report: ArtifactReport, save_dir: Path, stem: str) -> None:
    outputs = {
        "overall": report.overall,
        "hourly": report.hourly,
        "sleep_state": report.sleep_state,
    }
    for suffix, table in outputs.items():
        save_csv(table, save_dir / f"{stem}_artifact_report_{suffix}.csv")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epoch_seconds is not None and (
        not math.isfinite(args.epoch_seconds) or args.epoch_seconds <= 0
    ):
        parser.error("--epoch-seconds must be a positive finite number")

    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.utils.recording_selection import select_recordings
    from scripts.io.input_paths import artifact_path, scoring_path
    from scripts.io.output_paths import artifact_output_path

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(
        args.derivatives_root or get_derivatives_root()
    ).resolve(strict=False)
    pairs = select_recordings(
        rawdata_root,
        derivatives_root,
        subject=args.subject,
        date=args.date,
        session=args.session,
    )
    for edf_path, _ in pairs:
        artifacts = artifact_path(edf_path, rawdata_root, derivatives_root)
        if artifacts is None:
            raise FileNotFoundError(f"No artifact parquet found for {edf_path.name}")
        print(f"Artifacts: {artifacts}")
        artifact_epochs = pd.read_parquet(artifacts)
        scores = pd.read_parquet(scoring_path(edf_path, rawdata_root, derivatives_root))
        artifact_epochs = restrict_to_continuous_epochs(
            artifact_epochs, scores, epoch_seconds=args.epoch_seconds
        )
        report = build_artifact_report(
            artifact_epochs, epoch_seconds=args.epoch_seconds
        )
        _print_report(report)
        if args.save_dir is not None:
            save_dir = artifact_output_path(
                args.save_dir, edf_path, rawdata_root, derivatives_root
            )
            _save_report(report, save_dir, edf_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
