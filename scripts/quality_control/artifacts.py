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

try:
    from scripts.utils.epochs import infer_epoch_seconds
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.utils.epochs import infer_epoch_seconds


SLEEP_STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM", 3: "Undefined"}


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


def _channels_from_row(row: pd.Series) -> list[str]:
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
    """Expand flagged epoch rows to one row per affected channel."""
    required = {"epoch_id", "time_s", "sleep_state", "artifact"}
    missing = required - set(artifact_epochs.columns)
    if missing:
        raise ValueError(f"Artifact parquet is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    flagged = artifact_epochs.loc[artifact_epochs["artifact"].astype(bool)]
    for _, row in flagged.iterrows():
        for channel in _channels_from_row(row):
            rows.append(
                {
                    "epoch_id": int(row["epoch_id"]),
                    "time_s": float(row["time_s"]),
                    "hour": int(np.floor(float(row["time_s"]) / 3600.0)),
                    "sleep_state": sleep_state_name(row["sleep_state"]),
                    "channel": channel,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["epoch_id", "time_s", "hour", "sleep_state", "channel"],
    ).drop_duplicates()


def _period_lengths(times: pd.Series, epoch_seconds: float) -> list[float]:
    ordered = np.sort(times.astype(float).unique())
    if not len(ordered):
        return []
    lengths: list[float] = []
    run_epochs = 1
    for difference in np.diff(ordered):
        if difference <= epoch_seconds * 1.5:
            run_epochs += 1
        else:
            lengths.append(run_epochs * epoch_seconds)
            run_epochs = 1
    lengths.append(run_epochs * epoch_seconds)
    return lengths


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

    epoch_hours = artifact_epochs.assign(
        hour=np.floor(artifact_epochs["time_s"].astype(float) / 3600.0).astype(int)
    )
    hour_totals = epoch_hours.groupby("hour")["epoch_id"].nunique()
    hourly_rows: list[dict[str, object]] = []
    channels = sorted(expanded["channel"].unique())
    for channel in channels:
        channel_rows = expanded.loc[expanded["channel"] == channel]
        for hour, total_epochs in hour_totals.items():
            rows = channel_rows.loc[channel_rows["hour"] == hour]
            artifact_count = rows["epoch_id"].nunique()
            periods = _period_lengths(rows["time_s"], duration)
            hourly_rows.append(
                {
                    "channel": channel,
                    "hour": int(hour),
                    "artifact_periods": len(periods),
                    "artifact_epochs": artifact_count,
                    "artifact_duration_s": artifact_count * duration,
                    "hour_percent": 100.0 * artifact_count / total_epochs,
                    "longest_artifact_s": max(periods, default=0.0),
                }
            )
    hourly = pd.DataFrame(hourly_rows)

    epoch_states = artifact_epochs.assign(
        sleep_state=artifact_epochs["sleep_state"].map(sleep_state_name)
    )
    state_totals = epoch_states.groupby("sleep_state")["epoch_id"].nunique()
    state_rows: list[dict[str, object]] = []
    for channel in channels:
        channel_rows = expanded.loc[expanded["channel"] == channel]
        for state, total_epochs in state_totals.items():
            rows = channel_rows.loc[channel_rows["sleep_state"] == state]
            artifact_count = rows["epoch_id"].nunique()
            periods = _period_lengths(rows["time_s"], duration)
            state_rows.append(
                {
                    "channel": channel,
                    "sleep_state": state,
                    "artifact_periods": len(periods),
                    "artifact_epochs": artifact_count,
                    "artifact_duration_s": artifact_count * duration,
                    "state_percent": 100.0 * artifact_count / total_epochs,
                    "longest_artifact_s": max(periods, default=0.0),
                }
            )
    sleep_state = pd.DataFrame(state_rows)
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
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overall": report.overall,
        "hourly": report.hourly,
        "sleep_state": report.sleep_state,
    }
    for suffix, table in outputs.items():
        output = save_dir / f"{stem}_artifact_report_{suffix}.csv"
        table.to_csv(output, index=False)
        print(f"Saved: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epoch_seconds is not None and (
        not math.isfinite(args.epoch_seconds) or args.epoch_seconds <= 0
    ):
        parser.error("--epoch-seconds must be a positive finite number")

    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.io.input_paths import artifact_path
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
