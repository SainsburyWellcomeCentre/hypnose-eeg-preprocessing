"""Summarize epoch-level artifact decisions by channel, hour, and sleep state."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM", 3: "Undefined"}


@dataclass(frozen=True)
class ArtifactReport:
    epoch_seconds: float
    overall: pd.DataFrame
    hourly: pd.DataFrame
    sleep_state: pd.DataFrame


def infer_epoch_seconds(artifact_epochs: pd.DataFrame, fallback: float = 4.0) -> float:
    """Infer epoch length from positive consecutive time differences."""
    if "time_s" not in artifact_epochs.columns:
        raise ValueError("Artifact parquet is missing column 'time_s'")
    times = np.sort(artifact_epochs["time_s"].dropna().astype(float).unique())
    steps = np.diff(times)
    positive_steps = steps[steps > 0]
    return float(np.median(positive_steps)) if len(positive_steps) else fallback


def _state_name(value: object) -> str:
    if pd.isna(value):
        return "Undefined"
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "Undefined"
    return STATE_NAMES.get(numeric, str(numeric))


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
    else:
        if pd.notna(features):
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
                    "sleep_state": _state_name(row["sleep_state"]),
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
        sleep_state=artifact_epochs["sleep_state"].map(_state_name)
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
