"""General helpers for constructing and aligning fixed signal epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd


SLEEP_STATE_CODES = (0, 1, 2)


def align_epoch_states(
    scores: pd.DataFrame,
    *,
    n_epochs: int,
    epoch_seconds: float,
    time_column: str,
    state_column: str,
    unscored_numeric_state: int,
    unscored_text_state: str,
) -> pd.DataFrame:
    """Return one majority sleep-state row for every recording epoch."""
    missing = {time_column, state_column}.difference(scores.columns)
    if missing:
        raise ValueError(f"Sleep-scoring parquet is missing columns: {sorted(missing)}")

    selected = scores[[time_column, state_column]].dropna().copy()
    selected = selected.loc[
        (selected[time_column] >= 0)
        & (selected[time_column] < n_epochs * epoch_seconds)
    ].copy()
    selected["epoch_id"] = np.floor(
        selected[time_column] / epoch_seconds
    ).astype("int64")

    def majority(values: pd.Series) -> object:
        modes = values.mode()
        return modes.iloc[0] if not modes.empty else np.nan

    epoch_states = selected.groupby("epoch_id", as_index=False).agg(
        sleep_state=(state_column, majority),
        sleep_score_count=(state_column, "size"),
    )
    all_epochs = pd.DataFrame({"epoch_id": np.arange(n_epochs, dtype=np.int64)})
    epoch_states = all_epochs.merge(epoch_states, on="epoch_id", how="left")

    if pd.api.types.is_numeric_dtype(selected[state_column]):
        epoch_states["sleep_state"] = pd.to_numeric(
            epoch_states["sleep_state"]
        ).fillna(unscored_numeric_state)
    else:
        epoch_states["sleep_state"] = (
            epoch_states["sleep_state"]
            .astype("string")
            .fillna(unscored_text_state)
        )
    epoch_states["sleep_score_count"] = (
        epoch_states["sleep_score_count"].fillna(0).astype(int)
    )
    return epoch_states


def epoch_sleep_states(
    scores: pd.DataFrame,
    *,
    epoch_seconds: float,
    n_epochs: int,
) -> pd.Series:
    """Return one majority output-state code per complete signal epoch."""
    if "time_s" not in scores.columns:
        raise ValueError("Scoring parquet is missing column 'time_s'")
    label_column = "label_output" if "label_output" in scores.columns else "label"
    if label_column not in scores.columns:
        raise ValueError("Scoring parquet is missing 'label_output' or 'label'")

    selected = scores[["time_s", label_column]].copy()
    selected["epoch_id"] = np.floor(
        selected["time_s"].astype(float) / epoch_seconds
    ).astype("int64")
    if "kind" in scores.columns:
        selected["kind"] = scores["kind"].astype(str)
        non_signal_epochs = set(
            selected.loc[selected["kind"] != "signal", "epoch_id"].tolist()
        )
        selected = selected.loc[
            (selected["kind"] == "signal")
            & ~selected["epoch_id"].isin(non_signal_epochs)
        ]
    selected = selected.loc[
        (selected["epoch_id"] >= 0) & (selected["epoch_id"] < n_epochs)
    ]

    def majority(values: pd.Series) -> int:
        modes = pd.to_numeric(values, errors="coerce").dropna().astype(int).mode()
        return int(modes.iloc[0]) if len(modes) else 3

    return selected.groupby("epoch_id")[label_column].agg(majority).astype(int)


def infer_epoch_seconds(epoch_table: pd.DataFrame, fallback: float = 4.0) -> float:
    """Infer epoch length from positive consecutive time differences."""
    if "time_s" not in epoch_table.columns:
        raise ValueError("Epoch table is missing column 'time_s'")
    times = np.sort(epoch_table["time_s"].dropna().astype(float).unique())
    steps = np.diff(times)
    positive_steps = steps[steps > 0]
    return float(np.median(positive_steps)) if len(positive_steps) else fallback


def artifact_epoch_ids(
    artifact_table: pd.DataFrame, *, epoch_seconds: float
) -> set[int]:
    """Return analysis epoch IDs overlapped by flagged artifact intervals."""
    required = {"time_s", "artifact"}
    missing = required - set(artifact_table.columns)
    if missing:
        raise ValueError(f"Artifact parquet is missing columns: {sorted(missing)}")
    artifact_seconds = infer_epoch_seconds(artifact_table)
    flagged_times = artifact_table.loc[
        artifact_table["artifact"].astype(bool), "time_s"
    ].astype(float)
    excluded: set[int] = set()
    for start_s in flagged_times:
        first = int(np.floor(start_s / epoch_seconds))
        last = int(np.ceil((start_s + artifact_seconds) / epoch_seconds))
        excluded.update(range(first, last))
    return excluded


def complete_epoch_count(n_samples: int, sfreq: float, epoch_seconds: float) -> int:
    """Return the number of complete fixed-duration epochs in a signal."""
    epoch_samples = int(round(sfreq * epoch_seconds))
    if epoch_samples < 1:
        raise ValueError("epoch_seconds is too short for the sampling frequency")
    return n_samples // epoch_samples


def epoch_batch(
    samples: np.ndarray,
    *,
    epoch_count: int,
    epoch_samples: int,
) -> np.ndarray:
    """Reshape channels × samples into epochs × channels × samples."""
    expected_samples = epoch_count * epoch_samples
    if samples.ndim != 2 or samples.shape[1] != expected_samples:
        raise ValueError(
            "samples must have shape (channels, epoch_count * epoch_samples)"
        )
    reshaped = samples.reshape(samples.shape[0], epoch_count, epoch_samples)
    return np.moveaxis(reshaped, 1, 0)


def choose_chunk_epochs(
    sfreq: float,
    epoch_seconds: float,
    n_channels: int,
    target_memory_mb: float,
    working_array_factor: int,
    max_chunk_epochs: int,
) -> int:
    """Choose an epoch chunk size from signal dimensions and a memory budget."""
    values = (
        sfreq,
        epoch_seconds,
        n_channels,
        target_memory_mb,
        working_array_factor,
        max_chunk_epochs,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Epoch, channel, and memory parameters must be positive")
    epoch_samples = int(round(sfreq * epoch_seconds))
    estimated_bytes = (
        epoch_samples
        * n_channels
        * np.dtype(np.float64).itemsize
        * working_array_factor
    )
    target_bytes = int(target_memory_mb * 1024**2)
    return max(1, min(max_chunk_epochs, target_bytes // estimated_bytes))
