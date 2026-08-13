"""General helpers for working with artifact annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_epoch_seconds(artifact_epochs: pd.DataFrame, fallback: float = 4.0) -> float:
    """Infer epoch length from positive consecutive time differences."""
    if "time_s" not in artifact_epochs.columns:
        raise ValueError("Artifact parquet is missing column 'time_s'")
    times = np.sort(artifact_epochs["time_s"].dropna().astype(float).unique())
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
