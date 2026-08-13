"""General sleep-state constants and epoch-alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


SLEEP_STATE_CODES = (0, 1, 2)
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
