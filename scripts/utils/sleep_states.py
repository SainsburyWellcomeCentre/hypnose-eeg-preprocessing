"""Shared sleep-state display names and colors loaded from the sleep-scoring YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.utils.config import (
    DEFAULT_SLEEP_SCORING_CONFIG_PATH,
    load_config,
    nested_get,
)


@dataclass(frozen=True)
class SleepStates:
    sleep_state_names: dict[int, str]
    state_colors: dict[int, str]
    probability_columns: dict[int, str]


def load_sleep_states(path: str | Path = DEFAULT_SLEEP_SCORING_CONFIG_PATH) -> SleepStates:
    """Load and validate the shared sleep-state names and colors.

    Somnotate's prediction-probability columns follow every configured state name,
    lowercased and abbreviated to 5 characters (e.g. "Wake" -> "prob_wake",
    "Undefined" -> "prob_undef"), so they are derived rather than configured.
    """
    config = load_config(path)
    names = nested_get(config, ("sleep_scoring", "sleep_state_names"))
    colors = nested_get(config, ("sleep_scoring", "state_colors"))
    if not isinstance(names, Mapping) or not isinstance(colors, Mapping):
        raise ValueError(
            "sleep_scoring config requires sleep_state_names and state_colors mappings"
        )
    sleep_state_names = {int(code): str(label) for code, label in names.items()}
    state_colors = {int(code): str(color) for code, color in colors.items()}
    if not set(state_colors) <= set(sleep_state_names):
        raise ValueError("state_colors codes must all be defined in sleep_state_names")
    probability_columns = {
        code: f"prob_{label.lower()[:5]}" for code, label in sleep_state_names.items()
    }
    return SleepStates(
        sleep_state_names=sleep_state_names,
        state_colors=state_colors,
        probability_columns=probability_columns,
    )
