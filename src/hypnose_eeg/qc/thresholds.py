"""Shared quality-control review thresholds loaded from a pipeline YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Mapping

from hypnose_eeg.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    load_config,
    nested_get,
)


@dataclass(frozen=True)
class QCThresholds:
    duration_tolerance_s: float
    min_gap_s: float
    gap_scan_chunk_seconds: float
    max_gap_percent: float
    max_longest_gap_s: float
    confidence_threshold: float
    max_low_confidence_percent: float
    max_undefined_percent: float
    max_wake_percent: float
    max_nrem_percent: float
    max_rem_percent: float
    max_artifact_percent: float
    max_prescan_excluded_percent: float
    eeg_eeg_threshold: float
    eeg_emg_threshold: float
    max_correlation_review_percent: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if field.name in _PERCENT_FIELDS and not 0 <= value <= 100:
                raise ValueError(f"{field.name} must be between 0 and 100")
            if field.name in _FRACTION_FIELDS and not 0 <= value <= 1:
                raise ValueError(f"{field.name} must be between 0 and 1")
            if field.name in _POSITIVE_FIELDS and value <= 0:
                raise ValueError(f"{field.name} must be positive")
            if value < 0:
                raise ValueError(f"{field.name} must not be negative")


_PERCENT_FIELDS = frozenset(
    {
        "max_gap_percent",
        "max_low_confidence_percent",
        "max_undefined_percent",
        "max_wake_percent",
        "max_nrem_percent",
        "max_rem_percent",
        "max_artifact_percent",
        "max_prescan_excluded_percent",
        "max_correlation_review_percent",
    }
)
_FRACTION_FIELDS = frozenset(
    {"confidence_threshold", "eeg_eeg_threshold", "eeg_emg_threshold"}
)
_POSITIVE_FIELDS = frozenset({"min_gap_s", "gap_scan_chunk_seconds"})


def load_performance_check(
    path: str | Path = DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
) -> dict[str, int]:
    """Load and validate the shared pass/review/fail severity ranking."""
    config = load_config(path)
    order = nested_get(config, ("performance_check",))
    if not isinstance(order, Mapping):
        raise ValueError("quality_control config requires a performance_check mapping")
    required = {"pass", "review", "fail"}
    performance_check = {str(status).lower(): int(rank) for status, rank in order.items()}
    if set(performance_check) != required:
        raise ValueError(f"performance_check must define exactly {sorted(required)}")
    return performance_check


@cache
def default_performance_check() -> dict[str, int]:
    """Return the repository's severity ranking, loaded on first use."""
    return load_performance_check()


@cache
def default_qc_thresholds() -> QCThresholds:
    """Return the repository's quality-control thresholds, loaded on first use.

    Override single values with `with_overrides(default_qc_thresholds(), ...)`.
    """
    return load_qc_thresholds()


def load_qc_thresholds(path: str | Path = DEFAULT_QUALITY_CONTROL_CONFIG_PATH) -> QCThresholds:
    """Load and validate the shared quality-control review thresholds."""
    config = load_config(path)
    integrity = nested_get(config, ("quality_control", "recording_integrity"))
    scoring = nested_get(config, ("quality_control", "somnotate_scoring"))
    proportions = nested_get(config, ("quality_control", "sleep_state_proportions"))
    artifacts = nested_get(config, ("quality_control", "artifacts"))
    prescan = nested_get(config, ("quality_control", "artifact_prescan"))
    correlation = nested_get(config, ("quality_control", "channel_correlation"))
    if not all(
        isinstance(section, Mapping)
        for section in (integrity, scoring, proportions, artifacts, prescan, correlation)
    ):
        raise ValueError(
            "Quality-control config requires recording_integrity, somnotate_scoring, "
            "sleep_state_proportions, artifacts, artifact_prescan, and "
            "channel_correlation mappings"
        )
    return QCThresholds(
        duration_tolerance_s=float(integrity["duration_tolerance_s"]),
        min_gap_s=float(integrity["min_gap_s"]),
        gap_scan_chunk_seconds=float(integrity["gap_scan_chunk_seconds"]),
        max_gap_percent=float(integrity["max_gap_percent"]),
        max_longest_gap_s=float(integrity["max_longest_gap_s"]),
        confidence_threshold=float(scoring["confidence_threshold"]),
        max_low_confidence_percent=float(scoring["max_low_confidence_percent"]),
        max_undefined_percent=float(scoring["max_undefined_percent"]),
        max_wake_percent=float(proportions["max_wake_percent"]),
        max_nrem_percent=float(proportions["max_nrem_percent"]),
        max_rem_percent=float(proportions["max_rem_percent"]),
        max_artifact_percent=float(artifacts["max_artifact_percent"]),
        max_prescan_excluded_percent=float(prescan["max_excluded_percent"]),
        eeg_eeg_threshold=float(correlation["eeg_eeg_threshold"]),
        eeg_emg_threshold=float(correlation["eeg_emg_threshold"]),
        max_correlation_review_percent=float(
            correlation["max_correlation_review_percent"]
        ),
    )
