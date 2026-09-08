"""Shared quality-control review thresholds loaded from a pipeline YAML."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.config import (
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
    eeg_eeg_threshold: float
    eeg_emg_threshold: float
    max_correlation_review_percent: float


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


def load_qc_thresholds(path: str | Path = DEFAULT_QUALITY_CONTROL_CONFIG_PATH) -> QCThresholds:
    """Load and validate the shared quality-control review thresholds."""
    config = load_config(path)
    integrity = nested_get(config, ("quality_control", "recording_integrity"))
    scoring = nested_get(config, ("quality_control", "somnotate_scoring"))
    proportions = nested_get(config, ("quality_control", "sleep_state_proportions"))
    artifacts = nested_get(config, ("quality_control", "artifacts"))
    correlation = nested_get(config, ("quality_control", "channel_correlation"))
    if not all(
        isinstance(section, Mapping)
        for section in (integrity, scoring, proportions, artifacts, correlation)
    ):
        raise ValueError(
            "Quality-control config requires recording_integrity, somnotate_scoring, "
            "sleep_state_proportions, artifacts, and channel_correlation mappings"
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
        eeg_eeg_threshold=float(correlation["eeg_eeg_threshold"]),
        eeg_emg_threshold=float(correlation["eeg_emg_threshold"]),
        max_correlation_review_percent=float(
            correlation["max_correlation_review_percent"]
        ),
    )
