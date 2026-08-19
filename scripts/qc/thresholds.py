"""Shared quality-control review thresholds loaded from a pipeline YAML."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    from scripts.io.repository_paths import get_repo_root
    from scripts.utils.config import load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.repository_paths import get_repo_root
    from scripts.utils.config import load_config, nested_get


DEFAULT_CONFIG_PATH = get_repo_root() / "configs" / "pipelines" / "quality_control.yaml"


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
    max_artifact_percent: float
    eeg_eeg_threshold: float
    eeg_emg_threshold: float
    max_correlation_review_percent: float


def load_qc_thresholds(path: str | Path = DEFAULT_CONFIG_PATH) -> QCThresholds:
    """Load and validate the shared quality-control review thresholds."""
    config = load_config(path)
    integrity = nested_get(config, ("quality_control", "recording_integrity"))
    scoring = nested_get(config, ("quality_control", "somnotate_scoring"))
    artifacts = nested_get(config, ("quality_control", "artifacts"))
    correlation = nested_get(config, ("quality_control", "channel_correlation"))
    if not all(
        isinstance(section, Mapping)
        for section in (integrity, scoring, artifacts, correlation)
    ):
        raise ValueError(
            "Quality-control config requires recording_integrity, somnotate_scoring, "
            "artifacts, and channel_correlation mappings"
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
        max_artifact_percent=float(artifacts["max_artifact_percent"]),
        eeg_eeg_threshold=float(correlation["eeg_eeg_threshold"]),
        eeg_emg_threshold=float(correlation["eeg_emg_threshold"]),
        max_correlation_review_percent=float(
            correlation["max_correlation_review_percent"]
        ),
    )
