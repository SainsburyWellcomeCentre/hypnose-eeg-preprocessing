"""MNE import and recording-export operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from scripts.utils.config import (
    DEFAULT_SLEEP_SCORING_CONFIG_PATH,
    load_config,
    nested_get,
)


def import_mne() -> Any:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install MNE first: pip install mne") from exc

    return mne


def export_raw_edf(raw: Any, output_path: str | Path, overwrite: bool) -> None:
    output = Path(output_path)
    try:
        raw.export(output, fmt="edf", overwrite=overwrite, physical_range="auto")
    except TypeError:
        raw.export(output, fmt="edf", overwrite=overwrite)
    except RuntimeError as exc:
        raise RuntimeError(
            "EDF export requires the edfio package: pip install edfio"
        ) from exc


def load_channel_labels(
    config_path: str | Path = DEFAULT_SLEEP_SCORING_CONFIG_PATH,
) -> list[str]:
    """Load the configured EEG/EMG channel labels from the sleep-scoring YAML."""
    config = load_config(config_path)
    labels = nested_get(config, ("sleep_scoring", "channel_labels"))
    if not labels:
        raise ValueError(f"sleep_scoring.channel_labels is required in {config_path}")
    return [str(label) for label in labels]


def set_configured_channel_types(raw: Any, channel_labels: Sequence[str]) -> None:
    """Set each configured channel's MNE type explicitly, from its label text.

    `mne.io.read_raw_edf(..., infer_types=True)` only recognizes a channel's
    type from a space-separated "EEG "/"EMG " label prefix (e.g. "EMG EMG").
    `concatenate_recordings.py` exports its merged recording without that
    prefix (`raw.export(..., fmt="edf")` defaults to `add_ch_type=False`), so
    re-reading a concatenated EDF with `infer_types=True` silently mistypes
    EMG channels as EEG. Setting types from the configured labels sidesteps
    `infer_types` entirely, so it works regardless of how the file was made.
    """
    mapping: dict[str, str] = {}
    for label in channel_labels:
        bare = label
        for prefix in ("EEG ", "EMG "):
            if label.startswith(prefix):
                bare = label[len(prefix):]
                break
        modality = "emg" if label.upper().startswith("EMG") else "eeg"
        for candidate in (label, bare, f"EEG {bare}", f"EMG {bare}"):
            if candidate in raw.ch_names:
                mapping[candidate] = modality
                break
    if mapping:
        raw.set_channel_types(mapping, verbose="ERROR")
