"""Shared utilities, grouped into focused domain modules.

Names formerly exposed by ``scripts.utils.py`` remain available through lazy
imports. New code should generally import from the relevant domain module.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "SLEEP_STATE_CODES",
    "ArtifactReport",
    "artifact_path",
    "artifact_epoch_ids",
    "build_artifact_report",
    "coalesce",
    "compute_state_emg_rms",
    "compute_state_channel_correlations",
    "compute_state_spectra",
    "epoch_pearson_matrices",
    "epoch_sleep_states",
    "export_raw_edf",
    "import_mne",
    "load_config",
    "nested_get",
    "scoring_path",
    "session_derivatives_dir",
]

_EXPORT_MODULES = {
    "coalesce": ".config",
    "load_config": ".config",
    "nested_get": ".config",
    "export_raw_edf": ".mne_io",
    "import_mne": ".mne_io",
    "SLEEP_STATE_CODES": ".power_spectra",
    "artifact_epoch_ids": ".power_spectra",
    "compute_state_emg_rms": ".power_spectra",
    "compute_state_spectra": ".power_spectra",
    "epoch_sleep_states": ".power_spectra",
    "compute_state_channel_correlations": ".correlation",
    "epoch_pearson_matrices": ".correlation",
    "artifact_path": ".recording_paths",
    "scoring_path": ".recording_paths",
    "session_derivatives_dir": ".recording_paths",
    "ArtifactReport": ".artifact_reporting",
    "build_artifact_report": ".artifact_reporting",
}


def __getattr__(name: str) -> Any:
    """Load compatibility exports only when callers request them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
