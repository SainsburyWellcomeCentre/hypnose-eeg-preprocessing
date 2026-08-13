"""MNE import and recording-export operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
