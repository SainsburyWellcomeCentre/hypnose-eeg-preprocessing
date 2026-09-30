"""Construct categorized output paths in the shared session layout, and write CSVs and PDFs to them."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hypnose_helpers.io.layout import parse_session_dirname, parse_subject_dirname

from hypnose_eeg.io.input_paths import resolve_session_dir
from hypnose_eeg.io.output_layout import output_dir_name
from hypnose_eeg.utils.recording_selection import source_stem_from_fif

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure


OUTPUT_GROUPS = {
    "sleep_scoring",
    "sleep_scoring_qc",
    "artifacts",
    "quality_control",
}


def _session_dir_from_derivative_path(path: Path) -> Path | None:
    """Return the nearest shared-layout derivatives session containing a path."""
    for candidate in path.parents:
        if (
            parse_session_dirname(candidate.name) is not None
            and parse_subject_dirname(candidate.parent.name) is not None
        ):
            return candidate
    return None


def artifact_output_paths(
    fif_path: str | Path,
    *,
    output_suffix: str,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return CSV and parquet destinations for artifact detection output.

    Named after the analyzed recording's source stem (its data location within the
    shared session layout), not after whichever sleep-scoring file happened to
    produce the states used to detect artifacts, and with MNE/downsampling naming
    detail (`_raw`, `_resampled-<rate>hz`) stripped back out.
    """
    recording = Path(fif_path)
    if output_dir is not None:
        destination = Path(output_dir)
    else:
        session_dir = _session_dir_from_derivative_path(recording)
        destination = (session_dir or recording.parent) / output_dir_name("artifacts")
    stem = f"{source_stem_from_fif(recording)}_{output_suffix}"
    return (
        destination / f"{stem}.csv",
        destination / f"{stem}.parquet",
    )


def recording_output_name(output: str | Path, recording_path: str | Path) -> Path:
    """Prefix a relative output name with the stem of the recording it describes.

    Keeps every per-recording derivative discoverable by the same
    `sub-XXX_ses-YYY_recording-ZZZ_<output>` prefix already used by the artifact,
    sleep-scoring, and figure outputs, so a session directory holding more than
    one recording never collides on a shared name like `qc_summary.csv`.
    Absolute paths, and names already carrying the prefix, are returned unchanged.
    """
    requested = Path(output)
    if requested.is_absolute():
        return requested
    stem = source_stem_from_fif(Path(recording_path))
    if requested.name.startswith(f"{stem}_"):
        return requested
    return requested.with_name(f"{stem}_{requested.name}")


def session_output_dir(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
    output_group: str,
) -> Path:
    """Resolve a named per-session output directory through hypnose-helpers."""
    if output_group not in OUTPUT_GROUPS:
        raise ValueError(
            f"Unknown output group {output_group!r}; expected one of {sorted(OUTPUT_GROUPS)}"
        )
    session_dir = resolve_session_dir(
        recording_path, rawdata_root, derivatives_root, name="derivatives"
    )
    return session_dir / output_dir_name(output_group)


def session_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
    output_group: str,
) -> Path:
    """Resolve a relative output below a named shared session directory."""
    requested = Path(output)
    if requested.is_absolute():
        return requested
    if ".." in requested.parts:
        raise ValueError("Relative output paths cannot traverse outside the session")
    return session_output_dir(
        recording_path, rawdata_root, derivatives_root, output_group
    ) / requested


def quality_control_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve an output beneath the session's quality-control directory."""
    return session_output_path(
        output, recording_path, rawdata_root, derivatives_root, "quality_control"
    )


def sleep_scoring_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve an output beneath the session's Somnotate-scoring directory."""
    return session_output_path(
        output,
        recording_path,
        rawdata_root,
        derivatives_root,
        "sleep_scoring",
    )


def sleep_scoring_qc_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve an output beneath the session's sleep-scoring QC directory.

    Distinct from `sleep_scoring_output_path`: that one holds the Somnotate
    predictions themselves (written by score_recordings.py), while this one
    holds QC reports derived from them (written by hypnose_eeg/qc/sleep_scoring.py
    and hypnose_eeg/qc/summary_qc.py) -- the same split already used for
    `artifacts` (detector output) versus `quality_control` (QC reports).
    """
    return session_output_path(
        output,
        recording_path,
        rawdata_root,
        derivatives_root,
        "sleep_scoring_qc",
    )


def artifact_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve an output beneath the session's artifact directory."""
    return session_output_path(
        output, recording_path, rawdata_root, derivatives_root, "artifacts"
    )


def save_csv(df: "pd.DataFrame", path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories, and report the save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_csv_rows(
    rows: list[dict[str, Any]], fieldnames: list[str], path: str | Path
) -> Path:
    """Write dict rows to CSV, creating parent directories, and report the save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")
    return path


def save_pdf(fig: "Figure", path: str | Path) -> Path:
    """Save a figure as a provenance-tagged PDF at exactly `path`, and report the save.

    `hypnose_helpers`' `save_figure` always appends `_sub-..._date-...` tags to
    the filename; recording outputs already name their subject and session in
    the stem, so the tagged file is renamed to `path`.
    """
    from hypnose_helpers.provenance import provenance
    from hypnose_helpers.viz.save import save_figure

    path = Path(path)
    tagged = save_figure(
        fig,
        path.stem,
        fig_dir=path.parent,
        # Record the caller, not this wrapper, as the figure's origin.
        provenance=provenance(skip_modules=(__name__,)),
    )
    Path(tagged).replace(path)
    print(f"Saved: {path}")
    return path
