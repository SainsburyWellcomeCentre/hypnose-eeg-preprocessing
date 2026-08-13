"""Resolve session directories and recording-related derivative paths."""

from __future__ import annotations

from pathlib import Path

from hypnose_helpers.io.layout import (
    SessionLayout,
    parse_session_dirname,
    parse_subject_dirname,
)


def artifact_output_paths(
    sleep_parquet_path: str | Path,
    *,
    remove_from_stem: str,
    output_suffix: str,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return CSV and parquet destinations for artifact detection output."""
    scoring = Path(sleep_parquet_path)
    destination = Path(output_dir) if output_dir is not None else scoring.parent
    stem = scoring.stem.replace(remove_from_stem, "").rstrip("_-")
    return (
        destination / f"{stem}_{output_suffix}.csv",
        destination / f"{stem}_{output_suffix}.parquet",
    )


def session_derivatives_dir(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Map a recording under rawdata to its mirrored derivatives session."""
    recording = Path(recording_path)
    rawdata = Path(rawdata_root)
    derivatives = Path(derivatives_root)
    try:
        relative = recording.relative_to(rawdata)
    except ValueError as exc:
        raise ValueError(f"Recording is not beneath the rawdata root: {recording}") from exc
    if len(relative.parts) < 3:
        raise ValueError(
            f"Recording is not inside a subject/session hierarchy: {recording}"
        )
    return derivatives / relative.parts[0] / relative.parts[1]


def quality_control_dir(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve the shared per-session QC directory through hypnose-helpers."""
    recording = Path(recording_path)
    rawdata = Path(rawdata_root)
    try:
        relative = recording.relative_to(rawdata)
    except ValueError as exc:
        raise ValueError(f"Recording is not beneath the rawdata root: {recording}") from exc
    if len(relative.parts) < 3:
        raise ValueError(
            f"Recording is not inside a subject/session hierarchy: {recording}"
        )
    subject = parse_subject_dirname(relative.parts[0])
    session = parse_session_dirname(relative.parts[1])
    if subject is None or session is None:
        raise ValueError(
            f"Recording does not use the shared subject/session layout: {recording}"
        )
    session_number, date = session
    layout = SessionLayout(Path(derivatives_root), name="derivatives")
    selector = {"ses": session_number} if session_number is not None else {"date": date}
    session_ref = layout.find_session(subject, **selector)
    return session_ref.path / "quality_control"


def quality_control_output_path(
    output: str | Path,
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve a relative QC output below the shared session QC directory."""
    requested = Path(output)
    if requested.is_absolute():
        return requested
    if ".." in requested.parts:
        raise ValueError("Relative QC output paths cannot traverse outside the session")
    return quality_control_dir(
        recording_path, rawdata_root, derivatives_root
    ) / requested


def scoring_path(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve the unique Somnotate prediction parquet for a recording."""
    recording = Path(recording_path)
    session_dir = session_derivatives_dir(
        recording, rawdata_root, derivatives_root
    )
    filename = f"{recording.stem}_somnotate_predictions.parquet"
    expected = session_dir / "saved_results" / filename
    if expected.is_file():
        return expected
    matches = sorted(session_dir.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No scoring predictions found for {recording.name}")
    raise ValueError(f"Multiple scoring predictions found for {recording.name}: {matches}")


def artifact_path(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path | None:
    """Resolve an optional artifact-epoch parquet for a recording."""
    recording = Path(recording_path)
    session_dir = session_derivatives_dir(
        recording, rawdata_root, derivatives_root
    )
    exact_name = f"{recording.stem}_artifact_epochs.parquet"
    expected_dir = session_dir / "saved_results"
    for filename in (exact_name, "artifact_epochs.parquet"):
        expected = expected_dir / filename
        if expected.is_file():
            return expected

    matches = sorted(session_dir.rglob(exact_name))
    if not matches:
        matches = sorted(session_dir.rglob("artifact_epochs.parquet"))
    if not matches:
        matches = sorted(session_dir.rglob("*artifact_epochs.parquet"))
    if len(matches) > 1:
        raise ValueError(
            f"Multiple artifact files found for {recording.name}: {matches}"
        )
    return matches[0] if matches else None
