"""Resolve recording-related input files from the shared session layout."""

from __future__ import annotations

from pathlib import Path

from hypnose_helpers.io.layout import (
    SessionLayout,
    parse_session_dirname,
    parse_subject_dirname,
)


def resolve_session_dir(
    recording_path: str | Path,
    rawdata_root: str | Path,
    root: str | Path,
    *,
    name: str,
) -> Path:
    """Resolve a recording's session directory under a shared-layout root.

    Parses the subject/session out of the recording's rawdata path, then looks it up
    through `SessionLayout` -- robust to ses-vs-date selection, `_id-` subject suffixes,
    and duplicate directories, unlike copying the rawdata directory names verbatim.
    """
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
    layout = SessionLayout(Path(root), name=name)
    selector = {"ses": session_number} if session_number is not None else {"date": date}
    return layout.find_session(subject, **selector).path


def session_derivatives_dir(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Map a recording under rawdata to its derivatives session."""
    return resolve_session_dir(
        recording_path, rawdata_root, derivatives_root, name="derivatives"
    )


def scoring_path(
    recording_path: str | Path,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
) -> Path:
    """Resolve the unique Somnotate prediction parquet for a recording."""
    recording = Path(recording_path)
    session_dir = session_derivatives_dir(recording, rawdata_root, derivatives_root)
    filename = f"{recording.stem}_somnotate_predictions.parquet"
    expected = session_dir / "sleep_scoring" / filename
    if expected.is_file():
        return expected
    legacy = session_dir / "saved_results" / filename
    if legacy.is_file():
        return legacy
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
    session_dir = session_derivatives_dir(recording, rawdata_root, derivatives_root)
    exact_name = f"{recording.stem}_artifact_epochs.parquet"
    expected_dir = session_dir / "artifacts"
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
