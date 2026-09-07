"""Resolve EDF/FIF recording pairs from subject/date/session selectors."""

from __future__ import annotations

import re
from pathlib import Path

from hypnose_helpers.io.layout import SessionLayout, normalize_subjid


def is_concatenated_recording(path: Path) -> bool:
    """Whether `path` looks like a `concatenate_recordings.py`-produced file."""
    name = path.name.lower()
    return name.endswith("recording-concat.edf") or "_recording-concat" in name


def prefer_concatenated_recording(edf_paths: list[Path]) -> list[Path]:
    """Within one folder's EDFs, keep only the concatenated recording if several exist.

    `concatenate_recordings.py` combines a session's raw per-part EDFs into one
    `_recording-concat.edf` file alongside them. Once that has been run, processing
    every file in the folder duplicates work on the same underlying recording --
    and for the individual parts, without any of the gap-aware handling
    (concatenate_recordings.py's own gap-padding, or somnotate's
    `prepare_recording`) that only applies within a single file.

    If several EDFs are present but none is a concatenated recording, an empty
    list is returned rather than falling back to the ambiguous raw parts --
    callers should treat that as "not ready to process yet" and warn rather than
    silently score/downsample the parts independently.
    """
    if len(edf_paths) <= 1:
        return edf_paths
    return [path for path in edf_paths if is_concatenated_recording(path)]


def group_and_prefer_concatenated_recordings(edf_paths) -> list[Path]:
    """Apply `prefer_concatenated_recording` within each EDF's parent folder.

    For a flat, single-session list of EDFs, call `prefer_concatenated_recording`
    directly instead -- this is for callers (like `downsample_recordings.py`) whose
    glob spans many session folders in one flat list.
    """
    files_by_folder: dict[Path, list[Path]] = {}
    for edf_path in edf_paths:
        files_by_folder.setdefault(edf_path.parent, []).append(edf_path)

    selected: list[Path] = []
    for folder_files in files_by_folder.values():
        selected.extend(prefer_concatenated_recording(folder_files))
    return sorted(selected)


def source_stem_from_fif(path: Path) -> str:
    """Strip MNE/downsampling naming detail to recover the source recording's stem."""
    stem = re.sub(r"_raw$", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"_resampled-[0-9p.]+hz$", "", stem, flags=re.IGNORECASE)


def find_session_dirs(
    rawdata_root: Path,
    *,
    subject: str | int,
    date: str | int | None = None,
    session: str | int | None = None,
) -> list[Path]:
    """Resolve rawdata session directories using subject and date/session selectors."""
    layout = SessionLayout(rawdata_root, name="rawdata")
    sessions = layout.find_sessions(subject, ses=session, date=date)
    if not sessions:
        selector = f"date {date}" if date is not None else f"session {session}"
        raise FileNotFoundError(f"No {selector} found for {normalize_subjid(subject)}")
    return [session_ref.path for session_ref in sessions]


def select_recordings(
    rawdata_root: Path,
    derivatives_root: Path,
    *,
    subject: str | int,
    date: str | int | None = None,
    session: str | int | None = None,
) -> list[tuple[Path, Path]]:
    """Resolve matching EDF/FIF pairs using subject and date/session selectors."""
    session_dirs = find_session_dirs(rawdata_root, subject=subject, date=date, session=session)

    pairs: list[tuple[Path, Path]] = []
    for session_dir in session_dirs:
        relative_session = session_dir.relative_to(rawdata_root).as_posix()
        pairs.extend(
            pair_recordings(
                rawdata_root,
                derivatives_root,
                edf_pattern=f"{relative_session}/**/*.edf",
                fif_pattern=f"{relative_session}/**/*_raw.fif",
            )
        )

    if not pairs:
        listing = ", ".join(str(path) for path in session_dirs)
        raise FileNotFoundError(f"No matching EDF/FIF pairs beneath: {listing}")
    return pairs


def pair_recordings(
    rawdata_root: Path,
    derivatives_root: Path,
    edf_pattern: str = "**/*.edf",
    fif_pattern: str = "**/*_raw.fif",
) -> list[tuple[Path, Path]]:
    """Pair EDF/FIF files by source stem, preferring matching relative folders."""
    edfs = sorted(path for path in rawdata_root.glob(edf_pattern) if path.is_file())
    fif_files = sorted(
        path for path in derivatives_root.glob(fif_pattern) if path.is_file()
    )
    fif_by_stem: dict[str, list[Path]] = {}
    for fif_path in fif_files:
        fif_by_stem.setdefault(source_stem_from_fif(fif_path), []).append(fif_path)

    pairs: list[tuple[Path, Path]] = []
    for edf_path in edfs:
        candidates = fif_by_stem.get(edf_path.stem, [])
        if not candidates:
            continue
        relative_parent = edf_path.relative_to(rawdata_root).parent
        same_folder = [
            path
            for path in candidates
            if path.relative_to(derivatives_root).parent == relative_parent
        ]
        selected = same_folder or candidates
        if len(selected) > 1:
            listing = ", ".join(str(path) for path in selected)
            raise ValueError(f"Multiple FIF matches for {edf_path}: {listing}")
        pairs.append((edf_path, selected[0]))
    return pairs
