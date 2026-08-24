"""Resolve EDF/FIF recording pairs from subject/date/session selectors."""

from __future__ import annotations

import re
from pathlib import Path


SESSION_DIR_RE = re.compile(r"^ses-([^-_]+)_date-(\d{8})$")


def _source_stem_from_fif(path: Path) -> str:
    stem = re.sub(r"_raw$", "", path.stem, flags=re.IGNORECASE)
    return re.sub(r"_resampled-[0-9p.]+hz$", "", stem, flags=re.IGNORECASE)


def _subject_label(subject: str | int) -> str:
    value = str(subject)
    if value.startswith("sub-"):
        value = value[4:]
    return f"sub-{int(value):03d}" if value.isdigit() else f"sub-{value}"


def _session_matches(actual: str, requested: str | int) -> bool:
    requested_text = str(requested)
    if requested_text.startswith("ses-"):
        requested_text = requested_text[4:]
    if actual.isdigit() and requested_text.isdigit():
        return int(actual) == int(requested_text)
    return actual == requested_text


def select_recordings(
    rawdata_root: Path,
    derivatives_root: Path,
    *,
    subject: str | int,
    date: str | int | None = None,
    session: str | int | None = None,
) -> list[tuple[Path, Path]]:
    """Resolve matching EDF/FIF pairs using subject and date/session selectors."""
    subject_label = _subject_label(subject)
    subject_dirs = sorted(
        path
        for path in rawdata_root.iterdir()
        if path.is_dir() and path.name.startswith(subject_label)
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No rawdata directory found for {subject_label}")

    pairs: list[tuple[Path, Path]] = []
    matched_sessions: list[Path] = []
    for subject_dir in subject_dirs:
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            match = SESSION_DIR_RE.match(session_dir.name)
            if match is None:
                continue
            session_number, session_date = match.groups()
            if date is not None and session_date != str(date):
                continue
            if session is not None and not _session_matches(session_number, session):
                continue
            matched_sessions.append(session_dir)
            relative_session = session_dir.relative_to(rawdata_root).as_posix()
            pairs.extend(
                pair_recordings(
                    rawdata_root,
                    derivatives_root,
                    edf_pattern=f"{relative_session}/**/*.edf",
                    fif_pattern=f"{relative_session}/**/*_raw.fif",
                )
            )

    if not matched_sessions:
        selector = f"date {date}" if date is not None else f"session {session}"
        raise FileNotFoundError(f"No {selector} found for {subject_label}")
    if not pairs:
        listing = ", ".join(str(path) for path in matched_sessions)
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
        fif_by_stem.setdefault(_source_stem_from_fif(fif_path), []).append(fif_path)

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
