"""Resolve EDF/FIF recording pairs from subject/date/session selectors."""

from __future__ import annotations

import re
from pathlib import Path

from hypnose_helpers.io.layout import SessionLayout, normalize_subjid


def _source_stem_from_fif(path: Path) -> str:
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
