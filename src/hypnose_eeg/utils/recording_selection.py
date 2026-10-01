"""Resolve EDF/FIF recording pairs from subject/date/session selectors."""

from __future__ import annotations

import re
from pathlib import Path

from hypnose_helpers.io.layout import SessionLayout, SessionRef, normalize_subjid


# Suffix `trim_duplicate_channels.py` appends to a recording's stem when it
# writes a trimmed copy beside it. `preprocessing.trim_channels.output_suffix`
# in configs/pipelines/preprocessing.yaml can override it for that script and
# for concatenation, but the selection helpers below only recognise this value.
DEFAULT_TRIMMED_SUFFIX = "_trimmed"


def is_concatenated_recording(path: Path) -> bool:
    """Whether `path` looks like a `concatenate_recordings.py`-produced file."""
    name = path.name.lower()
    return name.endswith("recording-concat.edf") or "_recording-concat" in name


def is_trimmed_recording(path: Path, trimmed_suffix: str = DEFAULT_TRIMMED_SUFFIX) -> bool:
    """Whether `path` looks like an `trim_duplicate_channels.py`-produced file."""
    return path.stem.lower().endswith(trimmed_suffix.lower())


def prefer_trimmed_recordings(
    edf_paths: list[Path], trimmed_suffix: str = DEFAULT_TRIMMED_SUFFIX
) -> list[Path]:
    """Drop every EDF whose `<stem>_trimmed.edf` sibling is also in the list.

    `trim_duplicate_channels.py` writes a duplicate-free copy of a recording
    beside its source rather than editing the source, so wherever both are
    present the trimmed copy is the one to process. Siblings are matched by
    path, so a flat list spanning several folders is safe.
    """
    available = {path.parent / path.name.lower() for path in edf_paths}

    def has_trimmed_sibling(path: Path) -> bool:
        sibling = path.parent / f"{path.stem}{trimmed_suffix}{path.suffix}".lower()
        return sibling in available

    return [
        path
        for path in edf_paths
        if is_trimmed_recording(path, trimmed_suffix) or not has_trimmed_sibling(path)
    ]


def prefer_concatenated_recording(edf_paths: list[Path]) -> list[Path]:
    """Within one folder's EDFs, keep only the concatenated recording if several exist.

    `concatenate_recordings.py` combines a session's raw per-part EDFs into one
    `_recording-concat.edf` file alongside them. Once that has been run, processing
    every file in the folder duplicates work on the same underlying recording --
    and for the individual parts, without any of the gap-aware handling
    (concatenate_recordings.py's own gap-padding, or somnotate's
    `prepare_recording`) that only applies within a single file.

    A trimmed copy replaces its source first (`prefer_trimmed_recordings`), so a
    single-recording session whose only extra file is the trimmed copy resolves
    to that copy rather than looking like unconcatenated parts.

    If several EDFs are present but none is a concatenated recording, an empty
    list is returned rather than falling back to the ambiguous raw parts --
    callers should treat that as "not ready to process yet" and warn rather than
    silently score/downsample the parts independently.
    """
    edf_paths = prefer_trimmed_recordings(edf_paths)
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


def find_sessions(
    rawdata_root: Path,
    *,
    subject: str | int,
    date: str | int | None = None,
    session: str | int | None = None,
) -> list[SessionRef]:
    """Resolve rawdata sessions using subject and optional date/session selectors.

    With neither `date` nor `session` given, every session the subject has is
    returned, in directory order -- which is what a batch run over one subject
    iterates. Each ref carries the session number, date, and directory, so the
    caller can re-select one session without re-parsing its directory name.
    """
    layout = SessionLayout(rawdata_root, name="rawdata")
    sessions = layout.find_sessions(subject, ses=session, date=date)
    if not sessions:
        if date is not None:
            selector = f"date {date}"
        elif session is not None:
            selector = f"session {session}"
        else:
            selector = "sessions"
        raise FileNotFoundError(f"No {selector} found for {normalize_subjid(subject)}")
    return sessions


def find_session_dirs(
    rawdata_root: Path,
    *,
    subject: str | int,
    date: str | int | None = None,
    session: str | int | None = None,
) -> list[Path]:
    """Resolve rawdata session directories using subject and date/session selectors."""
    return [
        session_ref.path
        for session_ref in find_sessions(
            rawdata_root, subject=subject, date=date, session=session
        )
    ]


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
    """Pair EDF/FIF files by source stem, preferring matching relative folders.

    A trimmed copy stands in for its source EDF (`prefer_trimmed_recordings`),
    so a FIF left over from downsampling the untrimmed source is not paired.
    """
    edfs = prefer_trimmed_recordings(
        sorted(path for path in rawdata_root.glob(edf_pattern) if path.is_file())
    )
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
