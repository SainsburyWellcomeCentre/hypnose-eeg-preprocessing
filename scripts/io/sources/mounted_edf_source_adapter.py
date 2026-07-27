from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .source_adapter import EEGRecording, SourceAdapter, SourceCursor, utc_now


class MountedEdfSourceAdapter(SourceAdapter):
    """Read EDF recordings from a resolved mounted folder.

    The data-location profile resolves the machine-specific mount before this
    adapter is constructed, keeping storage details outside the adapter.
    """

    def __init__(
        self,
        source_id: str,
        source_dir: str,
        pattern: str = "**/*.edf",
    ) -> None:
        self._source_id = source_id
        self._source_dir = Path(source_dir)
        self._pattern = pattern

    @property
    def source_id(self) -> str:
        return self._source_id

    def open(self) -> None:
        if not self._source_dir.exists():
            raise FileNotFoundError(f"Mounted EDF source not found: {self._source_dir}")
        if not self._source_dir.is_dir():
            raise NotADirectoryError(f"Mounted EDF source is not a directory: {self._source_dir}")

    def close(self) -> None:
        return None

    def recordings(self, start_cursor: Optional[SourceCursor]) -> Iterator[tuple[EEGRecording, SourceCursor]]:
        try:
            import mne
        except ImportError as exc:
            raise ImportError("MountedEdfSourceAdapter requires MNE: pip install mne") from exc

        files = sorted(self._source_dir.glob(self._pattern))

        resume_file = start_cursor.last_file if start_cursor else ""

        for edf_path in files:
            file_key = str(edf_path.relative_to(self._source_dir))
            if resume_file and file_key <= resume_file:
                continue

            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sample_rate_hz = int(raw.info["sfreq"])
            base_timestamp = _recording_start(raw.info.get("meas_date"))
            source_metadata = _path_metadata(edf_path)
            data = raw.get_data()

            recording = EEGRecording(
                source_id=self.source_id,
                session_id=source_metadata["session_id"],
                timestamp_utc=base_timestamp,
                sample_rate_hz=sample_rate_hz,
                channels=list(raw.ch_names),
                samples=data.tolist(),
                metadata={
                    **source_metadata,
                    "source_file": file_key,
                    "source_path": str(edf_path),
                    "n_samples": raw.n_times,
                    "duration_seconds": raw.n_times / sample_rate_hz,
                },
            )
            next_cursor = SourceCursor(
                last_file=file_key,
                last_timestamp=base_timestamp.isoformat(),
            )
            yield recording, next_cursor


def _recording_start(meas_date: object) -> datetime:
    if isinstance(meas_date, datetime):
        if meas_date.tzinfo is None:
            return meas_date.replace(tzinfo=timezone.utc)
        return meas_date.astimezone(timezone.utc)
    return utc_now()


def _path_metadata(edf_path: Path) -> dict[str, str]:
    subject_id = _first_part_with_prefix(edf_path, "sub-") or "unknown-subject"
    session_id = _first_part_with_prefix(edf_path, "ses-") or edf_path.stem
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "recording_id": edf_path.stem,
    }


def _first_part_with_prefix(path: Path, prefix: str) -> Optional[str]:
    for part in path.parts:
        if part.startswith(prefix):
            return part
    return None
