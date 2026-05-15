from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .source_adapter import EEGChunk, SourceAdapter, SourceCursor


class FilesystemJsonlSourceAdapter(SourceAdapter):
    """Example adapter that tails JSONL files from an external folder.

    Expected line format:
    {
      "session_id": "s-001",
      "timestamp_utc": "2026-05-11T10:30:00Z",
      "sample_rate_hz": 256,
      "channels": ["Fp1", "Fp2"],
      "samples": [[...], [...]],
      "metadata": {"subject_id": "p-01"}
    }
    """

    def __init__(self, source_id: str, source_dir: str, pattern: str = "*.jsonl") -> None:
        self._source_id = source_id
        self._source_dir = Path(source_dir)
        self._pattern = pattern

    @property
    def source_id(self) -> str:
        return self._source_id

    def open(self) -> None:
        if not self._source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self._source_dir}")

    def close(self) -> None:
        # No persistent handles in this simple example.
        return None

    def stream(self, start_cursor: Optional[SourceCursor]) -> Iterator[tuple[EEGChunk, SourceCursor]]:
        files = sorted(self._source_dir.glob(self._pattern))

        resume_file = start_cursor.last_file if start_cursor else ""
        resume_offset = start_cursor.last_offset if start_cursor else 0

        for file_path in files:
            if resume_file and file_path.name < resume_file:
                continue

            with file_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if file_path.name == resume_file and line_idx <= resume_offset:
                        continue

                    payload = json.loads(line)
                    chunk = EEGChunk(
                        source_id=self.source_id,
                        session_id=payload["session_id"],
                        timestamp_utc=_normalize_ts(payload["timestamp_utc"]),
                        sample_rate_hz=int(payload["sample_rate_hz"]),
                        channels=list(payload["channels"]),
                        samples=list(payload["samples"]),
                        metadata=dict(payload.get("metadata", {})),
                    )
                    next_cursor = SourceCursor(
                        last_file=file_path.name,
                        last_offset=line_idx,
                        last_timestamp=payload["timestamp_utc"],
                    )
                    yield chunk, next_cursor


def _normalize_ts(value: str):
    # Keep a lightweight parser to avoid external dependencies in the example.
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))
