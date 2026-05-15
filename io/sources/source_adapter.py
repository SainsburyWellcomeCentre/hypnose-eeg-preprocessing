from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class EEGChunk:
    """A single chunk emitted by a source adapter."""

    source_id: str
    session_id: str
    timestamp_utc: datetime
    sample_rate_hz: int
    channels: list[str]
    samples: list[list[float]]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SourceCursor:
    """Opaque source position used for resumable ingestion."""

    last_file: str
    last_offset: int
    last_timestamp: str


class SourceAdapter(ABC):
    """Interface for external EEG sources.

    Implementations read from storage outside this repo (filesystem, S3, DB, etc.)
    and emit EEGChunk items in order.
    """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique source identifier persisted in checkpoints."""

    @abstractmethod
    def open(self) -> None:
        """Initialize external connections/resources."""

    @abstractmethod
    def close(self) -> None:
        """Release external connections/resources."""

    @abstractmethod
    def stream(self, start_cursor: Optional[SourceCursor]) -> Iterator[tuple[EEGChunk, SourceCursor]]:
        """Yield ordered chunks and next cursor values.

        Args:
            start_cursor: Last committed checkpoint cursor, or None for cold start.

        Yields:
            Tuples of (chunk, next_cursor). `next_cursor` is persisted only after
            downstream processing and sink writes complete successfully.
        """


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
