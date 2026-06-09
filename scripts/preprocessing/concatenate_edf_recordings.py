from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ConcatenationResult:
    session_path: str
    edf_output_path: str
    fif_output_path: str
    n_recordings: int
    status: str
    recordings: list[str]
    boundary_markers: list[str]
    n_channels: int | None = None
    sample_rate_hz: float | None = None
    n_samples: int | None = None
    duration_seconds: float | None = None
    existing_outputs: list[str] | None = None


class EdfSessionConcatenator:
    """Concatenate session folders that contain multiple EDF recordings."""

    def __init__(
        self,
        source_dir: str | Path,
        sink_dir: str | Path | None = None,
        edf_pattern: str = "**/*.edf",
        overwrite: bool = False,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.sink_dir = Path(sink_dir) if sink_dir is not None else None
        self.edf_pattern = edf_pattern
        self.overwrite = overwrite

    def find_multi_recording_sessions(self) -> dict[Path, list[Path]]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir.resolve()}")

        recording_files = sorted(
            path
            for path in self.source_dir.glob(self.edf_pattern)
            if "_recording" in path.name
            and "_recording-concat" not in path.name
            and path.suffix.lower() == ".edf"
        )

        sessions: dict[Path, list[Path]] = {}
        for edf_path in recording_files:
            sessions.setdefault(edf_path.parent, []).append(edf_path)

        return {
            session_dir: files
            for session_dir, files in sessions.items()
            if len(files) > 1
        }

    def output_paths(self, session_dir: Path, files: list[Path]) -> tuple[Path, Path]:
        output_dir = session_dir
        if self.sink_dir is not None:
            relative_session_dir = session_dir.relative_to(self.source_dir)
            output_dir = self.sink_dir / relative_session_dir

        base_name = self._concatenated_base_name(files)
        return output_dir / f"{base_name}.edf", output_dir / f"{base_name}_raw.fif"

    def concatenate_session(self, session_dir: Path, files: list[Path]) -> ConcatenationResult:
        mne = _import_mne()

        files = sorted(files)
        relative_session_dir = session_dir.relative_to(self.source_dir)
        edf_output_path, fif_output_path = self.output_paths(session_dir, files)
        existing_outputs = [path for path in (edf_output_path, fif_output_path) if path.exists()]
        boundary_markers = self._boundary_marker_descriptions(files)

        if existing_outputs and not self.overwrite:
            return ConcatenationResult(
                session_path=str(relative_session_dir),
                edf_output_path=str(edf_output_path),
                fif_output_path=str(fif_output_path),
                n_recordings=len(files),
                recordings=[file.name for file in files],
                boundary_markers=boundary_markers,
                status="skipped_exists",
                existing_outputs=[str(path) for path in existing_outputs],
            )

        raws = [
            mne.io.read_raw_edf(file, preload=True, infer_types=True, verbose=True)
            for file in files
        ]
        boundary_onsets = self._boundary_onsets(raws)
        concatenated = mne.concatenate_raws(raws)
        self._add_boundary_annotations(concatenated, boundary_onsets, boundary_markers)

        edf_output_path.parent.mkdir(parents=True, exist_ok=True)
        concatenated.save(fif_output_path, overwrite=self.overwrite)
        self._export_raw_edf(concatenated, edf_output_path)

        return ConcatenationResult(
            session_path=str(relative_session_dir),
            edf_output_path=str(edf_output_path),
            fif_output_path=str(fif_output_path),
            n_recordings=len(files),
            recordings=[file.name for file in files],
            boundary_markers=boundary_markers,
            n_channels=len(concatenated.ch_names),
            sample_rate_hz=float(concatenated.info["sfreq"]),
            n_samples=concatenated.n_times,
            duration_seconds=concatenated.n_times / float(concatenated.info["sfreq"]),
            status="written",
        )

    def concatenate_all(self) -> list[ConcatenationResult]:
        sessions = self.find_multi_recording_sessions()
        return [
            self.concatenate_session(session_dir, files)
            for session_dir, files in sorted(sessions.items(), key=lambda item: str(item[0]))
        ]

    def concatenate_first(self) -> ConcatenationResult:
        sessions = self.find_multi_recording_sessions()
        if not sessions:
            raise RuntimeError("No multi-recording EDF sessions found.")

        session_dir, files = next(iter(sorted(sessions.items(), key=lambda item: str(item[0]))))
        return self.concatenate_session(session_dir, files)

    def preview(self) -> list[ConcatenationResult]:
        sessions = self.find_multi_recording_sessions()
        results = []

        for session_dir, files in sorted(sessions.items(), key=lambda item: str(item[0])):
            files = sorted(files)
            relative_session_dir = session_dir.relative_to(self.source_dir)
            edf_output_path, fif_output_path = self.output_paths(session_dir, files)
            results.append(
                ConcatenationResult(
                    session_path=str(relative_session_dir),
                    edf_output_path=str(edf_output_path),
                    fif_output_path=str(fif_output_path),
                    n_recordings=len(files),
                    recordings=[file.name for file in files],
                    boundary_markers=self._boundary_marker_descriptions(files),
                    status="dry_run",
                )
            )

        return results

    def write_manifest(self, results: Iterable[ConcatenationResult], manifest_path: str | Path) -> Path:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = [_result_to_row(result) for result in results]
        fieldnames = [
            "session_path",
            "edf_output_path",
            "fif_output_path",
            "n_recordings",
            "status",
            "recordings",
            "boundary_markers",
            "n_channels",
            "sample_rate_hz",
            "n_samples",
            "duration_seconds",
            "existing_outputs",
        ]

        with manifest.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return manifest

    def _concatenated_base_name(self, files: list[Path]) -> str:
        first_stem = files[0].stem
        match = re.match(r"(?P<prefix>.+?)_recording.*$", first_stem)
        if match:
            return f"{match.group('prefix')}_recording-concat"
        return f"{first_stem}_concat"

    def _boundary_marker_descriptions(self, files: list[Path]) -> list[str]:
        return [
            f"CONCAT_BOUNDARY {previous.name} -> {current.name}"
            for previous, current in zip(files, files[1:])
        ]

    def _boundary_onsets(self, raws: list[Any]) -> list[float]:
        onsets = []
        elapsed = 0.0

        for raw in raws[:-1]:
            elapsed += raw.n_times / float(raw.info["sfreq"])
            onsets.append(elapsed)

        return onsets

    def _add_boundary_annotations(
        self,
        raw: Any,
        boundary_onsets: list[float],
        boundary_markers: list[str],
    ) -> None:
        for onset, description in zip(boundary_onsets, boundary_markers):
            raw.annotations.append(
                onset=onset,
                duration=0.0,
                description=description,
            )

    def _export_raw_edf(self, raw: Any, output_path: Path) -> None:
        try:
            raw.export(output_path, fmt="edf", overwrite=self.overwrite, physical_range="auto")
        except TypeError:
            raw.export(output_path, fmt="edf", overwrite=self.overwrite)
        except RuntimeError as exc:
            raise RuntimeError("EDF export requires the edfio package: pip install edfio") from exc


def _import_mne() -> Any:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install MNE first: pip install mne") from exc

    return mne


def _result_to_row(result: ConcatenationResult) -> dict[str, Any]:
    row = result.__dict__.copy()
    row["recordings"] = "|".join(result.recordings)
    row["boundary_markers"] = "|".join(result.boundary_markers)
    row["existing_outputs"] = "|".join(result.existing_outputs or [])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate sessions that contain multiple EDF recording files."
    )
    parser.add_argument("--source-dir", default="data/rawdata", help="Mounted raw EDF root.")
    parser.add_argument(
        "--sink-dir",
        default=None,
        help=(
            "Optional output root. If omitted, concatenated files are written "
            "beside their source EDF files."
        ),
    )
    parser.add_argument("--edf-pattern", default="**/*.edf", help="Glob used below source-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--first", action="store_true", help="Concatenate only the first matching session.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matching sessions without writing files.")
    parser.add_argument(
        "--manifest",
        default="data/rawdata/edf_concatenation_manifest.csv",
        help="CSV manifest path.",
    )
    args = parser.parse_args()

    concatenator = EdfSessionConcatenator(
        source_dir=args.source_dir,
        sink_dir=args.sink_dir,
        edf_pattern=args.edf_pattern,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        results = concatenator.preview()
    elif args.first:
        results = [concatenator.concatenate_first()]
    else:
        results = concatenator.concatenate_all()

    manifest = concatenator.write_manifest(results, args.manifest)
    for result in results:
        print(
            f"{result.status}: {result.session_path} "
            f"({result.n_recordings} recordings) -> {result.fif_output_path}"
        )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
