"""Combine multiple recording parts from the same session into one timeline.

Long EEG sessions may be split into several acquisition files. This script finds
sessions containing multiple parts, checks that their channel layouts are
compatible, concatenates them in filename or explicitly supplied order, and adds
annotations at every join so downstream analyses can identify the boundaries.
It writes a combined recording and a CSV manifest describing successes, skips,
and failures. Existing outputs are preserved unless overwrite is requested.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.io.mne_io import export_raw_edf, import_mne
    from scripts.utils.config import coalesce, load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.io.mne_io import export_raw_edf, import_mne
    from scripts.utils.config import coalesce, load_config, nested_get


@dataclass(frozen=True)
class ConcatenationResult:
    session_path: str
    edf_output_path: str
    n_recordings: int
    status: str
    recordings: list[str]
    boundary_markers: list[str]
    n_channels: int | None = None
    sample_rate_hz: float | None = None
    n_samples: int | None = None
    duration_seconds: float | None = None
    existing_outputs: list[str] | None = None
    error: str | None = None


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

        selected_sessions = {}
        for session_dir, files in sessions.items():
            selected_files = self._prefer_trimmed_recordings(files)
            if len(selected_files) > 1:
                selected_sessions[session_dir] = selected_files

        return selected_sessions

    def _prefer_trimmed_recordings(self, files: list[Path]) -> list[Path]:
        files_by_recording: dict[str, list[Path]] = {}
        for file in files:
            files_by_recording.setdefault(self._recording_key(file), []).append(file)

        selected = []
        for recording_files in files_by_recording.values():
            trimmed_files = [
                file
                for file in recording_files
                if file.stem.lower().endswith("_trimmed")
            ]
            selected.append(sorted(trimmed_files or recording_files)[0])

        return sorted(selected)

    def _recording_key(self, file: Path) -> str:
        stem = file.stem
        if stem.lower().endswith("_trimmed"):
            return stem[: -len("_trimmed")]
        return stem

    def select_recordings(
        self,
        session_dir: str | Path,
        recording_selectors: list[str],
    ) -> tuple[Path, list[Path]]:
        resolved_session_dir = self._resolve_session_dir(session_dir)
        available_files = sorted(
            path
            for path in resolved_session_dir.glob("*.edf")
            if path.is_file()
            and "_recording" in path.name
            and "_recording-concat" not in path.name
            and path.suffix.lower() == ".edf"
        )

        selected_files = [
            self._select_recording(resolved_session_dir, available_files, selector)
            for selector in recording_selectors
        ]
        if len(selected_files) < 2:
            raise ValueError("Select at least two EDF recordings to concatenate.")

        return resolved_session_dir, selected_files

    def preview_selected(
        self,
        session_dir: str | Path,
        recording_selectors: list[str],
    ) -> ConcatenationResult:
        resolved_session_dir, files = self.select_recordings(session_dir, recording_selectors)
        relative_session_dir = resolved_session_dir.relative_to(self.source_dir)
        edf_output_path = self.output_path(resolved_session_dir, files)
        return ConcatenationResult(
            session_path=str(relative_session_dir),
            edf_output_path=str(edf_output_path),
            n_recordings=len(files),
            recordings=[file.name for file in files],
            boundary_markers=self._boundary_marker_descriptions(files),
            status="dry_run",
        )

    def concatenate_selected(
        self,
        session_dir: str | Path,
        recording_selectors: list[str],
    ) -> ConcatenationResult:
        resolved_session_dir, files = self.select_recordings(session_dir, recording_selectors)
        return self.concatenate_session(resolved_session_dir, files, sort_files=False)

    def _resolve_session_dir(self, session_dir: str | Path) -> Path:
        path = Path(session_dir)
        if not path.is_absolute():
            path = self.source_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Session directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Session path is not a directory: {path}")
        try:
            path.relative_to(self.source_dir)
            return path
        except ValueError:
            pass

        try:
            relative_path = path.resolve().relative_to(self.source_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Session directory must be inside source_dir: {self.source_dir}") from exc
        return self.source_dir / relative_path

    def _select_recording(
        self,
        session_dir: Path,
        available_files: list[Path],
        selector: str,
    ) -> Path:
        selector_path = Path(selector)
        if selector_path.suffix.lower() == ".edf":
            exact_path = session_dir / selector_path
            if exact_path.exists() and exact_path.is_file():
                return exact_path
            raise FileNotFoundError(f"Selected EDF not found in session: {selector}")

        recording_key_matches = [
            file
            for file in available_files
            if self._recording_key(file) == selector
        ]
        if recording_key_matches:
            return self._prefer_trimmed_recordings(recording_key_matches)[0]

        exact_stem_matches = [
            file
            for file in available_files
            if file.stem == selector
        ]
        if exact_stem_matches:
            return self._prefer_trimmed_recordings(exact_stem_matches)[0]

        raise FileNotFoundError(f"Selected recording not found in session: {selector}")

    def output_path(self, session_dir: Path, files: list[Path]) -> Path:
        output_dir = session_dir
        if self.sink_dir is not None:
            relative_session_dir = session_dir.relative_to(self.source_dir)
            output_dir = self.sink_dir / relative_session_dir

        base_name = self._concatenated_base_name(files)
        return output_dir / f"{base_name}.edf"

    def concatenate_session(
        self,
        session_dir: Path,
        files: list[Path],
        sort_files: bool = True,
    ) -> ConcatenationResult:
        mne = import_mne()

        files = sorted(files) if sort_files else list(files)
        relative_session_dir = session_dir.relative_to(self.source_dir)
        edf_output_path = self.output_path(session_dir, files)
        boundary_markers = self._boundary_marker_descriptions(files)

        if not self.overwrite and edf_output_path.exists():
            return ConcatenationResult(
                session_path=str(relative_session_dir),
                edf_output_path=str(edf_output_path),
                n_recordings=len(files),
                recordings=[file.name for file in files],
                boundary_markers=boundary_markers,
                status="skipped_exists",
                existing_outputs=[str(edf_output_path)],
            )

        raws = [
            mne.io.read_raw_edf(file, preload=False, infer_types=True, verbose=True)
            for file in files
        ]
        info_error = self._raw_info_mismatch(raws, files)
        if info_error is not None:
            return ConcatenationResult(
                session_path=str(relative_session_dir),
                edf_output_path=str(edf_output_path),
                n_recordings=len(files),
                recordings=[file.name for file in files],
                boundary_markers=boundary_markers,
                status="failed_info_mismatch",
                error=info_error,
            )

        boundary_onsets = self._boundary_onsets(raws)
        concatenated = mne.concatenate_raws(raws)
        self._add_boundary_annotations(concatenated, boundary_onsets, boundary_markers)

        edf_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_raw_edf(concatenated, edf_output_path)
        gc.collect()

        result = ConcatenationResult(
            session_path=str(relative_session_dir),
            edf_output_path=str(edf_output_path),
            n_recordings=len(files),
            recordings=[file.name for file in files],
            boundary_markers=boundary_markers,
            n_channels=len(concatenated.ch_names),
            sample_rate_hz=float(concatenated.info["sfreq"]),
            n_samples=concatenated.n_times,
            duration_seconds=concatenated.n_times / float(concatenated.info["sfreq"]),
            status="written",
        )
        del concatenated, raws
        gc.collect()
        return result

    def concatenate_all(self) -> list[ConcatenationResult]:
        sessions = self.find_multi_recording_sessions()
        results = []

        for session_dir, files in sorted(sessions.items(), key=lambda item: str(item[0])):
            try:
                results.append(self.concatenate_session(session_dir, files))
            except Exception as exc:
                results.append(self._failed_result(session_dir, files, exc))

        return results

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
            edf_output_path = self.output_path(session_dir, files)
            results.append(
                ConcatenationResult(
                    session_path=str(relative_session_dir),
                    edf_output_path=str(edf_output_path),
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
            "n_recordings",
            "status",
            "recordings",
            "boundary_markers",
            "n_channels",
            "sample_rate_hz",
            "n_samples",
            "duration_seconds",
            "existing_outputs",
            "error",
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
        export_raw_edf(raw, output_path, overwrite=self.overwrite)

    def _raw_info_mismatch(self, raws: list[Any], files: list[Path]) -> str | None:
        if not raws:
            return None

        reference = raws[0]
        problems = []

        for raw, file in zip(raws[1:], files[1:]):
            if raw.info["nchan"] != reference.info["nchan"]:
                problems.append(
                    f"{file.name} has {raw.info['nchan']} channels; "
                    f"{files[0].name} has {reference.info['nchan']} channels"
                )
                continue

            if raw.ch_names != reference.ch_names:
                missing = sorted(set(reference.ch_names) - set(raw.ch_names))
                extra = sorted(set(raw.ch_names) - set(reference.ch_names))
                problems.append(
                    f"{file.name} channel names/order differ from {files[0].name}; "
                    f"missing={missing}; extra={extra}"
                )

        return "; ".join(problems) if problems else None

    def _failed_result(self, session_dir: Path, files: list[Path], exc: Exception) -> ConcatenationResult:
        files = sorted(files)
        relative_session_dir = session_dir.relative_to(self.source_dir)
        edf_output_path = self.output_path(session_dir, files)
        return ConcatenationResult(
            session_path=str(relative_session_dir),
            edf_output_path=str(edf_output_path),
            n_recordings=len(files),
            recordings=[file.name for file in files],
            boundary_markers=self._boundary_marker_descriptions(files),
            status="failed",
            existing_outputs=[str(edf_output_path)] if edf_output_path.exists() else [],
            error=f"{type(exc).__name__}: {exc}",
        )


def _result_to_row(result: ConcatenationResult) -> dict[str, Any]:
    row = result.__dict__.copy()
    row["recordings"] = "|".join(result.recordings)
    row["boundary_markers"] = "|".join(result.boundary_markers)
    row["existing_outputs"] = "|".join(result.existing_outputs or [])
    row["error"] = result.error or ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate sessions that contain multiple EDF recording files."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. CLI values override config values.",
    )
    parser.add_argument("--source-dir", default=None, help="Mounted raw EDF root.")
    parser.add_argument(
        "--sink-dir",
        default=None,
        help=(
            "Optional output root. If omitted, concatenated files are written "
            "beside their source EDF files."
        ),
    )
    parser.add_argument("--edf-pattern", default=None, help="Glob used below source-dir.")
    parser.add_argument(
        "--session-dir",
        default=None,
        help=(
            "Session/ephys directory to concatenate within. May be absolute or "
            "relative to source-dir."
        ),
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help=(
            "Explicit EDF filenames or stems to concatenate within --session-dir, "
            "in the order provided."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--first", action="store_true", help="Concatenate only the first matching session.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matching sessions without writing files.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    source_dir = coalesce(args.source_dir, nested_get(config, ("source", "uri")), get_rawdata_root())
    sink_dir = coalesce(args.sink_dir, nested_get(config, ("preprocessing", "concatenate", "sink_dir")))
    edf_pattern = coalesce(args.edf_pattern, nested_get(config, ("source", "glob")), "**/*.edf")
    manifest = coalesce(
        args.manifest,
        nested_get(config, ("preprocessing", "concatenate", "manifest")),
        get_derivatives_root() / "edf_concatenation_manifest.csv",
    )

    concatenator = EdfSessionConcatenator(
        source_dir=source_dir,
        sink_dir=sink_dir,
        edf_pattern=edf_pattern,
        overwrite=args.overwrite,
    )

    if (args.session_dir is None) != (args.recordings is None):
        raise ValueError("--session-dir and --recordings must be used together.")

    if args.session_dir is not None and args.recordings is not None:
        if args.dry_run:
            results = [concatenator.preview_selected(args.session_dir, args.recordings)]
        else:
            results = [concatenator.concatenate_selected(args.session_dir, args.recordings)]
    elif args.dry_run:
        results = concatenator.preview()
    elif args.first:
        results = [concatenator.concatenate_first()]
    else:
        results = concatenator.concatenate_all()

    manifest_path = concatenator.write_manifest(results, manifest)
    for result in results:
        print(
            f"{result.status}: {result.session_path} "
            f"({result.n_recordings} recordings) -> {result.edf_output_path}"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
