"""Check raw EDF and derivative FIF durations and enumerate recording gaps.

The checker never preloads a complete recording. It opens the derivative FIF only
for its duration metadata, compares that duration directly with the source EDF,
then scans only the EDF signal in bounded chunks for constant/non-finite runs.
Gap-like EDF annotations are also included in the gap report.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.io.mne_io import import_mne
    from scripts.io.output_paths import quality_control_output_path
    from scripts.qc.thresholds import DEFAULT_CONFIG_PATH as DEFAULT_QC_CONFIG_PATH, load_qc_thresholds
    from scripts.utils.config import coalesce
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.io.mne_io import import_mne
    from scripts.io.output_paths import quality_control_output_path
    from scripts.qc.thresholds import DEFAULT_CONFIG_PATH as DEFAULT_QC_CONFIG_PATH, load_qc_thresholds
    from scripts.utils.config import coalesce


GAP_ANNOTATION_TERMS = ("gap", "boundary", "discontinu", "dropout", "missing")
SESSION_DIR_RE = re.compile(r"^ses-([^-_]+)_date-(\d{8})$")


@dataclass(frozen=True)
class Gap:
    source: str
    path: str
    kind: str
    start_s: float
    end_s: float
    duration_s: float
    start_time: str
    end_time: str
    detail: str


@dataclass(frozen=True)
class IntegrityResult:
    recording: str
    edf_path: str
    fif_path: str
    edf_duration_s: float
    fif_duration_s: float
    edf_fif_difference_s: float
    edf_gap_count: int
    edf_gap_total_s: float
    edf_gap_percent: float
    edf_longest_gap_s: float
    status: str


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


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start_s, end_s in sorted(intervals):
        if merged and start_s <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
        else:
            merged.append((start_s, end_s))
    return merged


def _true_runs(mask: np.ndarray, offset: int) -> list[tuple[int, int]]:
    if not mask.size:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1) + offset
    ends = np.flatnonzero(changes == -1) + offset
    return list(zip(starts.tolist(), ends.tolist()))


def detect_signal_gaps(
    raw: Any,
    *,
    min_gap_s: float = 1.0,
    chunk_duration_s: float = 1800.0,
) -> list[tuple[float, float]]:
    """Find constant/non-finite runs in an MNE Raw without preloading it."""
    sfreq = float(raw.info["sfreq"])
    chunk_samples = max(2, int(round(chunk_duration_s * sfreq)))
    picks = [
        index
        for index, channel_type in enumerate(raw.get_channel_types())
        if channel_type in {"eeg", "emg"}
    ]
    if not picks:
        picks = list(range(len(raw.ch_names)))

    sample_intervals: list[tuple[int, int]] = []
    for start in range(0, raw.n_times, chunk_samples):
        read_start = max(0, start - 1)
        stop = min(start + chunk_samples, raw.n_times)
        data = raw.get_data(picks=picks, start=read_start, stop=stop)
        if data.shape[1] < 2:
            continue
        equal_edge = np.any(data[:, 1:] == data[:, :-1], axis=0)
        # Equal edges [start, end) describe samples [start, end] inclusively.
        sample_intervals.extend(
            (run_start, run_end + 1)
            for run_start, run_end in _true_runs(equal_edge, offset=read_start)
        )
        nonfinite_sample = np.any(~np.isfinite(data), axis=0)
        sample_intervals.extend(
            _true_runs(nonfinite_sample, offset=read_start)
        )

    merged_samples = _merge_intervals(sample_intervals)
    return [
        (start / sfreq, end / sfreq)
        for start, end in merged_samples
        if (end - start) / sfreq >= min_gap_s
    ]


def _clock_time(meas_date: datetime | None, offset_s: float) -> str:
    if meas_date is None:
        return ""
    value = meas_date.replace(tzinfo=None) + timedelta(seconds=offset_s)
    return value.strftime("%Y%m%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def collect_gaps(
    raw: Any,
    source: str,
    path: Path,
    *,
    min_gap_s: float,
    chunk_duration_s: float,
) -> list[Gap]:
    meas_date = raw.info.get("meas_date")
    gaps: list[Gap] = []
    for start_s, end_s in detect_signal_gaps(
        raw, min_gap_s=min_gap_s, chunk_duration_s=chunk_duration_s
    ):
        gaps.append(
            Gap(
                source=source,
                path=str(path),
                kind="constant_or_nonfinite_signal",
                start_s=start_s,
                end_s=end_s,
                duration_s=end_s - start_s,
                start_time=_clock_time(meas_date, start_s),
                end_time=_clock_time(meas_date, end_s),
                detail="One or more EEG/EMG channels were constant or non-finite.",
            )
        )

    for annotation in raw.annotations:
        description = str(annotation["description"])
        if not any(term in description.lower() for term in GAP_ANNOTATION_TERMS):
            continue
        start_s = float(annotation["onset"])
        duration_s = float(annotation["duration"])
        end_s = start_s + duration_s
        gaps.append(
            Gap(
                source=source,
                path=str(path),
                kind="annotation",
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                start_time=_clock_time(meas_date, start_s),
                end_time=_clock_time(meas_date, end_s),
                detail=description,
            )
        )
    return sorted(gaps, key=lambda gap: (gap.start_s, gap.kind))


def check_pair(
    edf_path: Path,
    fif_path: Path,
    *,
    duration_tolerance_s: float,
    min_gap_s: float,
    chunk_duration_s: float,
    max_gap_percent: float = 1.0,
    max_longest_gap_s: float = 600.0,
) -> tuple[IntegrityResult, list[Gap]]:
    mne = import_mne()
    edf = mne.io.read_raw_edf(
        edf_path, preload=False, infer_types=True, verbose="ERROR"
    )
    fif = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    try:
        edf_duration_s = edf.n_times / float(edf.info["sfreq"])
        fif_duration_s = fif.n_times / float(fif.info["sfreq"])
        edf_gaps = collect_gaps(
            edf,
            "edf",
            edf_path,
            min_gap_s=min_gap_s,
            chunk_duration_s=chunk_duration_s,
        )
    finally:
        edf.close()
        fif.close()

    duration_difference_s = fif_duration_s - edf_duration_s
    duration_ok = abs(duration_difference_s) <= duration_tolerance_s
    merged_gaps = _merge_intervals((gap.start_s, gap.end_s) for gap in edf_gaps)
    gap_durations = [end_s - start_s for start_s, end_s in merged_gaps]
    gap_total_s = sum(gap_durations)
    gap_percent = 100.0 * gap_total_s / edf_duration_s if edf_duration_s else 0.0
    longest_gap_s = max(gap_durations, default=0.0)
    gap_review = gap_percent >= max_gap_percent or longest_gap_s > max_longest_gap_s
    status = "pass" if duration_ok and not gap_review else "review"
    result = IntegrityResult(
        recording=edf_path.stem,
        edf_path=str(edf_path),
        fif_path=str(fif_path),
        edf_duration_s=edf_duration_s,
        fif_duration_s=fif_duration_s,
        edf_fif_difference_s=duration_difference_s,
        edf_gap_count=len(edf_gaps),
        edf_gap_total_s=gap_total_s,
        edf_gap_percent=gap_percent,
        edf_longest_gap_s=longest_gap_s,
        status=status,
    )
    return result, edf_gaps


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_gap(index: int, gap: Gap) -> str:
    """Format only one gap's duration for concise terminal output."""
    return f"    Gap {index}: {gap.duration_s:.3f} seconds"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edf", default=None, help="Check one explicit raw EDF.")
    parser.add_argument("--fif", default=None, help="Matching derivative FIF.")
    parser.add_argument(
        "--subject", "--subjid", dest="subject", default=None,
        help="Subject ID, for example 66 or sub-066.",
    )
    session_selector = parser.add_mutually_exclusive_group()
    session_selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    session_selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
    )
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument("--edf-pattern", default="**/*.edf")
    parser.add_argument("--fif-pattern", default="**/*_raw.fif")
    parser.add_argument(
        "--qc-config",
        default=str(DEFAULT_QC_CONFIG_PATH),
        help=f"Quality-control threshold YAML (default: {DEFAULT_QC_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--duration-tolerance", type=float, default=None, metavar="SECONDS"
    )
    parser.add_argument("--min-gap", type=float, default=None, metavar="SECONDS")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="EDF scan chunk size (default: from --qc-config).",
    )
    parser.add_argument(
        "--max-gap-percent",
        type=float,
        default=None,
        metavar="PERCENT",
        help="Flag for review when combined gap time reaches this percent of the "
        "recording (default: from --qc-config).",
    )
    parser.add_argument(
        "--max-longest-gap",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Flag for review when the single longest gap exceeds this many "
        "seconds (default: from --qc-config).",
    )
    parser.add_argument(
        "--summary",
        nargs="?",
        const="recording_integrity.csv",
        default=None,
        help="Optionally save the summary in the shared session QC directory.",
    )
    parser.add_argument(
        "--gaps",
        nargs="?",
        const="recording_gaps.csv",
        default=None,
        help="Optionally save gaps in the shared session QC directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        qc_thresholds = load_qc_thresholds(args.qc_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    args.duration_tolerance = coalesce(
        args.duration_tolerance, qc_thresholds.duration_tolerance_s
    )
    args.min_gap = coalesce(args.min_gap, qc_thresholds.min_gap_s)
    args.chunk_duration = coalesce(
        args.chunk_duration, qc_thresholds.gap_scan_chunk_seconds
    )
    args.max_gap_percent = coalesce(
        args.max_gap_percent, qc_thresholds.max_gap_percent
    )
    args.max_longest_gap = coalesce(
        args.max_longest_gap, qc_thresholds.max_longest_gap_s
    )

    for name, value in (
        ("minimum gap", args.min_gap),
        ("chunk duration", args.chunk_duration),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be a positive finite number")
    if not math.isfinite(args.duration_tolerance) or args.duration_tolerance < 0:
        parser.error("duration tolerance must be a non-negative finite number")
    for name, value in (
        ("max gap percent", args.max_gap_percent),
        ("max longest gap", args.max_longest_gap),
    ):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be a non-negative finite number")

    if bool(args.edf) != bool(args.fif):
        parser.error("--edf and --fif must be supplied together")
    has_session_selector = args.date is not None or args.session is not None
    if args.subject is None and has_session_selector:
        parser.error("--date or --session requires --subject")
    if args.subject is not None and not has_session_selector:
        parser.error("--subject requires either --date or --session")
    if args.edf and (args.subject is not None or has_session_selector):
        parser.error("use either explicit --edf/--fif paths or subject/session selectors")

    if args.edf:
        edf_path = Path(args.edf).resolve(strict=False)
        fif_path = Path(args.fif).resolve(strict=False)
        rawdata_root = Path(args.rawdata_root or edf_path.parent).resolve(strict=False)
        derivatives_root = Path(
            args.derivatives_root or fif_path.parent
        ).resolve(strict=False)
        pairs = [(edf_path, fif_path)]
    else:
        rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
        derivatives_root = Path(
            args.derivatives_root or get_derivatives_root()
        ).resolve(strict=False)
        if args.subject is not None:
            pairs = select_recordings(
                rawdata_root,
                derivatives_root,
                subject=args.subject,
                date=args.date,
                session=args.session,
            )
        else:
            pairs = pair_recordings(
                rawdata_root,
                derivatives_root,
                edf_pattern=args.edf_pattern,
                fif_pattern=args.fif_pattern,
            )
    if not pairs:
        parser.error("No matching EDF/FIF pairs found")

    results: list[IntegrityResult] = []
    gaps: list[Gap] = []
    for edf_path, fif_path in pairs:
        print(f"Checking {edf_path} <-> {fif_path}")
        result, pair_gaps = check_pair(
            edf_path,
            fif_path,
            duration_tolerance_s=args.duration_tolerance,
            min_gap_s=args.min_gap,
            chunk_duration_s=args.chunk_duration,
            max_gap_percent=args.max_gap_percent,
            max_longest_gap_s=args.max_longest_gap,
        )
        results.append(result)
        gaps.extend(pair_gaps)
        print(
            f"  {result.status}: EDF={result.edf_duration_s:.3f}s, "
            f"FIF={result.fif_duration_s:.3f}s, "
            f"difference={result.edf_fif_difference_s:+.3f}s, "
            f"gaps={len(pair_gaps)} ({result.edf_gap_total_s:.3f}s, "
            f"{result.edf_gap_percent:.3f}% of recording, "
            f"longest={result.edf_longest_gap_s:.3f}s)"
        )
        if pair_gaps:
            print("  Detected gaps:")
            for index, gap in enumerate(pair_gaps, start=1):
                print(_format_gap(index, gap))
        else:
            print("  Detected gaps: none")

    def output_path(requested: str) -> Path:
        path = Path(requested)
        if path.is_absolute():
            return path
        if len(pairs) != 1:
            parser.error("relative QC outputs require a single selected recording")
        return quality_control_output_path(
            path, pairs[0][0], rawdata_root, derivatives_root
        )

    if args.summary:
        summary_path = output_path(args.summary)
        result_rows = [asdict(result) for result in results]
        _write_csv(
            summary_path, result_rows, list(IntegrityResult.__dataclass_fields__)
        )
        print(f"Saved: {summary_path}")
    if args.gaps:
        gaps_path = output_path(args.gaps)
        gap_rows = [asdict(gap) for gap in gaps]
        _write_csv(gaps_path, gap_rows, list(Gap.__dataclass_fields__))
        print(f"Saved: {gaps_path}")
    return 1 if any(result.status != "pass" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
