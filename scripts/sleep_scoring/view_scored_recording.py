"""Open hypnose-somnotate's interactive view for one scored recording.

This is a visual quality-control tool: it displays the raw EEG/EMG traces and
Somnotate state predictions that already exist on disk. It does not rescore the
recording or calculate a numerical performance metric.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hypnose_helpers.io.selectors import parse_sessions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.io.input_paths import scoring_path
from scripts.io.output_layout import output_dir_name
from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root, get_repo_root
from scripts.utils.config import (
    DEFAULT_SLEEP_SCORING_CONFIG_PATH,
    coalesce,
    load_config,
    nested_get,
)


REMOTE_VIEWER_DOC = get_repo_root() / "docs" / "remote_visualization.md"


@dataclass(frozen=True)
class ScoringViewSettings:
    subject: str
    date: str | None
    repo_root: Path
    rawdata_root: Path
    derivatives_root: Path
    recording_index: int
    eeg_channel: int
    view_length_s: float
    hours: tuple[float, float] | None = None
    time_range: tuple[datetime, datetime] | None = None
    display_rate_hz: float | None = None
    show_artifacts: bool = False
    show_gaps: bool = True
    session: int | None = None


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [part.strip() for item in items for part in str(item).split(",") if part.strip()]


def _single_value(value: Any, *, option_name: str) -> str | None:
    values = _values(value)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(
            f"the interactive viewer requires exactly one {option_name}; got {len(values)}"
        )
    return values[0]


def _parse_time_range(value: Any) -> tuple[datetime, datetime] | None:
    if value is None:
        return None
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) != 2:
        raise ValueError("time_range must contain exactly START and END")
    try:
        parsed = tuple(datetime.strptime(str(item), "%Y%m%d %H:%M:%S") for item in values)
    except ValueError as exc:
        raise ValueError(
            "time_range values must use 'YYYYMMDD HH:MM:SS'"
        ) from exc
    start, end = parsed
    if end <= start:
        raise ValueError("time_range END must be later than START")
    return parsed


def _elapsed_range(
    settings: ScoringViewSettings, recording_start: datetime
) -> tuple[float, float, str, str]:
    """Convert either range representation to seconds from the EDF start."""
    if settings.hours is not None:
        start_hour, end_hour = settings.hours
        return (
            start_hour * 3600.0,
            end_hour * 3600.0,
            f"hours {start_hour:g}-{end_hour:g}",
            f"Hours {start_hour:g}-{end_hour:g} from session start",
        )

    assert settings.time_range is not None
    start_time, end_time = settings.time_range
    recording_start = recording_start.replace(tzinfo=None)
    start_s = (start_time - recording_start).total_seconds()
    end_s = (end_time - recording_start).total_seconds()
    if start_s < 0:
        raise ValueError(
            f"time_range START precedes EDF start "
            f"({recording_start:%Y%m%d %H:%M:%S})"
        )
    description = f"{start_time:%Y%m%d %H:%M:%S} to {end_time:%Y%m%d %H:%M:%S}"
    return start_s, end_s, description, f"{description} (real time)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually inspect one scored session with hypnose-somnotate view."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SLEEP_SCORING_CONFIG_PATH),
        help="Pipeline YAML path (default: configs/pipelines/sleep_scoring.yaml).",
    )
    parser.add_argument("--subject", "--subjid", dest="subject", default=None)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--date", default=None, help="Session date in YYYYMMDD form.")
    selector.add_argument(
        "--session",
        default=None,
        help="Session number, for example 1 or ses-1.",
    )
    parser.add_argument("--recording-index", type=int, default=None)
    parser.add_argument("--eeg-channel", type=int, choices=(0, 1), default=None)
    parser.add_argument("--view-length", type=float, default=None, metavar="SECONDS")
    parser.add_argument(
        "--display-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="Downsample the selected signals before display (for example: 128).",
    )
    parser.add_argument(
        "--show-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shade epochs flagged in the session artifacts directory.",
    )
    parser.add_argument(
        "--show-gaps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Shade 'gap'/'too_short' spans from the predictions parquet's own "
            "kind column (concatenation gaps, dropouts, segments too short to "
            "score). Default: true."
        ),
    )
    ranges = parser.add_mutually_exclusive_group()
    ranges.add_argument(
        "--hours",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help=(
            "Load only this elapsed-hour range from the start of the recording "
            "(for example: --hours 3 6)."
        ),
    )
    ranges.add_argument(
        "--time-range",
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help=(
            "Load a real clock-time range. Quote each value, for example: "
            "--time-range '20260717 03:00:00' '20260717 06:00:00'."
        ),
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    return parser


def settings_from_args(args: argparse.Namespace) -> ScoringViewSettings:
    config = load_config(args.config)
    view = nested_get(config, ("sleep_scoring_view",), {})
    if not isinstance(view, dict):
        raise ValueError("sleep_scoring_view config must be a YAML mapping")

    subject = _single_value(args.subject, option_name="subject")
    if subject is None:
        raise ValueError("a subject is required via --subject")

    if args.hours is not None:
        configured_hours = args.hours
        configured_time_range = None
    elif args.time_range is not None:
        configured_hours = None
        configured_time_range = args.time_range
    else:
        configured_hours = view.get("hours")
        configured_time_range = view.get("time_range")
    if configured_hours is not None and configured_time_range is not None:
        raise ValueError("use either hours or time_range, not both")

    hours = (
        None
        if configured_hours is None
        else tuple(float(value) for value in _values(configured_hours))
    )
    time_range = _parse_time_range(configured_time_range)

    date = _single_value(args.date, option_name="date")
    session_text = _single_value(args.session, option_name="session")
    session = (
        None if session_text is None else parse_sessions([session_text])[0]
    )
    if date is None and session is None and time_range is None:
        raise ValueError(
            "a date or session is required via --date/--session "
            "(or use --time-range)"
        )

    rawdata_root = Path(
        coalesce(args.rawdata_root, get_rawdata_root())
    ).expanduser().resolve(strict=False)
    derivatives_root = Path(
        coalesce(args.derivatives_root, get_derivatives_root())
    ).expanduser().resolve(strict=False)
    repo_root = Path(
        coalesce(args.repo_root, get_repo_root())
    ).expanduser().resolve(strict=False)

    recording_index = int(coalesce(args.recording_index, view.get("recording_index"), 0))
    eeg_channel = int(coalesce(args.eeg_channel, view.get("eeg_channel"), 0))
    view_length_s = float(coalesce(args.view_length, view.get("view_length_s"), 120.0))
    configured_display_rate = coalesce(
        args.display_rate, view.get("display_rate_hz")
    )
    display_rate_hz = (
        None if configured_display_rate is None else float(configured_display_rate)
    )
    show_artifacts = bool(coalesce(args.show_artifacts, view.get("show_artifacts"), False))
    show_gaps = bool(coalesce(args.show_gaps, view.get("show_gaps"), True))
    if recording_index < 0:
        raise ValueError("recording_index must not be negative")
    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")
    if view_length_s <= 0:
        raise ValueError("view_length_s must be positive")
    if display_rate_hz is not None and (
        not math.isfinite(display_rate_hz) or display_rate_hz <= 0
    ):
        raise ValueError("display_rate_hz must be a positive finite number")
    if hours is not None:
        if len(hours) != 2:
            raise ValueError("hours must contain exactly START and END")
        start_hour, end_hour = hours
        if not all(math.isfinite(value) for value in hours):
            raise ValueError("hours START and END must be finite")
        if start_hour < 0:
            raise ValueError("hours START must not be negative")
        if end_hour <= start_hour:
            raise ValueError("hours END must be greater than START")

    return ScoringViewSettings(
        subject=subject,
        date=date,
        repo_root=repo_root,
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
        recording_index=recording_index,
        eeg_channel=eeg_channel,
        view_length_s=view_length_s,
        hours=hours,
        time_range=time_range,
        display_rate_hz=display_rate_hz,
        show_artifacts=show_artifacts,
        show_gaps=show_gaps,
        session=session,
    )


def _downsample_signals(raw_signals: Any, source_hz: float, target_hz: float):
    """Polyphase-resample signals for display, preserving channels and duration."""
    if target_hz > source_hz:
        raise ValueError(
            f"display rate ({target_hz:g} Hz) exceeds EDF rate ({source_hz:g} Hz)"
        )
    if math.isclose(target_hz, source_hz):
        return raw_signals

    from scipy.signal import resample_poly

    ratio = Fraction(target_hz / source_hz).limit_denominator(10_000)
    return resample_poly(raw_signals, ratio.numerator, ratio.denominator, axis=0)


def _artifact_file(recording: Any) -> Path:
    """Resolve artifacts from the dedicated directory, with legacy fallback."""
    scoring_dir = Path(recording.output_dir)
    directories = (scoring_dir.parent / output_dir_name("artifacts"), scoring_dir)
    exact_name = f"{recording.edf_path.stem}_artifact_epochs.parquet"
    for directory in directories:
        exact = directory / exact_name
        if exact.is_file():
            return exact
        generic = directory / "artifact_epochs.parquet"
        if generic.is_file():
            return generic
        matches = sorted(directory.glob("*artifact_epochs.parquet"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple artifact epoch files found in {directory}; "
                f"expected {exact_name}"
            )
    searched = ", ".join(str(path) for path in directories)
    raise FileNotFoundError(f"No artifact_epochs.parquet file found in {searched}")


def _artifact_regions(
    artifact_table: Any,
    selected_start_s: float,
    selected_end_s: float,
) -> list[tuple[float, float]]:
    """Return merged artifact spans relative to the displayed interval."""
    required = {"time_s", "artifact"}
    missing = required - set(artifact_table.columns)
    if missing:
        raise ValueError(f"Artifact parquet is missing columns: {sorted(missing)}")

    times = artifact_table["time_s"].astype(float)
    positive_steps = times.diff().dropna()
    positive_steps = positive_steps[positive_steps > 0]
    epoch_s = float(positive_steps.median()) if len(positive_steps) else 4.0
    flagged = artifact_table.loc[artifact_table["artifact"].astype(bool), "time_s"]

    spans = sorted(
        (
            max(float(time_s), selected_start_s) - selected_start_s,
            min(float(time_s) + epoch_s, selected_end_s) - selected_start_s,
        )
        for time_s in flagged
        if float(time_s) < selected_end_s
        and float(time_s) + epoch_s > selected_start_s
    )
    merged: list[tuple[float, float]] = []
    for start_s, end_s in spans:
        if merged and start_s <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
        else:
            merged.append((start_s, end_s))
    return merged


def _shade_regions(
    fig: Any,
    regions: list[tuple[float, float]],
    *,
    color: str,
    label: str,
    text_color: str | None = None,
) -> None:
    """Shade spans on every viewer axis and label them on the signal axis."""
    if not regions or not fig.axes:
        return
    for axis in fig.axes:
        for start_s, end_s in regions:
            axis.axvspan(start_s, end_s, color=color, alpha=0.16, zorder=0)
    signal_axis = fig.axes[0]
    for start_s, end_s in regions:
        signal_axis.text(
            (start_s + end_s) / 2.0,
            0.98,
            label,
            transform=signal_axis.get_xaxis_transform(),
            color=text_color or color,
            fontsize=8,
            ha="center",
            va="top",
            rotation=90,
        )


def _label_artifacts(fig: Any, regions: list[tuple[float, float]]) -> None:
    """Shade artifact spans on every viewer axis and label the signal axis."""
    _shade_regions(fig, regions, color="red", label="Artifact", text_color="darkred")


def _run_custom_view(settings: ScoringViewSettings) -> int:
    """Display a selected/full recording with range, rate, and artifact controls."""
    from hypnose_somnotate.config import DEFAULT_CHANNEL_LABELS
    from hypnose_somnotate.io.loading import load_somnotate_vector
    from hypnose_somnotate.io.paths import find_recordings
    from hypnose_somnotate.visualization import plot_detailed_comparison
    from pyedflib import EdfReader
    from six import ensure_str

    from scripts.sleep_scoring.score_recordings import _match_channel_labels

    dates = [settings.date] if settings.date is not None else None
    recordings = find_recordings(settings.repo_root, [settings.subject], dates=dates)
    if settings.session is not None:
        recordings = [
            recording
            for recording in recordings
            if parse_sessions([recording.session]) == [settings.session]
        ]
    if not recordings:
        selector_text = (
            f" on date {settings.date}"
            if settings.date is not None
            else f" in session {settings.session}"
            if settings.session is not None
            else ""
        )
        raise ValueError(
            f"No recordings found for subject {settings.subject}{selector_text}"
        )

    if settings.time_range is not None:
        start_time = settings.time_range[0]
        recordings = [
            recording
            for recording in recordings
            if _edf_contains_time(recording.edf_path, start_time, EdfReader)
        ]
        if not recordings:
            scope = (
                f"date {settings.date}"
                if settings.date
                else f"session {settings.session}"
                if settings.session is not None
                else "the subject's recordings"
            )
            raise ValueError(
                f"No EDF in {scope} contains time_range START "
                f"({start_time:%Y%m%d %H:%M:%S})"
            )
    index = min(settings.recording_index, len(recordings) - 1)
    recording = recordings[index]
    pred_path = scoring_path(
        recording.edf_path,
        settings.rawdata_root,
        settings.derivatives_root,
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No somnotate predictions at {pred_path}. Score this recording first."
        )

    channel_labels = list(DEFAULT_CHANNEL_LABELS)
    with EdfReader(str(recording.edf_path)) as reader:
        labels = [
            ensure_str(reader.signal_label(i)).strip()
            for i in range(reader.signals_in_file)
        ]
        channel_labels = _match_channel_labels(
            channel_labels, set(labels), source=recording.edf_path.name
        )
        channel_indices = [labels.index(label) for label in channel_labels]

        sample_rates = [float(reader.getSampleFrequency(i)) for i in channel_indices]
        if len(set(sample_rates)) != 1:
            raise ValueError(
                f"Selected channels have different sample rates: {sample_rates}"
            )
        sampling_rate_hz = sample_rates[0]
        total_samples = min(reader.getNSamples()[i] for i in channel_indices)
        # Subject and session always identify the plot; hours/time_range only add
        # detail about which slice of that recording is shown.
        recording_label = f"{recording.subject} {recording.session} (date {recording.date})"
        if settings.hours is not None or settings.time_range is not None:
            start_s, end_s, range_description, detail = _elapsed_range(
                settings, reader.getStartdatetime()
            )
            title = f"{recording_label} — {detail}"
        else:
            start_s = 0.0
            end_s = total_samples / sampling_rate_hz
            range_description = "the complete recording"
            title = f"{recording_label} — {recording.edf_path.name}"
        first_sample = int(start_s * sampling_rate_hz)
        last_sample = min(int(end_s * sampling_rate_hz), total_samples)
        if first_sample >= total_samples:
            duration_h = total_samples / sampling_rate_hz / 3600.0
            raise ValueError(
                f"selected START is beyond the {duration_h:.3g}-hour recording"
            )

        import numpy as np

        raw_signals = np.column_stack(
            [
                reader.readSignal(i, first_sample, last_sample - first_sample)
                for i in channel_indices
            ]
        )

    plotted_rate_hz = sampling_rate_hz
    if settings.display_rate_hz is not None:
        raw_signals = _downsample_signals(
            raw_signals, sampling_rate_hz, settings.display_rate_hz
        )
        plotted_rate_hz = settings.display_rate_hz

    somnotate_vec = load_somnotate_vector(pred_path)
    # Somnotate predictions are one label per configured annotation interval.
    from hypnose_somnotate.somnotate_pipeline.utils import configuration

    epoch_s = float(configuration.time_resolution)
    first_epoch = int(start_s / epoch_s)
    loaded_end_s = last_sample / sampling_rate_hz
    last_epoch = min(int(loaded_end_s / epoch_s), len(somnotate_vec))
    somnotate_vec = somnotate_vec[first_epoch:last_epoch]
    if len(somnotate_vec) == 0:
        raise ValueError(f"No scoring predictions overlap {range_description}")

    print(
        f"Loading {range_description} from {recording.edf_path.name} "
        f"({len(raw_signals):,} samples at {plotted_rate_hz:g} Hz)\u2026"
    )
    fig, _viewer = plot_detailed_comparison(
        raw_signals,
        sampling_rate_hz=plotted_rate_hz,
        somnotate_vec=somnotate_vec,
        eeg_channel=settings.eeg_channel,
        view_length_s=settings.view_length_s,
    )
    if settings.show_gaps:
        import pandas as pd

        kind_table = pd.read_parquet(pred_path, columns=["time_s", "kind"])
        for kind_value, color, label in (
            ("gap", "dimgray", "Gap"),
            ("too_short", "darkorange", "Too short"),
        ):
            flagged = kind_table.assign(artifact=kind_table["kind"] == kind_value)
            regions = _artifact_regions(flagged[["time_s", "artifact"]], start_s, loaded_end_s)
            _shade_regions(fig, regions, color=color, label=label)
            if regions:
                print(f"Shaded {len(regions)} '{kind_value}' region(s) from the predictions parquet")
    if settings.show_artifacts:
        import pandas as pd

        artifact_path = _artifact_file(recording)
        artifact_table = pd.read_parquet(
            artifact_path, columns=["time_s", "artifact"]
        )
        regions = _artifact_regions(artifact_table, start_s, loaded_end_s)
        _label_artifacts(fig, regions)
        print(f"Labelled {len(regions)} artifact region(s) from {artifact_path.name}")
    fig.suptitle(title)

    import matplotlib.pyplot as plt

    plt.show()
    return 0


def _edf_contains_time(
    edf_path: Path,
    timestamp: datetime,
    reader_class: Any,
) -> bool:
    """Return whether an EDF's real clock span contains ``timestamp``."""
    with reader_class(str(edf_path)) as reader:
        recording_start = reader.getStartdatetime().replace(tzinfo=None)
        sample_counts = reader.getNSamples()
        if len(sample_counts) == 0:
            return False
        sampling_rate_hz = float(reader.getSampleFrequency(0))
        recording_end = recording_start + timedelta(
            seconds=float(sample_counts[0]) / sampling_rate_hz
        )
    return recording_start <= timestamp < recording_end


def _import_view_main() -> Callable[[list[str]], int]:
    try:
        from hypnose_somnotate.cli.view import main as view_main
    except ImportError as exc:
        raise ImportError(
            "The scoring viewer requires hypnose-somnotate and its visualization dependencies."
        ) from exc
    return view_main


def _require_graphical_display(environment: Mapping[str, str]) -> None:
    """Fail clearly before Qt aborts when an SSH session has no display tunnel."""
    if sys.platform.startswith("linux") and not (
        environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "No graphical display is available. Connect with SSH X11 forwarding "
            "and confirm that echo $DISPLAY is non-empty before launching the viewer. "
            f"See {REMOTE_VIEWER_DOC}."
        )


def run_view(
    settings: ScoringViewSettings,
    view_function: Callable[[list[str]], int] | None = None,
) -> int:
    if not settings.rawdata_root.is_dir():
        raise FileNotFoundError(f"Raw-data root not found: {settings.rawdata_root}")
    if not settings.derivatives_root.is_dir():
        raise FileNotFoundError(f"Derivatives root not found: {settings.derivatives_root}")

    _require_graphical_display(os.environ)

    os.environ["HYPNOSE_EEG_RAWDATA_ROOT"] = str(settings.rawdata_root)
    os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"] = str(settings.derivatives_root)

    if view_function is None:
        return _run_custom_view(settings)

    arguments = [
        "--sub",
        settings.subject,
        "--date",
        str(settings.date),
        "--recording-index",
        str(settings.recording_index),
        "--eeg-channel",
        str(settings.eeg_channel),
        "--view-length",
        str(settings.view_length_s),
        "--repo-root",
        str(settings.repo_root),
    ]
    return (view_function or _import_view_main())(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        return run_view(settings)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
