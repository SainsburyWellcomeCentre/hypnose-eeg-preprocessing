"""Plot mean EEG power spectra separated by predicted sleep state."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from hypnose_helpers.io.selectors import parse_subject
from hypnose_helpers.viz.save import save_figure
from hypnose_helpers.viz.styles import ensure_style

try:
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.analysis.emg import compute_state_emg_rms
    from scripts.analysis.power_spectra import compute_state_spectra
    from scripts.io.recording_paths import artifact_path, scoring_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.analysis.emg import compute_state_emg_rms
    from scripts.analysis.power_spectra import compute_state_spectra
    from scripts.io.recording_paths import artifact_path, scoring_path


STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM"}
STATE_COLORS = {0: "red", 1: "royalblue", 2: "goldenrod"}


def plot_state_spectra(
    frequencies: np.ndarray,
    spectra: dict[int, np.ndarray],
    counts: dict[int, int],
    channel_names: list[str],
    *,
    title: str,
):
    ensure_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(channel_names),
        1,
        figsize=(10, max(4, 3.2 * len(channel_names))),
        sharex=True,
        squeeze=False,
    )
    for channel_index, (axis, channel_name) in enumerate(
        zip(axes[:, 0], channel_names)
    ):
        for state, state_name in STATE_NAMES.items():
            if state not in spectra:
                continue
            axis.plot(
                frequencies,
                spectra[state][channel_index],
                color=STATE_COLORS[state],
                label=f"{state_name} ({counts[state]:,} epochs)",
            )
        axis.set_yscale("log")
        axis.set_ylabel("PSD (µV²/Hz)")
        axis.set_title(channel_name)
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_state_emg_rms(
    rms_by_state: dict[int, np.ndarray],
    channel_names: list[str],
    *,
    title: str,
):
    """Plot one EMG RMS histogram per sleep state and EMG channel."""
    ensure_style()
    import matplotlib.pyplot as plt

    states = [state for state in STATE_NAMES if state in rms_by_state]
    fig, axes = plt.subplots(
        len(channel_names),
        len(states),
        figsize=(5 * len(states), max(4, 3.2 * len(channel_names))),
        sharex="row",
        sharey="row",
        squeeze=False,
    )
    for channel_index, channel_name in enumerate(channel_names):
        combined = np.concatenate(
            [rms_by_state[state][:, channel_index] for state in states]
        )
        bin_edges = np.histogram_bin_edges(combined, bins="auto")
        for column_index, state in enumerate(states):
            axis = axes[channel_index, column_index]
            values = rms_by_state[state][:, channel_index]
            axis.hist(
                values,
                bins=bin_edges,
                color=STATE_COLORS[state],
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
            )
            axis.set_title(
                f"{channel_name} — {STATE_NAMES[state]}\n"
                f"{len(values):,} epochs"
            )
            axis.set_xlabel("EMG RMS (µV)")
            if column_index == 0:
                axis.set_ylabel("Epoch count")
            axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument("--epoch-seconds", type=float, default=4.0)
    parser.add_argument("--welch-seconds", type=float, default=2.0)
    parser.add_argument("--fmin", type=float, default=0.5, metavar="HZ")
    parser.add_argument("--fmax", type=float, default=55.0, metavar="HZ")
    parser.add_argument("--chunk-epochs", type=int, default=512)
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact epochs instead of excluding the matching artifact parquet.",
    )
    parser.add_argument("--save-dir", default=None, help="Optionally save PDF plots here.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in (
        ("epoch seconds", args.epoch_seconds),
        ("Welch seconds", args.welch_seconds),
        ("minimum frequency", args.fmin),
        ("maximum frequency", args.fmax),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be a positive finite number")
    if args.fmax <= args.fmin:
        parser.error("--fmax must be greater than --fmin")
    if args.chunk_epochs <= 0:
        parser.error("--chunk-epochs must be positive")
    if args.no_show and args.save_dir is None:
        parser.error("--no-show requires --save-dir")

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(
        args.derivatives_root or get_derivatives_root()
    ).resolve(strict=False)
    pairs = select_recordings(
        rawdata_root,
        derivatives_root,
        subject=args.subject,
        date=args.date,
        session=args.session,
    )

    figures = []
    for edf_path, fif_path in pairs:
        scores = scoring_path(edf_path, rawdata_root, derivatives_root)
        artifacts = (
            None
            if args.include_artifacts
            else artifact_path(edf_path, rawdata_root, derivatives_root)
        )
        print(f"FIF: {fif_path}")
        print(f"Scoring: {scores}")
        print(f"Artifacts: {artifacts or 'included/not available'}")
        frequencies, spectra, counts, channels = compute_state_spectra(
            fif_path,
            scores,
            artifact_path=artifacts,
            epoch_seconds=args.epoch_seconds,
            welch_seconds=args.welch_seconds,
            fmin_hz=args.fmin,
            fmax_hz=args.fmax,
            chunk_epochs=args.chunk_epochs,
        )
        for state, name in STATE_NAMES.items():
            print(f"  {name}: {counts[state]:,} epochs")
        fig = plot_state_spectra(
            frequencies,
            spectra,
            counts,
            channels,
            title=f"{args.subject} — {args.date or args.session} — sleep-state spectra",
        )
        figures.append(fig)
        emg_rms, emg_channels = compute_state_emg_rms(
            fif_path,
            scores,
            artifact_path=artifacts,
            epoch_seconds=args.epoch_seconds,
            chunk_epochs=args.chunk_epochs,
        )
        emg_fig = None
        if emg_channels and emg_rms:
            emg_fig = plot_state_emg_rms(
                emg_rms,
                emg_channels,
                title=(
                    f"{args.subject} — {args.date or args.session} — "
                    "EMG RMS by sleep state"
                ),
            )
            figures.append(emg_fig)
        else:
            print("EMG RMS: skipped (no EMG channels or scored finite epochs)")
        if args.save_dir is not None:
            output_path = save_figure(
                fig,
                f"{edf_path.stem}_sleep_state_power_spectra",
                fig_dir=args.save_dir,
                subjids=parse_subject(args.subject),
                dates=args.date,
            )
            print(f"Saved: {output_path}")
            if emg_fig is not None:
                emg_output_path = save_figure(
                    emg_fig,
                    f"{edf_path.stem}_sleep_state_emg_rms",
                    fig_dir=args.save_dir,
                    subjids=parse_subject(args.subject),
                    dates=args.date,
                )
                print(f"Saved: {emg_output_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
