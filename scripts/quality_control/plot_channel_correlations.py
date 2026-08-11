"""Plot epoch-wise EEG/EMG Pearson correlation by predicted sleep state."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.utils.correlation import compute_state_channel_correlations
    from scripts.utils.recording_paths import artifact_path, scoring_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.utils.correlation import compute_state_channel_correlations
    from scripts.utils.recording_paths import artifact_path, scoring_path


STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM"}
STATE_COLORS = {0: "red", 1: "royalblue", 2: "goldenrod"}


def plot_correlation_distributions(
    correlations: pd.DataFrame,
    *,
    title: str,
    bins: int,
):
    """Plot one sleep-state histogram for every channel pair."""
    import matplotlib.pyplot as plt

    pairs = list(
        correlations[["channel_1", "channel_2"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    fig, axes = plt.subplots(
        len(pairs),
        len(STATE_NAMES),
        figsize=(15, max(4, 3.1 * len(pairs))),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row, (channel_1, channel_2) in enumerate(pairs):
        pair_rows = correlations.loc[
            (correlations["channel_1"] == channel_1)
            & (correlations["channel_2"] == channel_2)
        ]
        for column, (state, state_name) in enumerate(STATE_NAMES.items()):
            axis = axes[row, column]
            values = pair_rows.loc[
                pair_rows["sleep_state"] == state, "pearson_r"
            ].to_numpy()
            if len(values):
                weights = np.full(len(values), 100.0 / len(values))
                axis.hist(
                    values,
                    bins=bin_edges,
                    weights=weights,
                    color=STATE_COLORS[state],
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.35,
                )
                axis.axvline(
                    np.median(values), color="black", linestyle="--", linewidth=1
                )
            axis.set_title(f"{state_name} ({len(values):,} epochs)")
            axis.set_xlabel("Pearson r")
            if column == 0:
                axis.set_ylabel(f"{channel_1} ↔ {channel_2}\nEpochs (%)")
            axis.set_xlim(-1.0, 1.0)
            axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
    parser.add_argument("--chunk-epochs", type=int, default=512)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--eeg-eeg-threshold",
        type=float,
        default=0.90,
        help="Review EEG–EEG epochs with absolute r above this value.",
    )
    parser.add_argument(
        "--eeg-emg-threshold",
        type=float,
        default=0.50,
        help="Review EEG–EMG epochs with absolute r above this value.",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact epochs instead of excluding the matching parquet.",
    )
    parser.add_argument("--save-dir", default=None, help="Optionally save PNG plots here.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    return parser


def _print_summary(correlations: pd.DataFrame) -> None:
    grouped = correlations.groupby(
        ["channel_1", "channel_2", "sleep_state"], sort=False
    )["pearson_r"]
    for (channel_1, channel_2, state), values in grouped:
        lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
        print(
            f"  {channel_1} ↔ {channel_2} — {STATE_NAMES[int(state)]}: "
            f"n={len(values):,}, median={median:.3f}, IQR={lower:.3f} to {upper:.3f}"
        )


def _review_threshold(
    channel_1_type: str,
    channel_2_type: str,
    *,
    eeg_eeg_threshold: float,
    eeg_emg_threshold: float,
) -> float | None:
    pair_types = {channel_1_type.lower(), channel_2_type.lower()}
    if pair_types == {"eeg"}:
        return eeg_eeg_threshold
    if pair_types == {"eeg", "emg"}:
        return eeg_emg_threshold
    return None


def _print_review_counts(
    correlations: pd.DataFrame,
    *,
    eeg_eeg_threshold: float,
    eeg_emg_threshold: float,
) -> None:
    """Print pair/state counts and unique epochs exceeding review thresholds."""
    print("Review-threshold epochs (absolute Pearson r):")
    flagged_epoch_ids: set[int] = set()
    grouped = correlations.groupby(
        [
            "channel_1",
            "channel_2",
            "channel_1_type",
            "channel_2_type",
            "sleep_state",
        ],
        sort=False,
    )
    for (
        channel_1,
        channel_2,
        channel_1_type,
        channel_2_type,
        state,
    ), rows in grouped:
        threshold = _review_threshold(
            str(channel_1_type),
            str(channel_2_type),
            eeg_eeg_threshold=eeg_eeg_threshold,
            eeg_emg_threshold=eeg_emg_threshold,
        )
        if threshold is None:
            continue
        flagged = rows["pearson_r"].abs() > threshold
        flagged_count = int(flagged.sum())
        flagged_epoch_ids.update(rows.loc[flagged, "epoch_id"].astype(int))
        percentage = 100.0 * flagged_count / len(rows)
        print(
            f"  {channel_1} ↔ {channel_2} — {STATE_NAMES[int(state)]}: "
            f"{flagged_count:,} / {len(rows):,} ({percentage:.2f}%) "
            f"above |r| > {threshold:.2f}"
        )
    valid_epoch_count = correlations["epoch_id"].nunique()
    percentage = 100.0 * len(flagged_epoch_ids) / valid_epoch_count
    print(
        f"  Unique epochs above any configured threshold: "
        f"{len(flagged_epoch_ids):,} / {valid_epoch_count:,} ({percentage:.2f}%)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.epoch_seconds) or args.epoch_seconds <= 0:
        parser.error("--epoch-seconds must be a positive finite number")
    if args.chunk_epochs <= 0:
        parser.error("--chunk-epochs must be positive")
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    for option, threshold in (
        ("--eeg-eeg-threshold", args.eeg_eeg_threshold),
        ("--eeg-emg-threshold", args.eeg_emg_threshold),
    ):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            parser.error(f"{option} must be between 0 and 1")
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
        correlations, channels = compute_state_channel_correlations(
            fif_path,
            scores,
            artifact_path=artifacts,
            epoch_seconds=args.epoch_seconds,
            chunk_epochs=args.chunk_epochs,
        )
        print(f"Channels: {', '.join(channels)}")
        _print_summary(correlations)
        _print_review_counts(
            correlations,
            eeg_eeg_threshold=args.eeg_eeg_threshold,
            eeg_emg_threshold=args.eeg_emg_threshold,
        )
        figure = plot_correlation_distributions(
            correlations,
            title=(
                f"{args.subject} — {args.date or args.session} — "
                "EEG/EMG Pearson correlation"
            ),
            bins=args.bins,
        )
        figures.append(figure)
        if args.save_dir is not None:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"{edf_path.stem}_sleep_state_correlations.png"
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {output_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
