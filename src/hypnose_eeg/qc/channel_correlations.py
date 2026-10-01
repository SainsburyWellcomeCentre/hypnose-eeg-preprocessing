"""Plot epoch-wise EEG/EMG Pearson correlation by predicted sleep state."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from hypnose_helpers.viz.styles import ensure_style

from hypnose_eeg.io.input_paths import artifact_path, scoring_path
from hypnose_eeg.io.output_paths import quality_control_output_path, save_pdf
from hypnose_eeg.io.repository_paths import resolve_data_roots
from hypnose_eeg.qc.spectra import default_spectra_config, load_spectra_config
from hypnose_eeg.utils.recording_selection import select_recordings
from hypnose_eeg.qc.thresholds import load_qc_thresholds
from hypnose_eeg.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    DEFAULT_SPECTRA_CONFIG_PATH,
    coalesce,
    with_overrides,
)
from hypnose_eeg.utils.sleep_states import default_sleep_states
from hypnose_eeg.analysis.correlation import compute_state_channel_correlations

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _plotted_states() -> dict[int, str]:
    """Scoreable states with a display color (Wake/NREM/REM), by code.

    "Undefined" has no color, so correlation plots exclude it.
    """
    states = default_sleep_states()
    return {
        code: name
        for code, name in states.sleep_state_names.items()
        if code in states.state_colors
    }


def plot_correlation_distributions(
    correlations: pd.DataFrame,
    *,
    title: str,
    bins: int,
):
    """Plot one sleep-state histogram for every channel pair."""
    ensure_style()
    import matplotlib.pyplot as plt

    pairs = list(
        correlations[["channel_1", "channel_2"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    plotted_states = _plotted_states()
    state_colors = default_sleep_states().state_colors
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    fig, axes = plt.subplots(
        len(pairs),
        len(plotted_states),
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
        for column, (state, state_name) in enumerate(plotted_states.items()):
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
                    color=state_colors[state],
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


@dataclass(frozen=True, eq=False)
class CorrelationResult:
    """One recording's epoch-wise channel correlations and the files they came from."""

    edf_path: Path
    fif_path: Path
    scoring_path: Path
    artifact_path: Path | None  # None when artifact epochs were included
    correlations: pd.DataFrame
    channels: list[str]

    @property
    def recording(self) -> str:
        return self.edf_path.stem


def compute_recording_correlations(
    edf_path: Path,
    fif_path: Path,
    *,
    rawdata_root: Path,
    derivatives_root: Path,
    epoch_seconds: float | None = None,
    chunk_epochs: int | None = None,
    include_artifacts: bool = False,
) -> CorrelationResult:
    """Correlate every channel pair per epoch for one EDF/FIF pair.

    `epoch_seconds`/`chunk_epochs` default to the spectra configuration's;
    artifact epochs are excluded unless `include_artifacts`.
    """
    spectra_config = default_spectra_config()
    epoch_seconds = coalesce(epoch_seconds, spectra_config.epoch_seconds)
    chunk_epochs = coalesce(chunk_epochs, spectra_config.chunk_epochs)
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be a positive finite number")
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")
    scores = scoring_path(edf_path, rawdata_root, derivatives_root)
    artifacts = (
        None
        if include_artifacts
        else artifact_path(edf_path, rawdata_root, derivatives_root)
    )
    correlations, channels = compute_state_channel_correlations(
        fif_path,
        scores,
        artifact_path=artifacts,
        epoch_seconds=epoch_seconds,
        chunk_epochs=chunk_epochs,
    )
    return CorrelationResult(
        edf_path=edf_path,
        fif_path=fif_path,
        scoring_path=scores,
        artifact_path=artifacts,
        correlations=correlations,
        channels=channels,
    )


def compute_session_correlations(
    subject: str | int,
    *,
    date: str | int | None = None,
    session: str | int | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    epoch_seconds: float | None = None,
    chunk_epochs: int | None = None,
    include_artifacts: bool = False,
) -> list[CorrelationResult]:
    """Correlate channel pairs for every recording in one subject's session.

    The roots default to the active data-location profile. Count the epochs
    above review thresholds with `correlation_review(result.correlations, ...)`.
    """
    rawdata_root, derivatives_root = resolve_data_roots(rawdata_root, derivatives_root)
    return [
        compute_recording_correlations(
            edf_path,
            fif_path,
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            epoch_seconds=epoch_seconds,
            chunk_epochs=chunk_epochs,
            include_artifacts=include_artifacts,
        )
        for edf_path, fif_path in select_recordings(
            rawdata_root, derivatives_root, subject=subject, date=date, session=session
        )
    ]


def plot_correlations(
    result: CorrelationResult, *, label: str | None = None, bins: int = 40
) -> "Figure":
    """Plot a result's per-pair, per-state correlation histograms."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    return plot_correlation_distributions(
        result.correlations,
        title=f"{label or result.recording} — EEG/EMG Pearson correlation",
        bins=bins,
    )


def save_correlations(
    result: CorrelationResult, save_dir: str | Path, figure: "Figure"
) -> list[Path]:
    """Write a `plot_correlations()` figure as a PDF into `save_dir`."""
    return [
        save_pdf(
            figure, Path(save_dir) / f"{result.recording}_sleep_state_correlations.pdf"
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--spectra-config",
        default=str(DEFAULT_SPECTRA_CONFIG_PATH),
        help=f"Spectra pipeline YAML (default: {DEFAULT_SPECTRA_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--epoch-seconds",
        type=float,
        default=None,
        help="Analysis epoch duration (default: from --spectra-config).",
    )
    parser.add_argument(
        "--chunk-epochs",
        type=int,
        default=None,
        help="FIF read chunk size in epochs (default: from --spectra-config).",
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--qc-config",
        default=str(DEFAULT_QUALITY_CONTROL_CONFIG_PATH),
        help=f"Quality-control threshold YAML (default: {DEFAULT_QUALITY_CONTROL_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--eeg-eeg-threshold",
        type=float,
        default=None,
        help="Review EEG–EEG epochs with absolute r above this value "
        "(default: from --qc-config).",
    )
    parser.add_argument(
        "--eeg-emg-threshold",
        type=float,
        default=None,
        help="Review EEG–EMG epochs with absolute r above this value "
        "(default: from --qc-config).",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact epochs instead of excluding the matching parquet.",
    )
    parser.add_argument(
        "--save-dir",
        nargs="?",
        const=".",
        default=None,
        help="Optionally save PDFs in the shared session QC directory.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    return parser


def _print_summary(correlations: pd.DataFrame) -> None:
    grouped = correlations.groupby(
        ["channel_1", "channel_2", "sleep_state"], sort=False
    )["pearson_r"]
    for (channel_1, channel_2, state), values in grouped:
        lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
        print(
            f"  {channel_1} ↔ {channel_2} — {_plotted_states()[int(state)]}: "
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


@dataclass(frozen=True, eq=False)
class CorrelationReview:
    """Epochs whose absolute Pearson r exceeds its pair type's review threshold."""

    # One row per channel pair and sleep state: channel_1, channel_2,
    # sleep_state (name), flagged_epochs, epochs, percent, threshold.
    by_pair: pd.DataFrame
    flagged_epochs: int  # unique epochs above any pair's threshold
    valid_epochs: int

    @property
    def percent(self) -> float:
        return 100.0 * self.flagged_epochs / self.valid_epochs if self.valid_epochs else 0.0


def correlation_review(
    correlations: pd.DataFrame,
    *,
    eeg_eeg_threshold: float,
    eeg_emg_threshold: float,
) -> CorrelationReview:
    """Count the epochs above review thresholds, per pair/state and overall.

    Pairs other than EEG–EEG and EEG–EMG have no threshold and are left out.
    """
    rows: list[dict[str, object]] = []
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
    ), pair_rows in grouped:
        threshold = _review_threshold(
            str(channel_1_type),
            str(channel_2_type),
            eeg_eeg_threshold=eeg_eeg_threshold,
            eeg_emg_threshold=eeg_emg_threshold,
        )
        if threshold is None:
            continue
        flagged = pair_rows["pearson_r"].abs() > threshold
        flagged_count = int(flagged.sum())
        flagged_epoch_ids.update(pair_rows.loc[flagged, "epoch_id"].astype(int))
        rows.append(
            {
                "channel_1": channel_1,
                "channel_2": channel_2,
                "sleep_state": _plotted_states()[int(state)],
                "flagged_epochs": flagged_count,
                "epochs": len(pair_rows),
                "percent": 100.0 * flagged_count / len(pair_rows),
                "threshold": threshold,
            }
        )
    return CorrelationReview(
        by_pair=pd.DataFrame(rows, columns=_REVIEW_COLUMNS),
        flagged_epochs=len(flagged_epoch_ids),
        valid_epochs=int(correlations["epoch_id"].nunique()),
    )


_REVIEW_COLUMNS = [
    "channel_1",
    "channel_2",
    "sleep_state",
    "flagged_epochs",
    "epochs",
    "percent",
    "threshold",
]


def _print_review_counts(
    correlations: pd.DataFrame,
    *,
    eeg_eeg_threshold: float,
    eeg_emg_threshold: float,
) -> None:
    """Print pair/state counts and unique epochs exceeding review thresholds."""
    review = correlation_review(
        correlations,
        eeg_eeg_threshold=eeg_eeg_threshold,
        eeg_emg_threshold=eeg_emg_threshold,
    )
    print("Review-threshold epochs (absolute Pearson r):")
    for row in review.by_pair.itertuples(index=False):
        print(
            f"  {row.channel_1} ↔ {row.channel_2} — {row.sleep_state}: "
            f"{row.flagged_epochs:,} / {row.epochs:,} ({row.percent:.2f}%) "
            f"above |r| > {row.threshold:.2f}"
        )
    print(
        f"  Unique epochs above any configured threshold: "
        f"{review.flagged_epochs:,} / {review.valid_epochs:,} ({review.percent:.2f}%)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        thresholds = with_overrides(
            load_qc_thresholds(args.qc_config),
            eeg_eeg_threshold=args.eeg_eeg_threshold,
            eeg_emg_threshold=args.eeg_emg_threshold,
        )
        spectra_config = with_overrides(
            load_spectra_config(args.spectra_config),
            epoch_seconds=args.epoch_seconds,
            chunk_epochs=args.chunk_epochs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    if args.no_show and args.save_dir is None:
        parser.error("--no-show requires --save-dir")

    rawdata_root, derivatives_root = resolve_data_roots(
        args.rawdata_root, args.derivatives_root
    )
    pairs = select_recordings(
        rawdata_root,
        derivatives_root,
        subject=args.subject,
        date=args.date,
        session=args.session,
    )

    for edf_path, fif_path in pairs:
        print(f"FIF: {fif_path}")
        result = compute_recording_correlations(
            edf_path,
            fif_path,
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            epoch_seconds=spectra_config.epoch_seconds,
            chunk_epochs=spectra_config.chunk_epochs,
            include_artifacts=args.include_artifacts,
        )
        print(f"Scoring: {result.scoring_path}")
        print(f"Artifacts: {result.artifact_path or 'included/not available'}")
        print(f"Channels: {', '.join(result.channels)}")
        _print_summary(result.correlations)
        _print_review_counts(
            result.correlations,
            eeg_eeg_threshold=thresholds.eeg_eeg_threshold,
            eeg_emg_threshold=thresholds.eeg_emg_threshold,
        )
        figure = plot_correlations(
            result, label=f"{args.subject} — {args.date or args.session}", bins=args.bins
        )
        if args.save_dir is not None:
            save_dir = quality_control_output_path(
                args.save_dir, edf_path, rawdata_root, derivatives_root
            )
            save_correlations(result, save_dir, figure)

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
