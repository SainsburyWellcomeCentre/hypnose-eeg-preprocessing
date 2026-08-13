"""Plot mean EEG power spectra separated by predicted sleep state."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from hypnose_helpers.io.selectors import parse_subject
from hypnose_helpers.viz.save import save_figure
from hypnose_helpers.viz.styles import ensure_style

try:
    from scripts.io.input_paths import artifact_path, scoring_path
    from scripts.io.output_paths import quality_control_output_path
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.analysis.emg import compute_state_emg_rms
    from scripts.analysis.power_spectra import compute_state_spectra
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.input_paths import artifact_path, scoring_path
    from scripts.io.output_paths import quality_control_output_path
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.analysis.emg import compute_state_emg_rms
    from scripts.analysis.power_spectra import compute_state_spectra


STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM"}
STATE_COLORS = {0: "red", 1: "royalblue", 2: "goldenrod"}
QUALITY_ORDER = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
FREQUENCY_BANDS_HZ = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


def _integrated_power(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if mask.sum() < 2:
        return np.full(spectrum.shape[0], np.nan)
    return np.trapz(spectrum[:, mask], frequencies[mask], axis=-1)


def build_spectral_quality_report(
    frequencies: np.ndarray,
    spectra: dict[int, np.ndarray],
    eeg_counts: dict[int, int],
    eeg_channels: list[str],
    emg_rms: dict[int, np.ndarray],
    emg_channels: list[str],
) -> pd.DataFrame:
    """Assess state-wise EEG band-power expectations and EMG RMS ordering."""
    rows: list[dict[str, object]] = []
    state_metrics: dict[int, dict[str, float | bool | int]] = {}
    for state, state_name in STATE_NAMES.items():
        eeg_epoch_count = int(eeg_counts.get(state, 0))
        spectrum = spectra.get(state)
        spectrum_finite = bool(
            spectrum is not None
            and spectrum.size
            and np.all(np.isfinite(spectrum))
        )
        channel_power = np.array([], dtype=float)
        relative_band_power = {band: math.nan for band in FREQUENCY_BANDS_HZ}
        if spectrum_finite and len(frequencies) > 1:
            channel_power = np.trapz(spectrum, frequencies, axis=-1)
            for band, (low_hz, high_hz) in FREQUENCY_BANDS_HZ.items():
                power = _integrated_power(spectrum, frequencies, low_hz, high_hz)
                valid = np.isfinite(power) & np.isfinite(channel_power) & (channel_power > 0)
                if np.any(valid):
                    relative_band_power[band] = float(
                        np.median(power[valid] / channel_power[valid])
                    )
        total_power_median = (
            float(np.median(channel_power)) if channel_power.size else math.nan
        )

        state_rms = emg_rms.get(state)
        emg_epoch_count = int(len(state_rms)) if state_rms is not None else 0
        rms_values = (
            np.asarray(state_rms, dtype=float).reshape(-1)
            if state_rms is not None
            else np.array([], dtype=float)
        )
        rms_finite = rms_values[np.isfinite(rms_values)]
        emg_finite_fraction = (
            float(len(rms_finite) / len(rms_values)) if len(rms_values) else math.nan
        )
        emg_median = float(np.median(rms_finite)) if len(rms_finite) else math.nan

        state_metrics[state] = {
            "eeg_epoch_count": eeg_epoch_count,
            "spectrum_finite": spectrum_finite,
            "total_power_median": total_power_median,
            "emg_epoch_count": emg_epoch_count,
            "emg_finite_fraction": emg_finite_fraction,
            "emg_median": emg_median,
            **relative_band_power,
        }

        rows.append(
            {
                "sleep_state": state_name,
                "sleep_state_code": state,
                "eeg_epoch_count": eeg_epoch_count,
                "eeg_channel_count": len(eeg_channels),
                "eeg_spectrum_finite": spectrum_finite,
                "eeg_total_power_median_uv2": total_power_median,
                **{
                    f"eeg_{band}_relative_power": relative_band_power[band]
                    for band in FREQUENCY_BANDS_HZ
                },
                "eeg_theta_delta_ratio": (
                    relative_band_power["theta"] / relative_band_power["delta"]
                    if relative_band_power["delta"] > 0
                    else math.nan
                ),
                "emg_epoch_count": emg_epoch_count,
                "emg_channel_count": len(emg_channels),
                "emg_rms_finite_fraction": emg_finite_fraction,
                "emg_rms_mean_uv": (
                    float(np.mean(rms_finite)) if len(rms_finite) else math.nan
                ),
                "emg_rms_median_uv": emg_median,
                "emg_rms_p05_uv": (
                    float(np.quantile(rms_finite, 0.05))
                    if len(rms_finite)
                    else math.nan
                ),
                "emg_rms_p95_uv": (
                    float(np.quantile(rms_finite, 0.95))
                    if len(rms_finite)
                    else math.nan
                ),
            }
        )

    report = pd.DataFrame(rows)
    report["expected_frequency_pattern"] = [
        "Wake beta relative power >= NREM",
        "NREM delta relative power >= Wake and REM",
        "REM theta/delta ratio >= NREM",
    ]
    wake, nrem, rem = (state_metrics[state] for state in STATE_NAMES)
    frequency_checks = {
        0: float(wake["beta"]) >= float(nrem["beta"]),
        1: float(nrem["delta"]) >= max(float(wake["delta"]), float(rem["delta"])),
        2: (
            float(rem["theta"]) / float(rem["delta"])
            >= float(nrem["theta"]) / float(nrem["delta"])
            if float(rem["delta"]) > 0 and float(nrem["delta"]) > 0
            else False
        ),
    }
    emg_checks = {
        0: float(wake["emg_median"]) >= max(
            float(nrem["emg_median"]), float(rem["emg_median"])
        ),
        1: float(wake["emg_median"]) >= float(nrem["emg_median"]) >= float(rem["emg_median"]),
        2: float(rem["emg_median"]) <= min(
            float(wake["emg_median"]), float(nrem["emg_median"])
        ),
    }

    statuses: list[str] = []
    reasons: list[str] = []
    spectral_statuses: list[str] = []
    emg_statuses: list[str] = []
    for state in STATE_NAMES:
        metrics = state_metrics[state]
        required_band_values = (
            (metrics["beta"], nrem["beta"])
            if state == 0
            else (metrics["delta"], wake["delta"], rem["delta"])
            if state == 1
            else (metrics["theta"], metrics["delta"], nrem["theta"], nrem["delta"])
        )

        if int(metrics["eeg_epoch_count"]) == 0:
            spectral_status = "REVIEW"
            spectral_reason = "No analyzable EEG epochs for this sleep state"
        elif not bool(metrics["spectrum_finite"]) or not math.isfinite(
            float(metrics["total_power_median"])
        ):
            spectral_status = "FAIL"
            spectral_reason = "EEG spectrum contains invalid values"
        elif float(metrics["total_power_median"]) <= 0:
            spectral_status = "FAIL"
            spectral_reason = "EEG total power is zero or negative"
        elif not all(math.isfinite(float(value)) for value in required_band_values):
            spectral_status = "REVIEW"
            spectral_reason = "Required frequency bands are outside the analyzed range"
        elif not frequency_checks[state]:
            spectral_status = "REVIEW"
            spectral_reason = "Expected sleep-state frequency pattern was not present"
        else:
            spectral_status = "PASS"
            spectral_reason = "Expected sleep-state frequency pattern was present"

        if not emg_channels or int(metrics["emg_epoch_count"]) == 0:
            emg_status = "REVIEW"
            emg_reason = "No analyzable EMG RMS epochs for this sleep state"
        elif float(metrics["emg_finite_fraction"]) < 1.0:
            emg_status = "FAIL"
            emg_reason = "EMG RMS contains invalid values"
        elif float(metrics["emg_median"]) <= 0:
            emg_status = "FAIL"
            emg_reason = "EMG RMS is zero or negative"
        elif not emg_checks[state]:
            emg_status = "REVIEW"
            emg_reason = "Expected Wake >= NREM >= REM EMG RMS ordering was not present"
        else:
            emg_status = "PASS"
            emg_reason = "Expected Wake >= NREM >= REM EMG RMS ordering was present"

        status = max(
            (spectral_status, emg_status), key=lambda value: QUALITY_ORDER[value]
        )
        spectral_statuses.append(spectral_status)
        emg_statuses.append(emg_status)
        statuses.append(status)
        reasons.append(f"EEG: {spectral_reason}; EMG: {emg_reason}")

    report["frequency_expectation_met"] = [frequency_checks[state] for state in STATE_NAMES]
    report["spectral_quality_status"] = spectral_statuses
    report["emg_expectation_met"] = [emg_checks[state] for state in STATE_NAMES]
    report["emg_quality_status"] = emg_statuses
    report["quality_status"] = statuses
    report["quality_reason"] = reasons
    recording_status = max(
        report["quality_status"], key=lambda status: QUALITY_ORDER[str(status)]
    )
    report.insert(0, "recording_quality_status", recording_status)
    return report


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
    parser.add_argument(
        "--save-dir",
        nargs="?",
        const=".",
        default=None,
        help="Optionally save PDFs and a quality CSV in the session QC directory.",
    )
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
        quality_report = build_spectral_quality_report(
            frequencies, spectra, counts, channels, emg_rms, emg_channels
        )
        print(
            "Spectral quality: "
            f"{quality_report['recording_quality_status'].iloc[0]}"
        )
        for row in quality_report.itertuples(index=False):
            print(
                f"  {row.sleep_state}: {row.quality_status} — "
                f"{row.quality_reason}"
            )
        if args.save_dir is not None:
            save_dir = quality_control_output_path(
                args.save_dir, edf_path, rawdata_root, derivatives_root
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            quality_output_path = (
                save_dir / f"{edf_path.stem}_sleep_state_spectral_quality.csv"
            )
            quality_report.to_csv(quality_output_path, index=False)
            print(f"Saved: {quality_output_path}")
            output_path = save_figure(
                fig,
                f"{edf_path.stem}_sleep_state_power_spectra",
                fig_dir=save_dir,
                subjids=parse_subject(args.subject),
                dates=args.date,
            )
            print(f"Saved: {output_path}")
            if emg_fig is not None:
                emg_output_path = save_figure(
                    emg_fig,
                    f"{edf_path.stem}_sleep_state_emg_rms",
                    fig_dir=save_dir,
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
