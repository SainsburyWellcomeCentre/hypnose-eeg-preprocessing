"""Plot mean EEG power spectra separated by predicted sleep state."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from hypnose_helpers.io.selectors import parse_subject
from hypnose_helpers.viz.save import save_figure
from hypnose_helpers.viz.styles import ensure_style

from hypnose_eeg.io.input_paths import artifact_path, scoring_path
from hypnose_eeg.io.output_paths import quality_control_output_path, save_csv
from hypnose_eeg.io.repository_paths import get_derivatives_root, get_rawdata_root
from hypnose_eeg.utils.recording_selection import select_recordings
from hypnose_eeg.analysis.emg import compute_state_emg_rms
from hypnose_eeg.analysis.power_spectra import compute_state_spectra, integrated_power
from hypnose_eeg.qc.thresholds import load_performance_check
from hypnose_eeg.utils.config import (
    DEFAULT_SPECTRA_CONFIG_PATH,
    load_config,
    nested_get,
    two_float_tuple,
)


# Spectral/EMG quality statuses are upper-cased ("PASS"/"REVIEW"/"FAIL").
PERFORMANCE_CHECK = {
    status.upper(): rank for status, rank in load_performance_check().items()
}


@dataclass(frozen=True)
class SpectraConfig:
    epoch_seconds: float
    welch_seconds: float
    fmin_hz: float
    fmax_hz: float
    chunk_epochs: int
    include_artifacts: bool
    frequency_bands_hz: dict[str, tuple[float, float]]
    sleep_state_names: dict[int, str]
    state_colors: dict[int, str]
    wake_code: int
    nrem_code: int
    rem_code: int
    determining_eeg_channel_number: int
    wake_delta_max_nrem_ratio: float
    nrem_delta_min_wake_ratio: float
    nrem_delta_min_rem_ratio: float
    rem_theta_delta_min_nrem_ratio: float
    emg_state_order: tuple[int, ...]


def load_spectra_config(path: str | Path = DEFAULT_SPECTRA_CONFIG_PATH) -> SpectraConfig:
    """Load and validate spectral analysis and quality definitions."""
    config = load_config(path)
    analysis = nested_get(config, ("spectra", "analysis"))
    bands = nested_get(config, ("spectra", "frequency_bands_hz"))
    names = nested_get(config, ("spectra", "sleep_state_names"))
    colors = nested_get(config, ("spectra", "state_colors"))
    quality = nested_get(config, ("spectra", "quality"))
    if not all(
        isinstance(section, Mapping)
        for section in (analysis, bands, names, colors, quality)
    ):
        raise ValueError(
            "Spectra config requires analysis, frequency bands, sleep state names, "
            "state colors, and quality mappings"
        )

    sleep_state_names = {int(code): str(label) for code, label in names.items()}
    state_colors = {int(code): str(color) for code, color in colors.items()}
    if set(sleep_state_names) != set(state_colors):
        raise ValueError("sleep_state_names and state_colors must define the same codes")

    frequency_bands = {
        str(name): two_float_tuple(value, f"{name} frequency band")
        for name, value in bands.items()
    }
    required_bands = {"delta", "theta", "alpha", "beta"}
    if required_bands != set(frequency_bands):
        raise ValueError(f"Frequency bands must be exactly {sorted(required_bands)}")
    for name, (low_hz, high_hz) in frequency_bands.items():
        if low_hz < 0 or high_hz <= low_hz:
            raise ValueError(f"Invalid {name} frequency band: [{low_hz}, {high_hz}]")

    wake_code = int(quality["wake_code"])
    nrem_code = int(quality["nrem_code"])
    rem_code = int(quality["rem_code"])
    codes = {wake_code, nrem_code, rem_code}
    if len(codes) != 3:
        raise ValueError("wake_code, nrem_code, and rem_code must be distinct")
    if codes != set(sleep_state_names):
        raise ValueError(
            "wake_code/nrem_code/rem_code must match the codes defined in "
            "sleep_state_names"
        )

    emg_order_value = quality["emg_state_order"]
    if isinstance(emg_order_value, str):
        emg_order_value = emg_order_value.strip("[]").split(",")
    emg_order = tuple(int(value) for value in emg_order_value)
    if set(emg_order) != codes or len(emg_order) != len(codes):
        raise ValueError("emg_state_order must contain every configured state exactly once")
    determining_channel = int(quality["determining_eeg_channel_number"])
    if determining_channel < 1:
        raise ValueError("determining_eeg_channel_number must be at least 1")
    return SpectraConfig(
        epoch_seconds=float(analysis["epoch_seconds"]),
        welch_seconds=float(analysis["welch_seconds"]),
        fmin_hz=float(analysis["minimum_frequency_hz"]),
        fmax_hz=float(analysis["maximum_frequency_hz"]),
        chunk_epochs=int(analysis["chunk_epochs"]),
        include_artifacts=bool(analysis["include_artifacts"]),
        frequency_bands_hz=frequency_bands,
        sleep_state_names=sleep_state_names,
        state_colors=state_colors,
        wake_code=wake_code,
        nrem_code=nrem_code,
        rem_code=rem_code,
        determining_eeg_channel_number=determining_channel,
        wake_delta_max_nrem_ratio=float(quality["wake_delta_max_nrem_ratio"]),
        nrem_delta_min_wake_ratio=float(quality["nrem_delta_min_wake_ratio"]),
        nrem_delta_min_rem_ratio=float(quality["nrem_delta_min_rem_ratio"]),
        rem_theta_delta_min_nrem_ratio=float(quality["rem_theta_delta_min_nrem_ratio"]),
        emg_state_order=emg_order,
    )


DEFAULT_SPECTRA_CONFIG = load_spectra_config()


def build_spectral_quality_report(
    frequencies: np.ndarray,
    spectra: dict[int, np.ndarray],
    eeg_counts: dict[int, int],
    eeg_channels: list[str],
    emg_rms: dict[int, np.ndarray],
    emg_channels: list[str],
    *,
    config: SpectraConfig = DEFAULT_SPECTRA_CONFIG,
) -> pd.DataFrame:
    """Assess band-power quality per channel using configured state expectations."""
    if not eeg_channels:
        raise ValueError("At least one EEG channel is required for spectral quality")
    if config.determining_eeg_channel_number > len(eeg_channels):
        raise ValueError(
            "Configured determining EEG channel is not present: "
            f"{config.determining_eeg_channel_number} > {len(eeg_channels)}"
        )
    emg_metrics: dict[int, dict[str, float | int]] = {}
    for state in config.sleep_state_names:
        values = emg_rms.get(state)
        epoch_count = int(len(values)) if values is not None else 0
        flattened = (
            np.asarray(values, dtype=float).reshape(-1)
            if values is not None
            else np.array([], dtype=float)
        )
        finite = flattened[np.isfinite(flattened)]
        emg_metrics[state] = {
            "epoch_count": epoch_count,
            "finite_fraction": (
                float(len(finite) / len(flattened)) if len(flattened) else math.nan
            ),
            "mean": float(np.mean(finite)) if len(finite) else math.nan,
            "median": float(np.median(finite)) if len(finite) else math.nan,
            "p05": float(np.quantile(finite, 0.05)) if len(finite) else math.nan,
            "p95": float(np.quantile(finite, 0.95)) if len(finite) else math.nan,
        }

    ordered_emg = [float(emg_metrics[state]["median"]) for state in config.emg_state_order]
    emg_order_met = all(
        first >= second for first, second in zip(ordered_emg, ordered_emg[1:])
    )
    emg_checks = {state: emg_order_met for state in config.sleep_state_names}

    channel_metrics: dict[tuple[int, int], dict[str, float | bool | int]] = {}
    for state in config.sleep_state_names:
        spectrum = spectra.get(state)
        for channel_index in range(len(eeg_channels)):
            channel_spectrum = (
                spectrum[channel_index : channel_index + 1]
                if spectrum is not None and channel_index < len(spectrum)
                else None
            )
            finite = bool(
                channel_spectrum is not None
                and channel_spectrum.size
                and np.all(np.isfinite(channel_spectrum))
            )
            total_power = math.nan
            relative = {band: math.nan for band in config.frequency_bands_hz}
            if finite and len(frequencies) > 1:
                total_power = float(np.trapz(channel_spectrum[0], frequencies))
                if total_power > 0:
                    for band, (low_hz, high_hz) in config.frequency_bands_hz.items():
                        power = integrated_power(
                            channel_spectrum, frequencies, low_hz, high_hz
                        )[0]
                        if np.isfinite(power):
                            relative[band] = float(power / total_power)
            channel_metrics[state, channel_index] = {
                "epoch_count": int(eeg_counts.get(state, 0)),
                "finite": finite,
                "total_power": total_power,
                **relative,
            }

    rows: list[dict[str, object]] = []
    expected_patterns = {
        config.wake_code: "Wake delta relative power < configured NREM ratio",
        config.nrem_code: "NREM delta relative power > configured Wake and REM ratios",
        config.rem_code: "REM theta/delta ratio > configured NREM ratio",
    }
    for state, state_name in config.sleep_state_names.items():
        emg = emg_metrics[state]
        for channel_index, channel_name in enumerate(eeg_channels):
            metrics = channel_metrics[state, channel_index]
            wake = channel_metrics[config.wake_code, channel_index]
            nrem = channel_metrics[config.nrem_code, channel_index]
            rem = channel_metrics[config.rem_code, channel_index]
            frequency_check = (
                float(wake["delta"])
                < config.wake_delta_max_nrem_ratio * float(nrem["delta"])
                if state == config.wake_code
                else float(nrem["delta"])
                > max(
                    config.nrem_delta_min_wake_ratio * float(wake["delta"]),
                    config.nrem_delta_min_rem_ratio * float(rem["delta"]),
                )
                if state == config.nrem_code
                else (
                    float(rem["theta"]) / float(rem["delta"])
                    > config.rem_theta_delta_min_nrem_ratio
                    * float(nrem["theta"])
                    / float(nrem["delta"])
                    if float(rem["delta"]) > 0 and float(nrem["delta"]) > 0
                    else False
                )
            )
            required = (
                (wake["delta"], nrem["delta"])
                if state == config.wake_code
                else (nrem["delta"], wake["delta"], rem["delta"])
                if state == config.nrem_code
                else (rem["theta"], rem["delta"], nrem["theta"], nrem["delta"])
            )

            if int(metrics["epoch_count"]) == 0:
                spectral_status = "REVIEW"
                spectral_reason = "No analyzable EEG epochs for this sleep state"
            elif not bool(metrics["finite"]) or not math.isfinite(
                float(metrics["total_power"])
            ):
                spectral_status = "FAIL"
                spectral_reason = "EEG spectrum contains invalid values"
            elif float(metrics["total_power"]) <= 0:
                spectral_status = "FAIL"
                spectral_reason = "EEG total power is zero or negative"
            elif not all(math.isfinite(float(value)) for value in required):
                spectral_status = "REVIEW"
                spectral_reason = "Required frequency bands are unavailable"
            elif not frequency_check:
                spectral_status = "REVIEW"
                spectral_reason = "Expected sleep-state frequency pattern was not present"
            else:
                spectral_status = "PASS"
                spectral_reason = "Expected sleep-state frequency pattern was present"

            if not emg_channels or int(emg["epoch_count"]) == 0:
                emg_status = "REVIEW"
                emg_reason = "No analyzable EMG RMS epochs for this sleep state"
            elif float(emg["finite_fraction"]) < 1.0:
                emg_status = "FAIL"
                emg_reason = "EMG RMS contains invalid values"
            elif float(emg["median"]) <= 0:
                emg_status = "FAIL"
                emg_reason = "EMG RMS is zero or negative"
            elif not emg_checks[state]:
                emg_status = "REVIEW"
                emg_reason = "Expected Wake >= NREM >= REM EMG ordering was absent"
            else:
                emg_status = "PASS"
                emg_reason = "Expected Wake >= NREM >= REM EMG ordering was present"

            quality_status = max(
                (spectral_status, emg_status), key=lambda value: PERFORMANCE_CHECK[value]
            )
            rows.append(
                {
                    "sleep_state": state_name,
                    "sleep_state_code": state,
                    "eeg_channel": channel_name,
                    "eeg_channel_number": channel_index + 1,
                    "determines_recording_quality": (
                        channel_index + 1 == config.determining_eeg_channel_number
                    ),
                    "eeg_epoch_count": metrics["epoch_count"],
                    "eeg_spectrum_finite": metrics["finite"],
                    "eeg_total_power_uv2": metrics["total_power"],
                    **{
                        f"eeg_{band}_relative_power": metrics[band]
                        for band in config.frequency_bands_hz
                    },
                    "eeg_theta_delta_ratio": (
                        float(metrics["theta"]) / float(metrics["delta"])
                        if float(metrics["delta"]) > 0
                        else math.nan
                    ),
                    "expected_frequency_pattern": expected_patterns[state],
                    "frequency_expectation_met": frequency_check,
                    "spectral_quality_status": spectral_status,
                    "emg_channels": ";".join(emg_channels),
                    "emg_epoch_count": emg["epoch_count"],
                    "emg_rms_finite_fraction": emg["finite_fraction"],
                    "emg_rms_mean_uv": emg["mean"],
                    "emg_rms_median_uv": emg["median"],
                    "emg_rms_p05_uv": emg["p05"],
                    "emg_rms_p95_uv": emg["p95"],
                    "emg_expectation_met": emg_checks[state],
                    "emg_quality_status": emg_status,
                    "quality_status": quality_status,
                    "quality_reason": f"EEG: {spectral_reason}; EMG: {emg_reason}",
                }
            )

    report = pd.DataFrame(rows)
    determining = report.loc[report["determines_recording_quality"]]
    recording_status = max(
        determining["quality_status"], key=lambda status: PERFORMANCE_CHECK[str(status)]
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
    config: SpectraConfig = DEFAULT_SPECTRA_CONFIG,
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
        for state, state_name in config.sleep_state_names.items():
            if state not in spectra:
                continue
            axis.plot(
                frequencies,
                spectra[state][channel_index],
                color=config.state_colors[state],
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
    config: SpectraConfig = DEFAULT_SPECTRA_CONFIG,
):
    """Plot one EMG RMS histogram per sleep state and EMG channel."""
    ensure_style()
    import matplotlib.pyplot as plt

    states = [state for state in config.sleep_state_names if state in rms_by_state]
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
                color=config.state_colors[state],
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
            )
            axis.set_title(
                f"{channel_name} — {config.sleep_state_names[state]}\n"
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
    parser.add_argument(
        "--spectra-config",
        default=str(DEFAULT_SPECTRA_CONFIG_PATH),
        help=f"Spectra pipeline YAML (default: {DEFAULT_SPECTRA_CONFIG_PATH}).",
    )
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument("--epoch-seconds", type=float, default=None)
    parser.add_argument("--welch-seconds", type=float, default=None)
    parser.add_argument("--fmin", type=float, default=None, metavar="HZ")
    parser.add_argument("--fmax", type=float, default=None, metavar="HZ")
    parser.add_argument("--chunk-epochs", type=int, default=None)
    parser.add_argument(
        "--include-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
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
    try:
        config = load_spectra_config(args.spectra_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.epoch_seconds = (
        config.epoch_seconds if args.epoch_seconds is None else args.epoch_seconds
    )
    args.welch_seconds = (
        config.welch_seconds if args.welch_seconds is None else args.welch_seconds
    )
    args.fmin = config.fmin_hz if args.fmin is None else args.fmin
    args.fmax = config.fmax_hz if args.fmax is None else args.fmax
    args.chunk_epochs = (
        config.chunk_epochs if args.chunk_epochs is None else args.chunk_epochs
    )
    args.include_artifacts = (
        config.include_artifacts
        if args.include_artifacts is None
        else args.include_artifacts
    )
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
        for state, name in config.sleep_state_names.items():
            print(f"  {name}: {counts[state]:,} epochs")
        fig = plot_state_spectra(
            frequencies,
            spectra,
            counts,
            channels,
            title=f"{args.subject} — {args.date or args.session} — sleep-state spectra",
            config=config,
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
                config=config,
            )
            figures.append(emg_fig)
        else:
            print("EMG RMS: skipped (no EMG channels or scored finite epochs)")
        quality_report = build_spectral_quality_report(
            frequencies,
            spectra,
            counts,
            channels,
            emg_rms,
            emg_channels,
            config=config,
        )
        print(
            "Spectral quality: "
            f"{quality_report['recording_quality_status'].iloc[0]}"
        )
        for row in quality_report.itertuples(index=False):
            print(
                f"  {row.eeg_channel} — {row.sleep_state}: "
                f"{row.quality_status} — "
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
            save_csv(quality_report, quality_output_path)
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
