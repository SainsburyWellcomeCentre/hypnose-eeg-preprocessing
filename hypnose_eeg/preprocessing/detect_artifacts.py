"""Detect state-dependent EEG artifacts in a FIF recording.

Aligns sleep states to epochs, extracts EEG/EMG features, and flags epochs using
per-channel, per-state robust outlier scores plus state-independent hard failures
(nonfinite, flatline, clipping). Writes epoch-level CSV and parquet outputs;
settings come from the pipeline YAML.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hypnose_eeg.analysis.power_spectra import bandpower
from hypnose_eeg.analysis.statistics import robust_upper_z
from hypnose_eeg.io.input_paths import scoring_path
from hypnose_eeg.io.mne_io import import_mne
from hypnose_eeg.io.output_paths import artifact_output_paths, save_csv
from hypnose_eeg.io.repository_paths import get_derivatives_root, get_rawdata_root
from hypnose_eeg.utils.config import (
    DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH,
    load_config,
    nested_get,
    two_float_tuple,
)
from hypnose_eeg.utils.provenance import write_provenance
from hypnose_eeg.utils.epochs import (
    align_epoch_states,
    choose_chunk_epochs,
    complete_epoch_count,
    epoch_batch,
)
from hypnose_eeg.utils.recording_selection import select_recordings


@dataclass(frozen=True)
class ArtifactDetector:
    """Configured artifact detector for one or many recording pairs."""

    epoch_seconds: float
    chunk_memory_mb: float
    chunk_working_array_factor: int
    max_chunk_epochs: int
    time_column: str
    state_column: str
    unscored_numeric_state: int
    unscored_text_state: str
    welch_segment_seconds: float
    psd_minimum_frequency_hz: float
    psd_maximum_frequency_hz: float
    high_frequency_band_hz: tuple[float, float]
    line_noise_band_hz: tuple[float, float]
    robust_z_threshold: float
    extreme_z_threshold: float
    min_std_uv: float
    max_edge_fraction: float
    robust_mad_scale: float
    soft_features_required: int
    emg_soft_features_required: int
    emg_supported_eeg_features_required: int
    eeg_score_features: tuple[str, ...]
    emg_score_features: tuple[str, ...]
    output_suffix: str
    overwrite: bool

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        **overrides: object,
    ) -> "ArtifactDetector":
        """Load a reusable detector configuration from the pipeline YAML."""
        config = load_config(config_path)
        root = ("artifact_detection",)

        def value(path: tuple[str, ...]) -> Any:
            configured = nested_get(config, root + path)
            if configured is None:
                raise ValueError(
                    f"Missing artifact detection config value: {'.'.join(root + path)}"
                )
            return configured

        high_band = two_float_tuple(
            value(("welch", "high_frequency_band_hz")), "high-frequency band"
        )
        line_band = two_float_tuple(
            value(("welch", "line_noise_band_hz")), "line-noise band"
        )

        settings: dict[str, object] = dict(
            epoch_seconds=float(value(("epoch_seconds",))),
            chunk_memory_mb=float(value(("chunking", "target_memory_mb"))),
            chunk_working_array_factor=int(value(("chunking", "working_array_factor"))),
            max_chunk_epochs=int(value(("chunking", "max_chunk_epochs"))),
            time_column=str(value(("sleep_scoring", "time_column"))),
            state_column=str(value(("sleep_scoring", "state_column"))),
            unscored_numeric_state=int(value(("sleep_scoring", "unscored_numeric_state"))),
            unscored_text_state=str(value(("sleep_scoring", "unscored_text_state"))),
            welch_segment_seconds=float(value(("welch", "segment_seconds"))),
            psd_minimum_frequency_hz=float(value(("welch", "minimum_frequency_hz"))),
            psd_maximum_frequency_hz=float(value(("welch", "maximum_frequency_hz"))),
            high_frequency_band_hz=high_band,
            line_noise_band_hz=line_band,
            robust_z_threshold=float(value(("thresholds", "robust_z"))),
            extreme_z_threshold=float(value(("thresholds", "extreme_z"))),
            min_std_uv=float(value(("thresholds", "minimum_standard_deviation_uv"))),
            max_edge_fraction=float(value(("thresholds", "maximum_edge_fraction"))),
            robust_mad_scale=float(value(("thresholds", "robust_mad_scale"))),
            soft_features_required=int(value(("thresholds", "soft_features_required"))),
            emg_soft_features_required=int(value(("thresholds", "emg_soft_features_required"))),
            emg_supported_eeg_features_required=int(
                value(("thresholds", "emg_supported_eeg_features_required"))
            ),
            eeg_score_features=tuple(value(("features", "eeg_score_features"))),
            emg_score_features=tuple(value(("features", "emg_score_features"))),
            output_suffix=str(value(("output", "suffix"))),
            overwrite=bool(value(("output", "overwrite"))),
        )
        settings.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**settings)

    def process(
        self,
        fif_path: str | Path,
        sleep_parquet_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict[str, object]:
        csv_path, parquet_path = artifact_output_paths(
            fif_path, output_suffix=self.output_suffix, output_dir=output_dir
        )
        existing = [path for path in (csv_path, parquet_path) if path.exists()]
        if existing and not self.overwrite:
            return {
                "fif_path": str(fif_path),
                "sleep_parquet_path": str(sleep_parquet_path),
                "status": "skipped_exists",
                "csv_path": str(csv_path),
                "parquet_path": str(parquet_path),
                "existing_outputs": [str(path) for path in existing],
            }

        artifact_epochs, csv_path, parquet_path = _detect_fif_artifacts(
            fif_path,
            sleep_parquet_path,
            output_dir=output_dir,
            detector=self,
        )
        n_epochs = len(artifact_epochs)
        n_artifacts = int(artifact_epochs["artifact"].sum())
        return {
            "fif_path": str(fif_path),
            "sleep_parquet_path": str(sleep_parquet_path),
            "status": "written",
            "csv_path": str(csv_path),
            "parquet_path": str(parquet_path),
            "n_epochs": n_epochs,
            "n_artifact_epochs": n_artifacts,
            "artifact_fraction": n_artifacts / n_epochs if n_epochs else 0.0,
        }

    def process_many(
        self,
        pairs: list[tuple[str | Path, str | Path]],
        output_dir: str | Path | None = None,
        continue_on_error: bool = True,
    ) -> list[dict[str, object]]:
        """Process pairs sequentially and optionally retain failures as results."""
        results = []
        for fif_path, sleep_parquet_path in pairs:
            try:
                results.append(self.process(fif_path, sleep_parquet_path, output_dir))
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(
                    {
                        "fif_path": str(fif_path),
                        "sleep_parquet_path": str(sleep_parquet_path),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results


def extract_features(
    raw: Any,
    n_epochs: int,
    epoch_seconds: float,
    chunk_memory_mb: float,
    chunk_working_array_factor: int,
    max_chunk_epochs: int,
    welch_segment_seconds: float,
    psd_minimum_frequency_hz: float,
    psd_maximum_frequency_hz: float,
    high_frequency_band_hz: tuple[float, float],
    line_noise_band_hz: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], int]:
    """Extract chunked EEG and optional EMG epoch features from an MNE Raw."""
    mne = import_mne()
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError("No channels typed as EEG were found in the FIF file.")
    emg_picks = mne.pick_types(raw.info, emg=True, exclude=[])
    eeg = raw.copy().pick(eeg_picks)
    emg = raw.copy().pick(emg_picks) if len(emg_picks) else None

    sfreq = float(raw.info["sfreq"])
    epoch_samples = int(round(epoch_seconds * sfreq))
    signal_channel_count = len(eeg.ch_names) + (len(emg.ch_names) if emg is not None else 0)
    chunk_epochs = choose_chunk_epochs(
        sfreq, epoch_seconds, signal_channel_count, chunk_memory_mb,
        chunk_working_array_factor, max_chunk_epochs,
    )
    nyquist = sfreq / 2
    psd_fmax = min(psd_maximum_frequency_hz, nyquist - sfreq / epoch_samples)
    if psd_fmax <= psd_minimum_frequency_hz:
        raise ValueError(f"Sampling frequency {sfreq:g} Hz is too low for spectral checks.")
    n_per_seg = min(epoch_samples, int(round(welch_segment_seconds * sfreq)))
    n_fft = 1 << (n_per_seg - 1).bit_length()
    n_overlap = n_per_seg // 2
    eeg_channel_names = np.array(eeg.ch_names)
    emg_channel_names = np.array(emg.ch_names) if emg is not None else None
    eeg_chunks: list[pd.DataFrame] = []
    emg_chunks: list[pd.DataFrame] = []

    for first_epoch in range(0, n_epochs, chunk_epochs):
        last_epoch = min(first_epoch + chunk_epochs, n_epochs)
        start = first_epoch * epoch_samples
        stop = last_epoch * epoch_samples
        epoch_count = last_epoch - first_epoch

        samples = eeg.get_data(start=start, stop=stop)
        batch = epoch_batch(
            samples, epoch_count=epoch_count, epoch_samples=epoch_samples
        ) * 1e6
        nonfinite = ~np.isfinite(batch).all(axis=-1)
        clean = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        peak_to_peak = np.ptp(clean, axis=-1)
        rms = np.sqrt(np.mean(clean**2, axis=-1))
        max_derivative = np.max(np.abs(np.diff(clean, axis=-1)), axis=-1)
        standard_deviation = np.std(clean, axis=-1)
        epoch_min = clean.min(axis=-1, keepdims=True)
        epoch_max = clean.max(axis=-1, keepdims=True)
        edge_fraction = np.maximum(
            np.mean(clean == epoch_min, axis=-1),
            np.mean(clean == epoch_max, axis=-1),
        )
        psd, frequencies = mne.time_frequency.psd_array_welch(
            clean,
            sfreq=sfreq,
            fmin=psd_minimum_frequency_hz,
            fmax=psd_fmax,
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            verbose=False,
        )
        high_frequency_power = bandpower(
            psd, frequencies, high_frequency_band_hz[0],
            min(high_frequency_band_hz[1], psd_fmax)
        )
        line_noise_power = bandpower(
            psd, frequencies, line_noise_band_hz[0],
            min(line_noise_band_hz[1], psd_fmax)
        )
        # Whole analysed band: near zero when the channel carries no EEG at all
        # (e.g. a disconnected headstage drifting slowly), which `std_uv` misses
        # because slow drift alone keeps the standard deviation up.
        broadband_power = bandpower(
            psd, frequencies, psd_minimum_frequency_hz, psd_fmax
        )

        if emg is not None:
            emg_samples = emg.get_data(start=start, stop=stop)
            emg_batch = epoch_batch(
                emg_samples,
                epoch_count=epoch_count,
                epoch_samples=epoch_samples,
            ) * 1e6
            emg_nonfinite = ~np.isfinite(emg_batch).all(axis=-1)
            emg_clean = np.nan_to_num(
                emg_batch, nan=0.0, posinf=0.0, neginf=0.0
            )
            emg_peak_to_peak = np.ptp(emg_clean, axis=-1)
            emg_rms = np.sqrt(np.mean(emg_clean**2, axis=-1))
            emg_max_derivative = np.max(
                np.abs(np.diff(emg_clean, axis=-1)), axis=-1
            )

        epoch_ids = first_epoch + np.arange(epoch_count)
        flat_epoch_ids = np.repeat(epoch_ids, len(eeg_channel_names))
        eeg_chunks.append(
            pd.DataFrame(
                {
                    "epoch_id": flat_epoch_ids,
                    "time_s": flat_epoch_ids * epoch_seconds,
                    "time_h": flat_epoch_ids * epoch_seconds / 3600,
                    "channel": np.tile(eeg_channel_names, epoch_count),
                    "peak_to_peak_uv": peak_to_peak.ravel(),
                    "rms_uv": rms.ravel(),
                    "max_derivative_uv_per_sample": max_derivative.ravel(),
                    "std_uv": standard_deviation.ravel(),
                    "edge_fraction": edge_fraction.ravel(),
                    "high_frequency_power_uv2": high_frequency_power.ravel(),
                    "line_noise_power_uv2": line_noise_power.ravel(),
                    "broadband_power_uv2": broadband_power.ravel(),
                    "nonfinite": nonfinite.ravel(),
                }
            )
        )
        if emg is not None:
            flat_emg_epoch_ids = np.repeat(epoch_ids, len(emg_channel_names))
            emg_chunks.append(
                pd.DataFrame(
                    {
                        "epoch_id": flat_emg_epoch_ids,
                        "emg_channel": np.tile(emg_channel_names, epoch_count),
                        "emg_peak_to_peak_uv": emg_peak_to_peak.ravel(),
                        "emg_rms_uv": emg_rms.ravel(),
                        "emg_max_derivative_uv_per_sample": emg_max_derivative.ravel(),
                        "emg_nonfinite": emg_nonfinite.ravel(),
                    }
                )
            )

    return (
        pd.concat(eeg_chunks, ignore_index=True) if eeg_chunks else pd.DataFrame(),
        pd.concat(emg_chunks, ignore_index=True) if emg_chunks else pd.DataFrame(),
        list(eeg.ch_names),
        list(emg.ch_names) if emg is not None else [],
        chunk_epochs,
    )


def classify_artifacts(
    eeg_features: pd.DataFrame,
    emg_features: pd.DataFrame,
    epoch_states: pd.DataFrame,
    n_epochs: int,
    robust_z_threshold: float,
    extreme_z_threshold: float,
    min_std_uv: float,
    max_edge_fraction: float,
    robust_mad_scale: float,
    soft_features_required: int,
    emg_soft_features_required: int,
    emg_supported_eeg_features_required: int,
    eeg_score_features: tuple[str, ...],
    emg_score_features: tuple[str, ...],
) -> pd.DataFrame:
    """Apply state-dependent robust rules and return one row per epoch."""
    eeg_features = eeg_features.merge(
        epoch_states, on="epoch_id", how="left", validate="many_to_one"
    )
    if not emg_features.empty:
        emg_features = emg_features.merge(
            epoch_states, on="epoch_id", how="left", validate="many_to_one"
        )
        for feature in emg_score_features:
            emg_features[f"z_{feature}"] = emg_features.groupby(
                ["emg_channel", "sleep_state"]
            )[feature].transform(
                lambda values: robust_upper_z(values, robust_mad_scale)
            )
        emg_z_columns = [f"z_{feature}" for feature in emg_score_features]
        emg_features["emg_soft_feature_count"] = (
            emg_features[emg_z_columns] > robust_z_threshold
        ).sum(axis=1)
        emg_features["emg_max_robust_z"] = emg_features[emg_z_columns].max(
            axis=1, skipna=True
        )
        emg_features["emg_extreme"] = (
            emg_features["emg_nonfinite"]
            | (emg_features["emg_soft_feature_count"] >= emg_soft_features_required)
            | (emg_features["emg_max_robust_z"] > extreme_z_threshold)
        )
        epoch_emg = emg_features.groupby("epoch_id", as_index=False).agg(
            emg_extreme=("emg_extreme", "any"),
            emg_rms_uv=("emg_rms_uv", "max"),
            emg_peak_to_peak_uv=("emg_peak_to_peak_uv", "max"),
            emg_max_robust_z=("emg_max_robust_z", "max"),
        )
    else:
        epoch_emg = pd.DataFrame(
            {
                "epoch_id": np.arange(n_epochs, dtype=np.int64),
                "emg_extreme": False,
                "emg_rms_uv": np.nan,
                "emg_peak_to_peak_uv": np.nan,
                "emg_max_robust_z": np.nan,
            }
        )

    for feature in eeg_score_features:
        eeg_features[f"z_{feature}"] = eeg_features.groupby(
            ["channel", "sleep_state"]
        )[feature].transform(
            lambda values: robust_upper_z(values, robust_mad_scale)
        )
    z_columns = [f"z_{feature}" for feature in eeg_score_features]
    eeg_features["soft_feature_count"] = (
        eeg_features[z_columns] > robust_z_threshold
    ).sum(axis=1)
    eeg_features["max_robust_z"] = eeg_features[z_columns].max(axis=1, skipna=True)
    eeg_features["hard_failure"] = (
        eeg_features["nonfinite"]
        | (eeg_features["std_uv"] < min_std_uv)
        | (eeg_features["edge_fraction"] > max_edge_fraction)
    )
    eeg_features = eeg_features.merge(
        epoch_emg, on="epoch_id", how="left", validate="many_to_one"
    )
    eeg_features["emg_supported_artifact"] = (
        eeg_features["emg_extreme"]
        & (eeg_features["soft_feature_count"] >= emg_supported_eeg_features_required)
    )
    eeg_features["artifact"] = (
        eeg_features["hard_failure"]
        | (eeg_features["soft_feature_count"] >= soft_features_required)
        | (eeg_features["max_robust_z"] > extreme_z_threshold)
        | eeg_features["emg_supported_artifact"]
    )

    reason_conditions = [
        ("nonfinite", eeg_features["nonfinite"]),
        ("flatline", eeg_features["std_uv"] < min_std_uv),
        ("clipping_or_repeated_edge", eeg_features["edge_fraction"] > max_edge_fraction),
        ("emg_outlier_with_eeg_outlier", eeg_features["emg_supported_artifact"]),
    ] + [
        (feature, eeg_features[f"z_{feature}"] > robust_z_threshold)
        for feature in eeg_score_features
    ]
    artifact_reason = np.full(len(eeg_features), "", dtype=object)
    for label, flagged in reason_conditions:
        flagged = flagged.to_numpy()
        artifact_reason = np.where(
            flagged,
            np.where(artifact_reason == "", label, artifact_reason + ";" + label),
            artifact_reason,
        )
    eeg_features["artifact_reason"] = artifact_reason

    artifact_epochs = eeg_features.groupby("epoch_id", as_index=False).agg(
        time_s=("time_s", "first"),
        time_h=("time_h", "first"),
        sleep_state=("sleep_state", "first"),
        artifact=("artifact", "any"),
        affected_channels=("artifact", "sum"),
        max_robust_z=("max_robust_z", "max"),
        emg_extreme=("emg_extreme", "first"),
        emg_rms_uv=("emg_rms_uv", "first"),
        emg_max_robust_z=("emg_max_robust_z", "first"),
    )
    reason_rows = eeg_features.loc[
        eeg_features["artifact"] & eeg_features["artifact_reason"].ne(""),
        ["epoch_id", "channel", "artifact_reason"],
    ].copy()
    reason_rows["channel_reason"] = (
        reason_rows["channel"].astype(str) + ":" + reason_rows["artifact_reason"]
    )
    epoch_reasons = reason_rows.groupby("epoch_id", as_index=False).agg(
        artifact_features=(
            "channel_reason",
            lambda values: " | ".join(dict.fromkeys(values)),
        )
    )
    artifact_epochs = artifact_epochs.merge(epoch_reasons, on="epoch_id", how="left")
    artifact_epochs["artifact_features"] = artifact_epochs[
        "artifact_features"
    ].fillna("")
    return artifact_epochs


def _detect_fif_artifacts(
    fif_path: str | Path,
    sleep_parquet_path: str | Path,
    *,
    detector: ArtifactDetector,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Run detection and write epoch-level CSV and parquet outputs."""
    mne = import_mne()
    fif_path = Path(fif_path)
    sleep_parquet_path = Path(sleep_parquet_path)
    if not fif_path.is_file():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")
    if fif_path.suffix.lower() != ".fif":
        raise ValueError(f"Expected a .fif recording: {fif_path}")
    if not sleep_parquet_path.is_file():
        raise FileNotFoundError(f"Sleep-scoring parquet not found: {sleep_parquet_path}")
    if sleep_parquet_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(f"Expected a parquet sleep-scoring file: {sleep_parquet_path}")
    if detector.epoch_seconds <= 0 or detector.chunk_memory_mb <= 0:
        raise ValueError("epoch_seconds and chunk_memory_mb must be positive.")

    csv_path, parquet_path = artifact_output_paths(
        fif_path,
        output_suffix=detector.output_suffix,
        output_dir=output_dir,
    )
    existing = [path for path in (csv_path, parquet_path) if path.exists()]
    if existing and not detector.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists: {paths}. Pass --overwrite to replace it.")

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_epochs = complete_epoch_count(raw.n_times, sfreq, detector.epoch_seconds)
    if n_epochs == 0:
        raise ValueError("The recording does not contain one complete analysis epoch.")

    print(f"FIF: {fif_path}")
    print(f"Sleep scores: {sleep_parquet_path}")
    print(f"Analyzing all {n_epochs:,} complete epochs")
    scores = pd.read_parquet(sleep_parquet_path)
    epoch_states = align_epoch_states(
        scores,
        n_epochs=n_epochs,
        epoch_seconds=detector.epoch_seconds,
        time_column=detector.time_column,
        state_column=detector.state_column,
        unscored_numeric_state=detector.unscored_numeric_state,
        unscored_text_state=detector.unscored_text_state,
    )
    eeg_features, emg_features, eeg_channels, emg_channels, chunk_epochs = extract_features(
        raw,
        n_epochs,
        detector.epoch_seconds,
        detector.chunk_memory_mb,
        detector.chunk_working_array_factor,
        detector.max_chunk_epochs,
        detector.welch_segment_seconds,
        detector.psd_minimum_frequency_hz,
        detector.psd_maximum_frequency_hz,
        detector.high_frequency_band_hz,
        detector.line_noise_band_hz,
    )
    print(f"EEG channels: {eeg_channels}")
    print(f"EMG channels: {emg_channels or 'none (EMG checks disabled)'}")
    print(
        f"Automatic chunk size: {chunk_epochs:,} epochs "
        f"(target working memory: {detector.chunk_memory_mb:g} MiB)"
    )
    artifact_epochs = classify_artifacts(
        eeg_features,
        emg_features,
        epoch_states,
        n_epochs,
        detector.robust_z_threshold,
        detector.extreme_z_threshold,
        detector.min_std_uv,
        detector.max_edge_fraction,
        detector.robust_mad_scale,
        detector.soft_features_required,
        detector.emg_soft_features_required,
        detector.emg_supported_eeg_features_required,
        detector.eeg_score_features,
        detector.emg_score_features,
    )

    save_csv(artifact_epochs, csv_path)
    artifact_epochs.to_parquet(parquet_path, index=False)
    print(
        f"Flagged {artifact_epochs['artifact'].sum():,} / {len(artifact_epochs):,} "
        f"epochs ({artifact_epochs['artifact'].mean():.2%})"
    )
    print(f"Saved: {parquet_path}")
    write_provenance(
        "artifact_detection",
        outputs=[parquet_path, csv_path],
        inputs={
            "fif_path": str(fif_path),
            "sleep_parquet_path": str(sleep_parquet_path),
            "eeg_channels": list(eeg_channels),
            "emg_channels": list(emg_channels),
        },
        parameters={
            "epoch_seconds": detector.epoch_seconds,
            "robust_z_threshold": detector.robust_z_threshold,
            "extreme_z_threshold": detector.extreme_z_threshold,
            "min_std_uv": detector.min_std_uv,
            "max_edge_fraction": detector.max_edge_fraction,
            "robust_mad_scale": detector.robust_mad_scale,
            "soft_features_required": detector.soft_features_required,
            "emg_soft_features_required": detector.emg_soft_features_required,
            "emg_supported_eeg_features_required": (
                detector.emg_supported_eeg_features_required
            ),
            "eeg_score_features": list(detector.eeg_score_features),
            "emg_score_features": list(detector.emg_score_features),
            "high_frequency_band_hz": list(detector.high_frequency_band_hz),
            "line_noise_band_hz": list(detector.line_noise_band_hz),
            "n_epochs": int(n_epochs),
            "sampling_rate_hz": sfreq,
        },
    )
    return artifact_epochs, csv_path, parquet_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "fif_path", nargs="?", default=None, help="Input MNE FIF recording."
    )
    parser.add_argument(
        "sleep_parquet_path",
        nargs="?",
        default=None,
        help="Matching sleep-scoring parquet file.",
    )
    parser.add_argument(
        "--edf", default=None,
        help="Raw EDF recording; resolves the matching sleep-scoring parquet. "
        "Requires --fif.",
    )
    parser.add_argument("--fif", default=None, help="Matching derivative FIF for --edf.")
    parser.add_argument(
        "--subject", "--subjid", dest="subject", default=None,
        help="Subject ID, for example 66 or sub-066. Alternative to fif_path/sleep_parquet_path.",
    )
    session_selector = parser.add_mutually_exclusive_group()
    session_selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    session_selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
    )
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH),
        help=f"Artifact detection YAML config (default: {DEFAULT_ARTIFACT_DETECTION_CONFIG_PATH}).",
    )
    parser.add_argument("--epoch-seconds", type=float, default=None)
    parser.add_argument(
        "--chunk-memory-mb",
        type=float,
        default=None,
        help="Target working memory for automatically sized extraction chunks.",
    )
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--state-column", default=None)
    parser.add_argument("--robust-z-threshold", type=float, default=None)
    parser.add_argument("--extreme-z-threshold", type=float, default=None)
    parser.add_argument("--min-std-uv", type=float, default=None)
    parser.add_argument("--max-edge-fraction", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory; defaults to the session's dedicated artifacts "
            "directory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def _resolve_paths(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the FIF and sleep-scoring parquet from explicit paths, --edf/--fif, or selectors."""
    has_explicit_path = args.fif_path is not None or args.sleep_parquet_path is not None
    has_edf_fif = args.edf is not None or args.fif is not None
    has_selector = args.subject is not None or args.date is not None or args.session is not None
    if sum([has_explicit_path, has_edf_fif, has_selector]) > 1:
        parser.error(
            "use exactly one of: fif_path/sleep_parquet_path, --edf/--fif, or "
            "--subject/--date/--session selectors"
        )
    if has_explicit_path:
        if args.fif_path is None or args.sleep_parquet_path is None:
            parser.error("fif_path and sleep_parquet_path must be supplied together")
        return Path(args.fif_path), Path(args.sleep_parquet_path)

    if has_edf_fif:
        if bool(args.edf) != bool(args.fif):
            parser.error("--edf and --fif must be supplied together")
        edf_path = Path(args.edf).resolve(strict=False)
        fif_path = Path(args.fif).resolve(strict=False)
        rawdata_root = Path(args.rawdata_root or edf_path.parent).resolve(strict=False)
        derivatives_root = Path(
            args.derivatives_root or fif_path.parent
        ).resolve(strict=False)
        sleep_parquet_path = scoring_path(edf_path, rawdata_root, derivatives_root)
        return fif_path, sleep_parquet_path

    if args.subject is None:
        parser.error("fif_path/sleep_parquet_path, --edf/--fif, or --subject is required")
    if args.date is None and args.session is None:
        parser.error("--subject requires either --date or --session")

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(
        args.derivatives_root or get_derivatives_root()
    ).resolve(strict=False)
    pairs = select_recordings(
        rawdata_root, derivatives_root, subject=args.subject, date=args.date, session=args.session
    )
    if len(pairs) != 1:
        parser.error(
            f"Selection matched {len(pairs)} recordings; narrow --date/--session to one recording"
        )
    edf_path, fif_path = pairs[0]
    sleep_parquet_path = scoring_path(edf_path, rawdata_root, derivatives_root)
    return fif_path, sleep_parquet_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fif_path, sleep_parquet_path = _resolve_paths(parser, args)
    detector = ArtifactDetector.from_yaml(
        args.config,
        epoch_seconds=args.epoch_seconds,
        chunk_memory_mb=args.chunk_memory_mb,
        time_column=args.time_column,
        state_column=args.state_column,
        robust_z_threshold=args.robust_z_threshold,
        extreme_z_threshold=args.extreme_z_threshold,
        min_std_uv=args.min_std_uv,
        max_edge_fraction=args.max_edge_fraction,
        overwrite=args.overwrite,
    )
    result = detector.process(fif_path, sleep_parquet_path, args.output_dir)
    if result["status"] == "skipped_exists":
        existing = ", ".join(result["existing_outputs"])
        print(f"Skipped, output exists: {existing}. Pass --overwrite to replace it.")


if __name__ == "__main__":
    main()
