"""Detect state-dependent EEG artifacts in a FIF recording.

The detector aligns sleep states from a parquet file to non-overlapping epochs,
extracts EEG and optional EMG features in bounded chunks, and estimates robust
outlier scores separately for every channel and sleep state. Hard signal failures
(nonfinite data, flatlining, and clipping-like repeated edge values) remain
state-independent.

Only epoch-level results are written. Outputs default to the directory containing
the sleep-scoring parquet file and are saved as both CSV and parquet. Tunable
detector settings and output naming rules are loaded from the pipeline YAML.
``ArtifactDetector`` reuses one immutable configuration for single-pair or
sequential multi-pair processing while retaining failures as structured results.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.io.mne_io import import_mne
    from scripts.utils.config import coalesce, load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.mne_io import import_mne
    from scripts.utils.config import coalesce, load_config, nested_get


EEG_SCORE_FEATURES = (
    "peak_to_peak_uv",
    "rms_uv",
    "max_derivative_uv_per_sample",
    "high_frequency_power_uv2",
    "line_noise_power_uv2",
)
EMG_SCORE_FEATURES = (
    "emg_peak_to_peak_uv",
    "emg_rms_uv",
    "emg_max_derivative_uv_per_sample",
)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/pipelines/artifact_detection.yaml"


def _two_float_tuple(configured: Any, name: str) -> tuple[float, float]:
    """Normalize a two-value YAML list, including the simple-parser fallback."""
    if isinstance(configured, str):
        configured = configured.strip().removeprefix("[").removesuffix("]").split(",")
    values = tuple(float(item) for item in configured)
    if len(values) != 2:
        raise ValueError(f"Configured {name} must contain [low, high].")
    return values


@dataclass(frozen=True)
class ArtifactDetectionConfig:
    """Immutable settings shared across one or many recording pairs."""

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
    remove_from_stem: str
    output_suffix: str
    overwrite: bool

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "ArtifactDetectionConfig":
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

        high_band = _two_float_tuple(
            value(("welch", "high_frequency_band_hz")), "high-frequency band"
        )
        line_band = _two_float_tuple(
            value(("welch", "line_noise_band_hz")), "line-noise band"
        )

        return cls(
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
            remove_from_stem=str(value(("output", "remove_from_stem"))),
            output_suffix=str(value(("output", "suffix"))),
            overwrite=bool(value(("output", "overwrite"))),
        )


@dataclass(frozen=True)
class ArtifactDetectionResult:
    """Outcome of processing one recording and sleep-score pair."""

    fif_path: str
    sleep_parquet_path: str
    status: str
    csv_path: str | None = None
    parquet_path: str | None = None
    n_epochs: int | None = None
    n_artifact_epochs: int | None = None
    artifact_fraction: float | None = None
    error: str | None = None


class ArtifactDetector:
    """Orchestrate repeatable artifact detection for one or many input pairs.

    Numerical feature extraction and classification remain standalone functions;
    this class owns shared configuration, output orchestration, and batch error
    handling.
    """

    def __init__(self, config: ArtifactDetectionConfig) -> None:
        self.config = config

    def process(
        self,
        fif_path: str | Path,
        sleep_parquet_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> ArtifactDetectionResult:
        artifact_epochs, csv_path, parquet_path = detect_fif_artifacts(
            fif_path,
            sleep_parquet_path,
            output_dir=output_dir,
            **asdict(self.config),
        )
        n_epochs = len(artifact_epochs)
        n_artifacts = int(artifact_epochs["artifact"].sum())
        return ArtifactDetectionResult(
            fif_path=str(fif_path),
            sleep_parquet_path=str(sleep_parquet_path),
            status="written",
            csv_path=str(csv_path),
            parquet_path=str(parquet_path),
            n_epochs=n_epochs,
            n_artifact_epochs=n_artifacts,
            artifact_fraction=n_artifacts / n_epochs if n_epochs else 0.0,
        )

    def process_many(
        self,
        pairs: list[tuple[str | Path, str | Path]],
        output_dir: str | Path | None = None,
        continue_on_error: bool = True,
    ) -> list[ArtifactDetectionResult]:
        """Process pairs sequentially and optionally retain failures as results."""
        results = []
        for fif_path, sleep_parquet_path in pairs:
            try:
                results.append(self.process(fif_path, sleep_parquet_path, output_dir))
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(
                    ArtifactDetectionResult(
                        fif_path=str(fif_path),
                        sleep_parquet_path=str(sleep_parquet_path),
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results


def _majority_state(values: pd.Series) -> Any:
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else np.nan


def align_sleep_states(
    parquet_path: str | Path,
    n_epochs: int,
    epoch_seconds: float,
    time_column: str,
    state_column: str,
    unscored_numeric_state: int,
    unscored_text_state: str,
) -> pd.DataFrame:
    """Return one sleep-state row for every selected epoch."""
    parquet_path = Path(parquet_path)
    sleep_df = pd.read_parquet(parquet_path)
    missing = {time_column, state_column}.difference(sleep_df.columns)
    if missing:
        raise ValueError(f"Sleep-scoring parquet is missing columns: {sorted(missing)}")

    scores = sleep_df[[time_column, state_column]].dropna().copy()
    scores = scores.loc[
        (scores[time_column] >= 0)
        & (scores[time_column] < n_epochs * epoch_seconds)
    ].copy()
    scores["epoch_id"] = np.floor(scores[time_column] / epoch_seconds).astype("int64")

    epoch_states = scores.groupby("epoch_id", as_index=False).agg(
        sleep_state=(state_column, _majority_state),
        sleep_score_count=(state_column, "size"),
    )
    selected_epochs = pd.DataFrame({"epoch_id": np.arange(n_epochs, dtype=np.int64)})
    epoch_states = selected_epochs.merge(epoch_states, on="epoch_id", how="left")

    if pd.api.types.is_numeric_dtype(scores[state_column]):
        unscored_state: int | str = unscored_numeric_state
        epoch_states["sleep_state"] = pd.to_numeric(
            epoch_states["sleep_state"]
        ).fillna(unscored_state)
    else:
        unscored_state = unscored_text_state
        epoch_states["sleep_state"] = (
            epoch_states["sleep_state"].astype("string").fillna(unscored_state)
        )
    epoch_states["sleep_score_count"] = (
        epoch_states["sleep_score_count"].fillna(0).astype(int)
    )
    return epoch_states


def _bandpower(
    psd: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies <= high)
    if mask.sum() < 2:
        return np.full(psd.shape[:-1], np.nan)
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def choose_chunk_epochs(
    sfreq: float,
    epoch_seconds: float,
    n_channels: int,
    target_memory_mb: float,
    working_array_factor: int,
    max_chunk_epochs: int,
) -> int:
    """Choose an epoch chunk size from signal dimensions and a memory budget.

    The working-array factor accounts for the input batch, cleaned copies,
    derivatives, Welch PSD output, and temporary arrays used during extraction.
    """
    if any(value <= 0 for value in (
        sfreq, epoch_seconds, n_channels, target_memory_mb,
        working_array_factor, max_chunk_epochs,
    )):
        raise ValueError(
            "sfreq, epoch_seconds, n_channels, and target_memory_mb must be positive."
        )
    epoch_samples = int(round(sfreq * epoch_seconds))
    estimated_bytes_per_epoch = (
        epoch_samples * n_channels * np.dtype(np.float64).itemsize
        * working_array_factor
    )
    target_bytes = int(target_memory_mb * 1024**2)
    return max(1, min(max_chunk_epochs, target_bytes // estimated_bytes_per_epoch))


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
    eeg_rows: list[dict[str, Any]] = []
    emg_rows: list[dict[str, Any]] = []

    for first_epoch in range(0, n_epochs, chunk_epochs):
        last_epoch = min(first_epoch + chunk_epochs, n_epochs)
        start = first_epoch * epoch_samples
        stop = last_epoch * epoch_samples
        epoch_count = last_epoch - first_epoch

        samples = eeg.get_data(start=start, stop=stop)
        batch = samples.reshape(len(eeg.ch_names), epoch_count, epoch_samples)
        batch = np.moveaxis(batch, 1, 0) * 1e6
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
        high_frequency_power = _bandpower(
            psd, frequencies, high_frequency_band_hz[0],
            min(high_frequency_band_hz[1], psd_fmax)
        )
        line_noise_power = _bandpower(
            psd, frequencies, line_noise_band_hz[0],
            min(line_noise_band_hz[1], psd_fmax)
        )

        if emg is not None:
            emg_samples = emg.get_data(start=start, stop=stop)
            emg_batch = emg_samples.reshape(
                len(emg.ch_names), epoch_count, epoch_samples
            )
            emg_batch = np.moveaxis(emg_batch, 1, 0) * 1e6
            emg_nonfinite = ~np.isfinite(emg_batch).all(axis=-1)
            emg_clean = np.nan_to_num(
                emg_batch, nan=0.0, posinf=0.0, neginf=0.0
            )
            emg_peak_to_peak = np.ptp(emg_clean, axis=-1)
            emg_rms = np.sqrt(np.mean(emg_clean**2, axis=-1))
            emg_max_derivative = np.max(
                np.abs(np.diff(emg_clean, axis=-1)), axis=-1
            )

        for offset in range(epoch_count):
            epoch_id = first_epoch + offset
            for channel_index, channel in enumerate(eeg.ch_names):
                eeg_rows.append(
                    {
                        "epoch_id": epoch_id,
                        "time_s": epoch_id * epoch_seconds,
                        "time_h": epoch_id * epoch_seconds / 3600,
                        "channel": channel,
                        "peak_to_peak_uv": peak_to_peak[offset, channel_index],
                        "rms_uv": rms[offset, channel_index],
                        "max_derivative_uv_per_sample": max_derivative[
                            offset, channel_index
                        ],
                        "std_uv": standard_deviation[offset, channel_index],
                        "edge_fraction": edge_fraction[offset, channel_index],
                        "high_frequency_power_uv2": high_frequency_power[
                            offset, channel_index
                        ],
                        "line_noise_power_uv2": line_noise_power[
                            offset, channel_index
                        ],
                        "nonfinite": nonfinite[offset, channel_index],
                    }
                )
            if emg is not None:
                for channel_index, channel in enumerate(emg.ch_names):
                    emg_rows.append(
                        {
                            "epoch_id": epoch_id,
                            "emg_channel": channel,
                            "emg_peak_to_peak_uv": emg_peak_to_peak[
                                offset, channel_index
                            ],
                            "emg_rms_uv": emg_rms[offset, channel_index],
                            "emg_max_derivative_uv_per_sample": emg_max_derivative[
                                offset, channel_index
                            ],
                            "emg_nonfinite": emg_nonfinite[offset, channel_index],
                        }
                    )

    return (
        pd.DataFrame(eeg_rows),
        pd.DataFrame(emg_rows),
        list(eeg.ch_names),
        list(emg.ch_names) if emg is not None else [],
        chunk_epochs,
    )


def _robust_upper_z(values: pd.Series, mad_scale: float) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    transformed = np.log10(np.clip(array, np.finfo(float).tiny, None))
    finite = np.isfinite(transformed)
    result = np.full(transformed.shape, np.nan)
    if not finite.any():
        return result
    median = np.median(transformed[finite])
    mad = np.median(np.abs(transformed[finite] - median))
    result[finite] = (
        0.0 if mad == 0 else (transformed[finite] - median) / (mad_scale * mad)
    )
    return result


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
) -> pd.DataFrame:
    """Apply state-dependent robust rules and return one row per epoch."""
    eeg_features = eeg_features.merge(
        epoch_states, on="epoch_id", how="left", validate="many_to_one"
    )
    if not emg_features.empty:
        emg_features = emg_features.merge(
            epoch_states, on="epoch_id", how="left", validate="many_to_one"
        )
        for feature in EMG_SCORE_FEATURES:
            emg_features[f"z_{feature}"] = emg_features.groupby(
                ["emg_channel", "sleep_state"]
            )[feature].transform(lambda values: _robust_upper_z(values, robust_mad_scale))
        emg_z_columns = [f"z_{feature}" for feature in EMG_SCORE_FEATURES]
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

    for feature in EEG_SCORE_FEATURES:
        eeg_features[f"z_{feature}"] = eeg_features.groupby(
            ["channel", "sleep_state"]
        )[feature].transform(lambda values: _robust_upper_z(values, robust_mad_scale))
    z_columns = [f"z_{feature}" for feature in EEG_SCORE_FEATURES]
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

    reasons: list[str] = []
    for row in eeg_features.to_dict(orient="records"):
        row_reasons = []
        if row["nonfinite"]:
            row_reasons.append("nonfinite")
        if row["std_uv"] < min_std_uv:
            row_reasons.append("flatline")
        if row["edge_fraction"] > max_edge_fraction:
            row_reasons.append("clipping_or_repeated_edge")
        if row["emg_supported_artifact"]:
            row_reasons.append("emg_outlier_with_eeg_outlier")
        row_reasons.extend(
            feature
            for feature in EEG_SCORE_FEATURES
            if row[f"z_{feature}"] > robust_z_threshold
        )
        reasons.append(";".join(row_reasons))
    eeg_features["artifact_reason"] = reasons

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


def output_paths(
    sleep_parquet_path: str | Path,
    remove_from_stem: str,
    output_suffix: str,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    sleep_parquet_path = Path(sleep_parquet_path)
    destination = Path(output_dir) if output_dir is not None else sleep_parquet_path.parent
    stem = sleep_parquet_path.stem.replace(remove_from_stem, "").rstrip("_-")
    return (
        destination / f"{stem}_{output_suffix}.csv",
        destination / f"{stem}_{output_suffix}.parquet",
    )


def detect_fif_artifacts(
    fif_path: str | Path,
    sleep_parquet_path: str | Path,
    *,
    epoch_seconds: float,
    chunk_memory_mb: float,
    chunk_working_array_factor: int,
    max_chunk_epochs: int,
    time_column: str,
    state_column: str,
    unscored_numeric_state: int,
    unscored_text_state: str,
    welch_segment_seconds: float,
    psd_minimum_frequency_hz: float,
    psd_maximum_frequency_hz: float,
    high_frequency_band_hz: tuple[float, float],
    line_noise_band_hz: tuple[float, float],
    robust_z_threshold: float,
    extreme_z_threshold: float,
    min_std_uv: float,
    max_edge_fraction: float,
    robust_mad_scale: float,
    soft_features_required: int,
    emg_soft_features_required: int,
    emg_supported_eeg_features_required: int,
    remove_from_stem: str,
    output_suffix: str,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
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
    if epoch_seconds <= 0 or chunk_memory_mb <= 0:
        raise ValueError("epoch_seconds and chunk_memory_mb must be positive.")

    csv_path, parquet_path = output_paths(
        sleep_parquet_path, remove_from_stem, output_suffix, output_dir
    )
    existing = [path for path in (csv_path, parquet_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists: {paths}. Pass --overwrite to replace it.")

    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    epoch_samples = int(round(epoch_seconds * sfreq))
    total_complete_epochs = raw.n_times // epoch_samples
    n_epochs = total_complete_epochs
    if n_epochs == 0:
        raise ValueError("The recording does not contain one complete analysis epoch.")

    print(f"FIF: {fif_path}")
    print(f"Sleep scores: {sleep_parquet_path}")
    print(f"Analyzing all {n_epochs:,} complete epochs")
    epoch_states = align_sleep_states(
        sleep_parquet_path,
        n_epochs,
        epoch_seconds,
        time_column,
        state_column,
        unscored_numeric_state,
        unscored_text_state,
    )
    eeg_features, emg_features, eeg_channels, emg_channels, chunk_epochs = extract_features(
        raw,
        n_epochs,
        epoch_seconds,
        chunk_memory_mb,
        chunk_working_array_factor,
        max_chunk_epochs,
        welch_segment_seconds,
        psd_minimum_frequency_hz,
        psd_maximum_frequency_hz,
        high_frequency_band_hz,
        line_noise_band_hz,
    )
    print(f"EEG channels: {eeg_channels}")
    print(f"EMG channels: {emg_channels or 'none (EMG checks disabled)'}")
    print(
        f"Automatic chunk size: {chunk_epochs:,} epochs "
        f"(target working memory: {chunk_memory_mb:g} MiB)"
    )
    artifact_epochs = classify_artifacts(
        eeg_features,
        emg_features,
        epoch_states,
        n_epochs,
        robust_z_threshold,
        extreme_z_threshold,
        min_std_uv,
        max_edge_fraction,
        robust_mad_scale,
        soft_features_required,
        emg_soft_features_required,
        emg_supported_eeg_features_required,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_epochs.to_csv(csv_path, index=False)
    artifact_epochs.to_parquet(parquet_path, index=False)
    print(
        f"Flagged {artifact_epochs['artifact'].sum():,} / {len(artifact_epochs):,} "
        f"epochs ({artifact_epochs['artifact'].mean():.2%})"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {parquet_path}")
    return artifact_epochs, csv_path, parquet_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fif_path", help="Input MNE FIF recording.")
    parser.add_argument("sleep_parquet_path", help="Matching sleep-scoring parquet file.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Artifact detection YAML config (default: {DEFAULT_CONFIG_PATH}).",
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
        help="Output directory; defaults beside the sleep-scoring parquet.",
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    config_root = ("artifact_detection",)

    def setting(path: tuple[str, ...]) -> Any:
        value = nested_get(config, config_root + path)
        if value is None:
            raise ValueError(f"Missing artifact detection config value: {'.'.join(config_root + path)}")
        return value

    high_frequency_band = _two_float_tuple(
        setting(("welch", "high_frequency_band_hz")), "high-frequency band"
    )
    line_noise_band = _two_float_tuple(
        setting(("welch", "line_noise_band_hz")), "line-noise band"
    )

    detector_config = ArtifactDetectionConfig(
        epoch_seconds=float(coalesce(args.epoch_seconds, setting(("epoch_seconds",)))),
        chunk_memory_mb=float(coalesce(args.chunk_memory_mb, setting(("chunking", "target_memory_mb")))),
        chunk_working_array_factor=int(setting(("chunking", "working_array_factor"))),
        max_chunk_epochs=int(setting(("chunking", "max_chunk_epochs"))),
        time_column=str(coalesce(args.time_column, setting(("sleep_scoring", "time_column")))),
        state_column=str(coalesce(args.state_column, setting(("sleep_scoring", "state_column")))),
        unscored_numeric_state=int(setting(("sleep_scoring", "unscored_numeric_state"))),
        unscored_text_state=str(setting(("sleep_scoring", "unscored_text_state"))),
        welch_segment_seconds=float(setting(("welch", "segment_seconds"))),
        psd_minimum_frequency_hz=float(setting(("welch", "minimum_frequency_hz"))),
        psd_maximum_frequency_hz=float(setting(("welch", "maximum_frequency_hz"))),
        high_frequency_band_hz=high_frequency_band,
        line_noise_band_hz=line_noise_band,
        robust_z_threshold=float(coalesce(args.robust_z_threshold, setting(("thresholds", "robust_z")))),
        extreme_z_threshold=float(coalesce(args.extreme_z_threshold, setting(("thresholds", "extreme_z")))),
        min_std_uv=float(coalesce(args.min_std_uv, setting(("thresholds", "minimum_standard_deviation_uv")))),
        max_edge_fraction=float(coalesce(args.max_edge_fraction, setting(("thresholds", "maximum_edge_fraction")))),
        robust_mad_scale=float(setting(("thresholds", "robust_mad_scale"))),
        soft_features_required=int(setting(("thresholds", "soft_features_required"))),
        emg_soft_features_required=int(setting(("thresholds", "emg_soft_features_required"))),
        emg_supported_eeg_features_required=int(setting(("thresholds", "emg_supported_eeg_features_required"))),
        remove_from_stem=str(setting(("output", "remove_from_stem"))),
        output_suffix=str(setting(("output", "suffix"))),
        overwrite=bool(coalesce(args.overwrite, setting(("output", "overwrite")))),
    )
    detector = ArtifactDetector(detector_config)
    detector.process(args.fif_path, args.sleep_parquet_path, args.output_dir)


if __name__ == "__main__":
    main()
