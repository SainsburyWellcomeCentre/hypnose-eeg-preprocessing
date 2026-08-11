"""Chunked cross-channel correlation utilities for scored recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .mne_io import import_mne
from .power_spectra import (
    SLEEP_STATE_CODES,
    artifact_epoch_ids,
    epoch_sleep_states,
)


def epoch_pearson_matrices(epoch_data: np.ndarray) -> np.ndarray:
    """Return channel correlation matrices for epochs × channels × samples."""
    data = np.asarray(epoch_data, dtype=float)
    if data.ndim != 3:
        raise ValueError("epoch_data must have shape (epochs, channels, samples)")
    if data.shape[-1] < 2:
        raise ValueError("At least two samples are required per epoch")

    finite_channels = np.all(np.isfinite(data), axis=-1)
    clean = np.where(np.isfinite(data), data, 0.0)
    centered = clean - clean.mean(axis=-1, keepdims=True)
    cross_products = np.einsum("ecs,eds->ecd", centered, centered)
    sums_of_squares = np.einsum("ecs,ecs->ec", centered, centered)
    denominators = np.sqrt(
        sums_of_squares[:, :, np.newaxis]
        * sums_of_squares[:, np.newaxis, :]
    )
    valid = (
        finite_channels[:, :, np.newaxis]
        & finite_channels[:, np.newaxis, :]
        & (denominators > 0)
    )
    correlations = np.full_like(cross_products, np.nan, dtype=float)
    np.divide(cross_products, denominators, out=correlations, where=valid)
    return np.clip(correlations, -1.0, 1.0)


def compute_state_channel_correlations(
    fif_path: str | Path,
    scoring_path: str | Path,
    *,
    artifact_path: str | Path | None,
    epoch_seconds: float,
    chunk_epochs: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate epoch-wise Pearson r for every EEG/EMG channel pair."""
    mne = import_mne()
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    try:
        signal_picks = mne.pick_types(raw.info, eeg=True, emg=True, exclude=[])
        if len(signal_picks) < 2:
            raise ValueError(
                f"FIF must contain at least two EEG/EMG channels: {fif_path}"
            )
        channel_names = [raw.ch_names[index] for index in signal_picks]
        channel_types = raw.get_channel_types(picks=signal_picks)
        channel_pairs = [
            (first, second)
            for first in range(len(channel_names))
            for second in range(first + 1, len(channel_names))
        ]
        sfreq = float(raw.info["sfreq"])
        epoch_samples = int(round(epoch_seconds * sfreq))
        if epoch_samples < 2:
            raise ValueError("epoch_seconds is too short for the FIF sampling rate")
        n_epochs = raw.n_times // epoch_samples
        if n_epochs == 0:
            raise ValueError("FIF contains no complete analysis epoch")

        scores = pd.read_parquet(scoring_path)
        epoch_states = epoch_sleep_states(
            scores, epoch_seconds=epoch_seconds, n_epochs=n_epochs
        )
        excluded_artifacts: set[int] = set()
        if artifact_path is not None:
            artifacts = pd.read_parquet(
                artifact_path, columns=["time_s", "artifact"]
            )
            excluded_artifacts = artifact_epoch_ids(
                artifacts, epoch_seconds=epoch_seconds
            )

        frames: list[pd.DataFrame] = []
        for first_epoch in range(0, n_epochs, chunk_epochs):
            last_epoch = min(first_epoch + chunk_epochs, n_epochs)
            epoch_ids = np.arange(first_epoch, last_epoch)
            labels = np.array(
                [epoch_states.get(int(epoch_id), 3) for epoch_id in epoch_ids],
                dtype=int,
            )
            keep = np.isin(labels, SLEEP_STATE_CODES)
            if excluded_artifacts:
                keep &= ~np.isin(epoch_ids, list(excluded_artifacts))
            if not np.any(keep):
                continue

            samples = raw.get_data(
                picks=signal_picks,
                start=first_epoch * epoch_samples,
                stop=last_epoch * epoch_samples,
            )
            batch = samples.reshape(
                len(signal_picks), last_epoch - first_epoch, epoch_samples
            )
            batch = np.moveaxis(batch, 1, 0)
            correlations = epoch_pearson_matrices(batch)

            for first_channel, second_channel in channel_pairs:
                values = correlations[:, first_channel, second_channel]
                pair_keep = keep & np.isfinite(values)
                if not np.any(pair_keep):
                    continue
                selected_epochs = epoch_ids[pair_keep]
                frames.append(
                    pd.DataFrame(
                        {
                            "epoch_id": selected_epochs,
                            "time_s": selected_epochs * epoch_seconds,
                            "sleep_state": labels[pair_keep],
                            "channel_1": channel_names[first_channel],
                            "channel_2": channel_names[second_channel],
                            "channel_1_type": channel_types[first_channel],
                            "channel_2_type": channel_types[second_channel],
                            "pearson_r": values[pair_keep],
                        }
                    )
                )

        if not frames:
            raise ValueError(
                "No scored, finite, non-constant channel pairs were available"
            )
        return pd.concat(frames, ignore_index=True), channel_names
    finally:
        raw.close()
