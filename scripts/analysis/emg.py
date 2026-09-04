"""Reusable EMG feature calculations for scored recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.io.mne_io import import_mne
from scripts.utils.epochs import (
    SLEEP_STATE_CODES,
    artifact_epoch_ids,
    epoch_sleep_states,
)


def compute_state_emg_rms(
    fif_path: str | Path,
    scoring_path: str | Path,
    *,
    artifact_path: str | Path | None,
    epoch_seconds: float,
    chunk_epochs: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Compute per-epoch EMG RMS in µV, grouped by sleep state."""
    mne = import_mne()
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    try:
        emg_picks = mne.pick_types(raw.info, emg=True, exclude=[])
        if len(emg_picks) == 0:
            return {}, []
        channel_names = [raw.ch_names[index] for index in emg_picks]
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

        state_batches: dict[int, list[np.ndarray]] = {
            state: [] for state in SLEEP_STATE_CODES
        }
        for first_epoch in range(0, n_epochs, chunk_epochs):
            last_epoch = min(first_epoch + chunk_epochs, n_epochs)
            epoch_ids = np.arange(first_epoch, last_epoch)
            labels = epoch_states.reindex(epoch_ids, fill_value=3).to_numpy(dtype=int)
            keep = np.isin(labels, SLEEP_STATE_CODES)
            if excluded_artifacts:
                keep &= ~np.isin(epoch_ids, list(excluded_artifacts))
            if not np.any(keep):
                continue

            samples = raw.get_data(
                picks=emg_picks,
                start=first_epoch * epoch_samples,
                stop=last_epoch * epoch_samples,
            )
            batch = samples.reshape(
                len(emg_picks), last_epoch - first_epoch, epoch_samples
            )
            batch = np.moveaxis(batch, 1, 0) * 1e6
            keep &= np.all(np.isfinite(batch), axis=(1, 2))
            if not np.any(keep):
                continue

            rms_uv = np.sqrt(np.mean(np.square(batch[keep]), axis=-1))
            selected_labels = labels[keep]
            for state in SLEEP_STATE_CODES:
                state_mask = selected_labels == state
                if np.any(state_mask):
                    state_batches[state].append(rms_uv[state_mask])

        grouped_rms = {
            state: np.concatenate(batches, axis=0)
            for state, batches in state_batches.items()
            if batches
        }
        return grouped_rms, channel_names
    finally:
        raw.close()
