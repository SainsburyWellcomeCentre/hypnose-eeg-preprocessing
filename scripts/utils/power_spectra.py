"""Sleep-state alignment and power-spectrum calculation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .mne_io import import_mne


SLEEP_STATE_CODES = (0, 1, 2)


def epoch_sleep_states(
    scores: pd.DataFrame,
    *,
    epoch_seconds: float,
    n_epochs: int,
) -> pd.Series:
    """Return one majority output-state code per complete signal epoch."""
    if "time_s" not in scores.columns:
        raise ValueError("Scoring parquet is missing column 'time_s'")
    label_column = "label_output" if "label_output" in scores.columns else "label"
    if label_column not in scores.columns:
        raise ValueError("Scoring parquet is missing 'label_output' or 'label'")

    selected = scores[["time_s", label_column]].copy()
    selected["epoch_id"] = np.floor(
        selected["time_s"].astype(float) / epoch_seconds
    ).astype("int64")
    if "kind" in scores.columns:
        selected["kind"] = scores["kind"].astype(str)
        non_signal_epochs = set(
            selected.loc[selected["kind"] != "signal", "epoch_id"].tolist()
        )
        selected = selected.loc[
            (selected["kind"] == "signal")
            & ~selected["epoch_id"].isin(non_signal_epochs)
        ]
    selected = selected.loc[
        (selected["epoch_id"] >= 0) & (selected["epoch_id"] < n_epochs)
    ]

    def majority(values: pd.Series) -> int:
        modes = pd.to_numeric(values, errors="coerce").dropna().astype(int).mode()
        return int(modes.iloc[0]) if len(modes) else 3

    return selected.groupby("epoch_id")[label_column].agg(majority).astype(int)


def artifact_epoch_ids(
    artifact_table: pd.DataFrame, *, epoch_seconds: float
) -> set[int]:
    """Return analysis epoch IDs overlapped by flagged artifact intervals."""
    required = {"time_s", "artifact"}
    missing = required - set(artifact_table.columns)
    if missing:
        raise ValueError(f"Artifact parquet is missing columns: {sorted(missing)}")
    times = artifact_table["time_s"].astype(float)
    positive_steps = times.diff().dropna()
    positive_steps = positive_steps[positive_steps > 0]
    artifact_seconds = float(positive_steps.median()) if len(positive_steps) else 4.0
    flagged_times = artifact_table.loc[
        artifact_table["artifact"].astype(bool), "time_s"
    ].astype(float)
    excluded: set[int] = set()
    for start_s in flagged_times:
        first = int(np.floor(start_s / epoch_seconds))
        last = int(np.ceil((start_s + artifact_seconds) / epoch_seconds))
        excluded.update(range(first, last))
    return excluded


def compute_state_spectra(
    fif_path: str | Path,
    scoring_path: str | Path,
    *,
    artifact_path: str | Path | None,
    epoch_seconds: float,
    welch_seconds: float,
    fmin_hz: float,
    fmax_hz: float,
    chunk_epochs: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int], list[str]]:
    """Compute mean EEG PSD per sleep state using bounded FIF reads."""
    mne = import_mne()
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    try:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise ValueError(f"FIF contains no EEG channels: {fif_path}")
        channel_names = [raw.ch_names[index] for index in eeg_picks]
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

        n_per_seg = min(epoch_samples, int(round(welch_seconds * sfreq)))
        if n_per_seg < 2:
            raise ValueError("welch_seconds is too short for the FIF sampling rate")
        n_fft = 1 << (n_per_seg - 1).bit_length()
        n_overlap = n_per_seg // 2
        if fmax_hz > sfreq / 2.0:
            raise ValueError(
                f"maximum frequency ({fmax_hz:g} Hz) exceeds the FIF Nyquist "
                f"frequency ({sfreq / 2.0:g} Hz)"
            )
        sums: dict[int, np.ndarray] = {}
        counts = {state: 0 for state in SLEEP_STATE_CODES}
        frequencies: np.ndarray | None = None

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
                picks=eeg_picks,
                start=first_epoch * epoch_samples,
                stop=last_epoch * epoch_samples,
            )
            batch = samples.reshape(
                len(eeg_picks), last_epoch - first_epoch, epoch_samples
            )
            batch = np.moveaxis(batch, 1, 0) * 1e6
            keep &= np.all(np.isfinite(batch), axis=(1, 2))
            if not np.any(keep):
                continue

            selected_batch = batch[keep]
            selected_labels = labels[keep]
            psd, frequencies = mne.time_frequency.psd_array_welch(
                selected_batch,
                sfreq=sfreq,
                fmin=fmin_hz,
                fmax=fmax_hz,
                n_fft=n_fft,
                n_per_seg=n_per_seg,
                n_overlap=n_overlap,
                average="mean",
                verbose=False,
            )
            for state in SLEEP_STATE_CODES:
                state_mask = selected_labels == state
                if not np.any(state_mask):
                    continue
                state_sum = psd[state_mask].sum(axis=0)
                sums[state] = sums.get(state, np.zeros_like(state_sum)) + state_sum
                counts[state] += int(state_mask.sum())

        if frequencies is None or not any(counts.values()):
            raise ValueError("No scored, finite EEG epochs were available for PSD analysis")
        means = {state: sums[state] / counts[state] for state in sums}
        return frequencies, means, counts, channel_names
    finally:
        raw.close()
