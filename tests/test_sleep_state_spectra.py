from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.quality_control.plot_spectra import build_parser, plot_state_emg_rms
from scripts.utils.power_spectra import (
    artifact_epoch_ids,
    compute_state_emg_rms,
    compute_state_spectra,
    epoch_sleep_states,
)
from scripts.utils.recording_paths import (
    artifact_path,
    scoring_path,
    session_derivatives_dir,
)


class SleepStateSpectraTests(unittest.TestCase):
    def test_spectrum_computation_is_available_from_utils(self) -> None:
        self.assertTrue(callable(compute_state_spectra))
        self.assertTrue(callable(compute_state_emg_rms))

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "Matplotlib is not installed"
    )
    def test_emg_rms_plot_has_one_histogram_per_sleep_state(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rms_by_state = {
            0: np.array([[1.0], [2.0]]),
            1: np.array([[0.5], [0.75]]),
            2: np.array([[0.2], [0.3]]),
        }
        figure = plot_state_emg_rms(rms_by_state, ["EMG"], title="test")
        try:
            self.assertEqual(
                [axis.get_title() for axis in figure.axes],
                [
                    "EMG — Wake\n2 epochs",
                    "EMG — NREM\n2 epochs",
                    "EMG — REM\n2 epochs",
                ],
            )
            self.assertTrue(
                all(axis.get_xlabel() == "EMG RMS (µV)" for axis in figure.axes)
            )
            self.assertEqual(figure.axes[0].get_ylabel(), "Epoch count")
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        by_date = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717"]
        )
        self.assertEqual(by_date.date, "20260717")
        by_session = build_parser().parse_args(
            ["--subject", "66", "--session", "1"]
        )
        self.assertEqual(by_session.session, "1")

    def test_scoring_and_artifact_paths_resolve_from_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys" / "recording.edf"
            saved = derivatives / "sub-066" / "ses-1_date-20260717" / "saved_results"
            edf.parent.mkdir(parents=True)
            saved.mkdir(parents=True)
            scoring = saved / "recording_somnotate_predictions.parquet"
            artifact = saved / "recording_artifact_epochs.parquet"
            scoring.touch()
            artifact.touch()
            self.assertEqual(
                session_derivatives_dir(edf, rawdata, derivatives), saved.parent
            )
            self.assertEqual(scoring_path(edf, rawdata, derivatives), scoring)
            self.assertEqual(artifact_path(edf, rawdata, derivatives), artifact)

    def test_generic_artifact_filename_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys" / "recording.edf"
            saved = derivatives / "sub-066" / "ses-1_date-20260717" / "saved_results"
            edf.parent.mkdir(parents=True)
            saved.mkdir(parents=True)
            artifact = saved / "artifact_epochs.parquet"
            artifact.touch()
            self.assertEqual(artifact_path(edf, rawdata, derivatives), artifact)

    def test_epoch_states_use_majority_and_artifacts_are_aligned(self) -> None:
        scores = pd.DataFrame(
            {
                "time_s": list(range(8)),
                "label_output": [0, 0, 1, 0, 2, 2, 2, 1],
                "kind": ["signal"] * 8,
            }
        )
        labels = epoch_sleep_states(scores, epoch_seconds=4.0, n_epochs=2)
        self.assertEqual(labels.to_dict(), {0: 0, 1: 2})
        artifacts = pd.DataFrame(
            {"time_s": [0.0, 4.0, 8.0], "artifact": [False, True, True]}
        )
        self.assertEqual(
            artifact_epoch_ids(artifacts, epoch_seconds=4.0), {1, 2}
        )

    def test_non_signal_rows_exclude_the_whole_analysis_epoch(self) -> None:
        scores = pd.DataFrame(
            {
                "time_s": list(range(8)),
                "label_output": [0] * 8,
                "kind": ["signal", "signal", "gap", "signal"] + ["signal"] * 4,
            }
        )
        labels = epoch_sleep_states(scores, epoch_seconds=4.0, n_epochs=2)
        self.assertEqual(labels.to_dict(), {1: 0})

    def test_artifact_span_covers_shorter_analysis_epochs(self) -> None:
        artifacts = pd.DataFrame(
            {"time_s": [0.0, 4.0, 8.0], "artifact": [False, True, False]}
        )
        self.assertEqual(
            artifact_epoch_ids(artifacts, epoch_seconds=2.0), {2, 3}
        )


if __name__ == "__main__":
    unittest.main()
