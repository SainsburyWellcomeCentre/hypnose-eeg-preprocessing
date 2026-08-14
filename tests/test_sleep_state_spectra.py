from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.quality_control.spectra import (
    DEFAULT_CONFIG_PATH,
    build_parser,
    build_spectral_quality_report,
    load_spectra_config,
    plot_state_emg_rms,
)
from scripts.analysis.emg import compute_state_emg_rms
from scripts.analysis.power_spectra import compute_state_spectra
from scripts.io.input_paths import (
    artifact_path,
    scoring_path,
    session_derivatives_dir,
)
from scripts.io.output_paths import (
    artifact_output_path,
    quality_control_output_path,
    sleep_scoring_output_path,
)
from scripts.utils.epochs import artifact_epoch_ids, epoch_sleep_states


class SleepStateSpectraTests(unittest.TestCase):
    def test_spectral_definitions_are_loaded_from_pipeline_config(self) -> None:
        config = load_spectra_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.frequency_bands_hz["theta"], (4.0, 10.0))
        self.assertEqual(config.determining_eeg_channel_number, 1)
        self.assertEqual(config.emg_state_order, (0, 1, 2))

    def test_quality_report_checks_expected_sleep_state_patterns(self) -> None:
        frequencies = np.array(
            [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30],
            dtype=float,
        )
        wake = np.ones((2, len(frequencies)))
        nrem = np.ones((2, len(frequencies)))
        rem = np.ones((2, len(frequencies)))
        wake[:, (frequencies >= 12) & (frequencies <= 30)] = 6.0
        nrem[:, frequencies <= 4] = 8.0
        rem[:, (frequencies >= 4) & (frequencies <= 8)] = 8.0
        report = build_spectral_quality_report(
            frequencies,
            {0: wake, 1: nrem, 2: rem},
            {0: 10, 1: 10, 2: 10},
            ["EEG1", "EEG2"],
            {
                0: np.array([[3.0], [4.0]]),
                1: np.array([[2.0], [2.5]]),
                2: np.array([[1.0], [1.5]]),
            },
            ["EMG"],
        ).set_index(["sleep_state", "eeg_channel"])

        self.assertEqual(report.loc[("Wake", "EEG1"), "quality_status"], "PASS")
        self.assertEqual(report.loc[("NREM", "EEG1"), "quality_status"], "PASS")
        self.assertEqual(report.loc[("REM", "EEG1"), "quality_status"], "PASS")
        self.assertTrue(report["frequency_expectation_met"].all())
        self.assertTrue(report["emg_expectation_met"].all())
        self.assertEqual(report.loc[("Wake", "EEG1"), "emg_rms_median_uv"], 3.5)
        self.assertEqual(report.loc[("Wake", "EEG1"), "eeg_epoch_count"], 10)
        self.assertTrue((report["recording_quality_status"] == "PASS").all())

    def test_invalid_spectrum_fails_quality_report(self) -> None:
        frequencies = np.arange(0.5, 30.5, 0.5)
        valid = np.ones((1, len(frequencies)))
        invalid = valid.copy()
        invalid[0, 0] = np.nan
        report = build_spectral_quality_report(
            frequencies,
            {0: valid, 1: invalid},
            {0: 5, 1: 5, 2: 0},
            ["EEG1"],
            {0: np.array([[3.0]]), 1: np.array([[2.0]])},
            ["EMG"],
        ).set_index("sleep_state")

        self.assertEqual(report.loc["NREM", "spectral_quality_status"], "FAIL")
        self.assertEqual(report.loc["REM", "spectral_quality_status"], "REVIEW")
        self.assertTrue((report["recording_quality_status"] == "FAIL").all())

    def test_only_first_eeg_channel_determines_recording_status(self) -> None:
        frequencies = np.arange(0.5, 30.5, 0.5)
        wake = np.ones((2, len(frequencies)))
        nrem = np.ones((2, len(frequencies)))
        rem = np.ones((2, len(frequencies)))
        nrem[0, frequencies <= 4] = 8.0
        rem[0, (frequencies >= 4) & (frequencies <= 8)] = 8.0
        wake[1] = np.nan
        nrem[1] = np.nan
        rem[1] = np.nan
        report = build_spectral_quality_report(
            frequencies,
            {0: wake, 1: nrem, 2: rem},
            {0: 5, 1: 5, 2: 5},
            ["EEG1", "EEG2"],
            {
                0: np.array([[3.0]]),
                1: np.array([[2.0]]),
                2: np.array([[1.0]]),
            },
            ["EMG"],
        )

        channel_1 = report.loc[report["eeg_channel"] == "EEG1"]
        channel_2 = report.loc[report["eeg_channel"] == "EEG2"]
        self.assertTrue((channel_1["quality_status"] == "PASS").all())
        self.assertTrue((channel_2["spectral_quality_status"] == "FAIL").all())
        self.assertTrue((report["recording_quality_status"] == "PASS").all())

    def test_spectrum_computation_is_available_from_analysis(self) -> None:
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
            session = derivatives / "sub-066" / "ses-1_date-20260717"
            scoring_dir = session / "sleep_scoring"
            artifact_dir = session / "artifacts"
            edf.parent.mkdir(parents=True)
            scoring_dir.mkdir(parents=True)
            artifact_dir.mkdir()
            scoring = scoring_dir / "recording_somnotate_predictions.parquet"
            artifact = artifact_dir / "recording_artifact_epochs.parquet"
            scoring.touch()
            artifact.touch()
            self.assertEqual(
                session_derivatives_dir(edf, rawdata, derivatives), session
            )
            self.assertEqual(scoring_path(edf, rawdata, derivatives), scoring)
            self.assertEqual(artifact_path(edf, rawdata, derivatives), artifact)
            self.assertEqual(
                quality_control_output_path(
                    "qc_summary.csv", edf, rawdata, derivatives
                ),
                session / "quality_control" / "qc_summary.csv",
            )
            self.assertEqual(
                sleep_scoring_output_path(
                    "somnotate_scoring_summary.csv", edf, rawdata, derivatives
                ),
                session / "sleep_scoring" / "somnotate_scoring_summary.csv",
            )
            self.assertEqual(
                artifact_output_path(
                    "recording_artifact_report", edf, rawdata, derivatives
                ),
                session / "artifacts" / "recording_artifact_report",
            )

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
