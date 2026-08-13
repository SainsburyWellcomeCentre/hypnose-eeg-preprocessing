from __future__ import annotations

import unittest

import pandas as pd

from scripts.quality_control.report_artifacts import build_parser
from scripts.quality_control.report_artifacts import build_artifact_report


class ArtifactReportingTests(unittest.TestCase):
    def test_report_separates_channels_hours_states_and_periods(self) -> None:
        table = pd.DataFrame(
            {
                "epoch_id": [0, 1, 2, 900, 901],
                "time_s": [0.0, 4.0, 8.0, 3600.0, 3604.0],
                "sleep_state": [0, 0, 1, 2, 2],
                "artifact": [True, True, False, True, True],
                "artifact_features": [
                    "EEG1:flatline | EEG2:line_noise",
                    "EEG1:flatline",
                    "",
                    "EEG2:emg_outlier_with_eeg_outlier",
                    "EEG2:high_frequency_power",
                ],
                "emg_extreme": [False, False, False, True, False],
            }
        )
        report = build_artifact_report(table)
        self.assertEqual(report.epoch_seconds, 4.0)
        overall = report.overall.set_index("channel")
        self.assertEqual(overall.loc["EEG1", "artifact_periods"], 1)
        self.assertEqual(overall.loc["EEG1", "artifact_duration_s"], 8.0)
        self.assertEqual(overall.loc["EEG2", "artifact_periods"], 2)
        self.assertEqual(overall.loc["EEG2", "longest_artifact_s"], 8.0)
        self.assertIn("EMG (combined)", overall.index)

        eeg2_hours = report.hourly.loc[report.hourly["channel"] == "EEG2"]
        self.assertEqual(set(eeg2_hours["hour"]), {0, 1})
        eeg1_hour_1 = report.hourly.loc[
            (report.hourly["channel"] == "EEG1") & (report.hourly["hour"] == 1)
        ].iloc[0]
        self.assertEqual(eeg1_hour_1["artifact_epochs"], 0)
        self.assertEqual(eeg1_hour_1["hour_percent"], 0.0)
        eeg2_rem = report.sleep_state.loc[
            (report.sleep_state["channel"] == "EEG2")
            & (report.sleep_state["sleep_state"] == "REM")
        ].iloc[0]
        self.assertEqual(eeg2_rem["artifact_epochs"], 2)
        self.assertEqual(eeg2_rem["state_percent"], 100.0)

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        self.assertEqual(
            build_parser().parse_args(
                ["--subject", "66", "--date", "20260717"]
            ).date,
            "20260717",
        )
        self.assertEqual(
            build_parser().parse_args(
                ["--subject", "66", "--session", "1"]
            ).session,
            "1",
        )


if __name__ == "__main__":
    unittest.main()
