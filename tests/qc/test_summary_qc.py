from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.qc.sleep_scoring import (
    prepare_scoring_output,
    sleep_state_proportion_report,
    sleep_state_proportions,
)
from scripts.qc.spectra import DEFAULT_SPECTRA_CONFIG
from scripts.qc.summary_qc import (
    REVIEW_COLUMNS,
    build_parser,
    overall_status,
    review_output_paths,
    sleep_state_proportion_section,
    spectral_quality_sections,
)


class SummaryQualityControlTests(unittest.TestCase):
    def test_overall_status_uses_most_severe_section(self) -> None:
        self.assertEqual(
            overall_status(pd.DataFrame({"status": ["pass", "review", "pass"]})),
            "review",
        )
        self.assertEqual(
            overall_status(pd.DataFrame({"status": ["review", "fail"]})),
            "fail",
        )
        self.assertEqual(overall_status(pd.DataFrame()), "fail")

    def test_review_schema_uses_time_ranges(self) -> None:
        self.assertEqual(REVIEW_COLUMNS[1:4], ["start_s", "end_s", "duration_s"])

    def test_review_outputs_include_csv_and_parquet(self) -> None:
        self.assertEqual(
            review_output_paths("qc_review_epochs.csv"),
            (
                Path("qc_review_epochs.csv"),
                Path("qc_review_epochs.parquet"),
            ),
        )
        self.assertEqual(
            review_output_paths("custom.parquet"),
            (Path("custom.csv"), Path("custom.parquet")),
        )

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--date", "20260717"]).date,
            "20260717",
        )
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--session", "1"]).session,
            "1",
        )

    def test_spectral_sections_use_configured_threshold_results(self) -> None:
        report = pd.DataFrame(
            {
                "determines_recording_quality": [True, True, True, False],
                "spectral_quality_status": ["PASS", "REVIEW", "PASS", "FAIL"],
                "emg_quality_status": ["PASS", "PASS", "PASS", "FAIL"],
                "frequency_expectation_met": [True, False, True, False],
                "emg_expectation_met": [True, True, True, False],
                "eeg_channel": ["EEG1", "EEG1", "EEG1", "EEG2"],
                "sleep_state": ["Wake", "NREM", "REM", "Wake"],
                "emg_rms_median_uv": [3.0, 2.0, 1.0, 4.0],
            }
        )

        sections = spectral_quality_sections(report, DEFAULT_SPECTRA_CONFIG)

        self.assertEqual(sections[0]["section"], "power_spectra")
        self.assertEqual(sections[0]["status"], "review")
        self.assertEqual(sections[0]["value"], "2/3")
        self.assertIn("wake_delta<nrem", str(sections[0]["threshold"]))
        self.assertIn("determining EEG channel 1=EEG1", str(sections[0]["detail"]))
        self.assertEqual(sections[1]["section"], "emg_rms")
        self.assertEqual(sections[1]["status"], "pass")
        self.assertEqual(sections[1]["threshold"], "Wake>=NREM>=REM")

    def test_sleep_state_proportion_section_flags_over_threshold_state(self) -> None:
        scores = pd.DataFrame(
            {
                "time_s": list(range(12)),
                "label_output": [0] * 8 + [1, 2, 3, 3],
                "kind": ["signal"] * 10 + ["gap", "gap"],
                "prob_wake": [0.9] * 8 + [0.10, 0.10, 0.0, 0.0],
                "prob_nrem": [0.05] * 8 + [0.80, 0.10, 0.0, 0.0],
                "prob_rem": [0.03] * 8 + [0.05, 0.85, 0.0, 0.0],
                "prob_undef": [0.02] * 8 + [0.05, 0.05, 1.0, 1.0],
            }
        )
        output = prepare_scoring_output(scores, confidence_threshold=0.80)
        proportion_report = sleep_state_proportion_report(
            sleep_state_proportions(output),
            max_wake_percent=75.0,
            max_nrem_percent=75.0,
            max_rem_percent=20.0,
        )

        section = sleep_state_proportion_section(proportion_report)

        self.assertEqual(section["section"], "sleep_state_proportions")
        self.assertEqual(section["status"], "review")
        self.assertIn("Wake=80.00%", section["value"])
        self.assertIn("Wake<=75%", section["threshold"])
        self.assertIn("Wake review", section["detail"])
        self.assertNotIn("Undefined", section["value"])

    def test_sleep_state_proportion_section_passes_within_thresholds(self) -> None:
        proportions = pd.DataFrame(
            {
                "sleep_state": ["Wake", "NREM", "REM", "Undefined"],
                "epochs": [3, 4, 1, 0],
                "signal_percent": [37.5, 50.0, 12.5, 0.0],
            }
        )
        proportion_report = sleep_state_proportion_report(
            proportions,
            max_wake_percent=75.0,
            max_nrem_percent=75.0,
            max_rem_percent=20.0,
        )

        section = sleep_state_proportion_section(proportion_report)

        self.assertEqual(section["status"], "pass")
        self.assertEqual(section["detail"], "within thresholds")


if __name__ == "__main__":
    unittest.main()
