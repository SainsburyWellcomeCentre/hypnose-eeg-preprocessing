from __future__ import annotations

import unittest

import pandas as pd

from hypnose_eeg.qc.sleep_scoring import (
    build_parser,
    prepare_scoring_output,
    sleep_state_proportion_report,
    sleep_state_proportions,
    scoring_summary,
)


class ScoringReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "label_output": [0, 1, 2, 3],
                "kind": ["signal", "signal", "signal", "gap"],
                "prob_wake": [0.90, 0.10, 0.05, 0.0],
                "prob_nrem": [0.05, 0.70, 0.15, 0.0],
                "prob_rem": [0.03, 0.15, 0.75, 0.0],
                "prob_undef": [0.02, 0.05, 0.05, 0.0],
            }
        )

    def test_output_selects_probability_for_predicted_state(self) -> None:
        output = prepare_scoring_output(self.scores, confidence_threshold=0.80)
        self.assertEqual(output["predicted_state"].tolist(), ["Wake", "NREM", "REM", "Undefined"])
        self.assertEqual(output["predicted_probability"].tolist(), [0.90, 0.70, 0.75, 0.0])
        self.assertEqual(output["low_confidence"].tolist(), [False, True, True, False])

    def test_summary_reports_duration_and_low_confidence_by_state(self) -> None:
        output = prepare_scoring_output(self.scores, confidence_threshold=0.80)
        summary = scoring_summary(output).set_index("sleep_state")
        self.assertEqual(summary.loc["NREM", "duration_s"], 1.0)
        self.assertEqual(summary.loc["NREM", "low_confidence_epochs"], 1)
        self.assertEqual(summary.loc["Undefined", "low_confidence_epochs"], 0)

    def test_cli_accepts_subject_with_date_or_session(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--date", "20260717"]).date,
            "20260717",
        )
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--session", "1"]).session,
            "1",
        )

    def test_missing_probability_columns_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "prob_rem"):
            prepare_scoring_output(
                self.scores.drop(columns="prob_rem"), confidence_threshold=0.80
            )

    def _wake_heavy_scores(self) -> pd.DataFrame:
        """10 signal epochs (8 Wake, 1 NREM, 1 REM) plus 2 gap epochs."""
        return pd.DataFrame(
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

    def test_sleep_state_proportions_use_signal_epochs_only(self) -> None:
        output = prepare_scoring_output(self._wake_heavy_scores(), confidence_threshold=0.80)
        proportions = sleep_state_proportions(output).set_index("sleep_state")

        self.assertEqual(proportions.loc["Wake", "epochs"], 8)
        self.assertAlmostEqual(proportions.loc["Wake", "signal_percent"], 80.0)
        self.assertAlmostEqual(proportions.loc["NREM", "signal_percent"], 10.0)
        self.assertAlmostEqual(proportions.loc["REM", "signal_percent"], 10.0)
        # The 2 gap epochs are Undefined but excluded from the denominator --
        # they must not appear as scored Undefined signal epochs either.
        self.assertEqual(proportions.loc["Undefined", "epochs"], 0)

    def test_sleep_state_proportion_report_flags_over_threshold_states(self) -> None:
        output = prepare_scoring_output(self._wake_heavy_scores(), confidence_threshold=0.80)
        report = sleep_state_proportion_report(
            sleep_state_proportions(output),
            max_wake_percent=75.0,
            max_nrem_percent=75.0,
            max_rem_percent=20.0,
        ).set_index("sleep_state")

        self.assertEqual(report.loc["Wake", "status"], "review")
        self.assertEqual(report.loc["NREM", "status"], "pass")
        self.assertEqual(report.loc["REM", "status"], "pass")
        self.assertEqual(report.loc["Undefined", "status"], "n/a")

    def test_sleep_state_proportion_report_flags_high_rem(self) -> None:
        output = prepare_scoring_output(self._wake_heavy_scores(), confidence_threshold=0.80)
        report = sleep_state_proportion_report(
            sleep_state_proportions(output),
            max_wake_percent=90.0,
            max_nrem_percent=75.0,
            max_rem_percent=5.0,
        ).set_index("sleep_state")

        self.assertEqual(report.loc["Wake", "status"], "pass")
        self.assertEqual(report.loc["REM", "status"], "review")


if __name__ == "__main__":
    unittest.main()
