from __future__ import annotations

import unittest

import pandas as pd

from scripts.quality_control.sleep_scoring import (
    build_parser,
    prepare_scoring_output,
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


if __name__ == "__main__":
    unittest.main()
