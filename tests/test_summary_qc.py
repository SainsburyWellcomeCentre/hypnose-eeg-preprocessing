from __future__ import annotations

import unittest

import pandas as pd

from scripts.quality_control.summary_qc import (
    REVIEW_COLUMNS,
    build_parser,
    overall_status,
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


if __name__ == "__main__":
    unittest.main()
