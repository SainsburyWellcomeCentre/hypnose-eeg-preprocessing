from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from scripts.quality_control.plot_channel_correlations import (
    _print_review_counts,
    build_parser,
)
from scripts.utils.correlation import epoch_pearson_matrices


class ChannelCorrelationTests(unittest.TestCase):
    def test_epoch_pearson_detects_positive_negative_and_constant_pairs(self) -> None:
        data = np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 4.0, 6.0, 8.0],
                    [4.0, 3.0, 2.0, 1.0],
                    [5.0, 5.0, 5.0, 5.0],
                ]
            ]
        )
        correlations = epoch_pearson_matrices(data)
        self.assertAlmostEqual(correlations[0, 0, 1], 1.0)
        self.assertAlmostEqual(correlations[0, 0, 2], -1.0)
        self.assertTrue(np.isnan(correlations[0, 0, 3]))

    def test_nonfinite_channel_does_not_produce_a_correlation(self) -> None:
        data = np.array([[[1.0, 2.0], [1.0, np.nan]]])
        correlations = epoch_pearson_matrices(data)
        self.assertTrue(np.isnan(correlations[0, 0, 1]))

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        by_date = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717"]
        )
        self.assertEqual(by_date.date, "20260717")
        by_session = build_parser().parse_args(
            ["--subject", "66", "--session", "1"]
        )
        self.assertEqual(by_session.session, "1")

    def test_review_counts_use_pair_thresholds_and_unique_epochs(self) -> None:
        correlations = pd.DataFrame(
            {
                "epoch_id": [0, 1, 0, 1],
                "sleep_state": [0, 0, 0, 0],
                "channel_1": ["EEG1", "EEG1", "EEG1", "EEG1"],
                "channel_2": ["EEG2", "EEG2", "EMG", "EMG"],
                "channel_1_type": ["eeg"] * 4,
                "channel_2_type": ["eeg", "eeg", "emg", "emg"],
                "pearson_r": [0.95, 0.20, -0.75, 0.10],
            }
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _print_review_counts(
                correlations,
                eeg_eeg_threshold=0.90,
                eeg_emg_threshold=0.50,
            )
        printed = output.getvalue()
        self.assertIn("EEG1 ↔ EEG2 — Wake: 1 / 2 (50.00%)", printed)
        self.assertIn("EEG1 ↔ EMG — Wake: 1 / 2 (50.00%)", printed)
        self.assertIn("Unique epochs above any configured threshold: 1 / 2", printed)


if __name__ == "__main__":
    unittest.main()
