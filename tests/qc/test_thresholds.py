from __future__ import annotations

import unittest

from hypnose_eeg.qc.summary_qc import summary_qc_settings
from hypnose_eeg.qc.thresholds import default_qc_thresholds
from hypnose_eeg.utils.config import with_overrides


class QCThresholdTests(unittest.TestCase):
    def test_unset_overrides_keep_the_configured_values(self) -> None:
        thresholds = default_qc_thresholds()
        self.assertEqual(with_overrides(thresholds, max_wake_percent=None), thresholds)

    def test_overrides_replace_single_values(self) -> None:
        thresholds = with_overrides(default_qc_thresholds(), max_wake_percent=60.0)
        self.assertEqual(thresholds.max_wake_percent, 60.0)
        self.assertEqual(thresholds.max_rem_percent, default_qc_thresholds().max_rem_percent)

    def test_out_of_range_values_are_refused(self) -> None:
        for name, value in (
            ("max_wake_percent", 150.0),
            ("max_gap_percent", -1.0),
            ("confidence_threshold", 1.5),
            ("eeg_emg_threshold", float("nan")),
            ("min_gap_s", 0.0),
            ("duration_tolerance_s", -0.5),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                with_overrides(default_qc_thresholds(), **{name: value})

    def test_zero_tolerance_is_allowed(self) -> None:
        thresholds = with_overrides(default_qc_thresholds(), duration_tolerance_s=0.0)
        self.assertEqual(thresholds.duration_tolerance_s, 0.0)


class SummaryQCSettingsTests(unittest.TestCase):
    def test_overrides_reach_thresholds_and_spectra_analysis(self) -> None:
        settings = summary_qc_settings(
            max_artifact_percent=9.0, epoch_seconds=2.0, fmax_hz=40.0
        )
        self.assertEqual(settings.thresholds.max_artifact_percent, 9.0)
        self.assertEqual(settings.spectra.epoch_seconds, 2.0)
        self.assertEqual(settings.spectra.fmax_hz, 40.0)
        self.assertTrue(settings.qc_config.endswith("quality_control.yaml"))

    def test_unset_overrides_keep_the_configuration(self) -> None:
        settings = summary_qc_settings(max_artifact_percent=None, epoch_seconds=None)
        self.assertEqual(settings.thresholds, default_qc_thresholds())

    def test_unknown_and_invalid_settings_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "bogus"):
            summary_qc_settings(bogus=1.0)
        with self.assertRaises(ValueError):
            summary_qc_settings(fmin_hz=50.0, fmax_hz=10.0)
        with self.assertRaises(ValueError):
            summary_qc_settings(max_correlation_review_percent=101.0)


if __name__ == "__main__":
    unittest.main()
