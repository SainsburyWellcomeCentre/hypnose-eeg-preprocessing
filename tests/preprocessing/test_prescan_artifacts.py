from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hypnose_eeg.preprocessing.prescan_artifacts import (
    PrescanSettings,
    artifact_periods,
    flag_prescan_epochs,
    period_table,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/pipelines/artifact_detection.yaml"
)


def _flags(n_epochs: int, *flagged_ranges: tuple[int, int]) -> np.ndarray:
    flags = np.zeros(n_epochs, dtype=bool)
    for start, stop in flagged_ranges:
        flags[start:stop] = True
    return flags


def _periods(flags: np.ndarray, **overrides: float) -> list[tuple[int, int]]:
    """Periods with 4 s epochs and the committed defaults unless overridden."""
    options = dict(merge_within_s=30.0, minimum_run_s=60.0, bridge_gap_s=300.0)
    options.update(overrides)
    return artifact_periods(flags, 4.0, **options)


class ArtifactPeriodTests(unittest.TestCase):
    def test_short_runs_are_left_to_the_post_scoring_detector(self) -> None:
        # 40 s of artifact, far from anything else: below minimum_run_s.
        self.assertEqual(_periods(_flags(1000, (100, 110))), [])

    def test_clean_epochs_inside_a_noisy_block_do_not_split_it(self) -> None:
        # Two 40 s runs 20 s apart -- each too short alone, one 100 s run together.
        self.assertEqual(_periods(_flags(1000, (100, 110), (115, 125))), [(100, 125)])

    def test_signal_shorter_than_bridge_gap_between_periods_is_absorbed(self) -> None:
        # 60 s periods separated by 296 s (74 epochs) of clean signal.
        flags = _flags(1000, (100, 115), (189, 204))
        self.assertEqual(_periods(flags), [(100, 204)])

    def test_signal_of_exactly_bridge_gap_or_longer_is_kept(self) -> None:
        # Exactly 300 s (75 epochs) of clean signal: not "less than 5 minutes".
        flags = _flags(1000, (100, 115), (190, 205))
        self.assertEqual(_periods(flags), [(100, 115), (190, 205)])

    def test_bridge_gap_is_configurable(self) -> None:
        flags = _flags(1000, (100, 115), (190, 205))
        self.assertEqual(_periods(flags, bridge_gap_s=600.0), [(100, 205)])
        self.assertEqual(_periods(flags, bridge_gap_s=0.0), [(100, 115), (190, 205)])

    def test_short_runs_do_not_bridge_long_periods(self) -> None:
        # A lone 8 s artifact between two periods is dropped before bridging, so
        # it cannot chain periods 8 minutes apart through two sub-5-minute gaps.
        flags = _flags(1000, (100, 115), (170, 172), (235, 250))
        self.assertEqual(_periods(flags), [(100, 115), (235, 250)])

    def test_periods_touching_the_recording_edges(self) -> None:
        self.assertEqual(_periods(_flags(100, (0, 20), (80, 100))), [(0, 100)])


class FlagPrescanEpochsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = PrescanSettings.from_yaml(CONFIG_PATH)

    def _features(self, n_epochs: int, rms_uv: np.ndarray) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for channel in ("EEG1", "EEG2"):
            jitter = np.exp(rng.normal(0, 0.1, n_epochs))
            rms = rms_uv * jitter
            rows.append(
                pd.DataFrame(
                    {
                        "epoch_id": np.arange(n_epochs),
                        "time_s": np.arange(n_epochs) * 4.0,
                        "channel": channel,
                        "peak_to_peak_uv": rms * 6,
                        "rms_uv": rms,
                        "max_derivative_uv_per_sample": rms / 2,
                        "high_frequency_power_uv2": rms**2 / 10,
                        "broadband_power_uv2": rms**2,
                        "edge_fraction": np.full(n_epochs, 0.002),
                        "nonfinite": np.zeros(n_epochs, dtype=bool),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_dead_signal_is_flagged_even_when_it_dominates_the_recording(self) -> None:
        # 80% dead (0.5 uV) and 20% real EEG (50 uV): the real EEG must not be
        # scored as extreme against the dead majority, and the dead part is flagged.
        rms = np.r_[np.full(800, 0.5), np.full(200, 50.0)]
        epochs = flag_prescan_epochs(self._features(1000, rms), self.settings)
        self.assertTrue(epochs["dead"].iloc[:800].all())
        self.assertFalse(epochs["flagged"].iloc[800:].any())

    def test_extreme_amplitude_is_flagged_against_live_epochs(self) -> None:
        rms = np.full(1000, 50.0)
        rms[500:520] = 50.0 * 1e3
        epochs = flag_prescan_epochs(self._features(1000, rms), self.settings)
        self.assertTrue(epochs["extreme"].iloc[500:520].all())
        self.assertFalse(epochs["flagged"].drop(index=range(500, 520)).any())

    def test_clipping_is_a_hard_failure(self) -> None:
        features = self._features(100, np.full(100, 50.0))
        features.loc[features["epoch_id"] == 10, "edge_fraction"] = 0.5
        epochs = flag_prescan_epochs(features, self.settings)
        self.assertEqual(epochs.index[epochs["hard_failure"]].tolist(), [10])

    def test_period_table_reports_span_and_reasons(self) -> None:
        epochs = pd.DataFrame(
            {
                "flagged": [True, False, True, True],
                "dead": [True, False, False, False],
                "hard_failure": [False, False, True, False],
                "extreme": [False, False, False, True],
            }
        )
        table = period_table(epochs, [(0, 4)], 4.0)
        row = table.iloc[0]
        self.assertEqual((row.start_s, row.end_s, row.duration_s), (0.0, 16.0, 16.0))
        self.assertEqual(row.flagged_fraction, 0.75)
        self.assertEqual(
            (row.dead_epochs, row.hard_failure_epochs, row.extreme_epochs), (1, 1, 1)
        )

    def test_settings_reject_negative_period_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "bridge_gap_s"):
            PrescanSettings.from_yaml(CONFIG_PATH, bridge_gap_s=-1.0)

    def test_cli_overrides_replace_yaml_values(self) -> None:
        settings = PrescanSettings.from_yaml(CONFIG_PATH, bridge_gap_s=120.0)
        self.assertEqual(settings.bridge_gap_s, 120.0)
        self.assertEqual(replace(settings, bridge_gap_s=300.0), self.settings)


if __name__ == "__main__":
    unittest.main()
