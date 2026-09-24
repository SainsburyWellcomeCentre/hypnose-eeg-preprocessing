from __future__ import annotations

import unittest

import numpy as np

from hypnose_eeg.analysis.statistics import robust_upper_z


class RobustStatisticsTests(unittest.TestCase):
    def test_robust_upper_z_identifies_high_values(self) -> None:
        scores = robust_upper_z([1.0, 2.0, 3.0, 100.0])
        self.assertGreater(scores[-1], scores[2])
        self.assertGreater(scores[-1], 0.0)

    def test_constant_values_have_zero_scores(self) -> None:
        np.testing.assert_array_equal(robust_upper_z([5.0, 5.0]), [0.0, 0.0])

    def test_nonfinite_values_remain_nan(self) -> None:
        scores = robust_upper_z([1.0, np.nan, np.inf])
        self.assertEqual(scores[0], 0.0)
        self.assertTrue(np.isnan(scores[1:]).all())

    def test_mad_scale_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            robust_upper_z([1.0, 2.0], mad_scale=0.0)


if __name__ == "__main__":
    unittest.main()
