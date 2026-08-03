from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.sleep_scoring.view_scored_recording import (
    ScoringViewSettings,
    _single_value,
    run_view,
)


class SleepScoringViewTests(unittest.TestCase):
    def test_single_value_rejects_ambiguous_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one subject"):
            _single_value([66, 67], option_name="subject")

    def test_run_view_forwards_somnotate_view_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            calls: list[list[str]] = []

            def fake_view(arguments: list[str]) -> int:
                calls.append(arguments)
                return 0

            settings = ScoringViewSettings(
                subject="66",
                date="20260717",
                repo_root=root,
                rawdata_root=rawdata,
                derivatives_root=derivatives,
                recording_index=1,
                eeg_channel=1,
                view_length_s=60.0,
            )

            result = run_view(settings, view_function=fake_view)

            self.assertEqual(result, 0)
            self.assertEqual(
                calls[0],
                [
                    "--sub", "66",
                    "--date", "20260717",
                    "--recording-index", "1",
                    "--eeg-channel", "1",
                    "--view-length", "60.0",
                    "--repo-root", str(root),
                ],
            )
            self.assertEqual(os.environ["HYPNOSE_EEG_RAWDATA_ROOT"], str(rawdata))
            self.assertEqual(os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"], str(derivatives))


if __name__ == "__main__":
    unittest.main()
