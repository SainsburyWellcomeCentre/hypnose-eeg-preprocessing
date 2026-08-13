from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.sleep_scoring.score_recordings import (
    SleepScoringSettings,
    _as_date_range,
    _relocate_scoring_outputs,
    _resolve_model_path,
    run_scoring,
)


class SleepScoringTests(unittest.TestCase):
    def test_native_scoring_outputs_are_relocated_from_saved_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "sub-066" / "ses-1_date-20260717"
            saved_results = session / "saved_results"
            saved_results.mkdir(parents=True)
            filenames = (
                "recording_somnotate_predictions.parquet",
                "recording_somnotate_segments.json",
                "recording_somnotate_predictions.txt",
            )
            for filename in filenames:
                (saved_results / filename).touch()

            relocated = _relocate_scoring_outputs(
                [saved_results / "recording_somnotate_predictions.parquet"]
            )

            destination = session / "sleep_scoring"
            self.assertEqual(
                relocated,
                [destination / "recording_somnotate_predictions.parquet"],
            )
            self.assertTrue(all((destination / name).is_file() for name in filenames))
            self.assertFalse(saved_results.exists())

    def test_model_name_resolves_below_derivatives(self) -> None:
        derivatives = Path("/data/derivatives")
        self.assertEqual(
            _resolve_model_path("baseline", derivatives),
            derivatives / "somnotate_training" / "baseline" / "model.pickle",
        )

    def test_date_range_accepts_config_and_compact_string(self) -> None:
        self.assertEqual(_as_date_range(["20260707", "20260718"]), ("20260707", "20260718"))
        self.assertEqual(_as_date_range("20260707-20260718"), ("20260707", "20260718"))

    def test_run_scoring_bridges_data_roots_and_forwards_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            expected_output = derivatives / "predictions.parquet"
            calls: list[dict] = []

            def fake_score_recordings(**kwargs):
                calls.append(kwargs)
                return [expected_output]

            settings = SleepScoringSettings(
                subjids=["66"],
                model_path=model,
                repo_root=root,
                rawdata_root=rawdata,
                derivatives_root=derivatives,
                dates=["20260717"],
                sessions=None,
                date_range=None,
                channel_labels=["EEG1", "EEG2", "EMG"],
                export_visbrain=False,
                sampling_rate_hz=512,
            )

            outputs = run_scoring(settings, score_function=fake_score_recordings)

            self.assertEqual(outputs, [expected_output])
            self.assertEqual(calls[0]["subjids"], ["66"])
            self.assertEqual(calls[0]["dates"], ["20260717"])
            self.assertEqual(calls[0]["model_path"], model)
            self.assertFalse(calls[0]["export_visbrain"])
            self.assertEqual(os.environ["HYPNOSE_EEG_RAWDATA_ROOT"], str(rawdata))
            self.assertEqual(os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"], str(derivatives))

    def test_run_scoring_resolves_session_numbers_per_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            (rawdata / "sub-066" / "ses-2_date-20260718").mkdir(parents=True)
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            calls: list[dict] = []

            def fake_score_recordings(**kwargs):
                calls.append(kwargs)
                return [derivatives / "predictions.parquet"]

            settings = SleepScoringSettings(
                subjids=["66"],
                model_path=model,
                repo_root=root,
                rawdata_root=rawdata,
                derivatives_root=derivatives,
                dates=None,
                sessions=[2],
                date_range=None,
                channel_labels=["EEG1", "EEG2", "EMG"],
                export_visbrain=False,
                sampling_rate_hz=512,
            )

            outputs = run_scoring(settings, score_function=fake_score_recordings)

            self.assertEqual(outputs, [derivatives / "predictions.parquet"])
            self.assertEqual(calls[0]["subjids"], ["66"])
            self.assertEqual(calls[0]["dates"], ["20260718"])
            self.assertIsNone(calls[0]["date_range"])


if __name__ == "__main__":
    unittest.main()
