from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hypnose_somnotate.preprocessing.gap_correction import (
    PreparedRecording,
    RecordingSegment,
    ScoringChunk,
)

from scripts.sleep_scoring.score_recordings import (
    SleepScoringSettings,
    _as_date_range,
    _channel_label_alias,
    _match_channel_labels,
    _resolve_model_path,
    run_scoring,
)


def _fake_prepared(duration_s: float = 10.0) -> PreparedRecording:
    """A minimal, valid PreparedRecording -- one signal segment, no gaps."""
    return PreparedRecording(
        segments=[RecordingSegment(0, "signal", 0.0, duration_s)],
        scoring_chunks=[ScoringChunk(0.0, duration_s, np.zeros((1, 3)))],
        strategy="trim_only",
        original_duration_s=duration_s,
        total_missing_s=0.0,
        missing_fraction=0.0,
        longest_gap_s=0.0,
        sampling_rate_hz=512.0,
        time_resolution_s=1.0,
    )


def _fake_predictions_df(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": np.arange(n, dtype=float),
            "label": np.zeros(n, dtype=int),
            "label_model": np.ones(n, dtype=int),
            "label_output": np.zeros(n, dtype=int),
            "segment_id": np.zeros(n, dtype=int),
            "kind": ["signal"] * n,
            "prob_wake": np.ones(n),
            "prob_nrem": np.zeros(n),
            "prob_rem": np.zeros(n),
            "prob_undef": np.zeros(n),
        }
    )


def _make_session(rawdata: Path, sub_label: str, session_dirname: str, edf_names: list[str]) -> Path:
    """Create a session's ephys/ folder with empty stand-in EDF files."""
    ephys_dir = rawdata / sub_label / session_dirname / "ephys"
    ephys_dir.mkdir(parents=True)
    for name in edf_names:
        (ephys_dir / name).touch()
    return ephys_dir


def _settings(
    *, root: Path, rawdata: Path, derivatives: Path, model: Path, **overrides
) -> SleepScoringSettings:
    """SleepScoringSettings with test-friendly defaults; pass overrides to vary one field."""
    fields = dict(
        subjids=["66"],
        model_path=model,
        repo_root=root,
        rawdata_root=rawdata,
        derivatives_root=derivatives,
        dates=None,
        sessions=None,
        date_range=None,
        channel_labels=["EEG1", "EEG2", "EMG"],
        export_visbrain=False,
        sampling_rate_hz=512,
        global_normalization=False,
        use_artifact_prescan=False,
    )
    fields.update(overrides)
    return SleepScoringSettings(**fields)


class SleepScoringTests(unittest.TestCase):
    def test_channel_label_alias_round_trips_prefix(self) -> None:
        self.assertEqual(_channel_label_alias("EEG EEG1A-B"), "EEG1A-B")
        self.assertEqual(_channel_label_alias("EEG1A-B"), "EEG EEG1A-B")
        self.assertEqual(_channel_label_alias("EMG EMG"), "EMG")
        self.assertEqual(_channel_label_alias("EMG"), "EMG EMG")

    def test_match_channel_labels_accepts_either_spelling(self) -> None:
        prefixed = {"EEG EEG1A-B", "EEG EEG2A-B", "EMG EMG"}
        bare = {"EEG1A-B", "EEG2A-B", "EMG"}
        requested = ["EEG EEG1A-B", "EEG EEG2A-B", "EMG EMG"]

        self.assertEqual(
            _match_channel_labels(requested, prefixed, source="raw.edf"),
            requested,
        )
        self.assertEqual(
            _match_channel_labels(requested, bare, source="concat.edf"),
            ["EEG1A-B", "EEG2A-B", "EMG"],
        )

    def test_match_channel_labels_raises_when_neither_spelling_present(self) -> None:
        with self.assertRaises(ValueError):
            _match_channel_labels(["EEG EEG1A-B"], {"ECG"}, source="raw.edf")

    def test_model_name_resolves_below_derivatives(self) -> None:
        derivatives = Path("/data/derivatives")
        self.assertEqual(
            _resolve_model_path("baseline", derivatives),
            derivatives / "somnotate_training" / "baseline" / "model.pickle",
        )

    def test_date_range_accepts_config_and_compact_string(self) -> None:
        self.assertEqual(_as_date_range(["20260707", "20260718"]), ("20260707", "20260718"))
        self.assertEqual(_as_date_range("20260707-20260718"), ("20260707", "20260718"))

    def test_run_scoring_scores_the_matched_recording_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            ephys_dir = _make_session(
                rawdata, "sub-066", "ses-001_date-20260717", ["sub-066_ses-001_recording-001.edf"]
            )
            calls: list[dict] = []

            def fake_score_recording(edf_path, model_path, **kwargs):
                calls.append({"edf_path": edf_path, "model_path": model_path, **kwargs})
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                dates=["20260717"], global_normalization=True,
            )

            outputs = run_scoring(settings, score_function=fake_score_recording)

            expected_output = (
                derivatives / "sub-066" / "ses-001_date-20260717" / "eeg" / "sleep_scoring"
                / "sub-066_ses-001_recording-001_somnotate_predictions.parquet"
            )
            self.assertEqual(outputs, [expected_output])
            self.assertTrue(expected_output.is_file())

            provenance = json.loads(
                (expected_output.parent
                 / "sub-066_ses-001_recording-001_somnotate_predictions_provenance.json"
                 ).read_text()
            )
            self.assertEqual(provenance["stage"], "sleep_scoring")
            self.assertIn("git", provenance)
            self.assertEqual(provenance["parameters"]["model"]["path"], str(model))
            self.assertEqual(
                provenance["parameters"]["model"]["sha256"],
                hashlib.sha256(model.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["inputs"]["subject"], "sub-066")

            self.assertEqual(calls[0]["edf_path"], ephys_dir / "sub-066_ses-001_recording-001.edf")
            self.assertEqual(calls[0]["model_path"], model)
            self.assertEqual(calls[0]["channel_labels"], ["EEG1", "EEG2", "EMG"])
            self.assertEqual(calls[0]["sampling_rate_hz"], 512)
            self.assertTrue(calls[0]["global_normalization"])

    def test_run_scoring_skips_a_recording_that_is_already_scored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            _make_session(
                rawdata, "sub-066", "ses-001_date-20260717", ["sub-066_ses-001_recording-001.edf"]
            )
            existing = (
                derivatives / "sub-066" / "ses-001_date-20260717" / "eeg" / "sleep_scoring"
                / "sub-066_ses-001_recording-001_somnotate_predictions.parquet"
            )
            existing.parent.mkdir(parents=True)
            existing.touch()
            calls: list[dict] = []

            def fake_score_recording(edf_path, model_path, **kwargs):
                calls.append({"edf_path": edf_path})
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                dates=["20260717"],
            )

            outputs = run_scoring(settings, score_function=fake_score_recording)

            self.assertEqual(outputs, [])
            self.assertEqual(calls, [])
            self.assertEqual(existing.stat().st_size, 0)

    def test_run_scoring_rescores_an_existing_recording_when_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            _make_session(
                rawdata, "sub-066", "ses-001_date-20260717", ["sub-066_ses-001_recording-001.edf"]
            )
            existing = (
                derivatives / "sub-066" / "ses-001_date-20260717" / "eeg" / "sleep_scoring"
                / "sub-066_ses-001_recording-001_somnotate_predictions.parquet"
            )
            existing.parent.mkdir(parents=True)
            existing.touch()

            def fake_score_recording(edf_path, model_path, **kwargs):
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                dates=["20260717"], overwrite=True,
            )

            outputs = run_scoring(settings, score_function=fake_score_recording)

            self.assertEqual(outputs, [existing])
            self.assertGreater(existing.stat().st_size, 0)

    def test_run_scoring_resolves_session_numbers_per_subject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            _make_session(
                rawdata, "sub-066", "ses-002_date-20260718", ["sub-066_ses-002_recording-001.edf"]
            )
            calls: list[dict] = []

            def fake_score_recording(edf_path, model_path, **kwargs):
                calls.append({"edf_path": edf_path, **kwargs})
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                sessions=[2],
            )

            outputs = run_scoring(settings, score_function=fake_score_recording)

            self.assertEqual(len(outputs), 1)
            self.assertEqual(
                calls[0]["edf_path"].name, "sub-066_ses-002_recording-001.edf"
            )

    def test_run_scoring_prefers_the_concatenated_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            _make_session(
                rawdata,
                "sub-066",
                "ses-001_date-20260717",
                [
                    "sub-066_ses-001_recording-001.edf",
                    "sub-066_ses-001_recording-002.edf",
                    "sub-066_ses-001_recording-concat.edf",
                ],
            )
            calls: list[dict] = []

            def fake_score_recording(edf_path, model_path, **kwargs):
                calls.append({"edf_path": edf_path, **kwargs})
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                dates=["20260717"],
            )

            outputs = run_scoring(settings, score_function=fake_score_recording)

            self.assertEqual(len(outputs), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["edf_path"].name, "sub-066_ses-001_recording-concat.edf")

    def test_run_scoring_skips_multi_part_session_without_a_concat_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            derivatives.mkdir()
            model = derivatives / "model.pickle"
            model.touch()
            _make_session(
                rawdata,
                "sub-066",
                "ses-001_date-20260717",
                ["sub-066_ses-001_recording-001.edf", "sub-066_ses-001_recording-002.edf"],
            )
            calls: list[dict] = []

            def fake_score_recording(edf_path, model_path, **kwargs):
                calls.append({"edf_path": edf_path, **kwargs})
                return _fake_predictions_df(), _fake_prepared()

            settings = _settings(
                root=root, rawdata=rawdata, derivatives=derivatives, model=model,
                dates=["20260717"],
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outputs = run_scoring(settings, score_function=fake_score_recording)

            self.assertEqual(outputs, [])
            self.assertEqual(calls, [])
            self.assertTrue(
                any("none is a concatenated recording" in str(w.message) for w in caught)
            )

    def _prescan_session(self, root: Path, periods: pd.DataFrame | None):
        """One scorable session, with prescan periods written if given.

        Returns (settings, calls, scorer): prescan enabled, and a fake scorer
        recording the keyword arguments it is called with.
        """
        rawdata = root / "rawdata"
        derivatives = root / "derivatives"
        rawdata.mkdir()
        derivatives.mkdir()
        model = derivatives / "model.pickle"
        model.touch()
        _make_session(
            rawdata, "sub-066", "ses-001_date-20260717", ["sub-066_ses-001_recording-001.edf"]
        )
        if periods is not None:
            prescan = (
                derivatives / "sub-066" / "ses-001_date-20260717" / "eeg" / "artifacts"
                / "sub-066_ses-001_recording-001_prescan_artifacts.parquet"
            )
            prescan.parent.mkdir(parents=True)
            periods.to_parquet(prescan)
        calls: list[dict] = []

        def fake_score_recording(edf_path, model_path, **kwargs):
            calls.append(kwargs)
            return _fake_predictions_df(), _fake_prepared()

        settings = _settings(
            root=root, rawdata=rawdata, derivatives=derivatives, model=model,
            dates=["20260717"], use_artifact_prescan=True,
        )
        return settings, calls, fake_score_recording

    def test_prescan_periods_are_passed_to_somnotate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            periods = pd.DataFrame(
                {"start_s": [0.0, 3600.0], "end_s": [120.0, 7200.0], "duration_s": [120.0, 3600.0]}
            )
            settings, calls, scorer = self._prescan_session(Path(directory), periods)

            outputs = run_scoring(settings, score_function=scorer)

            self.assertEqual(calls[0]["exclude_intervals_s"], [(0.0, 120.0), (3600.0, 7200.0)])
            provenance = json.loads(
                outputs[0].with_name(outputs[0].stem + "_provenance.json").read_text()
            )
            self.assertEqual(provenance["parameters"]["excluded_artifact_periods"], 2)
            self.assertIsNotNone(provenance["parameters"]["artifact_prescan"])

    def test_missing_prescan_scores_the_whole_recording_with_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings, calls, scorer = self._prescan_session(Path(directory), None)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_scoring(settings, score_function=scorer)

            self.assertIsNone(calls[0]["exclude_intervals_s"])
            self.assertTrue(any("No artifact prescan output" in str(w.message) for w in caught))

    def test_prescan_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            periods = pd.DataFrame({"start_s": [0.0], "end_s": [120.0], "duration_s": [120.0]})
            settings, calls, scorer = self._prescan_session(Path(directory), periods)
            settings = replace(settings, use_artifact_prescan=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_scoring(settings, score_function=scorer)

            self.assertIsNone(calls[0]["exclude_intervals_s"])
            self.assertFalse(any("prescan" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
