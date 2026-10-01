from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from hypnose_somnotate.preprocessing.gap_correction import (
    PreparedRecording,
    RecordingSegment,
    ScoringChunk,
)
from hypnose_somnotate.preprocessing.preprocessing import (
    NormalizationResult,
    load_normalization_stats,
    save_normalization_stats,
)

from hypnose_eeg.qc.summary_qc import normalization_section
from hypnose_eeg.sleep_scoring.reference_normalization import (
    normalization_stats_file,
    stats_metadata,
)
from hypnose_eeg.sleep_scoring.score_recordings import (
    SleepScoringSettings,
    build_parser,
    run_scoring,
    settings_from_args,
)

HOUR = 3600.0
N_BINS = 8
LABELS = ["EEG1", "EEG2", "EMG"]


def _stats(mean: float) -> list:
    return [(np.full(N_BINS, mean), np.ones(N_BINS)) for _ in LABELS]


def _prepared(signal_s: float) -> PreparedRecording:
    return PreparedRecording(
        segments=[RecordingSegment(0, "signal", 0.0, signal_s)],
        scoring_chunks=[ScoringChunk(0.0, signal_s, np.zeros((1, 3)))],
        strategy="trim_only",
        original_duration_s=signal_s,
        total_missing_s=0.0,
        missing_fraction=0.0,
        longest_gap_s=0.0,
        sampling_rate_hz=512.0,
        time_resolution_s=1.0,
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "label": [0],
            "label_model": [1],
            "label_output": [0],
            "segment_id": [0],
            "kind": ["signal"],
            "prob_wake": [1.0],
            "prob_nrem": [0.0],
            "prob_rem": [0.0],
            "prob_undef": [0.0],
        }
    )


class _Scorer:
    """Stands in for somnotate's score_recording, honouring `normalization_stats`."""

    def __init__(self, signal_s: float, own_mean: float = 0.0) -> None:
        self.signal_s = signal_s
        self.own = _stats(own_mean)
        self.applied: list | None = None

    def __call__(self, edf_path, model_path, *, normalization_stats=None, **kwargs):
        prepared = _prepared(self.signal_s)
        external = normalization_stats(prepared) if normalization_stats is not None else None
        self.applied = external
        prepared.normalization = NormalizationResult(
            source="reference" if external is not None else "self",
            applied=external if external is not None else self.own,
            own=self.own,
            signal_s=self.signal_s,
        )
        return _predictions(), prepared


class ReferenceNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.rawdata = self.root / "rawdata"
        self.derivatives = self.root / "derivatives"
        self.rawdata.mkdir()
        self.derivatives.mkdir()
        self.model = self.derivatives / "model.pickle"
        self.model.touch()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _session(self, ses: int, date: str, *, content: bytes = b"edf") -> Path:
        session_dir = f"ses-{ses:03d}_date-{date}"
        ephys = self.rawdata / "sub-066" / session_dir / "ephys"
        ephys.mkdir(parents=True)
        edf = ephys / f"sub-066_ses-{ses:03d}_recording-001.edf"
        edf.write_bytes(content)
        return edf

    def _cache(self, edf: Path, *, signal_hours: float, mean: float) -> Path:
        session_output = self.derivatives / "sub-066" / edf.parent.parent.name
        path = normalization_stats_file(session_output, edf)
        save_normalization_stats(
            path,
            _stats(mean),
            stats_metadata(
                edf,
                channel_labels=LABELS,
                sampling_rate_hz=512,
                signal_s=signal_hours * HOUR,
                excluded_artifact_periods=0,
            ),
        )
        return path

    def _settings(self, **overrides) -> SleepScoringSettings:
        fields = dict(
            subjids=["66"],
            model_path=self.model,
            repo_root=self.root,
            rawdata_root=self.rawdata,
            derivatives_root=self.derivatives,
            dates=["20260712"],
            sessions=None,
            date_range=None,
            channel_labels=LABELS,
            export_visbrain=False,
            sampling_rate_hz=512,
            global_normalization=True,
            use_artifact_prescan=False,
        )
        fields.update(overrides)
        return SleepScoringSettings(**fields)

    def _provenance(self, output: Path) -> dict:
        return json.loads(output.with_name(output.stem + "_provenance.json").read_text())

    def _run(self, settings, scorer):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs = run_scoring(settings, score_function=scorer)
        return outputs, [str(w.message) for w in caught]

    def test_short_recording_uses_the_nearest_earlier_long_session(self) -> None:
        older = self._session(1, "20260708")
        previous = self._session(2, "20260710")
        self._session(3, "20260712")
        later = self._session(4, "20260713")
        self._cache(older, signal_hours=20, mean=5.0)
        reference_path = self._cache(previous, signal_hours=22, mean=1.0)
        self._cache(later, signal_hours=22, mean=9.0)
        scorer = _Scorer(signal_s=3 * HOUR, own_mean=1.5)

        outputs, _ = self._run(self._settings(), scorer)

        self.assertTrue(np.allclose(scorer.applied[0][0], 1.0))
        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["reference_status"], "used")
        self.assertTrue(record["short_recording"])
        self.assertEqual(record["reference"]["session"], "ses-002")
        self.assertEqual(record["reference"]["days_from_recording"], -2)
        self.assertEqual(record["reference"]["stats_file"]["path"], str(reference_path))
        self.assertEqual(record["offset_z"], [0.5, 0.5, 0.5])

        # The short recording still saves its own statistics, marked as short.
        own_path = outputs[0].with_name("sub-066_ses-003_recording-001_somnotate_normalization.npz")
        _, metadata = load_normalization_stats(own_path)
        self.assertEqual(metadata["signal_s"], 3 * HOUR)

    def test_nearest_preference_may_pick_a_later_session(self) -> None:
        previous = self._session(2, "20260709")
        self._session(3, "20260712")
        later = self._session(4, "20260713")
        self._cache(previous, signal_hours=22, mean=1.0)
        self._cache(later, signal_hours=22, mean=9.0)
        scorer = _Scorer(signal_s=3 * HOUR)

        outputs, _ = self._run(self._settings(reference_prefer="nearest"), scorer)

        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["reference"]["session"], "ses-004")
        self.assertTrue(np.allclose(scorer.applied[0][0], 9.0))

    def test_long_recording_keeps_its_own_baseline_and_saves_it(self) -> None:
        previous = self._session(2, "20260710")
        self._session(3, "20260712")
        self._cache(previous, signal_hours=22, mean=1.0)
        scorer = _Scorer(signal_s=20 * HOUR)

        outputs, _ = self._run(self._settings(), scorer)

        self.assertIsNone(scorer.applied)
        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["source"], "self")
        self.assertEqual(record["reference_status"], "not_needed")
        self.assertIsNone(record["offset_z"])
        own_path = outputs[0].with_name("sub-066_ses-003_recording-001_somnotate_normalization.npz")
        self.assertIn(str(own_path), self._provenance(outputs[0])["outputs"])

    def test_sessions_beyond_the_age_limit_or_too_short_are_not_used(self) -> None:
        stale = self._session(1, "20260601")
        short = self._session(2, "20260711")
        self._session(3, "20260712")
        self._cache(stale, signal_hours=22, mean=1.0)
        self._cache(short, signal_hours=2, mean=1.0)
        scorer = _Scorer(signal_s=3 * HOUR)

        outputs, messages = self._run(self._settings(), scorer)

        self.assertIsNone(scorer.applied)
        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["reference_status"], "not_found")
        self.assertTrue(any("no reference session was found" in m for m in messages))

    def test_explicit_reference_session_overrides_the_age_limit(self) -> None:
        stale = self._session(1, "20260601")
        self._session(3, "20260712")
        self._cache(stale, signal_hours=22, mean=1.0)
        scorer = _Scorer(signal_s=3 * HOUR)

        outputs, _ = self._run(self._settings(reference_session="ses-1"), scorer)

        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["reference"]["session"], "ses-001")
        self.assertEqual(record["reference_session"], "ses-1")

    def test_reference_without_cached_statistics_is_computed_and_cached(self) -> None:
        previous = self._session(2, "20260710")
        self._session(3, "20260712")
        computed = []

        def fake_compute(edf_path, **kwargs):
            computed.append(edf_path)
            return _stats(2.0), _prepared(22 * HOUR)

        scorer = _Scorer(signal_s=3 * HOUR)
        with mock.patch("hypnose_somnotate.scoring.recording_normalization_stats", fake_compute):
            outputs, _ = self._run(self._settings(), scorer)
            self.assertEqual(computed, [previous])

            cache = normalization_stats_file(
                self.derivatives / "sub-066" / "ses-002_date-20260710", previous
            )
            self.assertTrue(cache.is_file())
            record = self._provenance(outputs[0])["parameters"]["normalization"]
            self.assertTrue(record["reference"]["computed_from_edf"])

            # A second run reads the cache instead of recomputing.
            self._run(self._settings(overwrite=True), _Scorer(signal_s=3 * HOUR))
            self.assertEqual(computed, [previous])

    def test_cached_statistics_for_a_changed_edf_are_recomputed(self) -> None:
        previous = self._session(2, "20260710")
        self._session(3, "20260712")
        self._cache(previous, signal_hours=22, mean=1.0)
        previous.write_bytes(b"re-concatenated edf")

        def fake_compute(edf_path, **kwargs):
            return _stats(4.0), _prepared(22 * HOUR)

        scorer = _Scorer(signal_s=3 * HOUR)
        with mock.patch("hypnose_somnotate.scoring.recording_normalization_stats", fake_compute):
            self._run(self._settings(), scorer)

        self.assertTrue(np.allclose(scorer.applied[0][0], 4.0))

    def test_reference_normalization_can_be_disabled(self) -> None:
        previous = self._session(2, "20260710")
        self._session(3, "20260712")
        self._cache(previous, signal_hours=22, mean=1.0)
        scorer = _Scorer(signal_s=3 * HOUR)

        outputs, _ = self._run(self._settings(reference_normalization=False), scorer)

        self.assertIsNone(scorer.applied)
        record = self._provenance(outputs[0])["parameters"]["normalization"]
        self.assertEqual(record["reference_status"], "disabled")

    def test_settings_read_config_and_cli_overrides(self) -> None:
        config = self.root / "scoring.yaml"
        config.write_text(
            "sleep_scoring:\n"
            "  model_path: model.pickle\n"
            "  reference_normalization:\n"
            "    enabled: true\n"
            "    min_signal_hours: 5\n"
            "    max_reference_age_days: 7\n"
            "    prefer: nearest\n"
        )
        base = [
            "--config", str(config), "--subject", "66",
            "--rawdata-root", str(self.rawdata), "--derivatives-root", str(self.derivatives),
        ]

        settings = settings_from_args(build_parser().parse_args(base))
        self.assertTrue(settings.reference_normalization)
        self.assertEqual(settings.min_signal_hours, 5.0)
        self.assertEqual(settings.max_reference_age_days, 7.0)
        self.assertEqual(settings.reference_prefer, "nearest")
        self.assertIsNone(settings.reference_session)

        overridden = settings_from_args(
            build_parser().parse_args(
                base + ["--no-reference-normalization", "--reference-session", "20260710",
                        "--min-signal-hours", "4"]
            )
        )
        self.assertFalse(overridden.reference_normalization)
        self.assertEqual(overridden.reference_session, "20260710")
        self.assertEqual(overridden.min_signal_hours, 4.0)


def _record(**normalization) -> dict:
    return {"parameters": {"normalization": normalization}}


class NormalizationSectionTests(unittest.TestCase):
    def test_reference_within_the_offset_limit_passes(self) -> None:
        section = normalization_section(
            _record(
                reference_status="used", short_recording=True, signal_hours=3.0,
                reference={"session": "ses-002", "date": "20260710", "signal_hours": 22.0,
                           "days_from_recording": -2},
                offset_z=[0.2, -0.4, 0.1],
            ),
            max_offset_z=1.0,
        )
        self.assertEqual(section["status"], "pass")
        self.assertAlmostEqual(section["value"], 0.4)
        self.assertIn("ses-002", section["detail"])

    def test_reference_beyond_the_offset_limit_is_reviewed(self) -> None:
        section = normalization_section(
            _record(reference_status="used", short_recording=True, signal_hours=3.0,
                    reference={"session": "ses-002"}, offset_z=[1.8, 0.1, None]),
            max_offset_z=1.0,
        )
        self.assertEqual(section["status"], "review")
        self.assertAlmostEqual(section["value"], 1.8)

    def test_short_recording_without_a_reference_is_reviewed(self) -> None:
        section = normalization_section(
            _record(reference_status="not_found", short_recording=True, signal_hours=3.0,
                    min_signal_hours=6.0, source="self", offset_z=None),
            max_offset_z=1.0,
        )
        self.assertEqual(section["status"], "review")
        self.assertIn("own statistics", section["detail"])

    def test_long_recording_and_older_provenance_pass(self) -> None:
        long_section = normalization_section(
            _record(reference_status="not_needed", short_recording=False, signal_hours=22.0,
                    source="self", offset_z=None),
            max_offset_z=1.0,
        )
        self.assertEqual(long_section["status"], "pass")
        self.assertEqual(normalization_section(None, max_offset_z=1.0)["status"], "pass")
        self.assertEqual(
            normalization_section({"parameters": {}}, max_offset_z=1.0)["status"], "pass"
        )


if __name__ == "__main__":
    unittest.main()
