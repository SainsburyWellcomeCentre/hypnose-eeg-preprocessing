from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hypnose_eeg.analysis.power_spectra import bandpower, integrated_power
from hypnose_eeg.io.mne_io import import_mne
from hypnose_eeg.io.output_paths import artifact_output_paths
from hypnose_eeg.preprocessing.detect_artifacts import (
    ArtifactDetector,
    _resolve_paths,
    build_parser,
    extract_features,
)
from hypnose_eeg.utils.config import two_float_tuple
from hypnose_eeg.utils.epochs import align_epoch_states, complete_epoch_count, epoch_batch


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/pipelines/artifact_detection.yaml"
)


class ArtifactDetectionTests(unittest.TestCase):
    def test_two_float_tuple_normalizes_yaml_values(self) -> None:
        self.assertEqual(two_float_tuple([20, 45], "band"), (20.0, 45.0))
        self.assertEqual(two_float_tuple("[49, 51]", "band"), (49.0, 51.0))
        with self.assertRaisesRegex(ValueError, "must contain"):
            two_float_tuple([1], "band")

    def test_detector_loads_yaml_and_applies_cli_overrides(self) -> None:
        detector = ArtifactDetector.from_yaml(
            CONFIG_PATH,
            epoch_seconds=8.0,
            overwrite=True,
        )
        self.assertEqual(detector.epoch_seconds, 8.0)
        self.assertTrue(detector.overwrite)
        self.assertEqual(detector.high_frequency_band_hz, (20.0, 45.0))

    def test_process_skips_instead_of_failing_when_outputs_exist(self) -> None:
        detector = ArtifactDetector.from_yaml(CONFIG_PATH, overwrite=False)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            fif_path = output_dir / "sub-066_ses-001_recording-concat_raw.fif"
            csv_path, parquet_path = artifact_output_paths(
                fif_path, output_suffix=detector.output_suffix, output_dir=output_dir
            )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.touch()
            parquet_path.touch()

            result = detector.process(fif_path, output_dir / "states.parquet", output_dir)

        self.assertEqual(result["status"], "skipped_exists")
        self.assertEqual(
            result["existing_outputs"], [str(csv_path), str(parquet_path)]
        )

    def test_epoch_helpers_count_and_reshape_complete_epochs(self) -> None:
        self.assertEqual(complete_epoch_count(21, sfreq=2.0, epoch_seconds=5.0), 2)
        samples = np.arange(40).reshape(2, 20)
        epochs = epoch_batch(samples, epoch_count=2, epoch_samples=10)
        self.assertEqual(epochs.shape, (2, 2, 10))
        np.testing.assert_array_equal(epochs[1, 0], samples[0, 10:20])

    def test_sleep_states_are_aligned_to_every_epoch(self) -> None:
        scores = pd.DataFrame(
            {"time_s": [0.0, 2.0, 4.0], "label": [0, 1, 2]}
        )
        aligned = align_epoch_states(
            scores,
            n_epochs=3,
            epoch_seconds=4.0,
            time_column="time_s",
            state_column="label",
            unscored_numeric_state=-1,
            unscored_text_state="unscored",
        )
        self.assertEqual(aligned["sleep_state"].tolist(), [0.0, 2.0, -1.0])
        self.assertEqual(aligned["sleep_score_count"].tolist(), [2, 1, 0])

    def test_bandpower_and_output_paths_are_domain_helpers(self) -> None:
        frequencies = np.array([0.0, 1.0, 2.0, 3.0])
        psd = np.ones((2, 4))
        np.testing.assert_allclose(bandpower(psd, frequencies, 1.0, 3.0), 2.0)
        np.testing.assert_allclose(
            integrated_power(psd, frequencies, 1.0, 3.0), 2.0
        )

        csv_path, parquet_path = artifact_output_paths(
            "/results/sample_raw.fif",
            output_suffix="artifact_epochs",
        )
        self.assertEqual(
            csv_path, Path("/results/eeg/artifacts/sample_artifact_epochs.csv")
        )
        self.assertEqual(
            parquet_path,
            Path("/results/eeg/artifacts/sample_artifact_epochs.parquet"),
        )

        csv_path, parquet_path = artifact_output_paths(
            "/derivatives/sub-066/ses-1_date-20260717/ephys/sample_raw.fif",
            output_suffix="artifact_epochs",
        )
        expected = Path(
            "/derivatives/sub-066/ses-1_date-20260717/eeg/artifacts/"
            "sample_artifact_epochs"
        )
        self.assertEqual(csv_path, expected.with_suffix(".csv"))
        self.assertEqual(parquet_path, expected.with_suffix(".parquet"))

        csv_path, parquet_path = artifact_output_paths(
            "/derivatives/sub-066/ses-1_date-20260717/ephys/"
            "sample_resampled-250hz_raw.fif",
            output_suffix="artifact_epochs",
        )
        self.assertEqual(csv_path, expected.with_suffix(".csv"))
        self.assertEqual(parquet_path, expected.with_suffix(".parquet"))

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        by_date = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717"]
        )
        self.assertEqual((by_date.subject, by_date.date), ("66", "20260717"))
        by_session = build_parser().parse_args(
            ["--subject", "66", "--session", "1"]
        )
        self.assertEqual(by_session.session, "1")

    def test_cli_resolves_fif_and_scoring_paths_from_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            raw_session = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys"
            deriv_session = derivatives / "sub-066" / "ses-1_date-20260717"
            raw_session.mkdir(parents=True)
            (deriv_session / "ephys").mkdir(parents=True)
            scoring_dir = deriv_session / "sleep_scoring"
            scoring_dir.mkdir()

            edf = raw_session / "recording.edf"
            fif = deriv_session / "ephys" / "recording_raw.fif"
            scoring = scoring_dir / "recording_somnotate_predictions.parquet"
            edf.touch()
            fif.touch()
            scoring.touch()

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--subject", "66", "--date", "20260717",
                    "--rawdata-root", str(rawdata),
                    "--derivatives-root", str(derivatives),
                ]
            )
            fif_path, sleep_parquet_path, edf_path = _resolve_paths(parser, args)
            self.assertEqual(fif_path, fif)
            self.assertEqual(sleep_parquet_path, scoring)
            self.assertEqual(edf_path, edf)

    def test_cli_accepts_source_edf_with_explicit_paths(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["recording_raw.fif", "scores.parquet", "--edf", "recording.edf"]
        )
        self.assertEqual(
            _resolve_paths(parser, args),
            (Path("recording_raw.fif"), Path("scores.parquet"), Path("recording.edf")),
        )
        args = parser.parse_args(["recording_raw.fif", "scores.parquet"])
        self.assertIsNone(_resolve_paths(parser, args)[2])

    def test_cli_rejects_mixing_positional_path_and_selectors(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["recording.fif", "scores.parquet", "--subject", "66", "--date", "20260717"]
        )
        with self.assertRaises(SystemExit):
            _resolve_paths(parser, args)


class SourceEdgeFractionTests(unittest.TestCase):
    """Clipping is read from the original-rate EDF, since resampling hides it."""

    SOURCE_SFREQ = 512.0
    N_EPOCHS = 20
    CLIPPED_EPOCHS = (5, 6)

    def setUp(self) -> None:
        mne = import_mne()
        rng = np.random.default_rng(0)
        n_samples = int(self.N_EPOCHS * 4.0 * self.SOURCE_SFREQ)
        data = rng.normal(0, 50e-6, (2, n_samples))
        # An EDF stores quantized values, so a signal held at the rail repeats
        # one value exactly; round to a 0.1 uV step as the digital range would.
        start, stop = (int(e * 4.0 * self.SOURCE_SFREQ) for e in self.CLIPPED_EPOCHS)
        data[0, start:stop] *= 8
        data = np.round(np.clip(data, -200e-6, 200e-6), 7)
        info = mne.create_info(["EEG1", "EEG2"], self.SOURCE_SFREQ, "eeg")
        self.source = mne.io.RawArray(data, info, verbose="ERROR")
        self.fif = self.source.copy().resample(128.0, verbose="ERROR")

    def _features(self, edge_source: object | None) -> pd.DataFrame:
        features, *_ = extract_features(
            self.fif, self.N_EPOCHS, 4.0, 128.0, 8, 2048, 2.0, 1.0, 55.0,
            (20.0, 45.0), (49.0, 51.0), edge_source=edge_source,
        )
        return features.set_index(["epoch_id", "channel"])["edge_fraction"]

    def test_resampled_fif_alone_hides_clipping(self) -> None:
        self.assertLess(self._features(None).max(), 0.02)

    def test_source_recording_reveals_clipping_in_its_epochs_only(self) -> None:
        edge_fraction = self._features(self.source)
        flagged = edge_fraction[edge_fraction > 0.02].index.tolist()
        self.assertEqual(flagged, [(5, "EEG1")])

    def test_epochs_beyond_the_source_get_no_edge_fraction(self) -> None:
        short_source = self.source.copy().crop(tmax=(self.N_EPOCHS - 1) * 4.0, include_tmax=False)
        edge_fraction = self._features(short_source)
        self.assertTrue(edge_fraction.loc[self.N_EPOCHS - 1].isna().all())
        self.assertFalse(edge_fraction.loc[: self.N_EPOCHS - 2].isna().any())

    def test_source_must_carry_the_fif_eeg_channels(self) -> None:
        renamed = self.source.copy().rename_channels({"EEG2": "Other"})
        with self.assertRaisesRegex(ValueError, "EEG2"):
            self._features(renamed)


if __name__ == "__main__":
    unittest.main()
