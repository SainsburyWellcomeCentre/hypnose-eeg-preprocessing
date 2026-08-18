from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.qc.recording_integrity import (
    Gap,
    _format_gap,
    _source_stem_from_fif,
    build_parser,
    check_pair,
    detect_signal_gaps,
    select_recordings,
)


class FakeRaw:
    def __init__(self, data: np.ndarray, sfreq: float) -> None:
        self._data = data
        self.info = {"sfreq": sfreq}
        self.n_times = data.shape[1]
        self.ch_names = [f"EEG{index}" for index in range(data.shape[0])]

    def get_channel_types(self) -> list[str]:
        return ["eeg"] * len(self.ch_names)

    def get_data(self, *, picks, start: int, stop: int) -> np.ndarray:
        return self._data[picks, start:stop]


class RecordingIntegrityTests(unittest.TestCase):
    def test_fif_is_only_read_for_duration_metadata(self) -> None:
        class FakeFileRaw(FakeRaw):
            def __init__(self, data: np.ndarray, sfreq: float) -> None:
                super().__init__(data, sfreq)
                self.info["meas_date"] = None
                self.annotations = []
                self.closed = False

            def close(self) -> None:
                self.closed = True

        edf = FakeFileRaw(np.arange(20, dtype=float)[None, :], 2.0)
        fif = FakeFileRaw(np.arange(10, dtype=float)[None, :], 1.0)

        def fail_if_fif_data_are_read(**_kwargs):
            raise AssertionError("FIF signal data should not be scanned")

        fif.get_data = fail_if_fif_data_are_read
        fake_mne = SimpleNamespace(
            io=SimpleNamespace(
                read_raw_edf=lambda *_args, **_kwargs: edf,
                read_raw_fif=lambda *_args, **_kwargs: fif,
            )
        )
        with patch(
            "scripts.qc.recording_integrity.import_mne",
            return_value=fake_mne,
        ):
            result, gaps = check_pair(
                Path("recording.edf"),
                Path("recording_raw.fif"),
                duration_tolerance_s=0.0,
                min_gap_s=1.0,
                chunk_duration_s=5.0,
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(gaps, [])
        self.assertTrue(edf.closed)
        self.assertTrue(fif.closed)

    def test_subject_and_date_or_session_arguments(self) -> None:
        by_date = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717"]
        )
        self.assertEqual((by_date.subject, by_date.date), ("66", "20260717"))
        by_session = build_parser().parse_args(
            ["--subject", "sub-066", "--session", "ses-1"]
        )
        self.assertEqual(by_session.session, "ses-1")

    def test_subject_selector_resolves_mirrored_recording_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            relative = Path("sub-066_mouse/ses-1_date-20260717/ephys")
            raw_dir = rawdata / relative
            derivative_dir = derivatives / relative
            raw_dir.mkdir(parents=True)
            derivative_dir.mkdir(parents=True)
            edf = raw_dir / "sub-066_ses-1_recording-concat.edf"
            fif = derivative_dir / "sub-066_ses-1_recording-concat_resampled-128hz_raw.fif"
            edf.touch()
            fif.touch()

            self.assertEqual(
                select_recordings(
                    rawdata, derivatives, subject="66", date="20260717"
                ),
                [(edf, fif)],
            )
            self.assertEqual(
                select_recordings(rawdata, derivatives, subject="066", session="1"),
                [(edf, fif)],
            )

    def test_expected_duration_is_not_a_cli_input(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(hasattr(args, "expected_hours"))
        self.assertFalse(hasattr(args, "expected_seconds"))
        self.assertIsNone(args.summary)
        self.assertIsNone(args.gaps)
        self.assertEqual(args.chunk_duration, 1800.0)

    def test_gap_terminal_output_only_includes_length(self) -> None:
        gap = Gap(
            source="edf",
            path="recording.edf",
            kind="constant_or_nonfinite_signal",
            start_s=12.0,
            end_s=16.5,
            duration_s=4.5,
            start_time="20260717 12:00:12",
            end_time="20260717 12:00:16.5",
            detail="Constant signal.",
        )
        output = _format_gap(1, gap)
        self.assertEqual(output.strip(), "Gap 1: 4.500 seconds")

    def test_fif_name_maps_back_to_edf_stem(self) -> None:
        path = Path("sub-066_recording-concat_resampled-128hz_raw.fif")
        self.assertEqual(_source_stem_from_fif(path), "sub-066_recording-concat")

    def test_gap_scan_finds_runs_across_chunk_boundaries(self) -> None:
        data = np.arange(40, dtype=float)[None, :]
        data[:, 8:18] = -1.0
        data[:, 30:32] = np.nan
        raw = FakeRaw(data, sfreq=2.0)
        gaps = detect_signal_gaps(
            raw, min_gap_s=1.0, chunk_duration_s=5.0
        )
        self.assertEqual(gaps, [(4.0, 9.0), (15.0, 16.0)])


if __name__ == "__main__":
    unittest.main()
