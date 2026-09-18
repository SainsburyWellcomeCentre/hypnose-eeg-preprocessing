from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from scripts.preprocessing.trim_duplicate_channels import (
    _resolve_edf_paths,
    build_parser,
    find_duplicate_labels,
    trim_duplicate_channels,
    read_edf_signal_labels,
    select_channels,
)


def _write_edf(path: Path, ch_names: list[str], duplicate_of: dict[int, int] | None = None) -> None:
    """Write a small EDF; `duplicate_of` maps a channel index to the index whose
    header label (and data) it should repeat, patched in after export because the
    exporter refuses non-unique labels -- exactly how the real files arise."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ch_names), 500)) * 1e-5
    for repeat, first in (duplicate_of or {}).items():
        data[repeat] = data[first]
    ch_types = ["emg" if name.startswith("EMG") else "eeg" for name in ch_names]
    info = mne.create_info(ch_names, sfreq=100.0, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.export(path, fmt="edf", overwrite=True, physical_range="auto", add_ch_type=True)
    with path.open("r+b") as handle:
        for repeat, first in (duplicate_of or {}).items():
            handle.seek(256 + first * 16)
            label = handle.read(16)
            handle.seek(256 + repeat * 16)
            handle.write(label)


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class ChannelSelectionTests(unittest.TestCase):
    def test_find_duplicate_labels_reports_every_index_of_a_repeated_label(self) -> None:
        labels = ["EEG1A-B", "EEG2A-B", "EMG", "EEG1A-B", "EMG", "EEG1A-B"]
        self.assertEqual(find_duplicate_labels(labels), {"EEG1A-B": [0, 3, 5], "EMG": [2, 4]})
        self.assertEqual(find_duplicate_labels(["a", "b"]), {})

    def test_select_channels_keeps_the_first_occurrence_of_each_label(self) -> None:
        labels = ["EEG1A-B", "EEG2A-B", "EMG", "EEG1A-B", "EMG"]
        self.assertEqual(select_channels(labels), [0, 1, 2])
        self.assertEqual(select_channels(["a", "b", "c"]), [0, 1, 2])

    def test_select_channels_keep_first_is_positional_and_ignores_labels(self) -> None:
        labels = ["EEG1A-B", "EEG1A-B", "EMG", "AUX"]
        self.assertEqual(select_channels(labels, keep_first=2), [0, 1])
        self.assertEqual(select_channels(labels, keep_first=10), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            select_channels(labels, keep_first=0)


class HeaderAndTrimTests(unittest.TestCase):
    def test_reads_raw_header_labels_before_mne_renames_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            edf = Path(directory) / "recording.edf"
            _write_edf(edf, ["EEG1A-B", "EEG2A-B", "EMG", "AUX"], duplicate_of={3: 0})

            labels = read_edf_signal_labels(edf)

            self.assertEqual(
                labels[:4], ["EEG EEG1A-B", "EEG EEG2A-B", "EMG EMG", "EEG EEG1A-B"]
            )
            raw = mne.io.read_raw_edf(edf, infer_types=True, verbose="ERROR")
            self.assertEqual(raw.ch_names, ["EEG1A-B-0", "EEG2A-B", "EMG", "EEG1A-B-1"])

    def test_rejects_a_file_that_is_not_an_edf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            not_edf = Path(directory) / "recording.edf"
            not_edf.write_bytes(b"nope")
            with self.assertRaises(ValueError):
                read_edf_signal_labels(not_edf)

    def test_drops_duplicates_and_restores_the_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            edf = Path(directory) / "recording.edf"
            _write_edf(edf, ["EEG1A-B", "EEG2A-B", "EMG", "AUX"], duplicate_of={3: 0})

            result = _quiet(trim_duplicate_channels, edf, n_samples=10)

            self.assertEqual(result.status, "written")
            self.assertEqual(result.kept, [0, 1, 2])
            self.assertEqual(result.dropped, [3])
            self.assertEqual(result.output_path, Path(directory) / "recording_trimmed.edf")
            trimmed = mne.io.read_raw_edf(result.output_path, infer_types=True, verbose="ERROR")
            self.assertEqual(trimmed.ch_names, ["EEG1A-B", "EEG2A-B", "EMG"])
            self.assertEqual(trimmed.get_channel_types(), ["eeg", "eeg", "emg"])
            source = mne.io.read_raw_edf(edf, infer_types=True, verbose="ERROR")
            np.testing.assert_allclose(trimmed.get_data(), source.get_data()[:3], atol=1e-7)

    def test_writes_nothing_for_a_recording_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            edf = Path(directory) / "recording.edf"
            _write_edf(edf, ["EEG1A-B", "EEG2A-B", "EMG"])

            result = _quiet(trim_duplicate_channels, edf, n_samples=10)

            self.assertEqual(result.status, "no_duplicates")
            self.assertIsNone(result.output_path)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["recording.edf"])

    def test_dry_run_reports_without_writing_and_existing_output_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            edf = Path(directory) / "recording.edf"
            _write_edf(edf, ["EEG1A-B", "EEG2A-B", "EMG", "AUX"], duplicate_of={3: 0})

            dry = _quiet(trim_duplicate_channels, edf, n_samples=10, dry_run=True)
            self.assertEqual(dry.status, "dry_run")
            self.assertFalse(dry.output_path.exists())

            first = _quiet(trim_duplicate_channels, edf, n_samples=10)
            mtime = first.output_path.stat().st_mtime_ns
            second = _quiet(trim_duplicate_channels, edf, n_samples=10)
            self.assertEqual(second.status, "skipped_exists")
            self.assertEqual(first.output_path.stat().st_mtime_ns, mtime)

            forced = _quiet(trim_duplicate_channels, edf, n_samples=10, overwrite=True)
            self.assertEqual(forced.status, "written")

    def test_keep_first_trims_by_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            edf = Path(directory) / "recording.edf"
            _write_edf(edf, ["EEG1A-B", "EEG2A-B", "EMG", "AUX"])

            result = _quiet(trim_duplicate_channels, edf, n_samples=10, keep_first=3)

            self.assertEqual(result.status, "written")
            trimmed = mne.io.read_raw_edf(result.output_path, infer_types=True, verbose="ERROR")
            self.assertEqual(trimmed.ch_names, ["EEG1A-B", "EEG2A-B", "EMG"])


class TrimDuplicateChannelsCliTests(unittest.TestCase):
    def test_cli_selects_subject_by_date_or_session(self) -> None:
        by_date = build_parser().parse_args(["--subject", "66", "--date", "20260717"])
        self.assertEqual((by_date.subject, by_date.date), ("66", "20260717"))
        by_session = build_parser().parse_args(["--subject", "66", "--session", "1"])
        self.assertEqual(by_session.session, "1")

    def test_cli_resolves_every_source_recording_in_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            session = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys"
            session.mkdir(parents=True)
            part_1 = session / "sub-066_ses-1_recording-1.edf"
            part_2 = session / "sub-066_ses-1_recording-2.edf"
            for path in (
                part_1,
                part_2,
                session / "sub-066_ses-1_recording-1_trimmed.edf",
                session / "sub-066_ses-1_recording-concat.edf",
            ):
                path.touch()

            parser = build_parser()
            by_date_args = parser.parse_args(
                ["--subject", "66", "--date", "20260717", "--rawdata-root", str(rawdata)]
            )
            self.assertEqual(_resolve_edf_paths(parser, by_date_args, "_trimmed"), [part_1, part_2])

            by_session_args = parser.parse_args(
                ["--subject", "66", "--session", "1", "--rawdata-root", str(rawdata)]
            )
            self.assertEqual(
                _resolve_edf_paths(parser, by_session_args, "_trimmed"), [part_1, part_2]
            )

    def test_cli_rejects_mixing_edf_path_and_selectors(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recording.edf", "--subject", "66", "--date", "20260717"])
        with self.assertRaises(SystemExit):
            _resolve_edf_paths(parser, args, "_trimmed")

    def test_cli_rejects_output_path_with_selectors(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--subject", "66", "--date", "20260717", "--output-path", "out.edf"]
        )
        with self.assertRaises(SystemExit):
            _resolve_edf_paths(parser, args, "_trimmed")

    def test_cli_errors_when_session_has_no_source_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            session = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys"
            session.mkdir(parents=True)
            (session / "sub-066_ses-1_recording-concat.edf").touch()

            parser = build_parser()
            args = parser.parse_args(
                ["--subject", "66", "--session", "1", "--rawdata-root", str(rawdata)]
            )
            with self.assertRaises(SystemExit):
                _resolve_edf_paths(parser, args, "_trimmed")


if __name__ == "__main__":
    unittest.main()
