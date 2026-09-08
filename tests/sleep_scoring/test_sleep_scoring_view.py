from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.sleep_scoring.view_scored_recording import (
    ScoringViewSettings,
    _artifact_file,
    _artifact_regions,
    _downsample_signals,
    _edf_contains_time,
    _elapsed_range,
    _require_graphical_display,
    _single_value,
    build_parser,
    run_view,
    settings_from_args,
)


class SleepScoringViewTests(unittest.TestCase):
    def test_session_number_is_an_alternative_to_date(self) -> None:
        args = build_parser().parse_args(["--subject", "66", "--session", "ses-2"])
        settings = settings_from_args(args)

        self.assertEqual(settings.session, 2)
        self.assertIsNone(settings.date)

    def test_artifact_file_prefers_dedicated_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            scoring_dir = session_dir / "sleep_scoring"
            artifact_dir = session_dir / "artifacts"
            scoring_dir.mkdir()
            artifact_dir.mkdir()
            recording = types.SimpleNamespace(
                output_dir=scoring_dir,
                edf_path=Path("recording.edf"),
            )
            legacy = scoring_dir / "recording_artifact_epochs.parquet"
            dedicated = artifact_dir / "recording_artifact_epochs.parquet"
            legacy.touch()
            dedicated.touch()

            self.assertEqual(_artifact_file(recording), dedicated)

    def test_hours_are_parsed_and_validated(self) -> None:
        args = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717", "--hours", "3", "6"]
        )
        settings = settings_from_args(args)
        self.assertEqual(settings.hours, (3.0, 6.0))

        args.hours = [6.0, 3.0]
        with self.assertRaisesRegex(ValueError, "END must be greater"):
            settings_from_args(args)

    def test_display_rate_and_artifact_overlay_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--subject", "66",
                "--date", "20260717",
                "--display-rate", "128",
                "--show-artifacts",
            ]
        )
        settings = settings_from_args(args)
        self.assertEqual(settings.display_rate_hz, 128.0)
        self.assertTrue(settings.show_artifacts)

    def test_show_gaps_defaults_true_and_can_be_disabled(self) -> None:
        default_args = build_parser().parse_args(["--subject", "66", "--date", "20260717"])
        self.assertTrue(settings_from_args(default_args).show_gaps)

        disabled_args = build_parser().parse_args(
            ["--subject", "66", "--date", "20260717", "--no-show-gaps"]
        )
        self.assertFalse(settings_from_args(disabled_args).show_gaps)

    def test_display_downsampling_reduces_samples(self) -> None:
        import numpy as np

        signals = np.arange(512 * 3, dtype=float).reshape(512, 3)
        scipy_module = types.ModuleType("scipy")
        signal_module = types.ModuleType("scipy.signal")
        signal_module.resample_poly = (
            lambda values, up, down, axis: values[:: down // up]
        )
        scipy_module.signal = signal_module
        with patch.dict(
            sys.modules, {"scipy": scipy_module, "scipy.signal": signal_module}
        ):
            downsampled = _downsample_signals(signals, 512.0, 128.0)
        self.assertEqual(downsampled.shape, (128, 3))

    def test_artifact_regions_are_clipped_and_merged_for_selected_range(self) -> None:
        table = pd.DataFrame(
            {
                "time_s": [8.0, 12.0, 16.0, 20.0, 24.0],
                "artifact": [True, True, True, False, True],
            }
        )
        self.assertEqual(
            _artifact_regions(table, selected_start_s=10.0, selected_end_s=26.0),
            [(0.0, 10.0), (14.0, 16.0)],
        )

    def test_real_time_range_does_not_assume_its_start_is_the_session_date(self) -> None:
        args = build_parser().parse_args(
            [
                "--subject", "66",
                "--time-range", "20260717 23:00:00", "20260718 02:00:00",
            ]
        )
        settings = settings_from_args(args)
        self.assertIsNone(settings.date)
        self.assertEqual(
            settings.time_range,
            (
                datetime(2026, 7, 17, 23, 0, 0),
                datetime(2026, 7, 18, 2, 0, 0),
            ),
        )

        args.time_range = ["20260718 02:00:00", "20260717 23:00:00"]
        with self.assertRaisesRegex(ValueError, "END must be later"):
            settings_from_args(args)

    def test_real_times_are_converted_relative_to_edf_start(self) -> None:
        settings = ScoringViewSettings(
            subject="66",
            date="20260717",
            repo_root=Path("."),
            rawdata_root=Path("."),
            derivatives_root=Path("."),
            recording_index=0,
            eeg_channel=0,
            view_length_s=120.0,
            time_range=(
                datetime(2026, 7, 17, 23, 0, 0),
                datetime(2026, 7, 18, 2, 0, 0),
            ),
        )
        start_s, end_s, _description, _title = _elapsed_range(
            settings, datetime(2026, 7, 17, 22, 30, 0)
        )
        self.assertEqual((start_s, end_s), (1800.0, 12600.0))

    def test_edf_lookup_finds_timestamp_on_following_day(self) -> None:
        class FakeReader:
            def __init__(self, _path: str) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                pass

            def getStartdatetime(self) -> datetime:
                return datetime(2026, 7, 17, 20, 0, 0)

            def getNSamples(self) -> list[int]:
                return [36 * 3600]

            def getSampleFrequency(self, _index: int) -> float:
                return 1.0

        self.assertTrue(
            _edf_contains_time(
                Path("recording.edf"),
                datetime(2026, 7, 18, 3, 0, 0),
                FakeReader,
            )
        )

    def test_single_value_rejects_ambiguous_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one subject"):
            _single_value([66, 67], option_name="subject")

    def test_missing_display_is_reported_before_qt_starts(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No graphical display"):
            _require_graphical_display({})

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

            with patch.dict(os.environ, {"DISPLAY": "localhost:10.0"}):
                result = run_view(settings, view_function=fake_view)
                self.assertEqual(os.environ["HYPNOSE_EEG_RAWDATA_ROOT"], str(rawdata))
                self.assertEqual(
                    os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"], str(derivatives)
                )

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

    def test_run_view_uses_range_loader_when_hours_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            settings = ScoringViewSettings(
                subject="66",
                date="20260717",
                repo_root=root,
                rawdata_root=rawdata,
                derivatives_root=derivatives,
                recording_index=0,
                eeg_channel=0,
                view_length_s=120.0,
                hours=(3.0, 6.0),
            )

            with (
                patch.dict(os.environ, {"DISPLAY": "localhost:10.0"}),
                patch(
                    "scripts.sleep_scoring.view_scored_recording._run_custom_view",
                    return_value=0,
                ) as range_view,
            ):
                self.assertEqual(run_view(settings), 0)
                range_view.assert_called_once_with(settings)


if __name__ == "__main__":
    unittest.main()
