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

from hypnose_eeg.sleep_scoring.view_scoring import (
    ScoredWindow,
    ScoringViewSettings,
    ShadedRegions,
    _artifact_file,
    _edf_contains_time,
    _elapsed_range,
    _fif_rate_hz,
    _single_value,
    artifact_regions,
    build_parser,
    downsample_signals,
    find_downsampled_fif,
    plot_scored_window,
    require_graphical_display,
    run_view,
    settings_from_args,
    view_settings,
)


class SleepScoringViewTests(unittest.TestCase):
    def test_session_number_is_an_alternative_to_date(self) -> None:
        args = build_parser().parse_args(["--subject", "66", "--session", "ses-2"])
        settings = settings_from_args(args)

        self.assertEqual(settings.session, 2)
        self.assertIsNone(settings.date)

    def test_artifact_file_prefers_dedicated_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys" / "recording.edf"
            session_dir = derivatives / "sub-066" / "ses-1_date-20260717"
            # Somnotate's native output folder, which `recording.output_dir` points at.
            legacy_dir = session_dir / "saved_results"
            artifact_dir = session_dir / "eeg" / "artifacts"
            edf.parent.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            artifact_dir.mkdir(parents=True)
            legacy = legacy_dir / "recording_artifact_epochs.parquet"
            dedicated = artifact_dir / "recording_artifact_epochs.parquet"
            legacy.touch()
            dedicated.touch()

            self.assertEqual(_artifact_file(edf, rawdata, derivatives), dedicated)

    def test_artifact_file_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys" / "recording.edf"
            edf.parent.mkdir(parents=True)
            (derivatives / "sub-066" / "ses-1_date-20260717").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "artifact_epochs.parquet"):
                _artifact_file(edf, rawdata, derivatives)

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
            downsampled = downsample_signals(signals, 512.0, 128.0)
        self.assertEqual(downsampled.shape, (128, 3))

    def test_fif_rate_hz_parses_integer_and_fractional_labels(self) -> None:
        self.assertEqual(
            _fif_rate_hz(Path("recording_resampled-128hz_raw.fif")), 128.0
        )
        self.assertEqual(
            _fif_rate_hz(Path("recording_resampled-99p5hz_raw.fif")), 99.5
        )
        self.assertIsNone(_fif_rate_hz(Path("recording_raw.fif")))

    def test_find_downsampled_fif_picks_lowest_sufficient_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf_path = (
                rawdata / "sub-066_id-1" / "ses-001_date-20260717" / "ephys"
                / "sub-066_ses-001_recording-concat.edf"
            )
            edf_path.parent.mkdir(parents=True)
            edf_path.touch()

            downsample_dir = (
                derivatives / "sub-066_id-1" / "ses-001_date-20260717" / "downsample"
            )
            downsample_dir.mkdir(parents=True)
            low_rate = downsample_dir / "sub-066_ses-001_recording-concat_resampled-64hz_raw.fif"
            target_rate = (
                downsample_dir / "sub-066_ses-001_recording-concat_resampled-128hz_raw.fif"
            )
            high_rate = (
                downsample_dir / "sub-066_ses-001_recording-concat_resampled-256hz_raw.fif"
            )
            for path in (low_rate, target_rate, high_rate):
                path.touch()

            found = find_downsampled_fif(edf_path, rawdata, derivatives, min_rate_hz=100.0)
            self.assertEqual(found, target_rate)

    def test_find_downsampled_fif_with_zero_min_rate_picks_lowest_available(self) -> None:
        """min_rate_hz=0.0 is what callers pass when no display rate was requested --
        any FIF beats the full-rate EDF, so the lowest-rate one (cheapest to read)
        should win."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf_path = (
                rawdata / "sub-066_id-1" / "ses-001_date-20260717" / "ephys"
                / "sub-066_ses-001_recording-concat.edf"
            )
            edf_path.parent.mkdir(parents=True)
            edf_path.touch()

            downsample_dir = (
                derivatives / "sub-066_id-1" / "ses-001_date-20260717" / "downsample"
            )
            downsample_dir.mkdir(parents=True)
            low_rate = downsample_dir / "sub-066_ses-001_recording-concat_resampled-64hz_raw.fif"
            high_rate = (
                downsample_dir / "sub-066_ses-001_recording-concat_resampled-256hz_raw.fif"
            )
            for path in (low_rate, high_rate):
                path.touch()

            found = find_downsampled_fif(edf_path, rawdata, derivatives, min_rate_hz=0.0)
            self.assertEqual(found, low_rate)

    def test_find_downsampled_fif_returns_none_when_no_rate_is_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            edf_path = (
                rawdata / "sub-066_id-1" / "ses-001_date-20260717" / "ephys"
                / "sub-066_ses-001_recording-concat.edf"
            )
            edf_path.parent.mkdir(parents=True)
            edf_path.touch()

            downsample_dir = (
                derivatives / "sub-066_id-1" / "ses-001_date-20260717" / "downsample"
            )
            downsample_dir.mkdir(parents=True)
            low_rate = downsample_dir / "sub-066_ses-001_recording-concat_resampled-64hz_raw.fif"
            low_rate.touch()

            found = find_downsampled_fif(edf_path, rawdata, derivatives, min_rate_hz=128.0)
            self.assertIsNone(found)

    def test_find_downsampled_fif_returns_none_outside_rawdata_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            scratch_edf = root / "scratch" / "recording.edf"
            scratch_edf.parent.mkdir(parents=True)
            scratch_edf.touch()

            found = find_downsampled_fif(scratch_edf, rawdata, derivatives, min_rate_hz=128.0)
            self.assertIsNone(found)

    def test_artifact_regions_are_clipped_and_merged_for_selected_range(self) -> None:
        table = pd.DataFrame(
            {
                "time_s": [8.0, 12.0, 16.0, 20.0, 24.0],
                "artifact": [True, True, True, False, True],
            }
        )
        self.assertEqual(
            artifact_regions(table, selected_start_s=10.0, selected_end_s=26.0),
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
        # The check only applies on Linux, where the viewer runs over SSH/X11.
        with patch.object(sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "No graphical display"):
                require_graphical_display({})

    def test_run_view_forwards_somnotate_view_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            rawdata.mkdir()
            derivatives.mkdir()
            calls: list[list[str]] = []
            roots_seen: list[tuple[str, str]] = []

            def fake_view(arguments: list[str]) -> int:
                calls.append(arguments)
                roots_seen.append(
                    (
                        os.environ["HYPNOSE_EEG_RAWDATA_ROOT"],
                        os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"],
                    )
                )
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

            with patch.dict(
                os.environ,
                {"DISPLAY": "localhost:10.0", "HYPNOSE_EEG_RAWDATA_ROOT": "/previous"},
            ):
                result = run_view(settings, view_function=fake_view)
                # The roots reach the somnotate viewer but do not leak into
                # the caller's environment afterwards.
                self.assertEqual(roots_seen, [(str(rawdata), str(derivatives))])
                self.assertEqual(os.environ["HYPNOSE_EEG_RAWDATA_ROOT"], "/previous")
                self.assertNotIn("HYPNOSE_EEG_DERIVATIVES_ROOT", os.environ)

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
                    "hypnose_eeg.sleep_scoring.view_scoring.show_scored_recording",
                    return_value=0,
                ) as range_view,
            ):
                self.assertEqual(run_view(settings), 0)
                range_view.assert_called_once_with(settings)

    def test_view_settings_is_the_keyword_form_of_the_cli(self) -> None:
        from_cli = settings_from_args(
            build_parser().parse_args(
                ["--subject", "66", "--session", "2", "--hours", "3", "6", "--no-show-gaps"]
            )
        )
        from_api = view_settings(66, session=2, hours=(3, 6), show_gaps=False)

        self.assertEqual(from_api, from_cli)
        self.assertEqual(from_api.hours, (3.0, 6.0))
        with self.assertRaisesRegex(ValueError, "a date or session is required"):
            view_settings(66)

    def test_scored_window_plots_with_shaded_regions_without_a_display(self) -> None:
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rate_hz = 256.0
        window = ScoredWindow(
            title="sub-066 ses-001 (date 20260717) — hours 0-0.05",
            signal_path=Path("rec.edf"),
            predictions_path=Path("rec_somnotate_predictions.parquet"),
            signals=np.random.default_rng(0).standard_normal((int(180 * rate_hz), 3)),
            sampling_rate_hz=rate_hz,
            predictions=np.repeat([1, 2, 3], 60),
            start_s=0.0,
            end_s=180.0,
            regions=[ShadedRegions("Gap", "dimgray", [(10.0, 20.0), (50.0, 60.0)])],
        )

        fig, viewer = plot_scored_window(window, view_length_s=60.0)
        try:
            self.assertIsNotNone(viewer)
            self.assertEqual(fig._suptitle.get_text(), window.title)
            gap_labels = [text for text in fig.axes[0].texts if text.get_text() == "Gap"]
            self.assertEqual(len(gap_labels), 2)
        finally:
            plt.close(fig)

    def test_render_scoring_saves_the_whole_window_as_a_png_in_hours(self) -> None:
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        import hypnose_eeg.io.output_paths as output_paths
        from hypnose_eeg.sleep_scoring.view_scoring import render_scoring

        rate_hz = 128.0  # the EMG band-pass needs more than 90 Hz
        window = ScoredWindow(
            title="sub-066 ses-001 (date 20260717) — Hours 1-1.1 from session start",
            signal_path=Path("rec_raw.fif"),
            predictions_path=Path("rec_somnotate_predictions.parquet"),
            signals=np.random.default_rng(0).standard_normal((int(360 * rate_hz), 3)),
            sampling_rate_hz=rate_hz,
            predictions=np.repeat([1, 2, 3], 40),
            start_s=3600.0,
            end_s=3960.0,
            regions=[],
            edf_path=Path("sub-066_ses-001.edf"),
        )
        seen = {}
        real_save_png = output_paths.save_png

        def capture(fig, path, *, dpi):
            data_axis = fig.axes[0]
            seen["xlim"] = data_axis.get_xlim()
            seen["tick"] = data_axis.xaxis.get_major_formatter()(0.0, 0)
            seen["xlabels"] = [axis.get_xlabel() for axis in fig.axes if axis.get_xlabel()]
            seen["width"] = fig.get_figwidth()
            return real_save_png(fig, path, dpi=dpi)

        settings = view_settings(66, session=1, hours=(1, 1.1))
        with tempfile.TemporaryDirectory() as tmp, patch(
            "hypnose_eeg.sleep_scoring.view_scoring.load_scored_window", return_value=window
        ), patch.object(output_paths, "save_png", side_effect=capture):
            path = render_scoring(
                settings,
                lambda loaded: Path(tmp) / f"{loaded.edf_path.stem}_overview.png",
                width_in=20,
                dpi=40,
            )

            self.assertEqual(path, Path(tmp) / "sub-066_ses-001_overview.png")
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertIn("Provenance", image.text)

        # The whole window is in view at once, labelled in hours from the session start.
        self.assertAlmostEqual(seen["xlim"][1] - seen["xlim"][0], 360.0, delta=1.0)
        self.assertEqual(seen["tick"], "1")
        self.assertEqual(seen["xlabels"], ["Time from session start [h]"])
        self.assertEqual(seen["width"], 20)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
