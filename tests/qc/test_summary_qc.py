from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.qc.sleep_scoring import (
    prepare_scoring_output,
    sleep_state_proportion_report,
    sleep_state_proportions,
)
from scripts.qc.spectra import DEFAULT_SPECTRA_CONFIG
from scripts.io.output_paths import recording_output_name
from scripts.qc.summary_qc import (
    DEFAULT_REVIEW_FILENAME,
    DEFAULT_SUMMARY_FILENAME,
    REVIEW_COLUMNS,
    SECTION_COLUMNS,
    artifact_prescan_section,
    build_parser,
    overall_status,
    paired_output_paths,
    sleep_state_proportion_section,
    spectral_quality_sections,
    summary_parquet_table,
)


def _scored(n_seconds: int, artifact_seconds: range = range(0)) -> pd.DataFrame:
    """One-second predictions for a recording, `kind == "artifact"` over `artifact_seconds`."""
    kind = ["signal"] * n_seconds
    for second in artifact_seconds:
        kind[second] = "artifact"
    return pd.DataFrame({"time_s": range(n_seconds), "kind": kind})


def _periods(*spans: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_s": [a for a, _ in spans],
            "end_s": [b for _, b in spans],
            "duration_s": [b - a for a, b in spans],
            "flagged_fraction": [1.0] * len(spans),
            "dead_epochs": [10] * len(spans),
            "hard_failure_epochs": [0] * len(spans),
            "extreme_epochs": [0] * len(spans),
        }
    )


class ArtifactPrescanSectionTests(unittest.TestCase):
    def _section(self, periods, scored):
        return artifact_prescan_section(
            periods, scored, scoring_epoch_s=1.0, max_excluded_percent=5.0
        )

    def test_more_than_five_percent_excluded_goes_to_review(self) -> None:
        section, reviews = self._section(_periods((0, 60)), _scored(1000, range(60)))
        self.assertEqual(section["status"], "review")
        self.assertAlmostEqual(section["value"], 6.0)
        self.assertEqual([(r["start_s"], r["end_s"]) for r in reviews], [(0.0, 60.0)])
        self.assertIn("0.02 h labelled artifact", section["detail"])

    def test_exactly_five_percent_passes(self) -> None:
        section, _ = self._section(_periods((100, 125), (500, 525)), _scored(1000))
        self.assertEqual(section["status"], "pass")
        self.assertAlmostEqual(section["value"], 5.0)

    def test_no_periods_passes(self) -> None:
        section, reviews = self._section(_periods(), _scored(1000))
        self.assertEqual((section["status"], section["value"], reviews), ("pass", 0.0, []))

    def test_missing_prescan_goes_to_review(self) -> None:
        section, reviews = self._section(None, _scored(1000))
        self.assertEqual(section["status"], "review")
        self.assertIn("no prescan output", section["detail"])
        self.assertEqual(reviews, [])

    def test_predictions_scored_without_the_prescan_are_called_out(self) -> None:
        section, _ = self._section(_periods((0, 10)), _scored(1000))
        self.assertEqual(section["status"], "pass")
        self.assertIn("rescore with --overwrite", section["detail"])

    def test_review_ranges_keep_a_numeric_threshold_column(self) -> None:
        _, reviews = self._section(_periods((0, 60)), _scored(1000))
        table = pd.DataFrame(reviews, columns=REVIEW_COLUMNS)
        self.assertTrue(pd.api.types.is_float_dtype(table["threshold"]))

    def test_cli_threshold_defaults_to_config(self) -> None:
        args = build_parser().parse_args(
            ["--subject", "66", "--session", "1", "--max-prescan-excluded-percent", "10"]
        )
        self.assertEqual(args.max_prescan_excluded_percent, 10.0)


class SummaryQualityControlTests(unittest.TestCase):
    def test_overall_status_uses_most_severe_section(self) -> None:
        self.assertEqual(
            overall_status(pd.DataFrame({"status": ["pass", "review", "pass"]})),
            "review",
        )
        self.assertEqual(
            overall_status(pd.DataFrame({"status": ["review", "fail"]})),
            "fail",
        )
        self.assertEqual(overall_status(pd.DataFrame()), "fail")

    def test_review_schema_uses_time_ranges(self) -> None:
        self.assertEqual(REVIEW_COLUMNS[1:4], ["start_s", "end_s", "duration_s"])

    def test_review_outputs_include_csv_and_parquet(self) -> None:
        self.assertEqual(
            paired_output_paths("qc_review_epochs.csv"),
            (
                Path("qc_review_epochs.csv"),
                Path("qc_review_epochs.parquet"),
            ),
        )
        self.assertEqual(
            paired_output_paths("custom.parquet"),
            (Path("custom.csv"), Path("custom.parquet")),
        )

    def test_summary_parquet_holds_mixed_values_as_their_csv_text(self) -> None:
        sections = pd.DataFrame(
            [
                ["integrity", "pass", "m", 0.5, 1.0, "d"],
                ["spectral", "review", "m", "3/4", "Wake>=NREM", "d"],
                ["prescan", "pass", "m", "n/a", float("nan"), "d"],
            ],
            columns=SECTION_COLUMNS,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qc_summary.parquet"
            summary_parquet_table(sections).to_parquet(path, index=False)
            table = pd.read_parquet(path)

        self.assertEqual(list(table.columns), SECTION_COLUMNS)
        self.assertEqual(list(table["value"]), ["0.5", "3/4", "n/a"])
        self.assertEqual(list(table["threshold"]), ["1.0", "Wake>=NREM", ""])

    def test_outputs_are_named_after_the_analyzed_recording(self) -> None:
        edf = Path("/raw/sub-066/ses-001/ephys/sub-066_ses-001_recording-concat.edf")
        self.assertEqual(
            recording_output_name(DEFAULT_SUMMARY_FILENAME, edf),
            Path("sub-066_ses-001_recording-concat_qc_summary.csv"),
        )
        self.assertEqual(
            paired_output_paths(recording_output_name(DEFAULT_REVIEW_FILENAME, edf)),
            (
                Path("sub-066_ses-001_recording-concat_qc_review_epochs.csv"),
                Path("sub-066_ses-001_recording-concat_qc_review_epochs.parquet"),
            ),
        )

    def test_naming_strips_derivative_detail_and_never_doubles_the_prefix(self) -> None:
        stem = "sub-066_ses-001_recording-concat"
        for source in (
            f"{stem}.edf",
            f"{stem}_raw.fif",
            f"{stem}_resampled-128hz_raw.fif",
        ):
            self.assertEqual(
                recording_output_name("qc_summary.csv", source),
                Path(f"{stem}_qc_summary.csv"),
            )
        already = f"{stem}_qc_summary.csv"
        self.assertEqual(
            recording_output_name(already, f"{stem}.edf"), Path(already)
        )

    def test_absolute_output_paths_are_left_alone(self) -> None:
        self.assertEqual(
            recording_output_name("/elsewhere/custom.csv", "sub-066_ses-001_recording-concat.edf"),
            Path("/elsewhere/custom.csv"),
        )

    def test_cli_selects_subject_by_date_or_session(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--date", "20260717"]).date,
            "20260717",
        )
        self.assertEqual(
            parser.parse_args(["--subject", "66", "--session", "1"]).session,
            "1",
        )

    def test_cli_saves_summary_and_review_epochs_by_default(self) -> None:
        args = build_parser().parse_args(["--subject", "66", "--session", "1"])
        self.assertEqual(args.summary, DEFAULT_SUMMARY_FILENAME)
        self.assertEqual(args.review_epochs, DEFAULT_REVIEW_FILENAME)

    def test_cli_accepts_custom_output_filenames(self) -> None:
        args = build_parser().parse_args(
            ["--subject", "66", "--session", "1",
             "--summary", "custom_summary.csv",
             "--review-epochs", "custom_review.csv"],
        )
        self.assertEqual(args.summary, "custom_summary.csv")
        self.assertEqual(args.review_epochs, "custom_review.csv")

    def test_cli_opt_outs_disable_each_output(self) -> None:
        args = build_parser().parse_args(
            ["--subject", "66", "--session", "1", "--no-summary", "--no-review-epochs"],
        )
        self.assertIsNone(args.summary)
        self.assertIsNone(args.review_epochs)

    def test_spectral_sections_use_configured_threshold_results(self) -> None:
        report = pd.DataFrame(
            {
                "determines_recording_quality": [True, True, True, False],
                "spectral_quality_status": ["PASS", "REVIEW", "PASS", "FAIL"],
                "emg_quality_status": ["PASS", "PASS", "PASS", "FAIL"],
                "frequency_expectation_met": [True, False, True, False],
                "emg_expectation_met": [True, True, True, False],
                "eeg_channel": ["EEG1", "EEG1", "EEG1", "EEG2"],
                "sleep_state": ["Wake", "NREM", "REM", "Wake"],
                "emg_rms_median_uv": [3.0, 2.0, 1.0, 4.0],
            }
        )

        sections = spectral_quality_sections(report, DEFAULT_SPECTRA_CONFIG)

        self.assertEqual(sections[0]["section"], "power_spectra")
        self.assertEqual(sections[0]["status"], "review")
        self.assertEqual(sections[0]["value"], "2/3")
        self.assertIn("wake_delta<nrem", str(sections[0]["threshold"]))
        self.assertIn("determining EEG channel 1=EEG1", str(sections[0]["detail"]))
        self.assertEqual(sections[1]["section"], "emg_rms")
        self.assertEqual(sections[1]["status"], "pass")
        self.assertEqual(sections[1]["threshold"], "Wake>=NREM>=REM")

    def test_sleep_state_proportion_section_flags_over_threshold_state(self) -> None:
        scores = pd.DataFrame(
            {
                "time_s": list(range(12)),
                "label_output": [0] * 8 + [1, 2, 3, 3],
                "kind": ["signal"] * 10 + ["gap", "gap"],
                "prob_wake": [0.9] * 8 + [0.10, 0.10, 0.0, 0.0],
                "prob_nrem": [0.05] * 8 + [0.80, 0.10, 0.0, 0.0],
                "prob_rem": [0.03] * 8 + [0.05, 0.85, 0.0, 0.0],
                "prob_undef": [0.02] * 8 + [0.05, 0.05, 1.0, 1.0],
            }
        )
        output = prepare_scoring_output(scores, confidence_threshold=0.80)
        proportion_report = sleep_state_proportion_report(
            sleep_state_proportions(output),
            max_wake_percent=75.0,
            max_nrem_percent=75.0,
            max_rem_percent=20.0,
        )

        section = sleep_state_proportion_section(proportion_report)

        self.assertEqual(section["section"], "sleep_state_proportions")
        self.assertEqual(section["status"], "review")
        self.assertIn("Wake=80.00%", section["value"])
        self.assertIn("Wake<=75%", section["threshold"])
        self.assertIn("Wake review", section["detail"])
        self.assertNotIn("Undefined", section["value"])

    def test_sleep_state_proportion_section_passes_within_thresholds(self) -> None:
        proportions = pd.DataFrame(
            {
                "sleep_state": ["Wake", "NREM", "REM", "Undefined"],
                "epochs": [3, 4, 1, 0],
                "signal_percent": [37.5, 50.0, 12.5, 0.0],
            }
        )
        proportion_report = sleep_state_proportion_report(
            proportions,
            max_wake_percent=75.0,
            max_nrem_percent=75.0,
            max_rem_percent=20.0,
        )

        section = sleep_state_proportion_section(proportion_report)

        self.assertEqual(section["status"], "pass")
        self.assertEqual(section["detail"], "within thresholds")


if __name__ == "__main__":
    unittest.main()
