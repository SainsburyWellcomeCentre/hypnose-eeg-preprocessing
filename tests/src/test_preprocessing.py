from __future__ import annotations

import unittest
from unittest.mock import call, patch

from src import preprocessing
from src._pipeline import StepFailed


class RunStepsTests(unittest.TestCase):
    @patch("src.preprocessing.run_step")
    def test_default_steps_run_trim_first_then_the_rest_in_order(self, mock_run_step):
        preprocessing.run_steps(subject="66", session="1")

        labels = [c.kwargs["label"] for c in mock_run_step.call_args_list]
        self.assertEqual(
            labels,
            [
                "preprocessing:trim",
                "preprocessing:concatenate",
                "preprocessing:downsample",
                "preprocessing:detect_artifacts",
            ],
        )

    @patch("src.preprocessing.run_step")
    def test_output_folder_overrides_reach_every_step_as_env(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", session="1",
            output_layout="/x/layout.yaml", output_dirs={"artifacts": "analysis/artifacts"},
        )
        for c in mock_run_step.call_args_list:
            self.assertEqual(
                c.kwargs["env"],
                {
                    "HYPNOSE_EEG_OUTPUT_LAYOUT": "/x/layout.yaml",
                    "HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "analysis/artifacts",
                },
            )

    @patch("src.preprocessing.run_step")
    def test_no_output_overrides_means_empty_env(self, mock_run_step):
        preprocessing.run_steps(subject="66", session="1")
        for c in mock_run_step.call_args_list:
            self.assertEqual(c.kwargs["env"], {})

    @patch("src.preprocessing.run_step")
    def test_cli_output_flags_are_forwarded(self, mock_run_step):
        preprocessing.main(
            ["--subject", "66", "--session", "1", "--steps", "trim",
             "--output-layout", "/x/layout.yaml", "--output-dir", "downsample=ds"]
        )
        self.assertEqual(
            mock_run_step.call_args.kwargs["env"],
            {"HYPNOSE_EEG_OUTPUT_LAYOUT": "/x/layout.yaml", "HYPNOSE_EEG_OUTPUT_DIR_DOWNSAMPLE": "ds"},
        )

    @patch("src.preprocessing.run_step")
    def test_concatenate_and_downsample_use_source_sink_flags(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", date="20260717",
            rawdata_root="/raw", derivatives_root="/deriv",
            steps=["concatenate", "downsample"],
        )

        for c in mock_run_step.call_args_list:
            module, args = c.args
            self.assertIn("--source-dir", args)
            self.assertIn("/raw", args)
            self.assertIn("--sink-dir", args)
            self.assertIn("/deriv", args)
            self.assertNotIn("--rawdata-root", args)

    @patch("src.preprocessing.run_step")
    def test_detect_artifacts_uses_rawdata_derivatives_flags(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", date="20260717",
            rawdata_root="/raw", derivatives_root="/deriv",
            steps=["detect_artifacts"],
        )

        module, args = mock_run_step.call_args.args
        self.assertEqual(module, "scripts.preprocessing.detect_artifacts")
        self.assertIn("--rawdata-root", args)
        self.assertIn("--derivatives-root", args)
        self.assertNotIn("--source-dir", args)

    @patch("src.preprocessing.run_step")
    def test_trim_has_no_derivatives_root_flag(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", date="20260717",
            rawdata_root="/raw", derivatives_root="/deriv",
            steps=["trim"],
        )

        module, args = mock_run_step.call_args.args
        self.assertIn("--rawdata-root", args)
        self.assertNotIn("--derivatives-root", args)
        self.assertNotIn("--sink-dir", args)

    @patch("src.preprocessing.run_step")
    def test_dry_run_forwarded_to_every_step_but_detect_artifacts(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", date="20260717", dry_run=True,
            steps=["trim", "concatenate", "downsample", "detect_artifacts"],
        )

        for c in mock_run_step.call_args_list:
            module, args = c.args
            if "detect_artifacts" in module:
                self.assertNotIn("--dry-run", args)
            else:
                self.assertIn("--dry-run", args)

    @patch("src.preprocessing.run_step")
    def test_overwrite_forwarded_to_every_step(self, mock_run_step):
        preprocessing.run_steps(subject="66", date="20260717", overwrite=True)

        for c in mock_run_step.call_args_list:
            _, args = c.args
            self.assertIn("--overwrite", args)

    @patch("src.preprocessing.run_step")
    def test_extra_args_forwarded_to_every_selected_step(self, mock_run_step):
        preprocessing.run_steps(
            subject="66", date="20260717",
            steps=["concatenate"], extra_args=["--first"],
        )

        _, args = mock_run_step.call_args.args
        self.assertIn("--first", args)

    @patch("src.preprocessing.run_step")
    def test_step_failure_stops_remaining_steps(self, mock_run_step):
        mock_run_step.side_effect = [
            None,
            StepFailed("preprocessing:downsample", "scripts.preprocessing.downsample_recordings", 1),
        ]
        with self.assertRaises(StepFailed):
            preprocessing.run_steps(
                subject="66", date="20260717", steps=["concatenate", "downsample", "detect_artifacts"],
            )
        self.assertEqual(mock_run_step.call_count, 2)


class MainTests(unittest.TestCase):
    @patch("src.preprocessing.run_steps")
    def test_main_reports_failure_and_returns_its_exit_code(self, mock_run_steps):
        mock_run_steps.side_effect = StepFailed("preprocessing:concatenate", "mod", 3)
        result = preprocessing.main(["--subject", "66", "--date", "20260717"])
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
