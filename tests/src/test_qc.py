from __future__ import annotations

import unittest
from unittest.mock import patch

from src import qc
from src._pipeline import StepFailed


class RunStepsTests(unittest.TestCase):
    @patch("src.qc.run_step")
    def test_default_step_is_summary_only(self, mock_run_step):
        qc.run_steps(subject="66", session="1")

        labels = [c.kwargs["label"] for c in mock_run_step.call_args_list]
        self.assertEqual(labels, ["qc:summary"])
        module = mock_run_step.call_args.args[0]
        self.assertEqual(module, "scripts.qc.summary_qc")

    @patch("src.qc.run_step")
    def test_steps_run_in_canonical_order_regardless_of_input_order(self, mock_run_step):
        qc.run_steps(subject="66", date="20260717", steps=["summary", "integrity", "spectra"])

        modules = [c.args[0] for c in mock_run_step.call_args_list]
        self.assertEqual(
            modules,
            [
                "scripts.qc.recording_integrity",
                "scripts.qc.spectra",
                "scripts.qc.summary_qc",
            ],
        )

    @patch("src.qc.run_step")
    def test_selector_and_extra_args_forwarded_to_every_step(self, mock_run_step):
        qc.run_steps(
            subject="66", date="20260717", rawdata_root="/raw", derivatives_root="/deriv",
            steps=["integrity", "artifacts"], extra_args=["--no-show"],
        )

        for c in mock_run_step.call_args_list:
            _, args = c.args
            self.assertEqual(
                args,
                [
                    "--subject", "66", "--date", "20260717",
                    "--rawdata-root", "/raw", "--derivatives-root", "/deriv",
                    "--no-show",
                ],
            )

    @patch("src.qc.run_step")
    def test_output_folder_overrides_reach_every_step_as_env(self, mock_run_step):
        qc.run_steps(
            subject="66", session="1", steps=["integrity", "summary"],
            output_dirs={"quality_control": "reports/qc"},
        )
        for c in mock_run_step.call_args_list:
            self.assertEqual(c.kwargs["env"], {"HYPNOSE_EEG_OUTPUT_DIR_QUALITY_CONTROL": "reports/qc"})

    @patch("src.qc.run_step")
    def test_cli_output_flags_are_forwarded(self, mock_run_step):
        qc.main(["--subject", "66", "--session", "1", "--output-layout", "/x/layout.yaml"])
        self.assertEqual(
            mock_run_step.call_args.kwargs["env"], {"HYPNOSE_EEG_OUTPUT_LAYOUT": "/x/layout.yaml"}
        )

    @patch("src.qc.run_step")
    def test_step_failure_propagates(self, mock_run_step):
        mock_run_step.side_effect = StepFailed("qc:summary", "scripts.qc.summary_qc", 1)
        with self.assertRaises(StepFailed):
            qc.run_steps(subject="66", date="20260717")


class MainTests(unittest.TestCase):
    @patch("src.qc.run_steps")
    def test_main_reports_failure_and_returns_its_exit_code(self, mock_run_steps):
        mock_run_steps.side_effect = StepFailed("qc:summary", "mod", 5)
        result = qc.main(["--subject", "66", "--date", "20260717"])
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
