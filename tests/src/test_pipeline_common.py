from __future__ import annotations

import argparse
import subprocess
import unittest
from unittest.mock import patch

from src._pipeline import StepFailed, add_selector_arguments, run_step


class RunStepTests(unittest.TestCase):
    @patch("src._pipeline.subprocess.run")
    def test_runs_module_with_python_from_repo_root(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        run_step("scripts.qc.summary_qc", ["--subject", "66"], label="qc:summary")

        called_args, called_kwargs = mock_run.call_args
        argv = called_args[0]
        self.assertEqual(argv[1:], ["-m", "scripts.qc.summary_qc", "--subject", "66"])
        self.assertTrue(str(called_kwargs["cwd"]).endswith("hypnose-eeg-preprocessing"))

    @patch("src._pipeline.subprocess.run")
    def test_nonzero_exit_raises_step_failed(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)
        with self.assertRaises(StepFailed) as ctx:
            run_step("scripts.qc.summary_qc", [], label="qc:summary")
        self.assertEqual(ctx.exception.returncode, 2)
        self.assertEqual(ctx.exception.module, "scripts.qc.summary_qc")
        self.assertIn("qc:summary", str(ctx.exception))


class SelectorArgumentsTests(unittest.TestCase):
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_selector_arguments(parser)
        return parser

    def test_subject_is_required(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args([])

    def test_date_and_session_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["--subject", "66", "--date", "20260717", "--session", "1"])

    def test_defaults(self):
        args = self._parser().parse_args(["--subject", "66"])
        self.assertEqual(args.subject, "66")
        self.assertIsNone(args.date)
        self.assertIsNone(args.session)
        self.assertIsNone(args.rawdata_root)
        self.assertIsNone(args.derivatives_root)


if __name__ == "__main__":
    unittest.main()
