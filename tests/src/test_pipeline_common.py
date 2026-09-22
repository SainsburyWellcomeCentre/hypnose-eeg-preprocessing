from __future__ import annotations

import argparse
import os
import subprocess
import unittest
from unittest.mock import patch

from src._pipeline import (
    StepFailed,
    add_selector_arguments,
    output_dir_overrides,
    output_layout_env,
    parse_output_dir_overrides,
    run_step,
)


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
    def test_no_env_override_inherits_the_process_environment(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        run_step("scripts.qc.summary_qc", [], label="qc:summary")
        self.assertIsNone(mock_run.call_args.kwargs["env"])

    @patch("src._pipeline.subprocess.run")
    def test_env_override_is_layered_over_the_process_environment(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        with patch.dict(os.environ, {"HYPNOSE_TEST_INHERITED": "yes"}):
            run_step(
                "scripts.qc.summary_qc", [], label="qc:summary",
                env={"HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "analysis/artifacts"},
            )
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS"], "analysis/artifacts")
        self.assertEqual(env["HYPNOSE_TEST_INHERITED"], "yes")

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
        self.assertIsNone(args.output_layout)
        self.assertIsNone(args.output_dir)

    def test_output_dir_is_repeatable(self):
        args = self._parser().parse_args(
            ["--subject", "66", "--output-dir", "artifacts=a", "--output-dir", "quality_control=q"]
        )
        self.assertEqual(args.output_dir, ["artifacts=a", "quality_control=q"])
        self.assertEqual(output_dir_overrides(self._parser(), args), {"artifacts": "a", "quality_control": "q"})

    def test_bad_output_dir_is_a_usage_error(self):
        parser = self._parser()
        for bad in ("artifacts", "=x", "figures=x", "artifacts="):
            args = parser.parse_args(["--subject", "66", "--output-dir", bad])
            with self.assertRaises(SystemExit):
                output_dir_overrides(parser, args)


class OutputLayoutEnvTests(unittest.TestCase):
    def test_nothing_requested_yields_empty_env(self):
        self.assertEqual(output_layout_env(), {})
        self.assertEqual(output_layout_env(None, {}), {})

    def test_layout_file_and_group_overrides_map_to_env_vars(self):
        env = output_layout_env("/x/layout.yaml", {"artifacts": "analysis/artifacts"})
        self.assertEqual(
            env,
            {
                "HYPNOSE_EEG_OUTPUT_LAYOUT": "/x/layout.yaml",
                "HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "analysis/artifacts",
            },
        )

    def test_output_root_maps_to_its_own_env_var(self):
        env = output_layout_env(None, None, "ephys")
        self.assertEqual(env, {"HYPNOSE_EEG_OUTPUT_ROOT": "ephys"})
        # `.` is a value, not an absent override: it drops the modality folder.
        self.assertEqual(output_layout_env(None, None, "."), {"HYPNOSE_EEG_OUTPUT_ROOT": "."})

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            output_layout_env(None, {"figures": "x"})

    def test_parse_overrides_requires_group_equals_folder(self):
        self.assertEqual(parse_output_dir_overrides(None), {})
        self.assertEqual(parse_output_dir_overrides(["artifacts = a/b "]), {"artifacts": "a/b"})
        with self.assertRaises(ValueError):
            parse_output_dir_overrides(["artifacts"])
        with self.assertRaises(ValueError):
            parse_output_dir_overrides(["figures=x"])


if __name__ == "__main__":
    unittest.main()
