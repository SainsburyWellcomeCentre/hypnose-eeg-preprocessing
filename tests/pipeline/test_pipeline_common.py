from __future__ import annotations

import argparse
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from hypnose_eeg.pipeline.steps import (
    StepFailed,
    add_selector_arguments,
    output_dir_overrides,
    output_layout_env,
    parse_output_dir_overrides,
    run_step,
)


class RunStepTests(unittest.TestCase):
    """`run_step` calls a CLI's `main(argv)` in this process."""

    def _module(self, main) -> str:
        module = types.ModuleType("fake_step")
        module.main = main
        patcher = patch.dict(sys.modules, {"fake_step": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        return "fake_step"

    def test_calls_main_with_the_arguments(self):
        main = Mock(return_value=0)
        run_step(self._module(main), ["--subject", "66"], label="qc:summary")
        main.assert_called_once_with(["--subject", "66"])

    def test_none_from_main_is_success(self):
        run_step(self._module(Mock(return_value=None)), [], label="preprocessing:trim")

    def test_nonzero_return_raises_step_failed(self):
        with self.assertRaises(StepFailed) as ctx:
            run_step(self._module(Mock(return_value=2)), [], label="qc:summary")
        self.assertEqual(ctx.exception.returncode, 2)
        self.assertEqual(ctx.exception.module, "fake_step")
        self.assertIsNone(ctx.exception.error)
        self.assertIn("qc:summary", str(ctx.exception))

    def test_system_exit_becomes_the_exit_status(self):
        def usage_error(argv):
            raise SystemExit(2)

        with self.assertRaises(StepFailed) as ctx:
            run_step(self._module(usage_error), [], label="qc:summary")
        self.assertEqual(ctx.exception.returncode, 2)
        self.assertIsNone(ctx.exception.error)

    def test_system_exit_zero_is_success(self):
        def done(argv):
            raise SystemExit(0)

        run_step(self._module(done), [], label="qc:summary")

    def test_an_exception_is_a_status_1_failure_carrying_the_error(self):
        def crash(argv):
            raise FileNotFoundError("no artifact parquet")

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(StepFailed) as ctx:
                run_step(self._module(crash), [], label="qc:summary")
        self.assertEqual(ctx.exception.returncode, 1)
        self.assertIsInstance(ctx.exception.error, FileNotFoundError)
        self.assertIn("no artifact parquet", str(ctx.exception))

    def test_keyboard_interrupt_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            run_step(self._module(Mock(side_effect=KeyboardInterrupt)), [], label="x")

    def test_env_applies_during_the_step_only(self):
        seen = {}

        def main(argv):
            seen.update(os.environ)
            return 0

        with patch.dict(os.environ, {"HYPNOSE_TEST_INHERITED": "yes"}):
            os.environ.pop("HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS", None)
            run_step(
                self._module(main), [], label="qc:summary",
                env={"HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "analysis/artifacts"},
            )
            self.assertNotIn("HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS", os.environ)
        self.assertEqual(seen["HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS"], "analysis/artifacts")
        self.assertEqual(seen["HYPNOSE_TEST_INHERITED"], "yes")

    def test_env_is_restored_after_a_failure(self):
        with patch.dict(os.environ, {"HYPNOSE_EEG_OUTPUT_ROOT": "eeg"}):
            with self.assertRaises(StepFailed):
                run_step(
                    self._module(Mock(return_value=1)), [], label="qc:summary",
                    env={"HYPNOSE_EEG_OUTPUT_ROOT": "ephys"},
                )
            self.assertEqual(os.environ["HYPNOSE_EEG_OUTPUT_ROOT"], "eeg")


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
