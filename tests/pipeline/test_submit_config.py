from __future__ import annotations

import io
import shlex
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hypnose_eeg.pipeline import submit_config
from hypnose_eeg.pipeline.run import build_parser

CONFIG = """\
sbatch:
  time: "48:00:00"
  mem: 64G
  cpus_per_task: 8
  partition: null
pipeline:
  subject: [66, "67:1-3"]
  all_sessions: true
  model: my-model
  task_unit: session
  overwrite: false
  stage: [qc]
  extra_args: [--channel-labels, EEG1]
"""


class SubmitConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parser = build_parser()

    def write(self, text: str) -> Path:
        path = Path(self._tmp.name) / "submit.yaml"
        path.write_text(text)
        return path

    def merged(self, text: str, *cli_args: str) -> list[str]:
        config = submit_config.read_submit_config(self.write(text), self.parser)
        return submit_config.merge_pipeline_args(config, list(cli_args), self.parser)

    def test_the_file_becomes_pipeline_arguments(self):
        self.assertEqual(
            self.merged(CONFIG),
            [
                "--subject", "66", "67:1-3", "--all-sessions", "--model", "my-model",
                "--task-unit", "session", "--stage", "qc",
                "--channel-labels", "EEG1",
            ],
        )

    def test_the_command_line_replaces_the_files_options(self):
        args = self.merged(CONFIG, "--model", "other", "--subject", "68")

        self.assertNotIn("my-model", args)
        self.assertEqual(args[-4:], ["--model", "other", "--subject", "68"])
        self.assertEqual(args.count("--subject"), 1)

    def test_a_command_line_selector_replaces_the_files_exclusive_one(self):
        args = self.merged(CONFIG, "--session", "2")

        self.assertNotIn("--all-sessions", args)
        self.assertEqual(args[-2:], ["--session", "2"])

    def test_aliases_and_equals_forms_count_as_given(self):
        args = self.merged(CONFIG, "--model-path=x", "--batch")

        self.assertNotIn("my-model", args)
        self.assertEqual(args.count("--all-sessions"), 0)

    def test_sbatch_values_are_read(self):
        config = submit_config.read_submit_config(self.write(CONFIG), self.parser)

        self.assertEqual(config.sbatch, {"time": "48:00:00", "mem": "64G", "cpus_per_task": "8"})

    def test_mistakes_are_refused(self):
        cases = {
            "unknown section": "slurm:\n  mem: 1G\n",
            "unknown sbatch key": "sbatch:\n  memory: 1G\n",
            "unquoted time": "sbatch:\n  time: 24:00:00\n",
            "unknown pipeline option": "pipeline:\n  subjects: [66]\n",
            "value for a flag": "pipeline:\n  overwrite: yes please\n",
            "true for an option with a value": "pipeline:\n  model: true\n",
            "extra args not a list": "pipeline:\n  extra_args: --x\n",
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                submit_config.read_submit_config(self.write(text), self.parser)

    def test_main_prints_shell_assignments(self):
        path = self.write(CONFIG)
        out = io.StringIO()
        with redirect_stdout(out):
            code = submit_config.main([str(path), "--", "--mem-note", "a b"])

        self.assertEqual(code, 0)
        lines = out.getvalue().splitlines()
        self.assertIn("CFG_TIME=48:00:00", lines)
        self.assertIn("CFG_PARTITION=''", lines)
        array = next(line for line in lines if line.startswith("CFG_PIPELINE_ARGS="))
        values = shlex.split(array[len("CFG_PIPELINE_ARGS=("):-1])
        self.assertEqual(values[-2:], ["--mem-note", "a b"])

    def test_main_reports_a_bad_file(self):
        with redirect_stderr(io.StringIO()) as err:
            code = submit_config.main([str(self.write("sbatch:\n  memory: 1G\n"))])

        self.assertEqual(code, 1)
        self.assertIn("unknown sbatch key", err.getvalue())


if __name__ == "__main__":
    unittest.main()
