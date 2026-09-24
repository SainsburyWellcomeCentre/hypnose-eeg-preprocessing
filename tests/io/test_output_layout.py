from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypnose_eeg.io import output_layout
from hypnose_eeg.io.output_layout import (
    DEFAULT_OUTPUT_DIR_NAMES,
    DEFAULT_OUTPUT_ROOT,
    OUTPUT_LAYOUT_ENV,
    OUTPUT_ROOT_ENV,
    output_dir_env_var,
    output_dir_name,
    output_dir_names,
    output_layout_config_path,
    output_root_dir,
)

# Every override variable this module honours, cleared before each test so the
# developer's shell environment can't leak into the assertions.
ALL_ENV_VARS = [OUTPUT_LAYOUT_ENV, OUTPUT_ROOT_ENV] + [
    output_dir_env_var(key) for key in DEFAULT_OUTPUT_DIR_NAMES
]


def rooted(*names: str) -> str:
    """The given folders as `output_dir_name` reports them, below the default root."""
    return "/".join((DEFAULT_OUTPUT_ROOT, *names))


class OutputLayoutTestCase(unittest.TestCase):
    def setUp(self) -> None:
        clean_env = {k: v for k, v in os.environ.items() if k not in ALL_ENV_VARS}
        patcher = patch.dict(os.environ, clean_env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class DefaultResolutionTests(OutputLayoutTestCase):
    def test_repository_layout_is_used_when_nothing_is_overridden(self):
        self.assertEqual(output_layout_config_path(), output_layout.DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH)
        self.assertEqual(output_root_dir(), "eeg")
        self.assertEqual(
            output_dir_names(),
            {key: rooted(folder) for key, folder in DEFAULT_OUTPUT_DIR_NAMES.items()},
        )

    def test_every_group_sits_below_the_modality_root(self):
        for folder in output_dir_names().values():
            self.assertTrue(folder.startswith("eeg/"), folder)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            output_dir_name("figures")
        with self.assertRaises(ValueError):
            output_dir_env_var("figures")

    def test_env_var_name_uses_upper_case_group(self):
        self.assertEqual(output_dir_env_var("sleep_scoring_qc"), "HYPNOSE_EEG_OUTPUT_DIR_SLEEP_SCORING_QC")


class OutputRootTests(OutputLayoutTestCase):
    def test_env_var_replaces_the_modality_root_for_every_group(self):
        os.environ[OUTPUT_ROOT_ENV] = "ephys"
        self.assertEqual(output_root_dir(), "ephys")
        self.assertEqual(output_dir_name("artifacts"), "ephys/artifacts")
        self.assertEqual(output_dir_name("downsample"), "ephys/downsample")

    def test_root_still_applies_to_a_relocated_group(self):
        os.environ[output_dir_env_var("artifacts")] = "analysis/artifacts"
        self.assertEqual(output_dir_name("artifacts"), "eeg/analysis/artifacts")

    def test_root_can_be_dropped(self):
        for value in (".", "", "   "):
            with self.subTest(value=value):
                os.environ[OUTPUT_ROOT_ENV] = value
                self.assertEqual(output_root_dir(), "")
                self.assertEqual(output_dir_name("artifacts"), "artifacts")

    def test_root_comes_from_the_layout_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "layout.yaml"
        path.write_text("output_root: ephys\noutput_dirs:\n  artifacts: detection\n")
        os.environ[OUTPUT_LAYOUT_ENV] = str(path)
        self.assertEqual(output_dir_name("artifacts"), "ephys/detection")
        # A layout file that says nothing about the root keeps the built-in one.
        path.write_text("output_dirs:\n  artifacts: detection\n")
        self.assertEqual(output_dir_name("artifacts"), "eeg/detection")

    def test_invalid_root_is_rejected(self):
        for value in ("/elsewhere", "../shared"):
            with self.subTest(value=value):
                os.environ[OUTPUT_ROOT_ENV] = value
                with self.assertRaises(ValueError):
                    output_dir_name("artifacts")


class EnvOverrideTests(OutputLayoutTestCase):
    def test_per_group_env_var_wins_over_everything(self):
        os.environ[output_dir_env_var("artifacts")] = "analysis/artifacts"
        self.assertEqual(output_dir_name("artifacts"), rooted("analysis/artifacts"))
        # Other groups are untouched.
        self.assertEqual(output_dir_name("sleep_scoring"), rooted("sleep_scoring"))

    def test_backslash_separators_are_normalised(self):
        os.environ[output_dir_env_var("artifacts")] = "analysis\\artifacts"
        self.assertEqual(output_dir_name("artifacts"), rooted("analysis/artifacts"))

    def test_absolute_paths_are_rejected(self):
        os.environ[output_dir_env_var("artifacts")] = "/elsewhere/artifacts"
        with self.assertRaises(ValueError):
            output_dir_name("artifacts")
        os.environ[output_dir_env_var("artifacts")] = "C:\\elsewhere"
        with self.assertRaises(ValueError):
            output_dir_name("artifacts")

    def test_traversal_outside_the_session_is_rejected(self):
        os.environ[output_dir_env_var("artifacts")] = "../shared/artifacts"
        with self.assertRaises(ValueError):
            output_dir_name("artifacts")

    def test_blank_override_is_rejected(self):
        os.environ[output_dir_env_var("artifacts")] = "   "
        with self.assertRaises(ValueError):
            output_dir_name("artifacts")
        os.environ[output_dir_env_var("artifacts")] = "./"
        with self.assertRaises(ValueError):
            output_dir_name("artifacts")


class LayoutFileOverrideTests(OutputLayoutTestCase):
    def _write_layout(self, body: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "layout.yaml"
        path.write_text(body)
        return path

    def test_alternative_layout_file_relocates_listed_groups_only(self):
        path = self._write_layout(
            "output_dirs:\n  artifacts: analysis/artifacts\n  quality_control: qc\n"
        )
        os.environ[OUTPUT_LAYOUT_ENV] = str(path)
        self.assertEqual(output_layout_config_path(), path)
        self.assertEqual(output_dir_name("artifacts"), rooted("analysis/artifacts"))
        self.assertEqual(output_dir_name("quality_control"), rooted("qc"))
        # Groups the file doesn't mention fall back to the built-in default, not
        # the repository yaml (which the external caller may not even have).
        self.assertEqual(output_dir_name("downsample"), rooted("downsample"))

    def test_per_group_env_var_still_wins_over_layout_file(self):
        path = self._write_layout("output_dirs:\n  artifacts: from_file\n")
        os.environ[OUTPUT_LAYOUT_ENV] = str(path)
        os.environ[output_dir_env_var("artifacts")] = "from_env"
        self.assertEqual(output_dir_name("artifacts"), rooted("from_env"))

    def test_missing_layout_file_fails_loudly(self):
        os.environ[OUTPUT_LAYOUT_ENV] = "/nonexistent/output_layout.yaml"
        with self.assertRaises(FileNotFoundError):
            output_dir_name("artifacts")

    def test_layout_file_expands_user_and_env_vars(self):
        path = self._write_layout("output_dirs:\n  artifacts: expanded\n")
        os.environ["HYPNOSE_TEST_LAYOUT_DIR"] = str(path.parent)
        os.environ[OUTPUT_LAYOUT_ENV] = "$HYPNOSE_TEST_LAYOUT_DIR/layout.yaml"
        self.assertEqual(output_dir_name("artifacts"), rooted("expanded"))


class DownstreamJoinTests(OutputLayoutTestCase):
    """Nested folders must compose with the helpers that join onto a session directory."""

    def test_session_output_dir_accepts_nested_folder(self):
        from hypnose_eeg.io import output_paths

        os.environ[output_dir_env_var("quality_control")] = "reports/qc"
        session = Path("/deriv/sub-001/ses-1_date-20260101")
        with patch.object(output_paths, "resolve_session_dir", return_value=session):
            resolved = output_paths.quality_control_output_path(
                "qc_summary.csv", "/raw/sub-001/ses-1_date-20260101/ephys/x.edf", "/raw", "/deriv"
            )
        self.assertEqual(resolved, session / "eeg" / "reports" / "qc" / "qc_summary.csv")

    def test_artifact_output_paths_accepts_nested_folder(self):
        from hypnose_eeg.io.output_paths import artifact_output_paths

        os.environ[output_dir_env_var("artifacts")] = "analysis/artifacts"
        fif = Path(
            "/deriv/sub-001/ses-1_date-20260101/eeg/downsample/"
            "sub-001_ses-1_recording-01_resampled-256hz_raw.fif"
        )
        csv_path, parquet_path = artifact_output_paths(fif, output_suffix="artifact_epochs")
        self.assertEqual(
            csv_path.parent, Path("/deriv/sub-001/ses-1_date-20260101/eeg/analysis/artifacts")
        )
        self.assertEqual(parquet_path.suffix, ".parquet")


if __name__ == "__main__":
    unittest.main()
