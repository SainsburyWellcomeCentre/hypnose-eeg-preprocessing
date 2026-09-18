from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.io import output_layout
from scripts.io.output_layout import (
    DEFAULT_OUTPUT_DIR_NAMES,
    OUTPUT_LAYOUT_ENV,
    output_dir_env_var,
    output_dir_name,
    output_dir_names,
    output_layout_config_path,
)

# Every override variable this module honours, cleared before each test so the
# developer's shell environment can't leak into the assertions.
ALL_ENV_VARS = [OUTPUT_LAYOUT_ENV] + [output_dir_env_var(key) for key in DEFAULT_OUTPUT_DIR_NAMES]


class OutputLayoutTestCase(unittest.TestCase):
    def setUp(self) -> None:
        clean_env = {k: v for k, v in os.environ.items() if k not in ALL_ENV_VARS}
        patcher = patch.dict(os.environ, clean_env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class DefaultResolutionTests(OutputLayoutTestCase):
    def test_repository_layout_is_used_when_nothing_is_overridden(self):
        self.assertEqual(output_layout_config_path(), output_layout.DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH)
        self.assertEqual(output_dir_names(), DEFAULT_OUTPUT_DIR_NAMES)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            output_dir_name("figures")
        with self.assertRaises(ValueError):
            output_dir_env_var("figures")

    def test_env_var_name_uses_upper_case_group(self):
        self.assertEqual(output_dir_env_var("sleep_scoring_qc"), "HYPNOSE_EEG_OUTPUT_DIR_SLEEP_SCORING_QC")


class EnvOverrideTests(OutputLayoutTestCase):
    def test_per_group_env_var_wins_over_everything(self):
        os.environ[output_dir_env_var("artifacts")] = "analysis/artifacts"
        self.assertEqual(output_dir_name("artifacts"), "analysis/artifacts")
        # Other groups are untouched.
        self.assertEqual(output_dir_name("sleep_scoring"), "sleep_scoring")

    def test_backslash_separators_are_normalised(self):
        os.environ[output_dir_env_var("artifacts")] = "analysis\\artifacts"
        self.assertEqual(output_dir_name("artifacts"), "analysis/artifacts")

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
        self.assertEqual(output_dir_name("artifacts"), "analysis/artifacts")
        self.assertEqual(output_dir_name("quality_control"), "qc")
        # Groups the file doesn't mention fall back to the built-in default, not
        # the repository yaml (which the external caller may not even have).
        self.assertEqual(output_dir_name("downsample"), "downsample")

    def test_per_group_env_var_still_wins_over_layout_file(self):
        path = self._write_layout("output_dirs:\n  artifacts: from_file\n")
        os.environ[OUTPUT_LAYOUT_ENV] = str(path)
        os.environ[output_dir_env_var("artifacts")] = "from_env"
        self.assertEqual(output_dir_name("artifacts"), "from_env")

    def test_missing_layout_file_fails_loudly(self):
        os.environ[OUTPUT_LAYOUT_ENV] = "/nonexistent/output_layout.yaml"
        with self.assertRaises(FileNotFoundError):
            output_dir_name("artifacts")

    def test_layout_file_expands_user_and_env_vars(self):
        path = self._write_layout("output_dirs:\n  artifacts: expanded\n")
        os.environ["HYPNOSE_TEST_LAYOUT_DIR"] = str(path.parent)
        os.environ[OUTPUT_LAYOUT_ENV] = "$HYPNOSE_TEST_LAYOUT_DIR/layout.yaml"
        self.assertEqual(output_dir_name("artifacts"), "expanded")


class DownstreamJoinTests(OutputLayoutTestCase):
    """Nested folders must compose with the helpers that join onto a session directory."""

    def test_session_output_dir_accepts_nested_folder(self):
        from scripts.io import output_paths

        os.environ[output_dir_env_var("quality_control")] = "reports/qc"
        session = Path("/deriv/sub-001/ses-1_date-20260101")
        with patch.object(output_paths, "resolve_session_dir", return_value=session):
            resolved = output_paths.quality_control_output_path(
                "qc_summary.csv", "/raw/sub-001/ses-1_date-20260101/ephys/x.edf", "/raw", "/deriv"
            )
        self.assertEqual(resolved, session / "reports" / "qc" / "qc_summary.csv")

    def test_artifact_output_paths_accepts_nested_folder(self):
        from scripts.io.output_paths import artifact_output_paths

        os.environ[output_dir_env_var("artifacts")] = "analysis/artifacts"
        fif = Path(
            "/deriv/sub-001/ses-1_date-20260101/downsample/"
            "sub-001_ses-1_recording-01_resampled-256hz_raw.fif"
        )
        csv_path, parquet_path = artifact_output_paths(fif, output_suffix="artifact_epochs")
        self.assertEqual(
            csv_path.parent, Path("/deriv/sub-001/ses-1_date-20260101/analysis/artifacts")
        )
        self.assertEqual(parquet_path.suffix, ".parquet")


if __name__ == "__main__":
    unittest.main()
