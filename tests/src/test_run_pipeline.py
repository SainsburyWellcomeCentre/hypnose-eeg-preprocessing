from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src import run_pipeline
from src._pipeline import StepFailed


class FullRunOrderingTests(unittest.TestCase):
    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_no_stage_interleaves_detect_artifacts_after_scoring(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        manager = Mock()
        manager.attach_mock(mock_preprocessing.run_steps, "preprocessing")
        manager.attach_mock(mock_sleep_scoring.run_steps, "sleep_scoring")
        manager.attach_mock(mock_qc.run_steps, "qc")

        result = run_pipeline.main(["--subject", "66", "--date", "20260717", "--model", "my-model"])

        self.assertEqual(result, 0)
        call_names = [c[0] for c in manager.mock_calls]
        self.assertEqual(call_names, ["preprocessing", "sleep_scoring", "preprocessing", "qc"])

        first_preprocessing_call = mock_preprocessing.run_steps.call_args_list[0]
        self.assertEqual(
            first_preprocessing_call.kwargs["steps"],
            ["trim", "concatenate", "downsample", "prescan_artifacts"],
        )

        second_preprocessing_call = mock_preprocessing.run_steps.call_args_list[1]
        self.assertEqual(second_preprocessing_call.kwargs["steps"], ["detect_artifacts"])

        scoring_call = mock_sleep_scoring.run_steps.call_args
        self.assertEqual(scoring_call.kwargs["steps"], ["score"])
        self.assertEqual(scoring_call.kwargs["model"], "my-model")

        qc_call = mock_qc.run_steps.call_args
        self.assertEqual(qc_call.kwargs["steps"], ["summary"])


    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_output_folder_overrides_reach_every_stage(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        result = run_pipeline.main(
            ["--subject", "66", "--session", "1",
             "--output-layout", "/x/layout.yaml",
             "--output-dir", "artifacts=analysis/artifacts",
             "--output-dir", "quality_control=reports/qc"]
        )
        self.assertEqual(result, 0)
        expected_dirs = {"artifacts": "analysis/artifacts", "quality_control": "reports/qc"}
        for mock_stage in (mock_preprocessing, mock_sleep_scoring, mock_qc):
            for call in mock_stage.run_steps.call_args_list:
                self.assertEqual(call.kwargs["output_layout"], "/x/layout.yaml")
                self.assertEqual(call.kwargs["output_dirs"], expected_dirs)

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_default_run_passes_no_output_overrides(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        run_pipeline.main(["--subject", "66", "--session", "1"])
        for mock_stage in (mock_preprocessing, mock_sleep_scoring, mock_qc):
            for call in mock_stage.run_steps.call_args_list:
                self.assertIsNone(call.kwargs["output_layout"])
                self.assertEqual(call.kwargs["output_dirs"], {})

    def test_bad_output_dir_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            run_pipeline.main(["--subject", "66", "--output-dir", "figures=x"])
        self.assertEqual(ctx.exception.code, 2)

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_overwrite_reaches_every_recomputing_stage(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        run_pipeline.main(["--subject", "66", "--date", "20260717", "--overwrite"])

        for call in mock_preprocessing.run_steps.call_args_list:
            self.assertTrue(call.kwargs["overwrite"])
        self.assertTrue(mock_sleep_scoring.run_steps.call_args.kwargs["overwrite"])

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_steps_with_existing_outputs_are_skipped_by_default(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        run_pipeline.main(["--subject", "66", "--date", "20260717"])

        for call in mock_preprocessing.run_steps.call_args_list:
            self.assertFalse(call.kwargs["overwrite"])
        self.assertFalse(mock_sleep_scoring.run_steps.call_args.kwargs["overwrite"])


class SingleStageTests(unittest.TestCase):
    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_restricting_to_preprocessing_runs_its_own_defaults(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        run_pipeline.main(["--subject", "66", "--date", "20260717", "--stage", "preprocessing"])

        mock_preprocessing.run_steps.assert_called_once()
        self.assertNotIn("steps", mock_preprocessing.run_steps.call_args.kwargs)
        mock_sleep_scoring.run_steps.assert_not_called()
        mock_qc.run_steps.assert_not_called()

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_multiple_stages_can_be_selected(self, mock_preprocessing, mock_sleep_scoring, mock_qc):
        run_pipeline.main(
            ["--subject", "66", "--session", "1", "--stage", "sleep_scoring", "qc"]
        )

        mock_preprocessing.run_steps.assert_not_called()
        mock_sleep_scoring.run_steps.assert_called_once()
        mock_qc.run_steps.assert_called_once()


class ViewerTests(unittest.TestCase):
    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_viewer_is_not_opened_by_default(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        run_pipeline.main(["--subject", "66", "--date", "20260717"])

        steps = [c.kwargs["steps"] for c in mock_sleep_scoring.run_steps.call_args_list]
        self.assertEqual(steps, [["score"]])

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_view_opens_the_viewer_after_every_other_stage(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        manager = Mock()
        manager.attach_mock(mock_preprocessing.run_steps, "preprocessing")
        manager.attach_mock(mock_sleep_scoring.run_steps, "sleep_scoring")
        manager.attach_mock(mock_qc.run_steps, "qc")

        result = run_pipeline.main(["--subject", "66", "--date", "20260717", "--view"])

        self.assertEqual(result, 0)
        call_names = [c[0] for c in manager.mock_calls]
        self.assertEqual(
            call_names,
            ["preprocessing", "sleep_scoring", "preprocessing", "qc", "sleep_scoring"],
        )
        view_call = mock_sleep_scoring.run_steps.call_args_list[-1]
        self.assertEqual(view_call.kwargs["steps"], ["view"])

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_view_works_without_rerunning_any_stage(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        result = run_pipeline.main(
            ["--subject", "66", "--session", "1", "--stage", "qc", "--view"]
        )

        self.assertEqual(result, 0)
        mock_preprocessing.run_steps.assert_not_called()
        mock_sleep_scoring.run_steps.assert_called_once()
        self.assertEqual(
            mock_sleep_scoring.run_steps.call_args.kwargs["steps"], ["view"]
        )

    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_failing_stage_stops_before_the_viewer(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 3)

        result = run_pipeline.main(["--subject", "66", "--date", "20260717", "--view"])

        self.assertEqual(result, 3)
        steps = [c.kwargs["steps"] for c in mock_sleep_scoring.run_steps.call_args_list]
        self.assertEqual(steps, [["score"]])


class FailureTests(unittest.TestCase):
    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_stage_failure_stops_the_pipeline_and_returns_its_exit_code(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        mock_preprocessing.run_steps.side_effect = [
            None,
            StepFailed("preprocessing:detect_artifacts", "hypnose_eeg.preprocessing.detect_artifacts", 7),
        ]

        result = run_pipeline.main(["--subject", "66", "--date", "20260717"])

        self.assertEqual(result, 7)
        mock_qc.run_steps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
