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
        self.assertEqual(first_preprocessing_call.kwargs["steps"], ["trim", "concatenate", "downsample"])

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


class FailureTests(unittest.TestCase):
    @patch("src.run_pipeline.qc")
    @patch("src.run_pipeline.sleep_scoring")
    @patch("src.run_pipeline.preprocessing")
    def test_stage_failure_stops_the_pipeline_and_returns_its_exit_code(
        self, mock_preprocessing, mock_sleep_scoring, mock_qc
    ):
        mock_preprocessing.run_steps.side_effect = [
            None,
            StepFailed("preprocessing:detect_artifacts", "scripts.preprocessing.detect_artifacts", 7),
        ]

        result = run_pipeline.main(["--subject", "66", "--date", "20260717"])

        self.assertEqual(result, 7)
        mock_qc.run_steps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
