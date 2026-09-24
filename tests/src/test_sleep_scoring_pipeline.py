from __future__ import annotations

import unittest
from unittest.mock import patch

from src import sleep_scoring
from src._pipeline import StepFailed


class RunStepsTests(unittest.TestCase):
    @patch("src.sleep_scoring.run_step")
    def test_default_step_is_score_only(self, mock_run_step):
        sleep_scoring.run_steps(subject="66", session="1")

        labels = [c.kwargs["label"] for c in mock_run_step.call_args_list]
        self.assertEqual(labels, ["sleep_scoring:score"])

    @patch("src.sleep_scoring.run_step")
    def test_model_only_forwarded_to_score_step(self, mock_run_step):
        sleep_scoring.run_steps(
            subject="66", date="20260717", model="my-model", steps=["score", "view"],
        )

        score_args = mock_run_step.call_args_list[0].args[1]
        view_args = mock_run_step.call_args_list[1].args[1]
        self.assertIn("--model", score_args)
        self.assertIn("my-model", score_args)
        self.assertNotIn("--model", view_args)

    @patch("src.sleep_scoring.run_step")
    def test_overwrite_only_forwarded_to_score_step_when_requested(self, mock_run_step):
        sleep_scoring.run_steps(
            subject="66", date="20260717", overwrite=True, steps=["score", "view"],
        )

        score_args = mock_run_step.call_args_list[0].args[1]
        view_args = mock_run_step.call_args_list[1].args[1]
        self.assertIn("--overwrite", score_args)
        self.assertNotIn("--overwrite", view_args)

    @patch("src.sleep_scoring.run_step")
    def test_scoring_skips_existing_predictions_by_default(self, mock_run_step):
        sleep_scoring.run_steps(subject="66", date="20260717")

        score_args = mock_run_step.call_args_list[0].args[1]
        self.assertNotIn("--overwrite", score_args)

    @patch("src.sleep_scoring.run_step")
    def test_output_folder_overrides_reach_every_step_as_env(self, mock_run_step):
        sleep_scoring.run_steps(
            subject="66", session="1", steps=["score", "view"],
            output_dirs={"sleep_scoring": "scoring/somnotate"},
        )
        for c in mock_run_step.call_args_list:
            self.assertEqual(
                c.kwargs["env"], {"HYPNOSE_EEG_OUTPUT_DIR_SLEEP_SCORING": "scoring/somnotate"}
            )

    @patch("src.sleep_scoring.run_step")
    def test_cli_output_flags_are_forwarded(self, mock_run_step):
        sleep_scoring.main(
            ["--subject", "66", "--session", "1", "--output-dir", "sleep_scoring=scoring"]
        )
        self.assertEqual(
            mock_run_step.call_args.kwargs["env"], {"HYPNOSE_EEG_OUTPUT_DIR_SLEEP_SCORING": "scoring"}
        )

    @patch("src.sleep_scoring.run_step")
    def test_view_not_run_unless_explicitly_selected(self, mock_run_step):
        sleep_scoring.run_steps(subject="66", date="20260717", steps=["score"])
        modules = [c.args[0] for c in mock_run_step.call_args_list]
        self.assertNotIn("hypnose_eeg.review.viewer", modules)

    @patch("src.sleep_scoring.run_step")
    def test_step_failure_propagates(self, mock_run_step):
        mock_run_step.side_effect = StepFailed(
            "sleep_scoring:score", "hypnose_eeg.sleep_scoring.score_recordings", 1
        )
        with self.assertRaises(StepFailed):
            sleep_scoring.run_steps(subject="66", date="20260717")


class MainTests(unittest.TestCase):
    @patch("src.sleep_scoring.run_steps")
    def test_main_reports_failure_and_returns_its_exit_code(self, mock_run_steps):
        mock_run_steps.side_effect = StepFailed("sleep_scoring:score", "mod", 4)
        result = sleep_scoring.main(["--subject", "66", "--date", "20260717"])
        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
