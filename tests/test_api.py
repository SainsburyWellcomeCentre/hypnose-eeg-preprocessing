from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hypnose_eeg import api


class FakeProfile:
    """Stands in for a `hypnose_helpers.io.paths.DataLocations`."""

    def get_rawdata_root(self) -> Path:
        return Path("/profile/rawdata")

    def get_derivatives_root(self) -> Path:
        return Path("/profile/derivatives")


class DataLocationsTests(unittest.TestCase):
    def test_defaults_leave_every_location_to_the_profile(self):
        self.assertEqual(
            api.DataLocations().stage_options(),
            dict(
                rawdata_root=None, derivatives_root=None, output_layout=None,
                output_root=None, output_dirs=None,
            ),
        )
        self.assertEqual(api.DataLocations().env(), {})

    def test_paths_reach_the_stages_as_text(self):
        options = api.DataLocations(
            rawdata_root=Path("/raw"), derivatives_root=Path("/deriv"),
            output_layout=Path("/x/layout.yaml"), output_root="ephys",
            output_dirs={"artifacts": "analysis/artifacts"},
        ).stage_options()
        self.assertEqual(options["rawdata_root"], str(Path("/raw")))
        self.assertEqual(options["derivatives_root"], str(Path("/deriv")))
        self.assertEqual(options["output_layout"], str(Path("/x/layout.yaml")))
        self.assertEqual(options["output_root"], "ephys")
        self.assertEqual(options["output_dirs"], {"artifacts": "analysis/artifacts"})

    def test_output_overrides_become_environment_variables(self):
        locations = api.DataLocations(
            output_root="ephys", output_dirs={"quality_control": "reports/qc"}
        )
        self.assertEqual(
            locations.env(),
            {
                "HYPNOSE_EEG_OUTPUT_ROOT": "ephys",
                "HYPNOSE_EEG_OUTPUT_DIR_QUALITY_CONTROL": "reports/qc",
            },
        )

    def test_a_helpers_profile_supplies_both_roots(self):
        locations = api.DataLocations.from_profile(FakeProfile(), output_root=".")
        self.assertEqual(locations.rawdata_root, Path("/profile/rawdata"))
        self.assertEqual(locations.derivatives_root, Path("/profile/derivatives"))
        self.assertEqual(locations.output_root, ".")


@patch("hypnose_eeg.api._preprocessing.run_steps")
class StageFunctionTests(unittest.TestCase):
    def test_preprocess_forwards_selection_and_locations(self, run_steps):
        api.preprocess(
            66, session=1, locations=api.DataLocations(rawdata_root="/raw"),
            steps=("trim", "concatenate"), overwrite=True,
        )
        kwargs = run_steps.call_args.kwargs
        self.assertEqual(kwargs["subject"], "66")
        self.assertEqual(kwargs["session"], "1")
        self.assertIsNone(kwargs["date"])
        self.assertEqual(kwargs["rawdata_root"], "/raw")
        self.assertEqual(kwargs["steps"], ["trim", "concatenate"])
        self.assertTrue(kwargs["overwrite"])

    def test_a_helpers_profile_is_accepted_as_locations(self, run_steps):
        api.preprocess(66, date=20260717, locations=FakeProfile())
        kwargs = run_steps.call_args.kwargs
        self.assertEqual(kwargs["date"], "20260717")
        self.assertEqual(kwargs["rawdata_root"], str(Path("/profile/rawdata")))
        self.assertEqual(kwargs["derivatives_root"], str(Path("/profile/derivatives")))

    def test_session_and_date_together_are_refused(self, run_steps):
        with self.assertRaises(ValueError):
            api.preprocess(66, session=1, date="20260717")
        run_steps.assert_not_called()

    def test_unrecognised_locations_are_refused(self, run_steps):
        with self.assertRaises(TypeError):
            api.preprocess(66, session=1, locations="/raw")

    def test_a_step_failure_raises(self, run_steps):
        run_steps.side_effect = api.StepFailed("preprocessing:trim", "mod", 2)
        with self.assertRaises(api.StepFailed):
            api.preprocess(66, session=1)


class PipelineFunctionTests(unittest.TestCase):
    @patch("hypnose_eeg.api._sleep_scoring.run_steps")
    def test_score_runs_only_the_score_step(self, run_steps):
        api.score(66, session=1, model=Path("model.pickle"))
        kwargs = run_steps.call_args.kwargs
        self.assertEqual(kwargs["steps"], ["score"])
        self.assertEqual(kwargs["model"], "model.pickle")

    @patch("hypnose_eeg.api._qc.run_steps")
    def test_quality_control_uses_the_stage_defaults(self, run_steps):
        api.quality_control(66, session=1)
        self.assertIsNone(run_steps.call_args.kwargs["steps"])

    @patch("hypnose_eeg.api._run.run_stages")
    def test_run_session_runs_the_stages_without_the_viewer(self, run_stages):
        api.run_session(66, session=1, stages=["qc"], model="m")
        common = run_stages.call_args.args[0]
        self.assertEqual(common["subject"], "66")
        self.assertEqual(common["session"], "1")
        kwargs = run_stages.call_args.kwargs
        self.assertEqual(kwargs["stages"], ["qc"])
        self.assertFalse(kwargs["view"])
        self.assertEqual(kwargs["extra"], [])

    @patch("hypnose_eeg.api._run.run_batch")
    def test_run_batch_takes_one_subject_or_several(self, run_batch):
        api.run_batch(66, keep_failed=True)
        self.assertEqual(run_batch.call_args.args[0], ["66"])
        self.assertTrue(run_batch.call_args.kwargs["keep_failed"])
        api.run_batch(["all"])
        self.assertEqual(run_batch.call_args.args[0], ["all"])

    @patch("hypnose_eeg.api.compute_session_qc")
    def test_session_qc_applies_the_output_overrides_while_computing(self, compute):
        seen = {}

        def capture(*args, **kwargs):
            seen["env"] = os.environ.get("HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS")
            return "result"

        compute.side_effect = capture
        with patch.dict(os.environ):
            os.environ.pop("HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS", None)
            result = api.session_qc(
                66, session=1,
                locations=api.DataLocations(output_dirs={"artifacts": "elsewhere"}),
            )
            self.assertNotIn("HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS", os.environ)
        self.assertEqual(result, "result")
        self.assertEqual(seen["env"], "elsewhere")
        self.assertEqual(compute.call_args.args, ("66",))
        self.assertEqual(compute.call_args.kwargs["session"], "1")

    @patch("hypnose_eeg.api.write_review_report")
    @patch("hypnose_eeg.api.review_subjects", return_value=["review"])
    def test_review_qc_reads_and_optionally_reports(self, review, write):
        locations = api.DataLocations(
            derivatives_root="/deriv", output_dirs={"quality_control": "qc"}
        )
        self.assertEqual(api.review_qc(66, locations=locations), ["review"])
        self.assertEqual(review.call_args.args[0], [66])
        self.assertEqual(review.call_args.kwargs["derivatives_root"], "/deriv")
        self.assertEqual(
            review.call_args.kwargs["env"],
            {"HYPNOSE_EEG_OUTPUT_DIR_QUALITY_CONTROL": "qc"},
        )
        write.assert_not_called()
        api.review_qc([66], report="review.csv")
        write.assert_called_once_with(["review"], "review.csv")


if __name__ == "__main__":
    unittest.main()
