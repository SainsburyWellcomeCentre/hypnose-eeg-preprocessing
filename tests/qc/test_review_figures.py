from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hypnose_eeg.qc import review_figures
from hypnose_eeg.qc.review_figures import (
    main,
    render_session_figures,
    review_figure_settings,
)

SESSION = "ses-001_date-20260717"


class ReviewFigureSettingsTests(unittest.TestCase):
    def test_defaults_come_from_the_qc_config(self):
        settings = review_figure_settings()

        self.assertEqual(settings.hours, (0.0, 12.0))
        self.assertEqual(settings.scoring, "review")
        self.assertEqual(settings.spectra, "review")
        self.assertTrue(settings.show_artifacts)

    def test_overrides_replace_only_what_is_given(self):
        settings = review_figure_settings(hours=[2, 8], spectra="always", dpi=None)

        self.assertEqual(settings.hours, (2.0, 8.0))
        self.assertEqual(settings.spectra, "always")
        self.assertEqual(settings.scoring, "review")
        self.assertEqual(settings.dpi, review_figure_settings().dpi)

    def test_invalid_settings_are_refused(self):
        for overrides in (
            {"spectra": "sometimes"}, {"scoring": "yes"}, {"hours": (6, 2)}, {"dpi": 0},
        ):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                review_figure_settings(**overrides)


@patch("hypnose_eeg.qc.spectra.save_spectra")
@patch("hypnose_eeg.qc.spectra.plot_spectra")
@patch("hypnose_eeg.qc.spectra.compute_session_spectra")
@patch("hypnose_eeg.sleep_scoring.view_scoring.render_scoring")
class RenderSessionFiguresTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.rawdata = root / "rawdata"
        self.derivatives = root / "derivatives"
        self.edf = self.rawdata / "sub-066" / SESSION / "ephys" / "sub-066_ses-001.edf"
        self.edf.parent.mkdir(parents=True)
        self.qc_dir = self.derivatives / "sub-066" / SESSION / "eeg" / "quality_control"
        (self.derivatives / "sub-066" / SESSION).mkdir(parents=True)

    def render(self, verdict, **overrides):
        with patch.object(review_figures, "read_qc_verdict", return_value=verdict) as read:
            paths = render_session_figures(
                66, session=1, rawdata_root=self.rawdata, derivatives_root=self.derivatives,
                settings=review_figure_settings(**overrides),
            )
        return paths, read

    def overview_target(self, render_scoring) -> Path:
        """Where the scoring overview of `self.edf` would be saved."""
        _settings, path = render_scoring.call_args.args
        return path(SimpleNamespace(edf_path=self.edf))

    def test_the_overview_covers_the_configured_hours_in_the_qc_folder(
        self, render_scoring, compute, plot, save
    ):
        render_scoring.return_value = Path("overview.png")

        paths, _ = self.render(("review", []), spectra="never")

        self.assertEqual(paths, [Path("overview.png")])
        view = render_scoring.call_args.args[0]
        self.assertEqual(view.hours, (0.0, 12.0))
        self.assertEqual(view.session, 1)
        self.assertTrue(view.show_artifacts)
        self.assertEqual(
            self.overview_target(render_scoring),
            self.qc_dir / "sub-066_ses-001_scoring_hours-0-12.png",
        )

    def test_a_passing_session_gets_no_figures(self, render_scoring, compute, plot, save):
        paths, read = self.render(("pass", []))

        self.assertEqual(paths, [])
        self.assertEqual(read.call_args.kwargs["session"], 1)
        render_scoring.assert_not_called()
        compute.assert_not_called()

    def test_a_session_under_review_gets_the_overview_and_its_spectra_as_pngs(
        self, render_scoring, compute, plot, save
    ):
        render_scoring.return_value = Path("overview.png")
        result = SimpleNamespace(edf_path=self.edf)
        compute.return_value = [result]
        plot.return_value = {}
        save.return_value = [Path("spectra.png")]

        for verdict in (("review", []), ("fail", ["artifacts"])):
            with self.subTest(verdict=verdict[0]):
                paths, read = self.render(verdict)

                self.assertEqual(paths, [Path("overview.png"), Path("spectra.png")])
                read.assert_called_once()
                saved_result, save_dir, _figures = save.call_args.args
                self.assertIs(saved_result, result)
                self.assertEqual(save_dir, self.qc_dir)
                self.assertEqual(save.call_args.kwargs["figure_format"], "png")

    def test_no_summary_means_no_figures_unless_asked_for(
        self, render_scoring, compute, plot, save
    ):
        compute.return_value = []

        self.assertEqual(self.render(None)[0], [])
        render_scoring.assert_not_called()
        compute.assert_not_called()

        self.render(None, scoring="always", spectra="always")
        render_scoring.assert_called_once()
        compute.assert_called_once()

    def test_each_figure_can_be_forced_for_a_passing_session(
        self, render_scoring, compute, plot, save
    ):
        compute.return_value = []

        self.render(("pass", []), scoring="always")
        render_scoring.assert_called_once()
        compute.assert_not_called()

        self.render(("pass", []), spectra="always")
        render_scoring.assert_called_once()
        compute.assert_called_once()

    def test_without_a_review_setting_the_verdict_is_not_read(
        self, render_scoring, compute, plot, save
    ):
        paths, read = self.render(("review", []), scoring="never", spectra="never")

        self.assertEqual(paths, [])
        read.assert_not_called()
        render_scoring.assert_not_called()
        compute.assert_not_called()


class MainTests(unittest.TestCase):
    @patch.object(review_figures, "render_session_figures", return_value=[Path("a.png")])
    def test_arguments_for_other_qc_steps_are_ignored(self, render):
        result = main(
            ["--subject", "66", "--session", "1", "--no-review-epochs",
             "--figure-hours", "0", "6", "--scoring", "always", "--spectra", "never"]
        )

        self.assertEqual(result, 0)
        settings = render.call_args.kwargs["settings"]
        self.assertEqual(settings.hours, (0.0, 6.0))
        self.assertEqual(settings.scoring, "always")
        self.assertEqual(settings.spectra, "never")
        self.assertEqual(render.call_args.kwargs["session"], "1")

    @patch.object(
        review_figures, "render_session_figures",
        side_effect=FileNotFoundError("No somnotate predictions"),
    )
    def test_missing_inputs_fail_the_step(self, render):
        self.assertEqual(main(["--subject", "66", "--session", "1"]), 1)


if __name__ == "__main__":
    unittest.main()
