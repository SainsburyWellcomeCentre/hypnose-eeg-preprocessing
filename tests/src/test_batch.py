from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypnose_helpers.io.layout import SessionRef

from src import run_pipeline
from src._batch import (
    SessionOutcome,
    erase_session_outputs,
    format_summary,
    missing_subject_outcome,
    report_path,
    resolve_subjects,
    session_selector,
)
from src._pipeline import StepFailed

SUBJECT = "66"
SESSION_DATES = {1: "20260717", 2: "20260718", 3: "20260719"}


def session_ref(root: Path, ses: int | None, date: str, subjid: int = 66) -> SessionRef:
    subject = f"sub-{subjid:03d}"
    subject_dir = root / subject
    token = f"{ses:03d}" if ses is not None else "pilot"
    return SessionRef(
        subjid=subjid,
        subject=subject,
        subject_dir=subject_dir,
        ses=ses,
        date=date,
        path=subject_dir / f"ses-{token}_date-{date}",
        session_index=1,
    )


def make_session_tree(root: Path, ses: int, date: str, *, files: dict[str, str]) -> Path:
    """Create `<root>/sub-066/ses-NNN_date-.../<relative files>` and return the session dir."""
    session_dir = root / "sub-066" / f"ses-{ses:03d}_date-{date}"
    for relative, content in files.items():
        path = session_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


class SessionSelectorTests(unittest.TestCase):
    def test_numbered_session_is_selected_by_number(self):
        ref = session_ref(Path("/rawdata"), 2, "20260718")
        self.assertEqual(session_selector(ref), {"session": "2", "date": None})

    def test_unnumbered_session_falls_back_to_its_date(self):
        ref = session_ref(Path("/rawdata"), None, "20260718")
        self.assertEqual(session_selector(ref), {"session": None, "date": "20260718"})


class EraseSessionOutputsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_the_output_root_is_removed_and_the_session_dir_kept(self):
        session_dir = make_session_tree(
            self.root, 1, "20260717",
            files={
                "eeg/sleep_scoring/predictions.parquet": "x",
                "eeg/quality_control/qc_summary.csv": "x",
                "behaviour/notes.txt": "keep me",
            },
        )

        erased = erase_session_outputs(self.root, subject=SUBJECT, session=1)

        self.assertEqual(erased, [session_dir / "eeg"])
        self.assertFalse((session_dir / "eeg").exists())
        self.assertTrue((session_dir / "behaviour" / "notes.txt").exists())

    def test_without_a_modality_root_only_the_known_groups_are_removed(self):
        session_dir = make_session_tree(
            self.root, 1, "20260717",
            files={
                "sleep_scoring/predictions.parquet": "x",
                "downsample/recording_raw.fif": "x",
                "scratch/mine.txt": "keep me",
            },
        )

        erased = erase_session_outputs(
            self.root, subject=SUBJECT, session=1,
            env={"HYPNOSE_EEG_OUTPUT_ROOT": "."},
        )

        self.assertEqual(
            sorted(erased),
            sorted([session_dir / "sleep_scoring", session_dir / "downsample"]),
        )
        self.assertTrue((session_dir / "scratch" / "mine.txt").exists())

    def test_relocated_output_groups_are_followed(self):
        session_dir = make_session_tree(
            self.root, 1, "20260717",
            files={"analysis/artifacts/epochs.parquet": "x"},
        )

        erased = erase_session_outputs(
            self.root, subject=SUBJECT, session=1,
            env={
                "HYPNOSE_EEG_OUTPUT_ROOT": ".",
                "HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "analysis/artifacts",
            },
        )

        self.assertEqual(erased, [session_dir / "analysis" / "artifacts"])
        self.assertFalse((session_dir / "analysis" / "artifacts").exists())

    def test_a_session_with_nothing_on_disk_erases_nothing(self):
        self.assertEqual(erase_session_outputs(self.root, subject=SUBJECT, session=9), [])

    def test_a_session_selected_by_date_is_found(self):
        session_dir = make_session_tree(
            self.root, 4, "20260720", files={"eeg/artifacts/epochs.parquet": "x"}
        )

        erased = erase_session_outputs(self.root, subject=SUBJECT, date="20260720")

        self.assertEqual(erased, [session_dir / "eeg"])

    def test_dry_run_reports_targets_without_removing_them(self):
        session_dir = make_session_tree(
            self.root, 1, "20260717", files={"eeg/artifacts/epochs.parquet": "x"}
        )

        erased = erase_session_outputs(self.root, subject=SUBJECT, session=1, dry_run=True)

        self.assertEqual(erased, [session_dir / "eeg"])
        self.assertTrue((session_dir / "eeg" / "artifacts" / "epochs.parquet").exists())

    def test_derived_edfs_are_left_in_rawdata_unless_asked_for(self):
        rawdata = self.root / "rawdata"
        derivatives = self.root / "derivatives"
        make_session_tree(derivatives, 1, "20260717", files={"eeg/artifacts/x.parquet": "x"})
        raw_session = make_session_tree(
            rawdata, 1, "20260717",
            files={
                "sub-066_ses-001_recording-1.edf": "raw",
                "sub-066_ses-001_recording-1_trimmed.edf": "derived",
                "sub-066_ses-001_recording-concat.edf": "derived",
            },
        )

        kept = erase_session_outputs(
            derivatives, subject=SUBJECT, session=1, rawdata_root=rawdata
        )
        self.assertTrue((raw_session / "sub-066_ses-001_recording-concat.edf").exists())
        self.assertNotIn(raw_session / "sub-066_ses-001_recording-concat.edf", kept)

        erased = erase_session_outputs(
            derivatives, subject=SUBJECT, session=1,
            rawdata_root=rawdata, erase_derived_edf=True,
        )

        self.assertFalse((raw_session / "sub-066_ses-001_recording-concat.edf").exists())
        self.assertFalse((raw_session / "sub-066_ses-001_recording-1_trimmed.edf").exists())
        self.assertTrue(
            (raw_session / "sub-066_ses-001_recording-1.edf").exists(),
            "a raw recording must never be erased",
        )
        self.assertEqual(len(erased), 2)

    def test_an_ambiguous_derivatives_session_is_refused(self):
        make_session_tree(self.root, 1, "20260717", files={"eeg/a.txt": "x"})
        make_session_tree(self.root, 2, "20260717", files={"eeg/a.txt": "x"})

        with self.assertRaises(ValueError):
            erase_session_outputs(self.root, subject=SUBJECT, date="20260717")

    def test_an_output_folder_symlinked_out_of_the_session_is_refused(self):
        session_dir = make_session_tree(self.root, 1, "20260717", files={})
        outside = self.root / "somewhere-else"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep me")
        (session_dir / "eeg").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(ValueError):
            erase_session_outputs(self.root, subject=SUBJECT, session=1)
        self.assertTrue((outside / "precious.txt").exists())

    def test_an_escaping_output_folder_is_refused(self):
        make_session_tree(self.root, 1, "20260717", files={"eeg/a.txt": "x"})

        with self.assertRaises(ValueError):
            erase_session_outputs(
                self.root, subject=SUBJECT, session=1,
                env={"HYPNOSE_EEG_OUTPUT_ROOT": "../.."},
            )


class ReportTests(unittest.TestCase):
    def test_the_report_lands_beside_the_subject_derivatives(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub-066_id-a1").mkdir()
            path = report_path(root, [SUBJECT], datetime(2026, 9, 22, 14, 15, 30))

        self.assertEqual(path.parent.name, "sub-066_id-a1")
        self.assertEqual(path.name, "batch_report_20260922-141530.csv")

    def test_an_absent_subject_still_gets_a_report_path(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            path = report_path(Path(tmp), [SUBJECT], datetime(2026, 9, 22, 14, 15, 30))

        self.assertEqual(path.parent.name, "sub-066")

    def test_a_bare_subject_value_is_not_read_as_several(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            path = report_path(Path(tmp), SUBJECT, datetime(2026, 9, 22, 14, 15, 30))

        self.assertEqual(path.parent.name, "sub-066")

    def test_several_subjects_share_one_report_at_the_derivatives_root(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub-066").mkdir()
            path = report_path(root, [66, 67], datetime(2026, 9, 22, 14, 15, 30))

        self.assertEqual(path, root / "batch_report_20260922-141530.csv")

    def test_a_failed_outcome_row_carries_the_step_and_what_was_erased(self):
        outcome = SessionOutcome(
            subject="sub-066", session=2, date="20260718",
            session_dir=Path("/rawdata/sub-066/ses-002_date-20260718"),
            status="failed", failed_step="preprocessing:concatenate", returncode=3,
            error="boom", erased=[Path("/derivatives/a"), Path("/derivatives/b")],
        )

        row = outcome.as_row()

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failed_step"], "preprocessing:concatenate")
        self.assertEqual(row["returncode"], 3)
        self.assertEqual(row["erased"], "/derivatives/a;/derivatives/b")


class ResolveSubjectsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_subject_ids_are_normalised_in_the_order_given(self):
        self.assertEqual(
            resolve_subjects(self.root, ["sub-067", "66", "066"]), [67, 66]
        )

    def test_a_comma_separated_list_is_accepted(self):
        self.assertEqual(resolve_subjects(self.root, ["66,67", "68"]), [66, 67, 68])

    def test_all_covers_every_subject_in_the_rawdata_tree(self):
        for name in ("sub-066", "sub-068_id-a1", "sub-067"):
            (self.root / name).mkdir()
        (self.root / "notes.txt").write_text("not a subject")

        self.assertEqual(resolve_subjects(self.root, ["all"]), [66, 67, 68])

    def test_all_cannot_be_combined_with_named_subjects(self):
        with self.assertRaises(ValueError):
            resolve_subjects(self.root, ["all", "66"])

    def test_all_on_an_empty_tree_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            resolve_subjects(self.root, ["all"])

    def test_a_nonsense_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_subjects(self.root, ["sixty-six"])


class SummaryTests(unittest.TestCase):
    def test_one_subject_is_listed_flat(self):
        summary = format_summary(
            [
                SessionOutcome("sub-066", 1, "20260717", Path("/r")),
                SessionOutcome(
                    "sub-066", 2, "20260718", Path("/r"), status="failed",
                    failed_step="qc:summary", error="qc:summary exited with status 2",
                ),
            ]
        )

        self.assertIn("Batch summary: 1/2 sessions completed", summary)
        self.assertNotIn("across", summary)
        self.assertIn("  [    OK] sub-066/ses-001_date-20260717", summary)

    def test_several_subjects_are_grouped_under_a_per_subject_tally(self):
        summary = format_summary(
            [
                SessionOutcome("sub-066", 1, "20260717", Path("/r")),
                SessionOutcome("sub-067", 1, "20260717", Path("/r")),
                SessionOutcome(
                    "sub-067", 2, "20260718", Path("/r"), status="failed",
                    failed_step="qc:summary", error="qc:summary exited with status 2",
                ),
                missing_subject_outcome(68, FileNotFoundError("No sessions found for sub-068")),
            ]
        )

        self.assertIn("Batch summary: 2/3 sessions completed across 3 subjects", summary)
        self.assertIn("  sub-066: 1/1 completed", summary)
        self.assertIn("  sub-067: 1/2 completed", summary)
        self.assertIn("  sub-068: No sessions found for sub-068", summary)

    def test_an_interrupted_session_does_not_count_as_completed(self):
        summary = format_summary(
            [
                SessionOutcome("sub-066", 1, "20260717", Path("/r")),
                SessionOutcome("sub-066", 2, "20260718", Path("/r"), status="interrupted"),
            ]
        )

        self.assertIn("Batch summary: 1/2 sessions completed", summary)


@patch("src.run_pipeline.qc")
@patch("src.run_pipeline.sleep_scoring")
@patch("src.run_pipeline.preprocessing")
class BatchRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.rawdata = self.root / "rawdata"
        self.derivatives = self.root / "derivatives"
        self.report = self.root / "report.csv"
        self.sessions = [
            session_ref(self.rawdata, ses, date) for ses, date in SESSION_DATES.items()
        ]
        finder = patch(
            "src.run_pipeline.find_sessions", return_value=self.sessions
        )
        self.find_sessions = finder.start()
        self.addCleanup(finder.stop)
        eraser = patch("src.run_pipeline.erase_session_outputs", return_value=[])
        self.erase = eraser.start()
        self.addCleanup(eraser.stop)

    def run_batch(self, *extra_argv: str) -> int:
        return run_pipeline.main(
            [
                "--subject", SUBJECT, "--all-sessions",
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
                "--report", str(self.report),
                *extra_argv,
            ]
        )

    def report_rows(self) -> list[dict[str, str]]:
        with self.report.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_every_session_runs_in_order(self, mock_preprocessing, mock_scoring, mock_qc):
        result = self.run_batch()

        self.assertEqual(result, 0)
        selectors = [
            (call.kwargs["session"], call.kwargs["date"])
            for call in mock_qc.run_steps.call_args_list
        ]
        self.assertEqual(selectors, [("1", None), ("2", None), ("3", None)])
        self.assertEqual([row["status"] for row in self.report_rows()], ["ok"] * 3)

    def test_a_failing_session_is_erased_noted_and_the_batch_continues(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_scoring.run_steps.side_effect = [
            None,
            StepFailed("sleep_scoring:score", "scripts.sleep_scoring.score_recordings", 4),
            None,
        ]
        self.erase.return_value = [self.derivatives / "sub-066/ses-002_date-20260718/eeg"]

        result = self.run_batch()

        self.assertEqual(result, 1)
        self.assertEqual(mock_scoring.run_steps.call_count, 3)
        # The third session still ran its remaining stages.
        self.assertEqual(mock_qc.run_steps.call_count, 2)

        self.erase.assert_called_once()
        self.assertEqual(self.erase.call_args.kwargs["session"], 2)
        self.assertEqual(self.erase.call_args.kwargs["date"], "20260718")

        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["ok", "failed", "ok"])
        failed = rows[1]
        self.assertEqual(failed["session"], "2")
        self.assertEqual(failed["failed_step"], "sleep_scoring:score")
        self.assertEqual(failed["returncode"], "4")
        self.assertIn("ses-002", failed["erased"])

    def test_an_unexpected_error_is_handled_like_a_failed_step(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_preprocessing.run_steps.side_effect = [
            FileNotFoundError("no EDF pairs"), None, None, None, None, None
        ]

        result = self.run_batch()

        self.assertEqual(result, 1)
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["failed", "ok", "ok"])
        self.assertEqual(rows[0]["failed_step"], "FileNotFoundError")
        self.assertEqual(rows[0]["returncode"], "")

    def test_keep_failed_records_the_failure_without_erasing(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "scripts.qc.summary_qc", 2), None, None
        ]

        result = self.run_batch("--keep-failed")

        self.assertEqual(result, 1)
        self.erase.assert_not_called()
        self.assertEqual(self.report_rows()[0]["erased"], "")

    def test_erase_derived_edf_reaches_the_eraser(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "scripts.qc.summary_qc", 2), None, None
        ]

        self.run_batch("--erase-derived-edf")

        self.assertTrue(self.erase.call_args.kwargs["erase_derived_edf"])
        self.assertEqual(self.erase.call_args.kwargs["rawdata_root"], self.rawdata)

    def test_output_folder_overrides_reach_the_eraser(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "scripts.qc.summary_qc", 2), None, None
        ]

        self.run_batch("--output-root", "analysis", "--output-dir", "artifacts=arts")

        self.assertEqual(
            self.erase.call_args.kwargs["env"],
            {
                "HYPNOSE_EEG_OUTPUT_ROOT": "analysis",
                "HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS": "arts",
            },
        )

    def test_a_failure_to_erase_does_not_stop_the_batch(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "scripts.qc.summary_qc", 2), None, None
        ]
        self.erase.side_effect = OSError("read-only file system")

        result = self.run_batch()

        self.assertEqual(result, 1)
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["failed", "ok", "ok"])
        self.assertIn("erase failed", rows[0]["error"])

    def test_an_interrupt_stops_the_batch_without_erasing(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_scoring.run_steps.side_effect = [None, KeyboardInterrupt(), None]

        result = self.run_batch()

        self.assertEqual(result, 1)
        self.erase.assert_not_called()
        self.assertEqual([row["status"] for row in self.report_rows()], ["ok", "interrupted"])

    def test_stage_and_overwrite_selection_apply_to_every_session(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_batch("--stage", "preprocessing", "--overwrite")

        self.assertEqual(result, 0)
        self.assertEqual(mock_preprocessing.run_steps.call_count, 3)
        mock_qc.run_steps.assert_not_called()
        for call in mock_preprocessing.run_steps.call_args_list:
            self.assertTrue(call.kwargs["overwrite"])
            self.assertNotIn("steps", call.kwargs)

    def test_an_unknown_subject_fails_before_running_anything(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        self.find_sessions.side_effect = FileNotFoundError("No sessions found for sub-066")

        result = self.run_batch()

        self.assertEqual(result, 1)
        mock_preprocessing.run_steps.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_the_viewer_is_refused_in_batch_mode(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx:
            self.run_batch("--view")

        self.assertEqual(ctx.exception.code, 2)
        mock_preprocessing.run_steps.assert_not_called()

    def test_all_sessions_cannot_be_combined_with_one_session(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx:
            run_pipeline.main(["--subject", SUBJECT, "--all-sessions", "--session", "1"])

        self.assertEqual(ctx.exception.code, 2)


@patch("src.run_pipeline.qc")
@patch("src.run_pipeline.sleep_scoring")
@patch("src.run_pipeline.preprocessing")
class MultiSubjectBatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.rawdata = self.root / "rawdata"
        self.derivatives = self.root / "derivatives"
        self.report = self.root / "report.csv"
        self.by_subject = {
            66: [session_ref(self.rawdata, 1, "20260717", 66)],
            67: [
                session_ref(self.rawdata, 1, "20260718", 67),
                session_ref(self.rawdata, 2, "20260719", 67),
            ],
        }

        def find(rawdata_root, *, subject, **kwargs):
            sessions = self.by_subject.get(int(subject))
            if not sessions:
                raise FileNotFoundError(f"No sessions found for sub-{int(subject):03d}")
            return sessions

        finder = patch("src.run_pipeline.find_sessions", side_effect=find)
        self.find_sessions = finder.start()
        self.addCleanup(finder.stop)
        eraser = patch("src.run_pipeline.erase_session_outputs", return_value=[])
        self.erase = eraser.start()
        self.addCleanup(eraser.stop)

    def run_batch(self, *subjects: str, extra_argv: tuple[str, ...] = ()) -> int:
        return run_pipeline.main(
            [
                "--subject", *subjects, "--all-sessions",
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
                "--report", str(self.report),
                *extra_argv,
            ]
        )

    def report_rows(self) -> list[dict[str, str]]:
        with self.report.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_every_subject_runs_every_session_in_order(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_batch("66", "67")

        self.assertEqual(result, 0)
        selectors = [
            (call.kwargs["subject"], call.kwargs["session"])
            for call in mock_qc.run_steps.call_args_list
        ]
        self.assertEqual(
            selectors, [("sub-066", "1"), ("sub-067", "1"), ("sub-067", "2")]
        )
        rows = self.report_rows()
        self.assertEqual([row["subject"] for row in rows], ["sub-066", "sub-067", "sub-067"])
        self.assertEqual([row["status"] for row in rows], ["ok"] * 3)

    def test_a_failure_erases_that_subjects_session_only(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_scoring.run_steps.side_effect = [
            StepFailed("sleep_scoring:score", "scripts.sleep_scoring.score_recordings", 4),
            None,
            None,
        ]

        result = self.run_batch("66", "67")

        self.assertEqual(result, 1)
        self.erase.assert_called_once()
        self.assertEqual(self.erase.call_args.kwargs["subject"], 66)
        self.assertEqual(self.erase.call_args.kwargs["session"], 1)
        # The other subject still ran both of its sessions.
        self.assertEqual(mock_qc.run_steps.call_count, 2)
        self.assertEqual(
            [row["status"] for row in self.report_rows()], ["failed", "ok", "ok"]
        )

    def test_a_subject_without_sessions_is_noted_and_the_rest_still_run(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_batch("66", "99", "67")

        self.assertEqual(result, 1)
        self.assertEqual(mock_qc.run_steps.call_count, 3)
        rows = self.report_rows()
        self.assertEqual(
            [(row["subject"], row["status"]) for row in rows],
            [("sub-066", "ok"), ("sub-099", "missing"), ("sub-067", "ok"), ("sub-067", "ok")],
        )
        self.assertIn("No sessions found for sub-099", rows[1]["error"])
        self.assertEqual(rows[1]["session"], "")
        self.erase.assert_not_called()

    def test_no_subject_with_sessions_writes_no_report(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_batch("98", "99")

        self.assertEqual(result, 1)
        mock_preprocessing.run_steps.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_all_selects_every_subject_in_the_rawdata_tree(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for subject in self.by_subject:
            (self.rawdata / f"sub-{subject:03d}").mkdir(parents=True)

        result = self.run_batch("all")

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.kwargs["subject"] for call in mock_qc.run_steps.call_args_list],
            ["sub-066", "sub-067", "sub-067"],
        )

    def test_the_combined_report_defaults_to_the_derivatives_root(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        self.derivatives.mkdir(parents=True)

        result = run_pipeline.main(
            [
                "--subject", "66", "67", "--all-sessions",
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
            ]
        )

        self.assertEqual(result, 0)
        reports = list(self.derivatives.glob("batch_report_*.csv"))
        self.assertEqual(len(reports), 1)

    def test_several_subjects_need_all_sessions(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx:
            run_pipeline.main(["--subject", "66", "67", "--session", "1"])

        self.assertEqual(ctx.exception.code, 2)
        mock_preprocessing.run_steps.assert_not_called()

    def test_all_needs_all_sessions_too(self, mock_preprocessing, mock_scoring, mock_qc):
        with self.assertRaises(SystemExit) as ctx:
            run_pipeline.main(["--subject", "all", "--session", "1"])

        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
