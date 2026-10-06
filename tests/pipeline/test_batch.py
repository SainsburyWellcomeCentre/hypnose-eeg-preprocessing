from __future__ import annotations

import csv
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hypnose_helpers.io.layout import SessionRef

from hypnose_eeg.pipeline import run as run_pipeline
from hypnose_eeg.pipeline.batch import (
    SessionOutcome,
    SessionSelection,
    erase_session_outputs,
    format_duration,
    format_summary,
    missing_subject_outcome,
    read_qc_verdict,
    report_path,
    resolve_selections,
    resolve_subjects,
    select_sessions,
    session_selector,
)
from hypnose_eeg.pipeline.steps import StepFailed

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


def qc_summary_csv(*rows: tuple[str, str]) -> str:
    """A `qc_summary.csv` body holding the given (section, status) rows."""
    lines = ["section,status,metric,value,threshold,detail"]
    lines += [f"{section},{status},m,1,1,d" for section, status in rows]
    return "\n".join(lines) + "\n"


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

    def test_a_label_keeps_runs_started_in_the_same_second_apart(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            path = report_path(
                Path(tmp), [SUBJECT], datetime(2026, 9, 22, 14, 15, 30),
                label="ses-002_date-20260718",
            )

        self.assertEqual(path.name, "batch_report_20260922-141530_ses-002_date-20260718.csv")

    def test_a_failed_outcome_row_carries_the_step_and_what_was_erased(self):
        erased = [Path("/derivatives/a"), Path("/derivatives/b")]
        outcome = SessionOutcome(
            subject="sub-066", session=2, date="20260718",
            session_dir=Path("/rawdata/sub-066/ses-002_date-20260718"),
            status="failed", failed_step="preprocessing:concatenate", returncode=3,
            error="boom", erased=erased,
        )

        row = outcome.as_row()

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failed_step"], "preprocessing:concatenate")
        self.assertEqual(row["returncode"], 3)
        # Paths are written with the platform's separator.
        self.assertEqual(row["erased"], f"{erased[0]};{erased[1]}")


class QCVerdictTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_summary(self, name: str, *rows: tuple[str, str]) -> Path:
        session_dir = make_session_tree(
            self.root, 1, "20260717",
            files={f"eeg/quality_control/{name}": qc_summary_csv(*rows)},
        )
        return session_dir / "eeg/quality_control" / name

    def test_the_worst_status_and_the_failed_sections_are_read(self):
        self.write_summary(
            "sub-066_ses-001_recording-concat_qc_summary.csv",
            ("integrity", "pass"), ("artifacts", "fail"),
            ("spectra", "review"), ("scoring", "fail"),
        )

        verdict = read_qc_verdict(self.root, subject=66, session=1)

        self.assertEqual(verdict, ("fail", ["artifacts", "scoring"]))

    def test_the_parquet_copy_is_read_in_preference_to_the_csv(self):
        path = self.write_summary("x_qc_summary.csv", ("integrity", "pass"))
        pd.DataFrame(
            {"section": ["integrity", "artifacts"], "status": ["pass", "fail"]}
        ).to_parquet(path.with_suffix(".parquet"), index=False)

        verdict = read_qc_verdict(self.root, subject=66, session=1)

        self.assertEqual(verdict, ("fail", ["artifacts"]))

    def test_a_passing_summary_has_no_failed_sections(self):
        self.write_summary("x_qc_summary.csv", ("integrity", "pass"), ("spectra", "review"))

        self.assertEqual(read_qc_verdict(self.root, subject=66, session=1), ("review", []))

    def test_a_summary_from_an_earlier_run_is_ignored(self):
        path = self.write_summary("x_qc_summary.csv", ("integrity", "fail"))
        old = time.time() - 3600
        os.utime(path, (old, old))

        self.assertIsNone(
            read_qc_verdict(self.root, subject=66, session=1, since=time.time())
        )

    def test_relocated_quality_control_folder_is_followed(self):
        make_session_tree(
            self.root, 1, "20260717",
            files={"analysis/qc/x_qc_summary.csv": qc_summary_csv(("a", "fail"))},
        )

        verdict = read_qc_verdict(
            self.root, subject=66, session=1,
            env={
                "HYPNOSE_EEG_OUTPUT_ROOT": "analysis",
                "HYPNOSE_EEG_OUTPUT_DIR_QUALITY_CONTROL": "qc",
            },
        )

        self.assertEqual(verdict, ("fail", ["a"]))

    def test_a_session_without_a_summary_has_no_verdict(self):
        make_session_tree(self.root, 1, "20260717", files={})

        self.assertIsNone(read_qc_verdict(self.root, subject=66, session=1))
        self.assertIsNone(read_qc_verdict(self.root, subject=66, session=9))


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


def sessions_filter(*values: str) -> tuple[tuple[str, str], ...]:
    return tuple(("session", value) for value in values)


class ResolveSelectionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_no_sessions_selects_every_session(self):
        self.assertEqual(resolve_selections(self.root, ["66"]), [SessionSelection(66)])

    def test_shared_sessions_apply_to_every_subject(self):
        self.assertEqual(
            resolve_selections(self.root, ["66", "67"], sessions=["1", "3"]),
            [
                SessionSelection(66, sessions_filter("1", "3")),
                SessionSelection(67, sessions_filter("1", "3")),
            ],
        )

    def test_a_subject_can_name_its_own_sessions(self):
        self.assertEqual(
            resolve_selections(self.root, ["66:1,3", "sub-067:2-4", "68"], sessions=["5"]),
            [
                SessionSelection(66, sessions_filter("1", "3")),
                SessionSelection(67, sessions_filter("2-4")),
                SessionSelection(68, sessions_filter("5")),
            ],
        )

    def test_dates_and_date_ranges_are_accepted(self):
        self.assertEqual(
            resolve_selections(self.root, ["66"], dates=["20260717", "20260720-20260730"]),
            [SessionSelection(66, (("date", "20260717"), ("date", "20260720-20260730")))],
        )

    def test_a_repeated_selection_runs_once(self):
        self.assertEqual(
            resolve_selections(self.root, ["66", "sub-066"], sessions=["1", "1"]),
            [SessionSelection(66, sessions_filter("1"))],
        )

    def test_all_with_sessions_covers_every_subject(self):
        for name in ("sub-066", "sub-067"):
            (self.root / name).mkdir()

        self.assertEqual(
            resolve_selections(self.root, ["all:2"]),
            [
                SessionSelection(66, sessions_filter("2")),
                SessionSelection(67, sessions_filter("2")),
            ],
        )

    def test_all_still_cannot_be_combined_with_named_subjects(self):
        with self.assertRaises(ValueError):
            resolve_selections(self.root, ["all:1", "66:2"])

    def test_malformed_selections_are_refused(self):
        for subjects, kwargs in [
            (["66"], {"sessions": ["one"]}),
            (["66"], {"sessions": ["2-x"]}),
            (["66"], {"dates": ["2026-07-17"]}),
            (["66:"], {}),
            (["66:first"], {}),
            (["66"], {"sessions": ["1"], "dates": ["20260717"]}),
        ]:
            with self.subTest(subjects=subjects, **kwargs), self.assertRaises(ValueError):
                resolve_selections(self.root, subjects, **kwargs)


class SelectSessionsTests(unittest.TestCase):
    def setUp(self):
        self.available = [
            session_ref(Path("/raw"), ses, date) for ses, date in SESSION_DATES.items()
        ]

    def select(self, *filters: tuple[str, str]):
        return select_sessions(SessionSelection(66, filters), self.available)

    def test_no_filters_keeps_every_session(self):
        self.assertEqual(
            select_sessions(SessionSelection(66), self.available), [(self.available, None)]
        )

    def test_each_value_is_selected_in_the_order_given(self):
        s1, _, s3 = self.available
        self.assertEqual(
            self.select(("session", "3"), ("session", "ses-001")), [([s3], None), ([s1], None)]
        )

    def test_a_range_selects_every_session_inside_it(self):
        _, s2, s3 = self.available
        self.assertEqual(self.select(("session", "2-5")), [([s2, s3], None)])

    def test_a_date_selects_its_session(self):
        self.assertEqual(self.select(("date", "20260718")), [([self.available[1]], None)])

    def test_a_session_the_subject_lacks_is_an_error_of_its_own(self):
        (found, none), (missing, error) = self.select(("session", "1"), ("session", "9"))

        self.assertEqual(found, [self.available[0]])
        self.assertIsNone(none)
        self.assertEqual(missing, [])
        self.assertIsInstance(error, FileNotFoundError)
        self.assertEqual(str(error), "No session 9 found for sub-066")


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

    def test_qc_failures_are_counted_and_listed(self):
        summary = format_summary(
            [
                SessionOutcome("sub-066", 1, "20260717", Path("/r"), qc_status="pass"),
                SessionOutcome(
                    "sub-066", 2, "20260718", Path("/r"), status="failed",
                    failed_step="qc:summary", returncode=1,
                    error="qc:summary (hypnose_eeg.qc.summary_qc) exited with status 1",
                    qc_status="fail", qc_failed_sections=["artifacts", "spectra"],
                    erased=[Path("/d/eeg")],
                ),
            ]
        )

        self.assertIn("Batch summary: 1/2 sessions completed, 1 failed QC", summary)
        self.assertIn(
            "[FAILED] sub-066/ses-002_date-20260718 -- QC FAIL (artifacts, spectra) "
            "(erased 1 path(s))",
            summary,
        )
        self.assertIn(
            "Failed QC:\n  sub-066/ses-002_date-20260718 (artifacts, spectra)", summary
        )

    def test_the_total_time_is_reported(self):
        summary = format_summary(
            [SessionOutcome("sub-066", 1, "20260717", Path("/r"))], elapsed_seconds=3723.4
        )

        self.assertNotIn("Failed QC", summary)
        self.assertTrue(summary.endswith("Total time: 1h 02m 03s"))

    def test_durations_use_the_shortest_form(self):
        self.assertEqual(format_duration(12.4), "12s")
        self.assertEqual(format_duration(245), "4m 05s")
        self.assertEqual(format_duration(7200), "2h 00m 00s")


@patch("hypnose_eeg.pipeline.run.qc")
@patch("hypnose_eeg.pipeline.run.sleep_scoring")
@patch("hypnose_eeg.pipeline.run.preprocessing")
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
            "hypnose_eeg.pipeline.run.find_sessions", return_value=self.sessions
        )
        self.find_sessions = finder.start()
        self.addCleanup(finder.stop)
        eraser = patch("hypnose_eeg.pipeline.run.erase_session_outputs", return_value=[])
        self.erase = eraser.start()
        self.addCleanup(eraser.stop)

    def run_batch(self, *extra_argv: str) -> int:
        return self.run_selected("--subject", SUBJECT, "--all-sessions", *extra_argv)

    def run_selected(self, *selection: str) -> int:
        return run_pipeline.main(
            [
                *selection,
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
                "--report", str(self.report),
            ]
        )

    def report_rows(self) -> list[dict[str, str]]:
        with self.report.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def qc_selectors(self, mock_qc) -> list[tuple[str | None, str | None]]:
        return [
            (call.kwargs["session"], call.kwargs["date"])
            for call in mock_qc.run_steps.call_args_list
        ]

    def test_selected_sessions_run_as_a_batch(self, mock_preprocessing, mock_scoring, mock_qc):
        result = self.run_selected("--subject", SUBJECT, "--session", "3", "1")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("3", None), ("1", None)])
        self.assertEqual([row["session"] for row in self.report_rows()], ["3", "1"])

    def test_a_session_range_runs_every_session_inside_it(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", SUBJECT, "--session", "2-3")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("2", None), ("3", None)])

    def test_a_session_selected_twice_runs_once(self, mock_preprocessing, mock_scoring, mock_qc):
        result = self.run_selected("--subject", SUBJECT, "--session", "1-2", "2")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("1", None), ("2", None)])

    def test_selected_dates_run_as_a_batch(self, mock_preprocessing, mock_scoring, mock_qc):
        result = self.run_selected("--subject", SUBJECT, "--date", "20260717,20260719")

        self.assertEqual(result, 0)
        # Each session is re-selected by its number, however it was chosen.
        self.assertEqual(self.qc_selectors(mock_qc), [("1", None), ("3", None)])

    def test_a_missing_session_is_reported_and_the_rest_still_run(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", SUBJECT, "--session", "1", "9")

        self.assertEqual(result, 1)
        self.assertEqual(self.qc_selectors(mock_qc), [("1", None)])
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["ok", "missing"])
        self.assertIn("No session 9 found for sub-066", rows[1]["error"])
        self.erase.assert_not_called()

    def test_a_malformed_session_fails_before_running_anything(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", SUBJECT, "--session", "1", "two")

        self.assertEqual(result, 1)
        mock_preprocessing.run_steps.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_one_session_chosen_either_way_is_a_single_run(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for selection in (("--subject", SUBJECT, "--session", "2"), ("--subject", f"{SUBJECT}:2")):
            mock_qc.reset_mock()
            with self.subTest(selection=selection):
                result = self.run_selected(*selection)

                self.assertEqual(result, 0)
                self.assertEqual(mock_qc.run_steps.call_args.kwargs["subject"], SUBJECT)
                self.assertEqual(self.qc_selectors(mock_qc), [("2", None)])
        # Single runs keep the batch machinery out of it: no planning, no report.
        self.find_sessions.assert_not_called()
        self.assertFalse(self.report.exists())

    def test_the_viewer_is_refused_for_several_selected_sessions(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx:
            self.run_selected("--subject", SUBJECT, "--session", "1", "2", "--view")

        self.assertEqual(ctx.exception.code, 2)
        mock_preprocessing.run_steps.assert_not_called()

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
            StepFailed("sleep_scoring:score", "hypnose_eeg.sleep_scoring.score_recordings", 4),
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
            StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 2), None, None
        ]

        result = self.run_batch("--keep-failed")

        self.assertEqual(result, 1)
        self.erase.assert_not_called()
        self.assertEqual(self.report_rows()[0]["erased"], "")

    def test_erase_derived_edf_reaches_the_eraser(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 2), None, None
        ]

        self.run_batch("--erase-derived-edf")

        self.assertTrue(self.erase.call_args.kwargs["erase_derived_edf"])
        self.assertEqual(self.erase.call_args.kwargs["rawdata_root"], self.rawdata)

    def test_output_folder_overrides_reach_the_eraser(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 2), None, None
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
            StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 2), None, None
        ]
        self.erase.side_effect = OSError("read-only file system")

        result = self.run_batch()

        self.assertEqual(result, 1)
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["failed", "ok", "ok"])
        self.assertIn("erase failed", rows[0]["error"])

    def test_a_qc_fail_fails_the_session_and_names_the_sections(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        def qc_summary(**kwargs):
            # summary_qc writes its section table, then exits 1 for a FAIL verdict.
            ses = int(kwargs["session"])
            rows = [("integrity", "pass"), ("artifacts", "fail" if ses == 2 else "pass")]
            make_session_tree(
                self.derivatives, ses, SESSION_DATES[ses],
                files={"eeg/quality_control/x_qc_summary.csv": qc_summary_csv(*rows)},
            )
            if ses == 2:
                raise StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 1)

        mock_qc.run_steps.side_effect = qc_summary

        result = self.run_batch()

        self.assertEqual(result, 1)
        self.erase.assert_called_once()
        self.assertEqual(self.erase.call_args.kwargs["session"], 2)
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["ok", "failed", "ok"])
        self.assertEqual([row["qc_status"] for row in rows], ["pass", "fail", "pass"])
        self.assertEqual(rows[1]["qc_failed_sections"], "artifacts")
        self.assertEqual(rows[1]["failed_step"], "qc:summary")

    def test_a_finished_session_reports_review(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        def qc_summary(**kwargs):
            ses = int(kwargs["session"])
            make_session_tree(
                self.derivatives, ses, SESSION_DATES[ses],
                files={"eeg/quality_control/x_qc_summary.csv": qc_summary_csv(
                    ("integrity", "pass"), ("spectra", "review" if ses == 3 else "pass"),
                )},
            )

        mock_qc.run_steps.side_effect = qc_summary

        result = self.run_batch()

        self.assertEqual(result, 0)
        rows = self.report_rows()
        self.assertEqual([row["qc_status"] for row in rows], ["pass", "pass", "review"])
        self.assertEqual([row["qc_failed_sections"] for row in rows], ["", "", ""])

    def write_old_summaries(self, review_session: int) -> None:
        """A QC summary per session from an earlier run, as a skipped QC stage leaves."""
        for ses, date in SESSION_DATES.items():
            make_session_tree(
                self.derivatives, ses, date,
                files={"eeg/quality_control/x_qc_summary.csv": qc_summary_csv(
                    ("integrity", "pass"),
                    ("spectra", "review" if ses == review_session else "pass"),
                )},
            )
        an_hour_ago = time.time() - 3600
        for path in self.derivatives.rglob("*_qc_summary.csv"):
            os.utime(path, (an_hour_ago, an_hour_ago))

    def test_a_skipped_qc_reports_the_summary_already_on_disk(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        self.write_old_summaries(review_session=3)

        result = self.run_batch()

        self.assertEqual(result, 0)
        rows = self.report_rows()
        self.assertEqual([row["qc_status"] for row in rows], ["pass", "pass", "review"])

    def test_without_the_qc_stage_an_old_summary_is_not_reported(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        self.write_old_summaries(review_session=3)

        result = self.run_batch("--stage", "preprocessing")

        self.assertEqual(result, 0)
        self.assertEqual([row["qc_status"] for row in self.report_rows()], ["", "", ""])

    def test_overwrite_reaches_the_qc_stage(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for argv in ((), ("--overwrite",)):
            mock_qc.reset_mock()
            with self.subTest(argv=argv):
                self.run_batch(*argv)

                self.assertEqual(
                    {call.kwargs["overwrite"] for call in mock_qc.run_steps.call_args_list},
                    {bool(argv)},
                )

    def test_a_qc_crash_without_a_fresh_summary_is_still_a_failure(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_qc.run_steps.side_effect = [
            StepFailed("qc:summary", "hypnose_eeg.qc.summary_qc", 1), None, None
        ]

        result = self.run_batch()

        self.assertEqual(result, 1)
        self.erase.assert_called_once()
        rows = self.report_rows()
        self.assertEqual([row["status"] for row in rows], ["failed", "ok", "ok"])
        self.assertEqual(rows[0]["qc_status"], "")

    def test_a_summary_step_that_raised_is_a_crash_not_a_verdict(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        def qc_summary(**kwargs):
            ses = int(kwargs["session"])
            if ses == 1:
                # A FAIL summary from this run is on disk, but the step then
                # raised: that is a crash, so the summary is not its verdict.
                make_session_tree(
                    self.derivatives, ses, SESSION_DATES[ses],
                    files={"eeg/quality_control/x_qc_summary.csv": qc_summary_csv(
                        ("artifacts", "fail"),
                    )},
                )
                raise StepFailed(
                    "qc:summary", "hypnose_eeg.qc.summary_qc", 1,
                    error=OSError("disk full"),
                )

        mock_qc.run_steps.side_effect = qc_summary

        result = self.run_batch()

        self.assertEqual(result, 1)
        rows = self.report_rows()
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["qc_status"], "")
        self.assertIn("OSError: disk full", rows[0]["error"])

    def test_the_batch_duration_is_on_every_row(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        self.run_batch()

        durations = {row["batch_duration_seconds"] for row in self.report_rows()}
        self.assertEqual(len(durations), 1)
        self.assertGreaterEqual(float(durations.pop()), 0)

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


@patch("hypnose_eeg.pipeline.run.qc")
@patch("hypnose_eeg.pipeline.run.sleep_scoring")
@patch("hypnose_eeg.pipeline.run.preprocessing")
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

        finder = patch("hypnose_eeg.pipeline.run.find_sessions", side_effect=find)
        self.find_sessions = finder.start()
        self.addCleanup(finder.stop)
        eraser = patch("hypnose_eeg.pipeline.run.erase_session_outputs", return_value=[])
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
            StepFailed("sleep_scoring:score", "hypnose_eeg.sleep_scoring.score_recordings", 4),
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

    def run_selected(self, *selection: str) -> int:
        return run_pipeline.main(
            [
                *selection,
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
                "--report", str(self.report),
            ]
        )

    def qc_selectors(self, mock_qc) -> list[tuple[str, str]]:
        return [
            (call.kwargs["subject"], call.kwargs["session"])
            for call in mock_qc.run_steps.call_args_list
        ]

    def test_a_shared_session_runs_for_every_subject(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", "66", "67", "--session", "1")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-066", "1"), ("sub-067", "1")])

    def test_sessions_can_be_chosen_per_subject(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", "66:1", "67:2")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-066", "1"), ("sub-067", "2")])

    def test_per_subject_sessions_take_precedence_over_shared_ones(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", "66", "67:2", "--session", "1")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-066", "1"), ("sub-067", "2")])

    def test_per_subject_sessions_combine_with_all_sessions_for_the_rest(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.run_selected("--subject", "67:2", "66", "--all-sessions")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-067", "2"), ("sub-066", "1")])

    def test_all_with_a_session_notes_the_subjects_without_it(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for subject in self.by_subject:
            (self.rawdata / f"sub-{subject:03d}").mkdir(parents=True)

        result = self.run_selected("--subject", "all", "--session", "2")

        self.assertEqual(result, 1)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-067", "2")])
        self.assertEqual(
            [(row["subject"], row["status"]) for row in self.report_rows()],
            [("sub-066", "missing"), ("sub-067", "ok")],
        )

    def test_subjects_without_chosen_sessions_are_refused(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for subjects in (["66", "67"], ["66:1", "67"], ["all"]):
            with self.subTest(subjects=subjects):
                with self.assertRaises(SystemExit) as ctx:
                    run_pipeline.main(["--subject", *subjects])

                self.assertEqual(ctx.exception.code, 2)
        mock_preprocessing.run_steps.assert_not_called()


@patch("hypnose_eeg.pipeline.run.qc")
@patch("hypnose_eeg.pipeline.run.sleep_scoring")
@patch("hypnose_eeg.pipeline.run.preprocessing")
class JobArrayTaskTests(unittest.TestCase):
    """`--list-tasks` / `--task-index`, which split a selection for a job array."""

    # The same two-subject rawdata as the multi-subject batch tests.
    setUp = MultiSubjectBatchTests.setUp
    report_rows = MultiSubjectBatchTests.report_rows
    qc_selectors = MultiSubjectBatchTests.qc_selectors

    def pipeline(self, *argv: str, report: bool = True) -> int:
        return run_pipeline.main(
            [
                *argv,
                "--rawdata-root", str(self.rawdata),
                "--derivatives-root", str(self.derivatives),
                *(("--report", str(self.report)) if report else ()),
            ]
        )

    def listed(self, *argv: str) -> tuple[int, list[str]]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.pipeline(*argv, "--list-tasks")
        return code, out.getvalue().splitlines()

    def test_each_subject_is_one_task_by_default(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        code, lines = self.listed("--subject", "66", "67", "--all-sessions")

        self.assertEqual(code, 0)
        self.assertEqual(
            lines,
            [
                "sub-066 ses-001_date-20260717",
                "sub-067 ses-001_date-20260718 ses-002_date-20260719",
            ],
        )
        mock_preprocessing.run_steps.assert_not_called()

    def test_each_session_is_one_task_by_session(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        code, lines = self.listed(
            "--subject", "66", "67", "--all-sessions", "--task-unit", "session"
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            lines,
            [
                "sub-066 ses-001_date-20260717",
                "sub-067 ses-001_date-20260718",
                "sub-067 ses-002_date-20260719",
            ],
        )

    def test_a_subject_selected_twice_is_still_one_task(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        code, lines = self.listed("--subject", "67:2", "67:1")

        self.assertEqual(code, 0)
        self.assertEqual(lines, ["sub-067 ses-002_date-20260719 ses-001_date-20260718"])

    def test_missing_selections_are_left_out_of_the_tasks(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        code, lines = self.listed("--subject", "66", "99", "--all-sessions")

        self.assertEqual(code, 0)
        self.assertEqual(lines, ["sub-066 ses-001_date-20260717"])

    def test_a_selection_without_sessions_lists_nothing_and_fails(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        code, lines = self.listed("--subject", "98", "99", "--all-sessions")

        self.assertEqual(code, 1)
        self.assertEqual(lines, [])

    def test_a_task_runs_only_its_subjects_sessions(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.pipeline("--subject", "66", "67", "--all-sessions", "--task-index", "1")

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-067", "1"), ("sub-067", "2")])

    def test_a_session_task_runs_as_a_batch_and_erases_its_failure(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        mock_scoring.run_steps.side_effect = StepFailed(
            "sleep_scoring:score", "hypnose_eeg.sleep_scoring.score_recordings", 4
        )

        result = self.pipeline(
            "--subject", "66", "67", "--all-sessions",
            "--task-unit", "session", "--task-index", "2",
        )

        self.assertEqual(result, 1)
        self.erase.assert_called_once()
        self.assertEqual(self.erase.call_args.kwargs["subject"], 67)
        self.assertEqual(self.erase.call_args.kwargs["session"], 2)
        self.assertEqual(
            [(row["subject"], row["session"], row["status"]) for row in self.report_rows()],
            [("sub-067", "2", "failed")],
        )

    def test_a_task_of_one_session_is_still_a_batch(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        result = self.pipeline("--subject", "66:1", "--task-index", "0")

        self.assertEqual(result, 0)
        self.assertEqual(self.report_rows()[0]["status"], "ok")

    def test_session_task_reports_are_named_after_their_session(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        for index in ("0", "1"):
            result = self.pipeline(
                "--subject", "67", "--all-sessions",
                "--task-unit", "session", "--task-index", index, report=False,
            )
            self.assertEqual(result, 0)

        reports = sorted(path.name for path in self.derivatives.glob("sub-067/*.csv"))
        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0].endswith("_ses-001_date-20260718.csv"))
        self.assertTrue(reports[1].endswith("_ses-002_date-20260719.csv"))

    def test_an_out_of_range_task_fails_without_running(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with redirect_stderr(io.StringIO()) as err:
            result = self.pipeline("--subject", "66", "67", "--all-sessions", "--task-index", "2")

        self.assertEqual(result, 1)
        self.assertIn("splits into 2 task(s) by subject", err.getvalue())
        mock_preprocessing.run_steps.assert_not_called()

    def test_the_viewer_is_refused_for_a_task(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(io.StringIO()):
            self.pipeline("--subject", "66:1", "--task-index", "0", "--view")

        self.assertEqual(ctx.exception.code, 2)

    def freeze_tasks(self, *argv: str) -> Path:
        """The `--list-tasks` output for `argv`, saved as submit.sh saves it."""
        code, lines = self.listed(*argv)
        self.assertEqual(code, 0)
        task_file = self.root / "tasks.txt"
        task_file.write_text("".join(f"{line}\n" for line in lines))
        return task_file

    def test_a_task_file_runs_the_session_listed_at_submission(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        selection = ("--subject", "67", "--all-sessions", "--task-unit", "session")
        task_file = self.freeze_tasks(*selection)
        # A session that sorts first arrives after submission, shifting the
        # selection's own task indices.
        self.by_subject[67].insert(0, session_ref(self.rawdata, 3, "20260716", 67))

        result = self.pipeline(
            *selection, "--task-index", "1", "--task-file", str(task_file)
        )

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-067", "2")])

    def test_a_subject_task_file_leaves_new_sessions_alone(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        selection = ("--subject", "66", "67", "--all-sessions")
        task_file = self.freeze_tasks(*selection)
        self.by_subject[67].append(session_ref(self.rawdata, 3, "20260720", 67))

        result = self.pipeline(
            *selection, "--task-index", "1", "--task-file", str(task_file)
        )

        self.assertEqual(result, 0)
        self.assertEqual(self.qc_selectors(mock_qc), [("sub-067", "1"), ("sub-067", "2")])

    def test_a_listed_session_that_is_gone_fails_without_running(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        selection = ("--subject", "67", "--all-sessions")
        task_file = self.freeze_tasks(*selection)
        del self.by_subject[67][1]

        with redirect_stderr(io.StringIO()) as err:
            result = self.pipeline(
                *selection, "--task-index", "0", "--task-file", str(task_file)
            )

        self.assertEqual(result, 1)
        self.assertIn("sub-067 no longer has ses-002_date-20260719", err.getvalue())
        mock_preprocessing.run_steps.assert_not_called()

    def test_a_task_file_index_out_of_range_fails_without_running(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        selection = ("--subject", "66", "67", "--all-sessions")
        task_file = self.freeze_tasks(*selection)

        with redirect_stderr(io.StringIO()) as err:
            result = self.pipeline(
                *selection, "--task-index", "2", "--task-file", str(task_file)
            )

        self.assertEqual(result, 1)
        self.assertIn("lists 2 task(s)", err.getvalue())
        mock_preprocessing.run_steps.assert_not_called()

    def test_a_task_file_needs_a_task_index(
        self, mock_preprocessing, mock_scoring, mock_qc
    ):
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(io.StringIO()):
            self.pipeline(
                "--subject", "66", "--all-sessions", "--task-file", str(self.root / "t.txt")
            )

        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
