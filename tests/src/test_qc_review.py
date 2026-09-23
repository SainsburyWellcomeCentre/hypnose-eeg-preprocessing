from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src import qc_review
from src.qc_review import format_review, resolve_subjects, review_subject

QC_FOLDER = "eeg/quality_control"


def qc_summary_csv(*rows: tuple[str, str]) -> str:
    """A `qc_summary.csv` body holding the given (section, status) rows."""
    lines = ["section,status,metric,value,threshold,detail"]
    lines += [f"{section},{status},m,1.2345,1,d" for section, status in rows]
    return "\n".join(lines) + "\n"


def review_epochs_csv(*rows: tuple[str, float]) -> str:
    """A `qc_review_epochs.csv` body holding the given (section, duration_s) rows."""
    lines = ["section,start_s,end_s,duration_s,sleep_state,channels,metric,value,threshold,reason"]
    lines += [f"{section},0,{duration},{duration},0,EEG1,m,1,1,r" for section, duration in rows]
    return "\n".join(lines) + "\n"


class QcReviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def session(self, ses: int, date: str, subject: str = "sub-066", **files: str) -> Path:
        session_dir = self.root / subject / f"ses-{ses:03d}_date-{date}"
        (session_dir / QC_FOLDER).mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (session_dir / QC_FOLDER / name).write_text(content)
        return session_dir

    def test_only_non_passing_sections_are_listed(self):
        self.session(
            1, "20260717",
            **{
                "rec_qc_summary.csv": qc_summary_csv(
                    ("integrity", "pass"), ("artifacts", "review"), ("emg_rms", "review")
                ),
                "rec_qc_review_epochs.csv": review_epochs_csv(
                    ("artifacts", 4), ("artifacts", 4), ("channel_correlation", 4)
                ),
            },
        )
        self.session(2, "20260718", **{"rec_qc_summary.csv": qc_summary_csv(("integrity", "pass"))})

        subject = review_subject(self.root, 66, qc_folder=QC_FOLDER)

        flagged = [session for session in subject.sessions if session.flagged]
        self.assertEqual([session.session for session in flagged], [1])
        self.assertEqual(flagged[0].status, "review")
        sections = {section.section: section for section in flagged[0].sections}
        self.assertEqual(sorted(sections), ["artifacts", "emg_rms"])
        self.assertEqual(sections["artifacts"].review_entries, 2)
        self.assertEqual(sections["artifacts"].review_seconds, 8.0)
        self.assertEqual(sections["emg_rms"].review_entries, 0)

    def test_a_fail_section_makes_the_session_fail(self):
        self.session(
            1, "20260717",
            **{"rec_qc_summary.csv": qc_summary_csv(("spectra", "review"), ("integrity", "fail"))},
        )

        session = review_subject(self.root, 66, qc_folder=QC_FOLDER).sessions[0]

        self.assertEqual(session.status, "fail")
        self.assertEqual(
            [(section.section, section.status) for section in session.sections],
            [("spectra", "review"), ("integrity", "fail")],
        )

    def test_every_recording_summary_in_a_session_is_read(self):
        self.session(
            1, "20260717",
            **{
                "a_qc_summary.csv": qc_summary_csv(("artifacts", "review")),
                "b_qc_summary.csv": qc_summary_csv(("spectra", "review")),
            },
        )

        session = review_subject(self.root, 66, qc_folder=QC_FOLDER).sessions[0]

        self.assertEqual(
            [(section.recording, section.section) for section in session.sections],
            [("a", "artifacts"), ("b", "spectra")],
        )

    def test_hidden_sidecar_files_are_ignored(self):
        self.session(
            1, "20260717",
            **{
                "rec_qc_summary.csv": qc_summary_csv(("integrity", "pass")),
                "._rec_qc_summary.csv": "\x00\x05\x16\x07\xb0",
            },
        )

        session = review_subject(self.root, 66, qc_folder=QC_FOLDER).sessions[0]

        self.assertEqual(session.status, "pass")

    def test_sessions_without_a_summary_are_named_not_flagged(self):
        self.session(1, "20260717", **{"rec_qc_summary.csv": qc_summary_csv(("a", "review"))})
        self.session(2, "20260718")

        subject = review_subject(self.root, 66, qc_folder=QC_FOLDER)
        text = format_review([subject])

        self.assertIn("1/1 checked sessions need review", text)
        self.assertIn("[REVIEW] sub-066/ses-001_date-20260717", text)
        self.assertIn("a: m=1.23 (threshold 1) -- d", text)
        self.assertIn("no QC summary (1): ses-002_date-20260718", text)

    def test_a_subject_without_derivatives_is_reported(self):
        subject = review_subject(self.root, 99, qc_folder=QC_FOLDER)

        self.assertIn("sub-099: no sessions", format_review([subject]))

    def test_all_covers_every_derivatives_subject(self):
        self.session(1, "20260717", subject="sub-066")
        self.session(1, "20260717", subject="sub-067_id-400")

        self.assertEqual(resolve_subjects(self.root, ["all"]), [66, 67])
        with self.assertRaises(ValueError):
            resolve_subjects(self.root, ["all", "66"])

    def test_main_writes_one_report_row_per_flagged_section(self):
        self.session(
            1, "20260717",
            **{"rec_qc_summary.csv": qc_summary_csv(("a", "review"), ("b", "pass"), ("c", "fail"))},
        )
        report = self.root / "out" / "review.csv"

        with redirect_stdout(io.StringIO()):
            code = qc_review.main(
                ["--subject", "66", "--derivatives-root", str(self.root), "--report", str(report)]
            )

        self.assertEqual(code, 0)
        with report.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["section"], row["status"]) for row in rows], [("a", "review"), ("c", "fail")])
        self.assertEqual({row["session_status"] for row in rows}, {"fail"})

    def test_main_follows_a_relocated_quality_control_folder(self):
        session_dir = self.root / "sub-066" / "ses-001_date-20260717" / "analysis" / "qc"
        session_dir.mkdir(parents=True)
        (session_dir / "rec_qc_summary.csv").write_text(qc_summary_csv(("a", "review")))

        out = io.StringIO()
        with redirect_stdout(out):
            qc_review.main(
                [
                    "--subject", "66", "--derivatives-root", str(self.root),
                    "--output-root", "analysis", "--output-dir", "quality_control=qc",
                ]
            )

        self.assertIn("[REVIEW] sub-066/ses-001_date-20260717", out.getvalue())


if __name__ == "__main__":
    unittest.main()
