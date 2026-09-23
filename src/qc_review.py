"""List every session whose QC summary asks for review, and which sections did.

Reads the `qc_summary.csv` that `scripts/qc/summary_qc.py` writes into each
session's quality-control directory, for every session the given subjects have
under the derivatives root (`--subject all` covers every subject there). A
session is listed when any section of its summary is not `pass`; each listed
section is shown with its metric, value, threshold, and detail, plus how many
entries it contributed to the session's `qc_review_epochs.csv`. Sessions with a
FAIL section are listed too, marked as such -- a batch run erases those, but a
single run leaves them in place.

Nothing is recomputed: only summaries already on disk are read, so run the QC
summary first (`python -m src.qc --subject 66 --session 1`, or a batch run).
Sessions without a summary are counted and named per subject, since a session
that was never checked is not one that passed.

    python -m src.qc_review --subject 65 66
    python -m src.qc_review --subject all --report qc_review.csv

`--report FILE` also writes one CSV row per flagged section. Output-folder
overrides (`--output-root`, `--output-dir quality_control=...`,
`--output-layout`) must match the ones the QC summary ran with, or its
summaries are not found.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypnose_helpers.io.layout import SessionLayout, SessionRef, normalize_subjid
from hypnose_helpers.io.selectors import parse_subjects

from src._batch import (
    ALL_SUBJECTS,
    QC_FAIL,
    QC_SUMMARY_SUFFIX,
    _applied_env,
    format_duration,
)
from src._pipeline import (
    add_output_layout_arguments,
    output_dir_overrides,
    output_layout_env,
)
from scripts.io.output_layout import output_dir_name
from scripts.io.output_paths import save_csv_rows
from scripts.io.repository_paths import get_derivatives_root
from scripts.qc.thresholds import load_performance_check

QC_PASS = "pass"
# The review ranges summary_qc writes beside each summary, under the same stem.
QC_REVIEW_SUFFIX = "qc_review_epochs.csv"

REPORT_FIELDS = [
    "subject",
    "session",
    "date",
    "session_dir",
    "recording",
    "session_status",
    "section",
    "status",
    "metric",
    "value",
    "threshold",
    "detail",
    "review_entries",
    "review_seconds",
    "summary_path",
]


@dataclass
class FlaggedSection:
    """One non-passing section of one recording's QC summary."""

    recording: str
    section: str
    status: str
    metric: str
    value: str
    threshold: str
    detail: str
    review_entries: int = 0
    review_seconds: float = 0.0
    summary_path: Path = Path()


@dataclass
class SessionReview:
    """A session's QC state: its worst status and the sections behind it."""

    subject: str
    session: int | None
    date: str
    session_dir: Path
    status: str = ""  # blank when the session has no QC summary
    sections: list[FlaggedSection] = field(default_factory=list)

    @property
    def label(self) -> str:
        session = f"ses-{self.session:03d}" if self.session is not None else "ses-?"
        return f"{self.subject}/{session}_date-{self.date}"

    @property
    def has_summary(self) -> bool:
        return bool(self.status)

    @property
    def flagged(self) -> bool:
        return self.has_summary and self.status != QC_PASS


@dataclass
class SubjectReview:
    """Every session one subject has under the derivatives root."""

    subject: str
    sessions: list[SessionReview] = field(default_factory=list)
    error: str = ""


def resolve_subjects(derivatives_root: Path, values: Sequence[str]) -> list[int]:
    """The subjects to check, from `--subject` values; `all` is every derivatives subject."""
    tokens = [str(value).strip() for value in values]
    if not any(token.lower() == ALL_SUBJECTS for token in tokens):
        return parse_subjects(tokens)
    if len(tokens) > 1:
        raise ValueError(
            f"--subject {ALL_SUBJECTS} already covers every subject; "
            f"do not list others alongside it"
        )
    subjects = [
        subjid
        for subjid, _ in SessionLayout(derivatives_root, name="derivatives").iter_subjects()
    ]
    if not subjects:
        raise FileNotFoundError(
            f"No subject directories under the derivatives root {derivatives_root}"
        )
    return subjects


def review_subject(
    derivatives_root: Path, subject: str | int, *, qc_folder: str
) -> SubjectReview:
    """Read the QC summaries of every session one subject has."""
    label = normalize_subjid(subject)
    layout = SessionLayout(derivatives_root, name="derivatives")
    try:
        refs = layout.find_sessions(subject, missing_ok=True)
    except (ValueError, OSError) as exc:
        return SubjectReview(label, error=str(exc))
    if not refs:
        return SubjectReview(label, error="no sessions under the derivatives root")
    return SubjectReview(
        label, sessions=[review_session(ref, qc_folder=qc_folder) for ref in refs]
    )


def review_session(session_ref: SessionRef, *, qc_folder: str) -> SessionReview:
    """A session's worst QC status and non-passing sections, across its recordings.

    A session holding several recordings (or a stale summary beside a newer one)
    has one summary each; all of them are read and the worst status wins, so
    nothing on disk that asks for review is hidden.
    """
    review = SessionReview(
        subject=normalize_subjid(session_ref.subjid),
        session=session_ref.ses,
        date=session_ref.date,
        session_dir=session_ref.path,
    )
    # Hidden files are skipped: macOS leaves `._<name>` AppleDouble sidecars on
    # network shares that match the pattern but are not CSV.
    summaries = [
        path
        for path in sorted((session_ref.path / qc_folder).glob(f"*{QC_SUMMARY_SUFFIX}"))
        if not path.name.startswith(".")
    ]
    if not summaries:
        return review

    severity = load_performance_check()
    statuses: list[str] = []
    for path in summaries:
        recording = path.name[: -len(QC_SUMMARY_SUFFIX)].rstrip("_")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            # summary_qc treats a summary with no sections as a failure too.
            statuses.append(QC_FAIL)
            continue
        flagged: list[FlaggedSection] = []
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            statuses.append(status)
            if severity.get(status, severity[QC_FAIL]) > severity[QC_PASS]:
                flagged.append(
                    FlaggedSection(
                        recording=recording,
                        section=str(row.get("section", "")).strip(),
                        status=status,
                        metric=str(row.get("metric", "")),
                        value=str(row.get("value", "")),
                        threshold=str(row.get("threshold", "")),
                        detail=str(row.get("detail", "")),
                        summary_path=path,
                    )
                )
        if flagged:
            _count_review_entries(path, flagged)
            review.sections.extend(flagged)

    worst = max(statuses, key=lambda status: severity.get(status, severity[QC_FAIL]))
    review.status = worst if worst in severity else QC_FAIL
    return review


def _count_review_entries(summary_path: Path, sections: list[FlaggedSection]) -> None:
    """Fill in how many review ranges (and seconds) each flagged section left."""
    stem = summary_path.name[: -len(QC_SUMMARY_SUFFIX)]
    review_path = summary_path.with_name(stem + QC_REVIEW_SUFFIX)
    if not review_path.is_file():
        return
    by_section = {section.section: section for section in sections}
    with review_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            section = by_section.get(str(row.get("section", "")).strip())
            if section is None:
                continue
            section.review_entries += 1
            try:
                section.review_seconds += float(row.get("duration_s") or 0)
            except ValueError:
                pass


def format_review(subjects: Sequence[SubjectReview]) -> str:
    """A per-subject listing of the sessions to review and the sections behind each."""
    sessions = [session for subject in subjects for session in subject.sessions]
    flagged = [session for session in sessions if session.flagged]
    checked = [session for session in sessions if session.has_summary]
    header = f"QC review: {len(flagged)}/{len(checked)} checked sessions need review"
    if len(subjects) > 1:
        header += f" across {len(subjects)} subjects"
    failed = sum(1 for session in flagged if session.status == QC_FAIL)
    if failed:
        header += f" ({failed} failed)"

    lines = [header]
    for subject in subjects:
        if subject.error:
            # Layout errors can span lines (duplicate sessions list each directory).
            error = subject.error.replace("\n", "\n      ")
            lines.append(f"  {subject.subject}: {error}")
            continue
        subject_checked = [session for session in subject.sessions if session.has_summary]
        subject_flagged = [session for session in subject_checked if session.flagged]
        lines.append(
            f"  {subject.subject}: {len(subject_flagged)}/{len(subject_checked)} "
            f"checked sessions need review"
        )
        for session in subject_flagged:
            lines.append(f"    [{session.status.upper():>6}] {session.label}")
            recordings = {section.recording for section in session.sections}
            for section in session.sections:
                lines.append(f"        {_section_line(section, show_recording=len(recordings) > 1)}")
        unchecked = [session for session in subject.sessions if not session.has_summary]
        if unchecked:
            names = ", ".join(_session_token(session) for session in unchecked)
            lines.append(f"    no QC summary ({len(unchecked)}): {names}")
    return "\n".join(lines)


def _section_line(section: FlaggedSection, *, show_recording: bool) -> str:
    """`artifacts: metric=6.48 (threshold 5.0) -- detail [N review entries, 24m 28s]`."""
    name = f"{section.recording} {section.section}" if show_recording else section.section
    status = "" if section.status == "review" else f" [{section.status.upper()}]"
    line = (
        f"{name}{status}: {section.metric}={_number(section.value)} "
        f"(threshold {_number(section.threshold)})"
    )
    if section.detail:
        line += f" -- {section.detail}"
    if section.review_entries:
        line += (
            f" [{section.review_entries} review entries, "
            f"{format_duration(section.review_seconds)}]"
        )
    return line


def _number(value: str) -> str:
    """A float trimmed to two decimals; anything else as written."""
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:g}" if number == round(number, 2) else f"{number:.2f}"


def _session_token(session: SessionReview) -> str:
    return session.label.split("/", 1)[1]


def write_review_report(subjects: Sequence[SubjectReview], path: str | Path) -> Path:
    """One CSV row per flagged section of every session that needs review."""
    rows = []
    for subject in subjects:
        for session in subject.sessions:
            for section in session.sections:
                rows.append(
                    {
                        "subject": session.subject,
                        "session": "" if session.session is None else session.session,
                        "date": session.date,
                        "session_dir": str(session.session_dir),
                        "recording": section.recording,
                        "session_status": session.status,
                        "section": section.section,
                        "status": section.status,
                        "metric": section.metric,
                        "value": section.value,
                        "threshold": section.threshold,
                        "detail": section.detail,
                        "review_entries": section.review_entries,
                        "review_seconds": round(section.review_seconds, 1),
                        "summary_path": str(section.summary_path),
                    }
                )
    return save_csv_rows(rows, REPORT_FIELDS, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List the sessions whose QC summary needs review, and which sections.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", "--subjid", dest="subject", nargs="+", required=True,
        metavar="SUBJECT",
        help="Subject IDs, for example 66 or sub-066, or "
        f"{ALL_SUBJECTS!r} for every subject in the derivatives tree.",
    )
    parser.add_argument(
        "--derivatives-root", default=None,
        help="Override the active profile's derivatives root.",
    )
    add_output_layout_arguments(parser)
    parser.add_argument(
        "--report", default=None, metavar="FILE",
        help="Also write one CSV row per flagged section to FILE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    derivatives_root = (
        Path(args.derivatives_root) if args.derivatives_root else get_derivatives_root()
    )
    try:
        env = output_layout_env(
            args.output_layout, output_dir_overrides(parser, args), args.output_root
        )
        subjects = resolve_subjects(derivatives_root, args.subject)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    with _applied_env(env):
        qc_folder = output_dir_name("quality_control")

    reviews = [
        review_subject(derivatives_root, subject, qc_folder=qc_folder)
        for subject in subjects
    ]
    print(format_review(reviews))
    if args.report:
        write_review_report(reviews, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
