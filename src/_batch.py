"""Session enumeration, failed-session cleanup, and reporting for batch runs.

`run_pipeline.py --all-sessions` processes every session one or more subjects
have, one after the other (`--subject all` covers every subject in the rawdata
tree). A session that fails must not stop the ones after it, and must
not leave half-written outputs behind that a later rerun would mistake for
completed work -- every stage skips a step whose output already exists, so a
truncated parquet or a downsampled FIF written before the crash would be
silently reused. So a failed session is erased (its derivative outputs removed)
and recorded in a CSV report, and the run carries on with the next session.

What "erased" covers is deliberately narrow: only the output folders this
pipeline itself writes below the failed session's *derivatives* directory,
resolved through the same `output_layout.py` precedence (including any
`--output-root`/`--output-dir` overrides in force) that put them there. Nothing
in rawdata is touched unless `erase_derived_edf=True` is asked for, and then
only files this pipeline produced -- `<stem>_trimmed.edf` and
`<stem>_recording-concat.edf` -- never a raw recording.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from hypnose_helpers.io.layout import SessionLayout, SessionRef, normalize_subjid
from hypnose_helpers.io.selectors import parse_subjects

from scripts.io.output_layout import output_dir_names, output_root_dir
from scripts.io.output_paths import save_csv_rows
from scripts.utils.recording_selection import (
    is_concatenated_recording,
    is_trimmed_recording,
)

# `--subject all`: every subject the rawdata tree holds, rather than a list
# typed out by hand. Spelled as a subject value so one flag covers both.
ALL_SUBJECTS = "all"

REPORT_FIELDS = [
    "started_at",
    "subject",
    "session",
    "date",
    "session_dir",
    "status",
    "duration_seconds",
    "failed_step",
    "returncode",
    "error",
    "erased",
]


@dataclass
class SessionOutcome:
    """One session's result in a batch run, and what was erased if it failed."""

    subject: str
    session: int | None
    date: str
    session_dir: Path
    status: str = "ok"
    started_at: str = ""
    duration_seconds: float = 0.0
    failed_step: str = ""
    returncode: int | None = None
    error: str = ""
    erased: list[Path] = field(default_factory=list)

    @property
    def label(self) -> str:
        if not self.date:
            return self.subject
        session = f"ses-{self.session:03d}" if self.session is not None else "ses-?"
        return f"{self.subject}/{session}_date-{self.date}"

    def as_row(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "subject": self.subject,
            "session": "" if self.session is None else self.session,
            "date": self.date,
            "session_dir": str(self.session_dir),
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 1),
            "failed_step": self.failed_step,
            "returncode": "" if self.returncode is None else self.returncode,
            "error": self.error,
            "erased": ";".join(str(path) for path in self.erased),
        }


def resolve_subjects(rawdata_root: str | Path, values: Sequence[str]) -> list[int]:
    """The subjects a batch run covers, from `--subject` values.

    Accepts the same forms as everywhere else (`66`, `066`, `sub-066`, and
    comma- or space-separated combinations), plus `all` for every subject
    directory under the rawdata root -- which is a whole-dataset run, so it is
    deliberately the only value allowed when it is given.
    """
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
        for subjid, _ in SessionLayout(Path(rawdata_root), name="rawdata").iter_subjects()
    ]
    if not subjects:
        raise FileNotFoundError(f"No subject directories under the rawdata root {rawdata_root}")
    return subjects


def missing_subject_outcome(subject: str | int, error: Exception) -> SessionOutcome:
    """A report row for a subject whose sessions could not be resolved at all.

    A batch spanning several subjects should no more stop at one unusable
    subject than at one unusable session, so this is recorded and reported
    rather than raised.
    """
    return SessionOutcome(
        subject=normalize_subjid(subject),
        session=None,
        date="",
        session_dir=Path(),
        status="missing",
        started_at=datetime.now().isoformat(timespec="seconds"),
        error=str(error),
    )


def session_selector(session_ref: SessionRef) -> dict[str, str | None]:
    """The `--session`/`--date` selector that re-selects exactly this session.

    Prefers the session number, which is what the stage CLIs and the lab book
    both quote; a directory whose session token is not numeric has no number to
    quote, so it is selected by date instead.
    """
    if session_ref.ses is not None:
        return {"session": str(session_ref.ses), "date": None}
    return {"session": None, "date": session_ref.date}


def erase_session_outputs(
    derivatives_root: str | Path,
    *,
    subject: str | int,
    session: int | None = None,
    date: str | None = None,
    env: Mapping[str, str] | None = None,
    rawdata_root: str | Path | None = None,
    erase_derived_edf: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Remove a session's pipeline outputs; return what was (or would be) removed.

    `env` carries the `HYPNOSE_EEG_OUTPUT_*` overrides the run used (see
    `src._pipeline.output_layout_env`), so the folders erased are the ones the
    run actually wrote rather than the repository defaults. A session with
    nothing on disk yet erases nothing rather than failing -- a session can fail
    on its very first step.
    """
    removed: list[Path] = []
    session_dir = _derivatives_session_dir(
        derivatives_root, subject=subject, session=session, date=date
    )
    if session_dir is not None:
        for target in _output_targets(session_dir, env or {}):
            if not target.is_dir():
                continue
            _check_inside(target, session_dir)
            if not dry_run:
                shutil.rmtree(target)
            removed.append(target)

    if erase_derived_edf and rawdata_root is not None:
        removed.extend(
            _erase_derived_edf(
                rawdata_root, subject=subject, session=session, date=date, dry_run=dry_run
            )
        )
    return removed


def report_path(
    derivatives_root: str | Path, subjects: Sequence[str | int], started_at: datetime
) -> Path:
    """Where a batch run's report lands: one timestamped CSV per run.

    A single-subject run keeps its report in that subject's derivatives
    directory, where anyone looking at the subject will find it; a run spanning
    several subjects writes one combined report at the derivatives root
    instead, since no one subject owns it.

    Timestamped rather than a fixed name so a rerun -- which is the normal
    response to a failed session -- cannot overwrite the report that recorded
    why the first run failed.
    """
    root = Path(derivatives_root)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    filename = f"batch_report_{stamp}.csv"
    # A lone subject is a sequence of characters to `list()`, which would read
    # as a multi-subject run and quietly move the report.
    subjects = [subjects] if isinstance(subjects, (str, int)) else list(subjects)
    if len(subjects) != 1:
        return root / filename

    layout = SessionLayout(root, name="derivatives")
    try:
        subject_dir = layout.subject_dir(subjects[0], missing_ok=True)
    except (ValueError, OSError):
        # A duplicate or unreadable subject directory must not cost the run its
        # report: fall back to the canonical `sub-NNN` name below.
        subject_dir = None
    return (subject_dir or root / normalize_subjid(subjects[0])) / filename


def write_report(outcomes: Sequence[SessionOutcome], path: str | Path) -> Path:
    """Write one row per session to CSV, through the shared CSV writer."""
    return save_csv_rows([outcome.as_row() for outcome in outcomes], REPORT_FIELDS, path)


def format_summary(outcomes: Sequence[SessionOutcome]) -> str:
    """A short per-session summary for the end of a batch run.

    Grouped under a per-subject tally once more than one subject ran, so a
    whole-dataset run says which subject the failures belong to without the
    reader counting lines.
    """
    subjects = list(dict.fromkeys(outcome.subject for outcome in outcomes))
    sessions = [outcome for outcome in outcomes if outcome.status != "missing"]
    completed = [outcome for outcome in sessions if outcome.status == "ok"]
    header = f"Batch summary: {len(completed)}/{len(sessions)} sessions completed"
    if len(subjects) > 1:
        header += f" across {len(subjects)} subjects"

    lines = [header]
    indent = "    " if len(subjects) > 1 else "  "
    for subject in subjects:
        rows = [outcome for outcome in outcomes if outcome.subject == subject]
        subject_sessions = [outcome for outcome in rows if outcome.status != "missing"]
        if len(subjects) > 1:
            if not subject_sessions:
                lines.append(f"  {subject}: {rows[0].error or 'no sessions found'}")
                continue
            done = sum(1 for outcome in subject_sessions if outcome.status == "ok")
            lines.append(f"  {subject}: {done}/{len(subject_sessions)} completed")
        for outcome in rows:
            lines.append(f"{indent}[{outcome.status.upper():>6}] {outcome.label}{_detail(outcome)}")
    return "\n".join(lines)


def _detail(outcome: SessionOutcome) -> str:
    """The trailing explanation on a summary line, for anything that did not pass."""
    if outcome.status == "ok" or not outcome.error:
        return ""
    # A StepFailed message already names the step, so only an error from
    # elsewhere needs the step spelled out in front of it.
    step = outcome.failed_step
    named = step and step not in outcome.error
    detail = f" -- {step}: {outcome.error}" if named else f" -- {outcome.error}"
    if outcome.erased:
        detail += f" (erased {len(outcome.erased)} path(s))"
    return detail


def _derivatives_session_dir(
    derivatives_root: str | Path,
    *,
    subject: str | int,
    session: int | None,
    date: str | None,
) -> Path | None:
    """The one derivatives session directory a selector resolves to, or None."""
    layout = SessionLayout(Path(derivatives_root), name="derivatives")
    selector = {"ses": session} if session is not None else {"date": date}
    matches = layout.find_sessions(subject, missing_ok=True, **selector)
    if not matches:
        return None
    if len(matches) > 1:
        listing = ", ".join(str(match.path) for match in matches)
        raise ValueError(
            f"Refusing to erase: {normalize_subjid(subject)} has several derivatives "
            f"sessions matching {selector}: {listing}"
        )
    return matches[0].path


def _output_targets(session_dir: Path, env: Mapping[str, str]) -> list[Path]:
    """The folders below one session directory that hold this pipeline's outputs.

    The configured modality root (`eeg/` by default) covers every group at once;
    when it has been configured away the groups sit directly in the session
    directory and are listed individually, so nothing else in there is touched.
    """
    with _applied_env(env):
        root = output_root_dir()
        if root:
            return [session_dir / root]
        folders = sorted(set(output_dir_names().values()))
    return [session_dir / folder for folder in folders]


def _erase_derived_edf(
    rawdata_root: str | Path,
    *,
    subject: str | int,
    session: int | None,
    date: str | None,
    dry_run: bool,
) -> list[Path]:
    """Remove the trimmed/concatenated EDFs the pipeline wrote beside the raw parts."""
    root = Path(rawdata_root)
    layout = SessionLayout(root, name="rawdata")
    selector = {"ses": session} if session is not None else {"date": date}
    removed: list[Path] = []
    for match in layout.find_sessions(subject, missing_ok=True, **selector):
        for path in sorted(match.path.rglob("*.edf")):
            if not (is_concatenated_recording(path) or is_trimmed_recording(path)):
                continue
            _check_inside(path, match.path)
            if not dry_run:
                path.unlink()
            removed.append(path)
    return removed


def _check_inside(target: Path, session_dir: Path) -> None:
    """Refuse to remove anything that is not strictly below the session directory."""
    resolved = target.resolve(strict=False)
    session = session_dir.resolve(strict=False)
    if resolved == session or session not in resolved.parents:
        raise ValueError(f"Refusing to erase {target}: not inside {session_dir}")


@contextmanager
def _applied_env(env: Mapping[str, str]) -> Iterator[None]:
    """Apply the run's output-layout environment for the duration of a lookup."""
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
