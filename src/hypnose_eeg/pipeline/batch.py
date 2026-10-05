"""Session selection, failed-session cleanup, and reporting for batch runs.

A batch run processes several sessions one after the other: every session one
or more subjects have (`run.py --all-sessions`, with `--subject all` covering
every subject in the rawdata tree), or only the ones selected -- by
`--session`/`--date` values and ranges shared by every subject, or per subject
as `--subject 66:1,3 67:2-4`. A session that fails must not stop the ones after it, and must
not leave half-written outputs behind that a later rerun would mistake for
completed work -- every stage skips a step whose output already exists, so a
truncated parquet or a downsampled FIF written before the crash would be
silently reused. So a failed session is erased (its derivative outputs removed)
and recorded in a CSV report, and the run carries on with the next session.

A QC summary that comes out FAIL fails its session the same way; the report
additionally records which QC sections failed, so it can be told apart from a
crash. A session that finishes is reported with its QC status (pass or review),
and the report carries the batch's total run time.

What "erased" covers is deliberately narrow: only the output folders this
pipeline itself writes below the failed session's *derivatives* directory,
resolved through the same `output_layout.py` precedence (including any
`--output-root`/`--output-dir` overrides in force) that put them there. Nothing
in rawdata is touched unless `erase_derived_edf=True` is asked for, and then
only files this pipeline produced -- `<stem>_trimmed.edf` and
`<stem>_recording-concat.edf` -- never a raw recording.
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from hypnose_helpers.io.layout import (
    SessionLayout,
    SessionRef,
    filter_sessions,
    normalize_subjid,
)
from hypnose_helpers.io.selectors import (
    flatten,
    parse_date_range,
    parse_dates,
    parse_session_range,
    parse_sessions,
    parse_subjects,
)

from hypnose_eeg.io.output_layout import output_dir_name, output_dir_names, output_root_dir
from hypnose_eeg.io.output_paths import save_csv_rows
from hypnose_eeg.pipeline.steps import applied_env
from hypnose_eeg.qc.thresholds import load_performance_check
from hypnose_eeg.utils.recording_selection import (
    is_concatenated_recording,
    is_trimmed_recording,
)

# `--subject all`: every subject the rawdata tree holds, rather than a list
# typed out by hand. Spelled as a subject value so one flag covers both.
ALL_SUBJECTS = "all"

# `--subject 66:1,3`: a subject followed by the sessions to run for it.
SUBJECT_SESSIONS_SEP = ":"

# A hyphen between digits marks an inclusive range (`2-5`, `20260707-20260718`);
# the hyphens in `ses-3` and `sub-066` follow letters, so are never read as one.
_RANGE_RE = re.compile(r"(?<=\d)-(?=\d)")

# The file suffixes `hypnose_eeg/qc/summary_qc.py` writes its section results
# under, prefixed with the recording stem: a CSV and a parquet copy of it.
QC_SUMMARY_SUFFIX = "qc_summary.csv"
QC_SUMMARY_PARQUET_SUFFIX = "qc_summary.parquet"
QC_FAIL = "fail"

REPORT_FIELDS = [
    "started_at",
    "subject",
    "session",
    "date",
    "session_dir",
    "status",
    "qc_status",
    "qc_failed_sections",
    "duration_seconds",
    "failed_step",
    "returncode",
    "error",
    "erased",
    "batch_duration_seconds",
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
    qc_status: str = ""
    qc_failed_sections: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if not self.date:
            return self.subject
        session = f"ses-{self.session:03d}" if self.session is not None else "ses-?"
        return f"{self.subject}/{session}_date-{self.date}"

    @property
    def qc_failed(self) -> bool:
        return self.qc_status == QC_FAIL

    def as_row(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "subject": self.subject,
            "session": "" if self.session is None else self.session,
            "date": self.date,
            "session_dir": str(self.session_dir),
            "status": self.status,
            "qc_status": self.qc_status,
            "qc_failed_sections": ";".join(self.qc_failed_sections),
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
    _check_all_stands_alone(tokens)
    subjects = [
        subjid
        for subjid, _ in SessionLayout(Path(rawdata_root), name="rawdata").iter_subjects()
    ]
    if not subjects:
        raise FileNotFoundError(f"No subject directories under the rawdata root {rawdata_root}")
    return subjects


@dataclass(frozen=True)
class SessionSelection:
    """One subject of a batch run, and which of its sessions to run.

    `filters` holds one `(key, value)` per session number (`"session"`) or
    date (`"date"`) asked for, each a single value or an inclusive `A-B` range;
    no filters selects every session the subject has.
    """

    subject: int
    filters: tuple[tuple[str, str], ...] = ()


def resolve_selections(
    rawdata_root: str | Path,
    values: Sequence[str],
    *,
    sessions: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
) -> list[SessionSelection]:
    """The subjects and sessions a batch run covers, from `--subject` values.

    A value can carry its own sessions as `SUBJECT:SESSIONS` (`66:1,3`,
    `67:2-5`); every other value takes `sessions` or `dates` when given, and
    every session the subject has when not. Subjects are read as
    `resolve_subjects` reads them, `all` included. Session numbers and dates
    are validated here, so a typo fails the run before any session starts.
    """
    if sessions and dates:
        raise ValueError("Select sessions by number or by date, not both")
    shared = _session_filters("session", sessions) + _session_filters("date", dates)
    split = [str(value).strip().partition(SUBJECT_SESSIONS_SEP) for value in values]
    _check_all_stands_alone([subject for subject, _, _ in split])

    selections: list[SessionSelection] = []
    for subject_text, has_own, own in split:
        filters = shared
        if has_own:
            filters = _session_filters("session", [own])
            if not filters:
                raise ValueError(
                    f"No sessions named after {subject_text}{SUBJECT_SESSIONS_SEP}; "
                    f"expected for example {subject_text}{SUBJECT_SESSIONS_SEP}1,3"
                )
        selections += [
            SessionSelection(subject, filters)
            for subject in resolve_subjects(rawdata_root, [subject_text])
        ]
    # A subject listed twice the same way is one selection, not two runs.
    return list(dict.fromkeys(selections))


def select_sessions(
    selection: SessionSelection, available: Sequence[SessionRef]
) -> list[tuple[list[SessionRef], FileNotFoundError | None]]:
    """`available` narrowed to `selection`, as one entry per filter it holds.

    A filter that matches nothing comes back as an error in its own entry
    rather than as a quietly shorter run: asking for sessions 1 and 9 of a
    subject that has no session 9 should say so. A range only has to match
    one session.
    """
    if not selection.filters:
        return [(list(available), None)]
    entries: list[tuple[list[SessionRef], FileNotFoundError | None]] = []
    for key, value in selection.filters:
        if key == "session":
            matched = filter_sessions(available, ses=value)
        else:
            matched = filter_sessions(available, date=value)
        error = None if matched else FileNotFoundError(
            f"No {key} {value} found for {normalize_subjid(selection.subject)}"
        )
        entries.append((matched, error))
    return entries


def _session_filters(key: str, values: Sequence[str] | None) -> tuple[tuple[str, str], ...]:
    """Validated `(key, value)` filters for session numbers or dates, duplicates dropped."""
    if key == "session":
        parse_values, parse_range = parse_sessions, parse_session_range
    else:
        parse_values, parse_range = parse_dates, parse_date_range
    filters: list[tuple[str, str]] = []
    for token in flatten(values):
        if is_range(token):
            parse_range(token)
        else:
            parse_values([token])
        if (key, token) not in filters:
            filters.append((key, token))
    return tuple(filters)


def is_range(value: str) -> bool:
    """Whether a session or date value is an inclusive `A-B` range."""
    return bool(_RANGE_RE.search(value))


def _check_all_stands_alone(tokens: Sequence[str]) -> None:
    """Refuse `all` listed beside other subjects -- it is a whole-dataset run."""
    if len(tokens) > 1 and any(token.lower() == ALL_SUBJECTS for token in tokens):
        raise ValueError(
            f"--subject {ALL_SUBJECTS} already covers every subject; "
            f"do not list others alongside it"
        )


def missing_subject_outcome(subject: str | int, error: Exception) -> SessionOutcome:
    """A report row for a subject whose sessions could not be resolved at all,
    or for a session or date selected for it that it does not have.

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
    `hypnose_eeg.pipeline.steps.output_layout_env`), so the folders erased are the ones the
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
    derivatives_root: str | Path,
    subjects: Sequence[str | int],
    started_at: datetime,
    *,
    label: str | None = None,
) -> Path:
    """Where a batch run's report lands: one timestamped CSV per run.

    A single-subject run keeps its report in that subject's derivatives
    directory, where anyone looking at the subject will find it; a run spanning
    several subjects writes one combined report at the derivatives root
    instead, since no one subject owns it.

    Timestamped rather than a fixed name so a rerun -- which is the normal
    response to a failed session -- cannot overwrite the report that recorded
    why the first run failed. `label` is appended to the name, so runs that
    start in the same second -- job-array tasks for one subject's sessions --
    keep a report each.
    """
    root = Path(derivatives_root)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    filename = f"batch_report_{stamp}_{label}.csv" if label else f"batch_report_{stamp}.csv"
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


def qc_summary_files(qc_dir: Path) -> dict[str, Path]:
    """Each recording's QC summary in `qc_dir`, keyed by its filename prefix.

    The parquet copy is preferred; the CSV is used only where no parquet sits
    beside it (summaries written before the parquet copy existed). Hidden files
    are skipped: macOS leaves `._<name>` AppleDouble sidecars on network shares
    that match the pattern but hold no table.
    """
    found: dict[str, Path] = {}
    # Parquet last, so it replaces the CSV of the same recording.
    for suffix in (QC_SUMMARY_SUFFIX, QC_SUMMARY_PARQUET_SUFFIX):
        for path in qc_dir.glob(f"*{suffix}"):
            if not path.name.startswith("."):
                found[path.name[: -len(suffix)]] = path
    return dict(sorted(found.items()))


def read_qc_summary(path: Path) -> list[dict[str, str]]:
    """A QC summary's section rows as text, from its parquet or CSV form."""
    if path.suffix == ".parquet":
        return [
            {key: "" if value is None else str(value) for key, value in row.items()}
            for row in pq.read_table(path).to_pylist()
        ]
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_qc_verdict(
    derivatives_root: str | Path,
    *,
    subject: str | int,
    session: int | None = None,
    date: str | None = None,
    env: Mapping[str, str] | None = None,
    since: float | None = None,
) -> tuple[str, list[str]] | None:
    """A session's overall QC status and failed sections, from its QC summary.

    Only summaries modified at or after `since` (a `time.time()` value) count,
    so a summary left by an earlier run is not reported as this run's verdict.
    A session holding several recordings reports the worst of them. Returns
    None when no such summary exists -- the QC summary was not among the
    stages run, or it never got as far as writing one.
    """
    session_dir = _derivatives_session_dir(
        derivatives_root, subject=subject, session=session, date=date
    )
    if session_dir is None:
        return None
    with applied_env(env or {}):
        qc_dir = session_dir / output_dir_name("quality_control")
    # Coarse filesystem timestamps can round a fresh write to just before `since`.
    cutoff = None if since is None else since - 2
    summaries = [
        path for path in qc_summary_files(qc_dir).values()
        if cutoff is None or path.stat().st_mtime >= cutoff
    ]
    if not summaries:
        return None

    severity = load_performance_check()
    statuses: list[str] = []
    failed: list[str] = []
    for path in summaries:
        for row in read_qc_summary(path):
            status = str(row.get("status", "")).strip().lower()
            statuses.append(status)
            if severity.get(status, severity[QC_FAIL]) >= severity[QC_FAIL]:
                failed.append(str(row.get("section", "")).strip())
    if not statuses:
        # summary_qc treats a summary with no sections as a failure too.
        return QC_FAIL, []
    worst = max(statuses, key=lambda status: severity.get(status, severity[QC_FAIL]))
    if worst not in severity:
        worst = QC_FAIL
    return worst, list(dict.fromkeys(failed))


def write_report(
    outcomes: Sequence[SessionOutcome],
    path: str | Path,
    *,
    batch_duration_seconds: float | None = None,
) -> Path:
    """Write one row per session to CSV, through the shared CSV writer.

    The batch's total run time is repeated on every row rather than given a row
    of its own, so the report stays one row per session for anyone filtering it.
    """
    rows = [outcome.as_row() for outcome in outcomes]
    total = "" if batch_duration_seconds is None else round(batch_duration_seconds, 1)
    for row in rows:
        row["batch_duration_seconds"] = total
    return save_csv_rows(rows, REPORT_FIELDS, path)


def format_summary(
    outcomes: Sequence[SessionOutcome], *, elapsed_seconds: float | None = None
) -> str:
    """A short per-session summary for the end of a batch run.

    Grouped under a per-subject tally once more than one subject ran, so a
    whole-dataset run says which subject the failures belong to without the
    reader counting lines. Sessions that failed QC are listed again at the end
    with the sections that failed.
    """
    subjects = list(dict.fromkeys(outcome.subject for outcome in outcomes))
    sessions = [outcome for outcome in outcomes if outcome.status != "missing"]
    completed = [outcome for outcome in sessions if outcome.status == "ok"]
    header = f"Batch summary: {len(completed)}/{len(sessions)} sessions completed"
    if len(subjects) > 1:
        header += f" across {len(subjects)} subjects"
    qc_failed = [outcome for outcome in sessions if outcome.qc_failed]
    if qc_failed:
        header += f", {len(qc_failed)} failed QC"

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
    if qc_failed:
        lines.append("Failed QC:")
        for outcome in qc_failed:
            lines.append(f"  {outcome.label}{_qc_detail(outcome)}")
    if elapsed_seconds is not None:
        lines.append(f"Total time: {format_duration(elapsed_seconds)}")
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """`1h 02m 03s`, `4m 05s`, or `12s` -- whichever is the shortest that fits."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _detail(outcome: SessionOutcome) -> str:
    """The trailing explanation on a summary line, for anything that did not pass."""
    if outcome.qc_failed:
        detail = f" -- QC FAIL{_qc_detail(outcome)}"
        if outcome.erased:
            detail += f" (erased {len(outcome.erased)} path(s))"
        return detail
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


def _qc_detail(outcome: SessionOutcome) -> str:
    """The failed QC sections, if the summary named any."""
    if not outcome.qc_failed_sections:
        return ""
    return f" ({', '.join(outcome.qc_failed_sections)})"


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
    with applied_env(env):
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
