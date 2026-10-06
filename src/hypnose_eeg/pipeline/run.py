"""Run the full Hypnose pipeline: preprocessing -> sleep_scoring -> qc.

Each stage is one of `preprocessing.py`, `sleep_scoring.py`, or `qc.py` in
this package; see those modules for their own step lists and defaults. With no
`--stage` given, all three run in the order the pipeline actually requires --
artifact detection depends on sleep scoring, so it is deferred until after
scoring even though it is a preprocessing step. The artifact *prescan* runs
before scoring, which leaves the long artifact periods it finds unscored:

  1. preprocessing: trim, concatenate, downsample, prescan_artifacts
  2. sleep_scoring: score
  3. preprocessing: detect_artifacts
  4. qc: summary, figures (review PNGs; see `hypnose_eeg/qc/review_figures.py`)

Restricting to one or more `--stage` values runs each named stage's own
default step set instead (for example `--stage preprocessing` runs
trim, concatenate, downsample, prescan_artifacts, and detect_artifacts together,
which assumes sleep scoring has already been done for that session).

A step whose outputs already exist is skipped and the run continues with the
next one, so an interrupted or partially-completed session can be resumed by
rerunning the same command. Pass `--overwrite` to recompute regardless.

One subject with one `--session` or `--date` is a single run. Anything more
is a batch run, one session after the other:

  --subject 66 67 --all-sessions      every session each subject has
  --subject all --all-sessions        every session of every rawdata subject
  --subject 66 67 --session 1 3       sessions 1 and 3 of each subject
  --subject 66 --session 2-5          sessions 2 to 5 (inclusive range)
  --subject 66:1,3 67:2-4             sessions chosen per subject
  --subject 66:1 67 --session 2       per-subject sessions, `--session` for the rest

`--date` takes values and ranges the same way. A selected session or date that
a subject does not have is reported as `missing` rather than skipped silently.
A session that fails does not stop the batch: its derivative
outputs are erased -- so the half-finished ones cannot be mistaken for
completed work by a later rerun -- the failure is recorded in a CSV report, and
the next session starts. A subject with no resolvable sessions is recorded the
same way rather than stopping the subjects after it. `--keep-failed` notes a
failure without erasing anything. See `batch.py` for exactly what erasing
covers. A QC FAIL counts as a failed session like any other, and the report
names the QC sections that failed; a session that finishes is reported with its
QC status (pass or review). The report also records how long the whole batch
took.

`--view` opens the interactive scoring viewer
(`hypnose_eeg/sleep_scoring/view_scoring.py`) once the selected stages
finish, so a run can end in a look at the traces and predictions it produced.
It runs last -- after artifact detection and the QC summary, so `--show-artifacts`
has artifacts to show -- and needs a display. It only reads outputs already on
disk, so it is also available on its own (`--stage qc --view`, or with no
stage work at all). Viewer-only options such as `--hours` cannot be given here,
because unrecognized arguments go to every stage; pass those to
`hypnose-eeg-score --steps view` instead.

Sleep scoring, artifact detection, and the QC summary each write a
`<output stem>_provenance.json` sidecar naming the git commit that produced the
output -- and, for sleep scoring, which model was used.

A selection can also be split into independent tasks for a SLURM job array
(see `slurm/README.md`). `--list-tasks` prints one line per task --
`--task-unit subject` (the default) gives each subject its own task, running
its sessions in order; `--task-unit session` gives every session its own --
and `--task-index N` runs only task N of the same selection, as a batch run
(failed sessions erased and reported). With `--task-file FILE` -- a saved
`--list-tasks` output -- task N is line N of that file instead, so a task runs
the sessions frozen at submission even if rawdata has gained sessions since.
`--merge-reports DIR` combines the per-task reports an array left in DIR into
one batch report for the whole job.
A short recording is normalized against another session of the same animal,
so sessions of one subject run in parallel only with `--task-unit session`.

Unrecognized arguments are forwarded verbatim to every stage that runs, so a
flag understood by only one stage (`--model` aside, which this script does
know about) is best passed by invoking that stage's script directly.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from hypnose_eeg.pipeline import preprocessing, qc, sleep_scoring
from hypnose_eeg.pipeline.batch import (
    ALL_SUBJECTS,
    REPORT_FIELDS,
    SUBJECT_SESSIONS_SEP,
    SessionOutcome,
    SessionSelection,
    erase_session_outputs,
    format_summary,
    is_range,
    missing_subject_outcome,
    read_qc_verdict,
    report_path,
    resolve_selections,
    select_sessions,
    session_selector,
    write_report,
)
from hypnose_eeg.pipeline.steps import (
    StepFailed,
    add_output_layout_arguments,
    output_dir_overrides,
    output_layout_env,
)
from hypnose_helpers.io.layout import normalize_subjid, parse_session_dirname, parse_subject
from hypnose_helpers.io.selectors import flatten

from hypnose_eeg.io.output_paths import save_csv_rows
from hypnose_eeg.io.repository_paths import (
    get_derivatives_root,
    get_rawdata_root,
    resolve_data_roots,
)
from hypnose_eeg.utils.recording_selection import find_sessions

STAGE_ORDER = ["preprocessing", "sleep_scoring", "qc"]

# How `--list-tasks`/`--task-index` split a selection: one task per subject
# (its sessions in order) or one per session.
TASK_UNITS = ["subject", "session"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Hypnose EEG pipeline.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", "--subjid", dest="subject", nargs="+", required=True,
        metavar="SUBJECT",
        help="Subject ID, for example 66 or sub-066; several may be given, or "
        f"{ALL_SUBJECTS!r} for every subject in the rawdata tree. "
        f"SUBJECT{SUBJECT_SESSIONS_SEP}SESSIONS (66{SUBJECT_SESSIONS_SEP}1,3 or "
        f"67{SUBJECT_SESSIONS_SEP}2-4) selects that subject's own sessions.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--date", nargs="+", default=None, metavar="DATE",
        help="Session date(s): YYYYMMDD, or an inclusive range such as "
        "20260707-20260718.",
    )
    selector.add_argument(
        "--session", nargs="+", default=None, metavar="SESSION",
        help="Session number(s), for example 1 or ses-1, or an inclusive range "
        "such as 2-5. More than one session is run as a batch.",
    )
    selector.add_argument(
        "--all-sessions", "--batch", dest="all_sessions", action="store_true",
        help="Run every session each selected subject has, one after the other, "
        "continuing past a session that fails (see --keep-failed and --report).",
    )
    parser.add_argument(
        "--rawdata-root", default=None, help="Override the active profile's raw EDF root."
    )
    parser.add_argument(
        "--derivatives-root", default=None,
        help="Override the active profile's derivatives root.",
    )
    add_output_layout_arguments(parser)
    parser.add_argument(
        "--model", "--model-path", dest="model", default=None,
        help="Somnotate model name or model.pickle path (forwarded to sleep scoring).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute every stage's outputs instead of skipping the steps whose "
        "outputs already exist.",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Open the interactive scoring viewer after the selected stages finish "
        "(opens a plot window; requires a display).",
    )
    parser.add_argument(
        "--stage", nargs="+", choices=STAGE_ORDER, default=None,
        help="Restrict the run to these stages, each using its own default step set "
        "(default: all three, in dependency order).",
    )
    batch = parser.add_argument_group("batch options (runs covering more than one session)")
    batch.add_argument(
        "--keep-failed", action="store_true",
        help="Record a failed session in the report but leave its outputs on disk "
        "(default: erase them, so a rerun recomputes the session from scratch).",
    )
    batch.add_argument(
        "--erase-derived-edf", action="store_true",
        help="When erasing a failed session, also delete the trimmed and "
        "concatenated EDFs the pipeline wrote beside its raw recordings.",
    )
    batch.add_argument(
        "--report", default=None, metavar="FILE",
        help="Where to write the per-session CSV report (default: "
        "<derivatives>/sub-XXX/batch_report_<timestamp>.csv for one subject, "
        "<derivatives>/batch_report_<timestamp>.csv for several).",
    )
    tasks = parser.add_argument_group("job-array options (see slurm/README.md)")
    task_mode = tasks.add_mutually_exclusive_group()
    task_mode.add_argument(
        "--list-tasks", action="store_true",
        help="Print the tasks the selection splits into, one per line, and exit "
        "without running anything.",
    )
    task_mode.add_argument(
        "--task-index", type=int, default=None, metavar="N",
        help="Run only task N (counted from 0) of the selection, as a batch run.",
    )
    task_mode.add_argument(
        "--merge-reports", default=None, metavar="DIR",
        help="Merge the per-task reports a job array left in DIR (task-N.csv) "
        "into one batch report, and exit without running anything. With "
        "--task-file, tasks that left no report are reported too.",
    )
    task_mode.add_argument(
        "--print-derivatives-root", action="store_true",
        help="Print the derivatives root this run would write to (after "
        "--derivatives-root and the data profile) and exit without running "
        "anything; slurm/submit.sh keeps its logs and reports below it.",
    )
    tasks.add_argument(
        "--task-unit", choices=TASK_UNITS, default="subject",
        help="What one task covers: every selected session of one subject, run in "
        "order (default), or one session.",
    )
    tasks.add_argument(
        "--task-file", default=None, metavar="FILE",
        help="With --task-index N, run line N of this saved --list-tasks output "
        "instead of task N of the selection as rawdata stands now.",
    )
    return parser


def run_stages(
    common: dict,
    *,
    stages: list[str] | None,
    model: str | None,
    overwrite: bool,
    view: bool,
    extra: list[str],
) -> None:
    """Run the selected stages for one session; raise `StepFailed` on the first failure."""
    if stages is None:
        preprocessing.run_steps(
            **common, overwrite=overwrite,
            steps=["trim", "concatenate", "downsample", "prescan_artifacts"],
            extra_args=extra,
        )
        sleep_scoring.run_steps(
            **common, model=model, overwrite=overwrite,
            steps=["score"], extra_args=extra,
        )
        preprocessing.run_steps(
            **common, overwrite=overwrite,
            steps=["detect_artifacts"], extra_args=extra,
        )
        qc.run_steps(
            **common, overwrite=overwrite, steps=["summary", "figures"], extra_args=extra
        )
    else:
        if "preprocessing" in stages:
            preprocessing.run_steps(**common, overwrite=overwrite, extra_args=extra)
        if "sleep_scoring" in stages:
            sleep_scoring.run_steps(
                **common, model=model, overwrite=overwrite, extra_args=extra
            )
        if "qc" in stages:
            qc.run_steps(**common, overwrite=overwrite, extra_args=extra)
    if view:
        sleep_scoring.run_steps(**common, steps=["view"], extra_args=extra)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    output_dirs = output_dir_overrides(parser, args)
    if args.print_derivatives_root:
        print(resolve_data_roots(args.rawdata_root, args.derivatives_root)[1])
        return 0
    if args.merge_reports is not None:
        try:
            merge_task_reports(
                Path(args.merge_reports),
                task_file=args.task_file,
                rawdata_root=args.rawdata_root,
                derivatives_root=args.derivatives_root,
                report=args.report,
            )
        except (OSError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.task_file is not None and args.task_index is None:
        parser.error("--task-file needs --task-index: which line of the file to run")
    # A job-array task always runs as a batch, so a failed session is erased
    # and reported even when the task covers only one session.
    as_tasks = args.list_tasks or args.task_index is not None
    single = (
        None if args.all_sessions or as_tasks
        else _single_session(args.subject, args.session, args.date)
    )
    if single is None and args.view:
        parser.error(
            "--view needs a single session: the viewer waits for a window to be "
            "closed, which a batch run has nobody to do. Review a session "
            "afterwards with --session N --stage qc --view."
        )

    options = dict(
        rawdata_root=args.rawdata_root,
        derivatives_root=args.derivatives_root,
        output_layout=args.output_layout,
        output_root=args.output_root,
        output_dirs=output_dirs,
    )
    if single is None:
        unselected = [
            subject for subject in args.subject if SUBJECT_SESSIONS_SEP not in subject
        ]
        if unselected and not (args.all_sessions or args.session or args.date):
            parser.error(
                f"choose the sessions to run for {' '.join(unselected)}: --session, "
                f"--date, SUBJECT{SUBJECT_SESSIONS_SEP}SESSIONS, or --all-sessions "
                "for every session."
            )
        if args.list_tasks:
            return _print_tasks(
                args.subject, sessions=args.session, dates=args.date,
                rawdata_root=args.rawdata_root, unit=args.task_unit,
            )
        try:
            result = run_batch(
                args.subject, sessions=args.session, dates=args.date, **options,
                stages=args.stage, model=args.model, overwrite=args.overwrite,
                extra_args=extra, keep_failed=args.keep_failed,
                erase_derived_edf=args.erase_derived_edf, report=args.report,
                task_index=args.task_index, task_unit=args.task_unit,
                task_file=args.task_file,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        return 0 if result.ok else 1

    try:
        run_stages(
            dict(**single, **options),
            stages=args.stage, model=args.model, overwrite=args.overwrite,
            view=args.view, extra=extra,
        )
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


def _single_session(
    subjects: Sequence[str], sessions: Sequence[str] | None, dates: Sequence[str] | None
) -> dict[str, str | None] | None:
    """The stage selectors of a run covering one session, or None for a batch run.

    One subject with at most one session number or date -- `--subject 66
    --session 1` or `--subject 66:1` alike -- runs without the batch machinery,
    as it always has (and so can end in `--view`); one subject with no session
    at all is left for the stages to resolve. Several subjects, `all`, several
    sessions, or a range make a batch run.
    """
    if len(subjects) != 1:
        return None
    subject, has_own, own = str(subjects[0]).strip().partition(SUBJECT_SESSIONS_SEP)
    if subject.lower() == ALL_SUBJECTS or len(flatten(subject)) != 1:
        return None
    key = "session" if has_own or sessions else "date"
    values = flatten([own] if has_own else (sessions or dates))
    if len(values) > 1 or (values and is_range(values[0])):
        return None
    selector: dict[str, str | None] = {"session": None, "date": None}
    if values:
        selector[key] = values[0]
    return dict(subject=subject, **selector)


@dataclass
class BatchResult:
    """What a batch run did: one outcome per session, plus unresolvable selections."""

    outcomes: list[SessionOutcome]
    session_count: int
    elapsed_seconds: float = 0.0
    # None when nothing ran or the report could not be written.
    report: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether sessions ran and none of them failed, nor any selection came up missing."""
        return self.session_count > 0 and all(
            outcome.status == "ok" for outcome in self.outcomes
        )


@dataclass(frozen=True)
class _BatchSettings:
    """What a batch run applies to every session: stage options and failure handling."""

    options: dict
    stages: list[str] | None
    model: str | None
    overwrite: bool
    extra: list[str]
    keep_failed: bool
    erase_derived_edf: bool
    env: dict[str, str]
    rawdata_root: Path
    derivatives_root: Path


def run_batch(
    subjects: Sequence[str],
    *,
    sessions: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    output_layout: str | Path | None = None,
    output_root: str | None = None,
    output_dirs: dict[str, str] | None = None,
    stages: list[str] | None = None,
    model: str | None = None,
    overwrite: bool = False,
    extra_args: list[str] | None = None,
    keep_failed: bool = False,
    erase_derived_edf: bool = False,
    report: str | Path | None = None,
    task_index: int | None = None,
    task_unit: str = "subject",
    task_file: str | Path | None = None,
) -> BatchResult:
    """Run the selected sessions of every selected subject, erasing and noting the failures.

    `subjects` are subject IDs, or `"all"` for every subject in the rawdata
    tree; one written `SUBJECT:SESSIONS` (`"66:1,3"`) runs only those
    sessions of it. The rest run the `sessions` or `dates` given -- values or
    inclusive `A-B` ranges -- or every session they have. A session that fails
    is recorded and, unless `keep_failed`, has its outputs erased, and the
    batch carries on; the per-session CSV report goes to `report`, or to a
    timestamped file below the derivatives root. Raises
    `FileNotFoundError`/`ValueError` when the subjects themselves cannot be
    resolved or a session or date is malformed.

    `task_index` runs only that task of the selection, split by `task_unit` as
    `plan_tasks` splits it -- one task of a job array. Selections that come up
    missing belong to no task; `--list-tasks` reports them. With `task_file`,
    a saved `--list-tasks` output, the task is line `task_index` of that file
    instead, and the selection is not resolved again: sessions added to
    rawdata since the file was written are left alone.
    """
    rawdata = Path(rawdata_root) if rawdata_root else get_rawdata_root()
    derivatives = Path(derivatives_root) if derivatives_root else get_derivatives_root()
    report_label = None
    if task_file is not None:
        if task_index is None:
            raise ValueError("A task file needs a task index: which line of it to run")
        plan = _read_task(rawdata, Path(task_file), task_index)
        resolved_subjects = [subject for subject, _, _ in plan]
    else:
        selections = resolve_selections(rawdata, subjects, sessions=sessions, dates=dates)
        resolved_subjects = list(
            dict.fromkeys(selection.subject for selection in selections)
        )
        plan = _plan_batch(rawdata, selections)
        if task_index is not None:
            tasks = _split_tasks(plan, task_unit)
            if not 0 <= task_index < len(tasks):
                raise ValueError(
                    f"Task index {task_index} is out of range: the selection splits "
                    f"into {len(tasks)} task(s) by {task_unit}"
                )
            plan = tasks[task_index]
            resolved_subjects = [subject for subject, _, _ in plan]
    if task_index is not None:
        if task_unit == "session":
            # Tasks for one subject's sessions start together and would
            # otherwise write the same timestamped report.
            report_label = plan[0][1][0].path.name
    total = sum(len(sessions) for _, sessions, _ in plan)
    if not total:
        # Nothing resolved for any subject: the errors are already on stderr and
        # there is no run to report on.
        return BatchResult(
            outcomes=[
                missing_subject_outcome(subject, error)
                for subject, _, error in plan
                if error is not None
            ],
            session_count=0,
        )

    batch = _BatchSettings(
        options=dict(
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            output_layout=output_layout,
            output_root=output_root,
            output_dirs=output_dirs,
        ),
        stages=stages,
        model=model,
        overwrite=overwrite,
        extra=list(extra_args or []),
        keep_failed=keep_failed,
        erase_derived_edf=erase_derived_edf,
        env=output_layout_env(output_layout, output_dirs, output_root),
        rawdata_root=rawdata,
        derivatives_root=derivatives,
    )
    started = datetime.now()
    batch_clock = time.monotonic()
    outcomes: list[SessionOutcome] = []
    print(f"Batch: {total} session(s) across {len(resolved_subjects)} subject(s)")

    position = 0
    interrupted = False
    for subject, sessions, error in plan:
        if error is not None:
            outcomes.append(missing_subject_outcome(subject, error))
            continue
        for session_ref in sessions:
            position += 1
            print(f"\n===== [{position}/{total}] {session_ref} =====")
            outcome = _run_one_session(batch, session_ref, subject=subject)
            outcomes.append(outcome)
            interrupted = outcome.status == "interrupted"
            if interrupted:
                break
        if interrupted:
            break

    elapsed = time.monotonic() - batch_clock
    print()
    print(format_summary(outcomes, elapsed_seconds=elapsed))
    destination: Path | None = (
        Path(report) if report
        else report_path(derivatives, resolved_subjects, started, label=report_label)
    )
    try:
        write_report(outcomes, destination, batch_duration_seconds=elapsed)
    except OSError as exc:
        # The report is a record of the run, not the run itself -- losing it is
        # worth a loud warning, not throwing away the sessions that succeeded.
        print(f"WARNING: could not write the batch report: {exc}", file=sys.stderr)
        destination = None
    return BatchResult(
        outcomes=outcomes,
        session_count=total,
        elapsed_seconds=elapsed,
        report=destination,
    )


def _plan_batch(
    rawdata_root: Path, selections: list[SessionSelection]
) -> list[tuple[int, list, Exception | None]]:
    """Resolve each subject's sessions up front, keeping the ones that do not resolve.

    Listing every subject before running any of them means an unusable subject
    -- or a selected session it does not have -- is reported in its place in
    the run rather than only discovered hours in, and never stops the subjects
    after it. A session selected more than once runs once.
    """
    plan: list[tuple[int, list, Exception | None]] = []
    planned: set[Path] = set()
    for selection in selections:
        subject = selection.subject
        try:
            available = find_sessions(rawdata_root, subject=subject)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            plan.append((subject, [], exc))
            continue
        for sessions, error in select_sessions(selection, available):
            if error is not None:
                print(f"FAILED: {error}", file=sys.stderr)
                plan.append((subject, [], error))
                continue
            fresh = [session for session in sessions if session.path not in planned]
            planned.update(session.path for session in fresh)
            if fresh:
                plan.append((subject, fresh, None))
    return plan


def plan_tasks(
    subjects: Sequence[str],
    *,
    sessions: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
    rawdata_root: str | Path | None = None,
    unit: str = "subject",
) -> list[list[tuple[int, list, None]]]:
    """The tasks a selection splits into for a job array, in the order run.

    Each task is a batch plan of its own: every selected session of one
    subject (`unit="subject"`), in the order a batch run would take them, or a
    single session (`unit="session"`). `run_batch(task_index=N)` runs task N.
    Selections that come up missing are printed to stderr and left out.
    """
    rawdata = Path(rawdata_root) if rawdata_root else get_rawdata_root()
    selections = resolve_selections(rawdata, subjects, sessions=sessions, dates=dates)
    return _split_tasks(_plan_batch(rawdata, selections), unit)


def _split_tasks(
    plan: list[tuple[int, list, Exception | None]], unit: str
) -> list[list[tuple[int, list, None]]]:
    """`plan` split into one plan per subject or per session, missing entries dropped."""
    if unit not in TASK_UNITS:
        raise ValueError(f"Unknown task unit {unit!r}; expected one of {TASK_UNITS}")
    runnable = [(subject, sessions) for subject, sessions, error in plan if error is None]
    if unit == "session":
        return [
            [(subject, [session], None)]
            for subject, sessions in runnable
            for session in sessions
        ]
    # A subject selected more than once (`66:1 66:3`) is still one task.
    by_subject: dict[int, list] = {}
    for subject, sessions in runnable:
        by_subject.setdefault(subject, []).extend(sessions)
    return [[(subject, sessions, None)] for subject, sessions in by_subject.items()]


def _read_task(
    rawdata_root: Path, task_file: Path, task_index: int
) -> list[tuple[int, list, None]]:
    """Task `task_index` of a saved `--list-tasks` output, as a batch plan.

    The line names the subject and its session directories; each is looked
    up in rawdata by that name, so the task runs what was listed even if the
    subject has gained sessions since. A session directory that has gone
    fails the task instead of running a different one.
    """
    lines = task_file.read_text().splitlines()
    if not 0 <= task_index < len(lines):
        raise ValueError(
            f"Task index {task_index} is out of range: {task_file} lists "
            f"{len(lines)} task(s)"
        )
    subject_name, *session_names = lines[task_index].split()
    if not session_names:
        raise ValueError(
            f"Line {task_index} of {task_file} names no sessions: "
            f"{lines[task_index]!r}"
        )
    subject = parse_subject(subject_name)
    available = {
        session.path.name: session
        for session in find_sessions(rawdata_root, subject=subject)
    }
    gone = [name for name in session_names if name not in available]
    if gone:
        raise FileNotFoundError(
            f"{normalize_subjid(subject)} no longer has {', '.join(gone)} in "
            f"{rawdata_root} (task {task_index} of {task_file})"
        )
    return [(subject, [available[name] for name in session_names], None)]


# The per-task reports a job array leaves for `merge_task_reports`.
_TASK_REPORT_RE = re.compile(r"^task-(\d+)\.csv$")


def merge_task_reports(
    report_dir: Path,
    *,
    task_file: str | Path | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    report: str | Path | None = None,
) -> Path:
    """Merge a job array's per-task reports into one batch report; return its path.

    Each array task writes its rows to `report_dir/task-N.csv`; they are
    combined in task order. With `task_file` -- the `--list-tasks` output the
    array ran -- a task that left no report (cancelled, out of time or memory,
    or failed before it ran) gets a `no_report` row for each of its sessions,
    so the report still accounts for every session the job was given. The
    batch duration on every row is the job's, from its first task starting to
    its last one finishing. The report goes to `report`, or where a batch
    run's report would, labelled `job-<report_dir name>`; the per-task files
    are removed once it is written.
    """
    task_reports = {
        int(match.group(1)): path
        for path in (report_dir.iterdir() if report_dir.is_dir() else [])
        if (match := _TASK_REPORT_RE.match(path.name))
    }
    task_lines = Path(task_file).read_text().splitlines() if task_file else []
    if not task_reports and not task_lines:
        raise FileNotFoundError(f"No task reports to merge in {report_dir}")

    rawdata = Path(rawdata_root) if rawdata_root else get_rawdata_root()
    rows: list[dict[str, object]] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    for index in range(max([*task_reports, len(task_lines) - 1]) + 1):
        path = task_reports.get(index)
        if path is None:
            if index < len(task_lines):
                rows.extend(_no_report_rows(rawdata, task_lines[index], index))
            continue
        with path.open(newline="") as handle:
            task_rows = list(csv.DictReader(handle))
        rows.extend(task_rows)
        span = _task_span(task_rows)
        if span is not None:
            starts.append(span[0])
            ends.append(span[1])

    duration = round((max(ends) - min(starts)).total_seconds(), 1) if starts else ""
    for row in rows:
        row["batch_duration_seconds"] = duration
    derivatives = Path(derivatives_root) if derivatives_root else get_derivatives_root()
    destination = Path(report) if report else report_path(
        derivatives,
        list(dict.fromkeys(str(row["subject"]) for row in rows)),
        min(starts) if starts else datetime.now(),
        label=f"job-{report_dir.name}",
    )
    save_csv_rows(rows, REPORT_FIELDS, destination)

    for path in task_reports.values():
        path.unlink()
    try:
        report_dir.rmdir()
    except OSError:
        pass  # something else was left in it; keep it
    counts = Counter(str(row["status"]) for row in rows)
    print(
        f"Merged {len(task_reports)} task report(s): "
        + ", ".join(f"{count} {status}" for status, count in counts.items())
    )
    return destination


def _task_span(rows: list[dict[str, str]]) -> tuple[datetime, datetime] | None:
    """When one task's batch started and finished, from its report rows."""
    try:
        start = min(
            datetime.fromisoformat(row["started_at"]) for row in rows if row.get("started_at")
        )
        duration = float(rows[0]["batch_duration_seconds"])
    except (IndexError, KeyError, ValueError):
        return None
    return start, start + timedelta(seconds=duration)


def _no_report_rows(rawdata_root: Path, line: str, index: int) -> list[dict[str, object]]:
    """Report rows for the sessions of a task that left no report of its own."""
    subject_name, *session_names = line.split()
    subject = parse_subject(subject_name)
    try:
        available = {
            session.path.name: session.path
            for session in find_sessions(rawdata_root, subject=subject)
        }
    except (FileNotFoundError, ValueError):
        available = {}
    error = (
        f"array task {index} left no report: it was cancelled, ran out of time or "
        "memory, or failed before running (see its logs in <derivatives>/slurm/logs and sacct)"
    )
    rows: list[dict[str, object]] = []
    for name in session_names:
        ses, date = parse_session_dirname(name) or (None, "")
        rows.append(
            SessionOutcome(
                subject=normalize_subjid(subject),
                session=ses,
                date=date,
                session_dir=available.get(name, Path(name)),
                status="no_report",
                error=error,
            ).as_row()
        )
    return rows


def _print_tasks(
    subjects: Sequence[str],
    *,
    sessions: Sequence[str] | None,
    dates: Sequence[str] | None,
    rawdata_root: str | None,
    unit: str,
) -> int:
    """Print one line per task -- its subject, then its session directories."""
    try:
        tasks = plan_tasks(
            subjects, sessions=sessions, dates=dates, rawdata_root=rawdata_root, unit=unit
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    for task in tasks:
        for subject, task_sessions, _ in task:
            names = " ".join(session.path.name for session in task_sessions)
            print(f"{normalize_subjid(subject)} {names}")
    if not tasks:
        print("FAILED: the selection holds no sessions to run", file=sys.stderr)
        return 1
    return 0


def _run_one_session(
    batch: _BatchSettings,
    session_ref,
    *,
    subject: int,
) -> SessionOutcome:
    """Run one session's stages, turning any failure into an outcome to carry on from."""
    outcome = SessionOutcome(
        subject=session_ref.subject,
        session=session_ref.ses,
        date=session_ref.date,
        session_dir=session_ref.path,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    clock = time.monotonic()
    wall_start = time.time()
    # A QC stage that found a summary already on disk reports that one, so
    # when QC runs, the summary on disk is its verdict however old; without
    # QC, only a summary written during this session's run would count.
    runs_qc = batch.stages is None or "qc" in batch.stages
    qc_lookup = dict(
        subject=subject, session=session_ref.ses, date=session_ref.date,
        env=batch.env, since=None if runs_qc else wall_start,
    )
    try:
        run_stages(
            dict(
                subject=normalize_subjid(subject),
                **session_selector(session_ref),
                **batch.options,
            ),
            stages=batch.stages, model=batch.model, overwrite=batch.overwrite,
            view=False, extra=batch.extra,
        )
    except KeyboardInterrupt:
        # An interrupt is the operator stopping the batch, not this session
        # failing: leave its outputs alone and report what ran so far.
        print("\nInterrupted; stopping the batch.", file=sys.stderr)
        outcome.status = "interrupted"
    except Exception as exc:  # noqa: BLE001 - one session must not stop the batch
        # Read before erasing: the summary is one of the outputs erased, and
        # it is what tells a QC FAIL apart from a crash in the same step.
        verdict = (
            _read_qc_verdict(batch.derivatives_root, outcome, qc_lookup)
            if _is_qc_verdict(exc) else None
        )
        if verdict is not None and verdict[0] == "fail":
            outcome.qc_status, outcome.qc_failed_sections = verdict
        outcome.status = "failed"
        outcome.error = str(exc)
        if isinstance(exc, StepFailed):
            outcome.failed_step = exc.label
            outcome.returncode = exc.returncode
        else:
            outcome.failed_step = type(exc).__name__
        print(f"FAILED: {outcome.label}: {exc}", file=sys.stderr)
        _erase_failed_session(batch, outcome, subject=subject)
    else:
        verdict = _read_qc_verdict(batch.derivatives_root, outcome, qc_lookup)
        if verdict is not None:
            outcome.qc_status, outcome.qc_failed_sections = verdict
    outcome.duration_seconds = time.monotonic() - clock
    return outcome


def _is_qc_verdict(exc: Exception) -> bool:
    """Whether a failure could be `summary_qc` reporting FAIL (it returns 1 for that).

    A step that raised is a crash whatever its status, so only a plain status
    of 1 from the summary step can be a verdict.
    """
    return (
        isinstance(exc, StepFailed)
        and exc.label == "qc:summary"
        and exc.returncode == 1
        and exc.error is None
    )


def _read_qc_verdict(
    derivatives_root: Path, outcome: SessionOutcome, lookup: dict
) -> tuple[str, list[str]] | None:
    """This run's QC verdict for a session, or None if it has none to read."""
    try:
        return read_qc_verdict(derivatives_root, **lookup)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"WARNING: could not read the QC summary of {outcome.label}: {exc}", file=sys.stderr)
        return None


def _erase_failed_session(
    batch: _BatchSettings,
    outcome: SessionOutcome,
    *,
    subject: int,
) -> None:
    """Erase a failed session's outputs, unless `keep_failed`, recording what went."""
    if batch.keep_failed:
        print(f"Keeping the outputs of {outcome.label} (--keep-failed).")
        return
    try:
        outcome.erased = erase_session_outputs(
            batch.derivatives_root,
            subject=subject,
            session=outcome.session,
            date=outcome.date,
            env=batch.env,
            rawdata_root=batch.rawdata_root,
            erase_derived_edf=batch.erase_derived_edf,
        )
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not erase {outcome.label}: {exc}", file=sys.stderr)
        outcome.error = f"{outcome.error}; erase failed: {exc}"
        return
    for path in outcome.erased:
        print(f"Erased: {path}")
    if not outcome.erased:
        print(f"Nothing to erase for {outcome.label}.")


if __name__ == "__main__":
    raise SystemExit(main())
