"""Run the full Hypnose pipeline: preprocessing -> sleep_scoring -> qc.

Each stage is one of `src/preprocessing.py`, `src/sleep_scoring.py`, or
`src/qc.py`; see those modules for their own step lists and defaults. With no
`--stage` given, all three run in the order the pipeline actually requires --
artifact detection depends on sleep scoring, so it is deferred until after
scoring even though it is a preprocessing step. The artifact *prescan* runs
before scoring, which leaves the long artifact periods it finds unscored:

  1. preprocessing: trim, concatenate, downsample, prescan_artifacts
  2. sleep_scoring: score
  3. preprocessing: detect_artifacts
  4. qc: summary

Restricting to one or more `--stage` values runs each named stage's own
default step set instead (for example `--stage preprocessing` runs
trim, concatenate, downsample, prescan_artifacts, and detect_artifacts together,
which assumes sleep scoring has already been done for that session).

A step whose outputs already exist is skipped and the run continues with the
next one, so an interrupted or partially-completed session can be resumed by
rerunning the same command. Pass `--overwrite` to recompute regardless.

`--all-sessions` takes subject IDs alone and runs every session each of them
has, one subject after the other; `--subject all` covers every subject in the
rawdata tree. A session that fails does not stop the batch: its derivative
outputs are erased -- so the half-finished ones cannot be mistaken for
completed work by a later rerun -- the failure is recorded in a CSV report, and
the next session starts. A subject with no resolvable sessions is recorded the
same way rather than stopping the subjects after it. `--keep-failed` notes a
failure without erasing anything. See `src/_batch.py` for exactly what erasing
covers. A QC FAIL counts as a failed session like any other, and the report
names the QC sections that failed; a session that finishes is reported with its
QC status (pass or review). The report also records how long the whole batch
took.

`--view` opens the interactive scoring viewer
(`scripts/sleep_scoring/view_scored_recording.py`) once the selected stages
finish, so a run can end in a look at the traces and predictions it produced.
It runs last -- after artifact detection and the QC summary, so `--show-artifacts`
has artifacts to show -- and needs a display. It only reads outputs already on
disk, so it is also available on its own (`--stage qc --view`, or with no
stage work at all). Viewer-only options such as `--hours` cannot be given here,
because unrecognized arguments go to every stage; pass those to
`python -m src.sleep_scoring --steps view` instead.

Sleep scoring, artifact detection, and the QC summary each write a
`<output stem>_provenance.json` sidecar naming the git commit that produced the
output -- and, for sleep scoring, which model was used.

Unrecognized arguments are forwarded verbatim to every stage that runs, so a
flag understood by only one stage (`--model` aside, which this script does
know about) is best passed by invoking that stage's script directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, qc, sleep_scoring
from src._batch import (
    ALL_SUBJECTS,
    SessionOutcome,
    erase_session_outputs,
    format_summary,
    missing_subject_outcome,
    read_qc_verdict,
    report_path,
    resolve_subjects,
    session_selector,
    write_report,
)
from src._pipeline import (
    StepFailed,
    add_output_layout_arguments,
    output_dir_overrides,
    output_layout_env,
)
from hypnose_helpers.io.layout import normalize_subjid

from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
from scripts.utils.recording_selection import find_sessions

STAGE_ORDER = ["preprocessing", "sleep_scoring", "qc"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Hypnose EEG pipeline.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", "--subjid", dest="subject", nargs="+", required=True,
        metavar="SUBJECT",
        help="Subject ID, for example 66 or sub-066. Several may be given with "
        f"--all-sessions, as may {ALL_SUBJECTS!r} for every subject in the rawdata tree.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
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
    batch = parser.add_argument_group("batch options (with --all-sessions)")
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
        qc.run_steps(**common, steps=["summary"], extra_args=extra)
    else:
        if "preprocessing" in stages:
            preprocessing.run_steps(**common, overwrite=overwrite, extra_args=extra)
        if "sleep_scoring" in stages:
            sleep_scoring.run_steps(
                **common, model=model, overwrite=overwrite, extra_args=extra
            )
        if "qc" in stages:
            qc.run_steps(**common, extra_args=extra)
    if view:
        sleep_scoring.run_steps(**common, steps=["view"], extra_args=extra)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    output_dirs = output_dir_overrides(parser, args)
    if args.all_sessions and args.view:
        parser.error(
            "--view cannot be combined with --all-sessions: the viewer waits for a "
            "window to be closed, which a batch run has nobody to do. Review a "
            "session afterwards with --session N --stage qc --view."
        )

    options = dict(
        rawdata_root=args.rawdata_root,
        derivatives_root=args.derivatives_root,
        output_layout=args.output_layout,
        output_root=args.output_root,
        output_dirs=output_dirs,
    )
    if args.all_sessions:
        return run_batch(args, extra, options)
    if len(args.subject) > 1 or args.subject[0].lower() == ALL_SUBJECTS:
        parser.error(
            "several subjects can only be run with --all-sessions; one session of "
            "one subject is a single run."
        )

    try:
        run_stages(
            dict(subject=args.subject[0], date=args.date, session=args.session, **options),
            stages=args.stage, model=args.model, overwrite=args.overwrite,
            view=args.view, extra=extra,
        )
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


def run_batch(args: argparse.Namespace, extra: list[str], options: dict) -> int:
    """Run every session of every selected subject, erasing and noting the failures."""
    rawdata_root = Path(args.rawdata_root) if args.rawdata_root else get_rawdata_root()
    derivatives_root = (
        Path(args.derivatives_root) if args.derivatives_root else get_derivatives_root()
    )
    try:
        subjects = resolve_subjects(rawdata_root, args.subject)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    plan = _plan_batch(rawdata_root, subjects)
    total = sum(len(sessions) for _, sessions, _ in plan)
    if not total:
        # Nothing resolved for any subject: the errors are already on stderr and
        # there is no run to report on.
        return 1

    env = output_layout_env(args.output_layout, options["output_dirs"], args.output_root)
    started = datetime.now()
    batch_clock = time.monotonic()
    outcomes: list[SessionOutcome] = []
    print(f"Batch: {total} session(s) across {len(subjects)} subject(s)")

    position = 0
    interrupted = False
    for subject, sessions, error in plan:
        if error is not None:
            outcomes.append(missing_subject_outcome(subject, error))
            continue
        for session_ref in sessions:
            position += 1
            print(f"\n===== [{position}/{total}] {session_ref} =====")
            outcome = _run_one_session(
                args, extra, options, session_ref,
                subject=subject, env=env,
                derivatives_root=derivatives_root, rawdata_root=rawdata_root,
            )
            outcomes.append(outcome)
            interrupted = outcome.status == "interrupted"
            if interrupted:
                break
        if interrupted:
            break

    elapsed = time.monotonic() - batch_clock
    print()
    print(format_summary(outcomes, elapsed_seconds=elapsed))
    destination = (
        Path(args.report) if args.report
        else report_path(derivatives_root, subjects, started)
    )
    try:
        write_report(outcomes, destination, batch_duration_seconds=elapsed)
    except OSError as exc:
        # The report is a record of the run, not the run itself -- losing it is
        # worth a loud warning, not throwing away the sessions that succeeded.
        print(f"WARNING: could not write the batch report: {exc}", file=sys.stderr)
    return 0 if all(outcome.status == "ok" for outcome in outcomes) else 1


def _plan_batch(
    rawdata_root: Path, subjects: list[int]
) -> list[tuple[int, list, Exception | None]]:
    """Resolve each subject's sessions up front, keeping the ones that do not resolve.

    Listing every subject before running any of them means an unusable subject
    is reported in its place in the run rather than only discovered hours in,
    and never stops the subjects after it.
    """
    plan: list[tuple[int, list, Exception | None]] = []
    for subject in subjects:
        try:
            plan.append((subject, find_sessions(rawdata_root, subject=subject), None))
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            plan.append((subject, [], exc))
    return plan


def _run_one_session(
    args: argparse.Namespace,
    extra: list[str],
    options: dict,
    session_ref,
    *,
    subject: int,
    env: dict[str, str],
    derivatives_root: Path,
    rawdata_root: Path,
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
    qc_lookup = dict(
        subject=subject, session=session_ref.ses, date=session_ref.date,
        env=env, since=wall_start,
    )
    try:
        run_stages(
            dict(
                subject=normalize_subjid(subject),
                **session_selector(session_ref),
                **options,
            ),
            stages=args.stage, model=args.model, overwrite=args.overwrite,
            view=False, extra=extra,
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
            _read_qc_verdict(derivatives_root, outcome, qc_lookup)
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
        _erase_failed_session(
            args, outcome, derivatives_root, rawdata_root, env, subject=subject
        )
    else:
        verdict = _read_qc_verdict(derivatives_root, outcome, qc_lookup)
        if verdict is not None:
            outcome.qc_status, outcome.qc_failed_sections = verdict
    outcome.duration_seconds = time.monotonic() - clock
    return outcome


def _is_qc_verdict(exc: Exception) -> bool:
    """Whether a failure could be `summary_qc` reporting FAIL (it exits 1 for that)."""
    return isinstance(exc, StepFailed) and exc.label == "qc:summary" and exc.returncode == 1


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
    args: argparse.Namespace,
    outcome: SessionOutcome,
    derivatives_root: Path,
    rawdata_root: Path,
    env: dict[str, str],
    *,
    subject: int,
) -> None:
    """Erase a failed session's outputs, unless `--keep-failed`, recording what went."""
    if args.keep_failed:
        print(f"Keeping the outputs of {outcome.label} (--keep-failed).")
        return
    try:
        outcome.erased = erase_session_outputs(
            derivatives_root,
            subject=subject,
            session=outcome.session,
            date=outcome.date,
            env=env,
            rawdata_root=rawdata_root,
            erase_derived_edf=args.erase_derived_edf,
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
