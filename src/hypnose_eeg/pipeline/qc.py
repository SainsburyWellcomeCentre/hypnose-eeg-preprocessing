"""Run Hypnose session quality-control checks.

`summary` wraps `hypnose_eeg/qc/summary_qc.py`, which already runs every
session-level QC section (EDF/FIF integrity, Somnotate confidence, artifact
burden, EEG/EMG channel correlation, sleep-state power spectra, EMG RMS) and
prints one PASS/REVIEW/FAIL decision -- it is the default step. It also writes
both of its outputs into the session's quality-control directory on every run:
`qc_summary.csv` with the section results and `qc_review_epochs.csv` with the
unified review ranges, each with a `.parquet` copy.

`figures` (`hypnose_eeg/qc/review_figures.py`), the other default step, then
saves PNGs for review without needing a display, when the summary is not a
plain pass: an overview of the first 12 hours of the scored recording, and the
sleep-state power spectra and EMG RMS figures. A summary that FAILs stops
the stage before `figures`; rerun with `--steps figures` to draw them anyway.

Like the other stages, QC skips work already done: a session that already has
a QC summary skips `summary`, and `figures` with it -- they were drawn from
that summary -- reporting the verdict on disk instead (a stored FAIL still
fails the stage). Pass `--overwrite` to recompute them. `--steps figures` on
its own always draws, and the other steps always run.

The remaining steps wrap the individual `hypnose_eeg/qc/*.py` reports for when
a single section's plots or table are wanted on their own.

Unrecognized arguments are forwarded verbatim to every selected step (for
example `--no-summary`/`--no-review-epochs` to stop `summary` writing either
file, or `--no-show --save-dir` for `spectra`/`channel_correlations`, which
otherwise open plot windows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hypnose_helpers.io.selectors import parse_sessions

from hypnose_eeg.io.repository_paths import get_derivatives_root
from hypnose_eeg.pipeline.batch import QC_FAIL, read_qc_verdict
from hypnose_eeg.pipeline.steps import (
    StepFailed,
    add_selector_arguments,
    output_layout_env,
    output_dir_overrides,
    run_step,
)

STEP_ORDER = [
    "integrity", "sleep_scoring", "spectra", "channel_correlations", "artifacts",
    "summary", "figures",
]
STEP_MODULES = {
    "integrity": "hypnose_eeg.qc.recording_integrity",
    "sleep_scoring": "hypnose_eeg.qc.sleep_scoring",
    "spectra": "hypnose_eeg.qc.spectra",
    "channel_correlations": "hypnose_eeg.qc.channel_correlations",
    "artifacts": "hypnose_eeg.qc.artifacts",
    "summary": "hypnose_eeg.qc.summary_qc",
    "figures": "hypnose_eeg.qc.review_figures",
}
# `figures` reads the verdict `summary` writes, so it runs after it.
DEFAULT_STEPS = ["summary", "figures"]


def run_steps(
    *,
    subject: str | None,
    date: str | None = None,
    session: str | None = None,
    rawdata_root: str | None = None,
    derivatives_root: str | None = None,
    overwrite: bool = False,
    steps: list[str] | None = None,
    extra_args: list[str] | None = None,
    output_layout: str | Path | None = None,
    output_dirs: dict[str, str] | None = None,
    output_root: str | None = None,
) -> None:
    """Run the selected QC steps, in order.

    Unless `overwrite`, a session that already has a QC summary skips the
    `summary` step and the `figures` drawn from it; a stored FAIL raises
    `StepFailed` as the summary step would have.

    `output_layout`/`output_root`/`output_dirs` relocate named output folders
    within each session's derivatives directory (see
    `hypnose_eeg.pipeline.steps.output_layout_env`); leave them unset to keep the
    repository's `configs/output_layout.yaml`.
    """
    selected = steps if steps is not None else DEFAULT_STEPS
    extra_args = list(extra_args or [])
    env = output_layout_env(output_layout, output_dirs, output_root)
    stored_status = None
    if not overwrite and "summary" in selected:
        stored_status = _stored_qc_status(
            subject, session=session, date=date,
            derivatives_root=derivatives_root, env=env,
        )
    for step in STEP_ORDER:
        if step not in selected:
            continue
        if stored_status is not None and step in ("summary", "figures"):
            if step == "summary":
                print(
                    f"==> qc:summary skipped: a QC summary already exists "
                    f"({stored_status.upper()}); pass --overwrite to recompute it "
                    "and its figures.",
                    flush=True,
                )
                if stored_status == QC_FAIL:
                    raise StepFailed(f"qc:{step}", STEP_MODULES[step], 1)
            continue
        args: list[str] = []
        if subject:
            args += ["--subject", subject]
        if date:
            args += ["--date", date]
        elif session:
            args += ["--session", session]
        if rawdata_root:
            args += ["--rawdata-root", rawdata_root]
        if derivatives_root:
            args += ["--derivatives-root", derivatives_root]
        args += extra_args
        run_step(STEP_MODULES[step], args, label=f"qc:{step}", env=env)


def _stored_qc_status(
    subject: str | None,
    *,
    session: str | None,
    date: str | None,
    derivatives_root: str | None,
    env: dict[str, str],
) -> str | None:
    """The overall status of the session's QC summary on disk, or None if it has none."""
    if not subject or not (session or date):
        return None
    try:
        verdict = read_qc_verdict(
            derivatives_root or get_derivatives_root(),
            subject=subject,
            session=None if session is None else parse_sessions([session])[0],
            date=date,
            env=env,
        )
    except (OSError, ValueError):
        # Unreadable or ambiguous: let the summary step run and say so itself.
        return None
    return None if verdict is None else verdict[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run session quality-control checks.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selector_arguments(parser)
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, default=None,
        help=f"QC sections to run (default: {' '.join(DEFAULT_STEPS)} -- the summary "
        "covers every section, then the review figures are saved).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute the QC summary and review figures of a session that "
        "already has them (default: skip them).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    try:
        run_steps(
            subject=args.subject,
            date=args.date,
            session=args.session,
            rawdata_root=args.rawdata_root,
            derivatives_root=args.derivatives_root,
            overwrite=args.overwrite,
            steps=args.steps,
            extra_args=extra,
            output_layout=args.output_layout,
            output_root=args.output_root,
            output_dirs=output_dir_overrides(parser, args),
        )
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
