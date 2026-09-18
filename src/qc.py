"""Run Hypnose session quality-control checks.

`summary` wraps `scripts/qc/summary_qc.py`, which already runs every
session-level QC section (EDF/FIF integrity, Somnotate confidence, artifact
burden, EEG/EMG channel correlation, sleep-state power spectra, EMG RMS) and
prints one PASS/REVIEW/FAIL decision -- it is the default step. It also writes
both of its outputs into the session's quality-control directory on every run:
`qc_summary.csv` with the section results, and `qc_review_epochs.csv` plus its
typed `.parquet` copy with the unified review ranges. The remaining steps wrap
the individual `scripts/qc/*.py` reports for when a single section's plots or
table are wanted on their own.

Unrecognized arguments are forwarded verbatim to every selected step (for
example `--no-summary`/`--no-review-epochs` to stop `summary` writing either
file, or `--no-show --save-dir` for `spectra`/`channel_correlations`, which
otherwise open plot windows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src._pipeline import StepFailed, add_selector_arguments, run_step

STEP_ORDER = ["integrity", "sleep_scoring", "spectra", "channel_correlations", "artifacts", "summary"]
STEP_MODULES = {
    "integrity": "scripts.qc.recording_integrity",
    "sleep_scoring": "scripts.qc.sleep_scoring",
    "spectra": "scripts.qc.spectra",
    "channel_correlations": "scripts.qc.channel_correlations",
    "artifacts": "scripts.qc.artifacts",
    "summary": "scripts.qc.summary_qc",
}
DEFAULT_STEPS = ["summary"]


def run_steps(
    *,
    subject: str | None,
    date: str | None = None,
    session: str | None = None,
    rawdata_root: str | None = None,
    derivatives_root: str | None = None,
    steps: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Run the selected QC steps, in order."""
    selected = steps if steps is not None else DEFAULT_STEPS
    extra_args = list(extra_args or [])
    for step in STEP_ORDER:
        if step not in selected:
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
        run_step(STEP_MODULES[step], args, label=f"qc:{step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run session quality-control checks.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selector_arguments(parser)
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, default=None,
        help=f"QC sections to run (default: {' '.join(DEFAULT_STEPS)}, which covers "
        "every section).",
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
            steps=args.steps,
            extra_args=extra,
        )
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
