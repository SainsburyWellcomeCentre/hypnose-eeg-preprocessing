"""Run the Hypnose preprocessing pipeline: trim, concatenate, downsample, prescan_artifacts, detect_artifacts.

Each step wraps its matching `hypnose_eeg/preprocessing/*.py` CLI (run as a
subprocess) rather than duplicating that logic here. Run this module directly
for just the preprocessing stage, or use `src/run_pipeline.py` for the full
preprocessing -> sleep_scoring -> qc pipeline.

`trim` (`trim_duplicate_channels.py`) runs first so that concatenation and
everything after it see duplicate-free recordings: it writes a `_trimmed` copy
of any source EDF whose header repeats a channel label and leaves clean
recordings untouched, so it is a no-op for most sessions. Its manual
`--keep-first N` mode is not part of the pipeline; run the script directly for
that.

`prescan_artifacts` scans the downsampled FIF for long periods of dead or
extreme signal *before* sleep scoring, which then leaves them unscored.
`detect_artifacts` needs the sleep-scoring output, so a full run
(`src/run_pipeline.py`) runs it after scoring.

Unrecognized arguments are forwarded verbatim to every selected step (for
example `--target-sfreq 256` for `downsample`); step-specific flags that only
some steps understand, such as `--config`, are best passed by running that
step's own script directly instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src._pipeline import (
    StepFailed,
    add_selector_arguments,
    output_layout_env,
    output_dir_overrides,
    run_step,
)

STEP_ORDER = ["trim", "concatenate", "downsample", "prescan_artifacts", "detect_artifacts"]
STEP_MODULES = {
    "trim": "hypnose_eeg.preprocessing.trim_duplicate_channels",
    "concatenate": "hypnose_eeg.preprocessing.concatenate_recordings",
    "downsample": "hypnose_eeg.preprocessing.downsample_recordings",
    "prescan_artifacts": "hypnose_eeg.preprocessing.prescan_artifacts",
    "detect_artifacts": "hypnose_eeg.preprocessing.detect_artifacts",
}
DEFAULT_STEPS = list(STEP_ORDER)
# Steps that write artifact outputs rather than recordings: they take the
# derivatives root and have no --dry-run.
ARTIFACT_STEPS = ("prescan_artifacts", "detect_artifacts")


def _step_args(
    step: str,
    *,
    subject: str | None,
    date: str | None,
    session: str | None,
    rawdata_root: str | None,
    derivatives_root: str | None,
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    args: list[str] = []
    if subject:
        args += ["--subject", subject]
    if date:
        args += ["--date", date]
    elif session:
        args += ["--session", session]

    if step in ("concatenate", "downsample"):
        if rawdata_root:
            args += ["--source-dir", rawdata_root]
        if derivatives_root:
            args += ["--sink-dir", derivatives_root]
    else:
        if rawdata_root:
            args += ["--rawdata-root", rawdata_root]
        if derivatives_root and step in ARTIFACT_STEPS:
            args += ["--derivatives-root", derivatives_root]
    if dry_run and step not in ARTIFACT_STEPS:
        args.append("--dry-run")

    if overwrite:
        args.append("--overwrite")
    return args


def run_steps(
    *,
    subject: str | None,
    date: str | None = None,
    session: str | None = None,
    rawdata_root: str | None = None,
    derivatives_root: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    steps: list[str] | None = None,
    extra_args: list[str] | None = None,
    output_layout: str | Path | None = None,
    output_dirs: dict[str, str] | None = None,
    output_root: str | None = None,
) -> None:
    """Run the selected preprocessing steps, in pipeline order.

    `output_layout`/`output_root`/`output_dirs` relocate named output folders
    within each session's derivatives directory (see
    `src._pipeline.output_layout_env`); leave them unset to keep the
    repository's `configs/output_layout.yaml`.
    """
    selected = steps if steps is not None else DEFAULT_STEPS
    extra_args = list(extra_args or [])
    env = output_layout_env(output_layout, output_dirs, output_root)
    for step in STEP_ORDER:
        if step not in selected:
            continue
        args = _step_args(
            step,
            subject=subject,
            date=date,
            session=session,
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            overwrite=overwrite,
            dry_run=dry_run,
        ) + extra_args
        run_step(STEP_MODULES[step], args, label=f"preprocessing:{step}", env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline stage.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selector_arguments(parser)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview matching sessions/files without writing (trim/concatenate/downsample only).",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, default=None,
        help=f"Steps to run, in pipeline order (default: {' '.join(DEFAULT_STEPS)}).",
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
            dry_run=args.dry_run,
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
