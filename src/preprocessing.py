"""Run the Hypnose preprocessing pipeline: concatenate, trim, downsample, detect_artifacts.

Each step wraps its matching `scripts/preprocessing/*.py` CLI (run as a
subprocess) rather than duplicating that logic here. Run this module directly
for just the preprocessing stage, or use `src/run_pipeline.py` for the full
preprocessing -> sleep_scoring -> qc pipeline.

`trim` (`inspect_and_trim_channels.py`) is deliberately excluded from the
default step set: it is a manual remediation step for channel-count errors
surfaced by concatenation, not a routine stage (see `docs/README_TODO.md`).
Select it explicitly with `--steps trim` when needed, and note that it only
inspects unless `--write-trimmed` is also forwarded.

Unrecognized arguments are forwarded verbatim to every selected step (for
example `--target-sfreq 256` for `downsample`); step-specific flags that only
some steps understand, such as `--config`, are best passed by running that
step's own script directly instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src._pipeline import StepFailed, add_selector_arguments, run_step

STEP_ORDER = ["concatenate", "trim", "downsample", "detect_artifacts"]
STEP_MODULES = {
    "concatenate": "scripts.preprocessing.concatenate_recordings",
    "trim": "scripts.preprocessing.inspect_and_trim_channels",
    "downsample": "scripts.preprocessing.downsample_recordings",
    "detect_artifacts": "scripts.preprocessing.detect_artifacts",
}
DEFAULT_STEPS = ["concatenate", "downsample", "detect_artifacts"]


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
        if dry_run:
            args.append("--dry-run")
    else:
        if rawdata_root:
            args += ["--rawdata-root", rawdata_root]
        if derivatives_root and step == "detect_artifacts":
            args += ["--derivatives-root", derivatives_root]

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
) -> None:
    """Run the selected preprocessing steps, in pipeline order."""
    selected = steps if steps is not None else DEFAULT_STEPS
    extra_args = list(extra_args or [])
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
        run_step(STEP_MODULES[step], args, label=f"preprocessing:{step}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline stage.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selector_arguments(parser)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview matching sessions/files without writing (concatenate/downsample only).",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, default=None,
        help=f"Steps to run, in pipeline order (default: {' '.join(DEFAULT_STEPS)}; "
        "trim is manual/on-demand).",
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
        )
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
