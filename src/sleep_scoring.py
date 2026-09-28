"""Run the Hypnose sleep-scoring pipeline: score, and optionally view.

`score` wraps `hypnose_eeg/sleep_scoring/score_recordings.py` and is the default
step. `view` wraps the interactive `hypnose_eeg/sleep_scoring/view_scoring.py`
viewer; it is excluded by default because it opens a plot window rather than
running unattended, but remains available via `--steps view`.

A recording whose predictions parquet already exists is skipped rather than
rescored, so a partially-completed session can be resumed; pass `--overwrite`
to rescore it regardless.

Unrecognized arguments are forwarded verbatim to every selected step (for
example `--hours 3 6` for `view`, or `--channel-labels ...` for `score`).
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

STEP_ORDER = ["score", "view"]
STEP_MODULES = {
    "score": "hypnose_eeg.sleep_scoring.score_recordings",
    "view": "hypnose_eeg.sleep_scoring.view_scoring",
}
DEFAULT_STEPS = ["score"]


def _step_args(
    step: str,
    *,
    subject: str | None,
    date: str | None,
    session: str | None,
    rawdata_root: str | None,
    derivatives_root: str | None,
    model: str | None,
    overwrite: bool,
) -> list[str]:
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
    if step == "score" and model:
        args += ["--model", model]
    if step == "score" and overwrite:
        args.append("--overwrite")
    return args


def run_steps(
    *,
    subject: str | None,
    date: str | None = None,
    session: str | None = None,
    rawdata_root: str | None = None,
    derivatives_root: str | None = None,
    model: str | None = None,
    overwrite: bool = False,
    steps: list[str] | None = None,
    extra_args: list[str] | None = None,
    output_layout: str | Path | None = None,
    output_dirs: dict[str, str] | None = None,
    output_root: str | None = None,
) -> None:
    """Run the selected sleep-scoring steps, in order.

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
            model=model,
            overwrite=overwrite,
        ) + extra_args
        run_step(STEP_MODULES[step], args, label=f"sleep_scoring:{step}", env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sleep-scoring pipeline stage.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selector_arguments(parser)
    parser.add_argument(
        "--model", "--model-path", dest="model", default=None,
        help="Somnotate model name or model.pickle path (forwarded to the score step).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rescore recordings whose predictions already exist (default: skip them).",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, default=None,
        help=f"Steps to run (default: {' '.join(DEFAULT_STEPS)}; view opens an "
        "interactive window).",
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
            model=args.model,
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
