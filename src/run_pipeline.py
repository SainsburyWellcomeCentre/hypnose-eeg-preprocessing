"""Run the full Hypnose pipeline: preprocessing -> sleep_scoring -> qc.

Each stage is one of `src/preprocessing.py`, `src/sleep_scoring.py`, or
`src/qc.py`; see those modules for their own step lists and defaults. With no
`--stage` given, all three run in the order the pipeline actually requires --
artifact detection depends on sleep scoring, so it is deferred until after
scoring even though it is a preprocessing step:

  1. preprocessing: trim, concatenate, downsample
  2. sleep_scoring: score
  3. preprocessing: detect_artifacts
  4. qc: summary

Restricting to one or more `--stage` values runs each named stage's own
default step set instead (for example `--stage preprocessing` runs
trim, concatenate, downsample, and detect_artifacts together, which assumes sleep
scoring has already been done for that session).

A step whose outputs already exist is skipped and the run continues with the
next one, so an interrupted or partially-completed session can be resumed by
rerunning the same command. Pass `--overwrite` to recompute regardless.

Sleep scoring, artifact detection, and the QC summary each write a
`<output stem>_provenance.json` sidecar naming the git commit that produced the
output -- and, for sleep scoring, which model was used.

Unrecognized arguments are forwarded verbatim to every stage that runs, so a
flag understood by only one stage (`--model` aside, which this script does
know about) is best passed by invoking that stage's script directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing, qc, sleep_scoring
from src._pipeline import (
    StepFailed,
    add_output_layout_arguments,
    output_dir_overrides,
)

STAGE_ORDER = ["preprocessing", "sleep_scoring", "qc"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Hypnose EEG pipeline.", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", "--subjid", dest="subject", required=True,
        help="Subject ID, for example 66 or sub-066.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
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
        "--stage", nargs="+", choices=STAGE_ORDER, default=None,
        help="Restrict the run to these stages, each using its own default step set "
        "(default: all three, in dependency order).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    common = dict(
        subject=args.subject,
        date=args.date,
        session=args.session,
        rawdata_root=args.rawdata_root,
        derivatives_root=args.derivatives_root,
        output_layout=args.output_layout,
        output_dirs=output_dir_overrides(parser, args),
    )
    stages = args.stage

    try:
        if stages is None:
            preprocessing.run_steps(
                **common, overwrite=args.overwrite,
                steps=["trim", "concatenate", "downsample"], extra_args=extra,
            )
            sleep_scoring.run_steps(
                **common, model=args.model, overwrite=args.overwrite,
                steps=["score"], extra_args=extra,
            )
            preprocessing.run_steps(
                **common, overwrite=args.overwrite,
                steps=["detect_artifacts"], extra_args=extra,
            )
            qc.run_steps(**common, steps=["summary"], extra_args=extra)
        else:
            if "preprocessing" in stages:
                preprocessing.run_steps(**common, overwrite=args.overwrite, extra_args=extra)
            if "sleep_scoring" in stages:
                sleep_scoring.run_steps(
                    **common, model=args.model, overwrite=args.overwrite, extra_args=extra
                )
            if "qc" in stages:
                qc.run_steps(**common, extra_args=extra)
    except StepFailed as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
