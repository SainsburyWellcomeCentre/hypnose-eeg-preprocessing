"""Shared subprocess-orchestration helpers for the src/ pipeline entry points.

`preprocessing.py`, `sleep_scoring.py`, and `qc.py` each run the existing
`scripts/` CLIs as subprocesses (`python -m scripts....`) instead of
reimplementing their logic, so this module holds the bits they all share:
running one step and reporting a clear failure, the `--subject`/
`--date`/`--session`/`--rawdata-root`/`--derivatives-root` selector flags
common to nearly every wrapped script, and the `--output-layout`/
`--output-dir` overrides that relocate named output folders within each
session's derivatives directory.

The output-folder overrides reach the wrapped scripts through the
`HYPNOSE_EEG_OUTPUT_LAYOUT`/`HYPNOSE_EEG_OUTPUT_DIR_<GROUP>` environment
variables (`scripts/io/output_layout.py`) rather than per-script flags, since
every script already resolves its folders through `output_dir_name()` and the
data-location roots are overridable the same way.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from scripts.io.output_layout import (
    DEFAULT_OUTPUT_DIR_NAMES,
    OUTPUT_LAYOUT_ENV,
    output_dir_env_var,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class StepFailed(RuntimeError):
    """A wrapped `scripts/` CLI exited with a non-zero status."""

    def __init__(self, label: str, module: str, returncode: int) -> None:
        super().__init__(f"{label} ({module}) exited with status {returncode}")
        self.label = label
        self.module = module
        self.returncode = returncode


def run_step(
    module: str,
    args: Sequence[str],
    *,
    label: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run `python -m <module> <args>` from the repo root; raise on failure.

    `env` holds extra environment variables layered over the current process
    environment for just this step (used for the output-folder overrides).
    """
    print(f"==> {label}: python -m {module} {' '.join(args)}")
    step_env = {**os.environ, **env} if env else None
    result = subprocess.run(
        [sys.executable, "-m", module, *args], cwd=REPO_ROOT, env=step_env
    )
    if result.returncode != 0:
        raise StepFailed(label, module, result.returncode)


def output_layout_env(
    output_layout: str | Path | None = None,
    output_dirs: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables carrying output-folder overrides to a wrapped script.

    `output_layout` is a path to an alternative `output_layout.yaml`;
    `output_dirs` maps output-group keys (`artifacts`, `sleep_scoring`, ...) to
    folders relative to the session directory. Either may be omitted, and an
    empty mapping is returned when neither is given, so the wrapped script
    falls back to the repository's own layout.
    """
    env: dict[str, str] = {}
    if output_layout:
        env[OUTPUT_LAYOUT_ENV] = str(output_layout)
    for key, folder in (output_dirs or {}).items():
        if key not in DEFAULT_OUTPUT_DIR_NAMES:
            raise ValueError(
                f"Unknown output group {key!r}; expected one of "
                f"{sorted(DEFAULT_OUTPUT_DIR_NAMES)}"
            )
        env[output_dir_env_var(key)] = str(folder)
    return env


def parse_output_dir_overrides(values: Sequence[str] | None) -> dict[str, str]:
    """Turn repeated `--output-dir GROUP=FOLDER` values into a `{group: folder}` mapping."""
    overrides: dict[str, str] = {}
    for value in values or ():
        key, separator, folder = value.partition("=")
        key = key.strip()
        folder = folder.strip()
        if not separator or not key or not folder:
            raise ValueError(f"--output-dir expects GROUP=FOLDER, got {value!r}")
        if key not in DEFAULT_OUTPUT_DIR_NAMES:
            raise ValueError(
                f"--output-dir group {key!r} is not one of {sorted(DEFAULT_OUTPUT_DIR_NAMES)}"
            )
        overrides[key] = folder
    return overrides


def output_dir_overrides(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[str, str]:
    """`parse_output_dir_overrides` for parsed `--output-dir` flags, failing as a usage error."""
    try:
        return parse_output_dir_overrides(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))


def add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the subject/date/session/rawdata-root/derivatives-root flags."""
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


def add_output_layout_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags that relocate named output folders within a session."""
    parser.add_argument(
        "--output-layout", default=None, metavar="FILE",
        help="Alternative output_layout.yaml naming the per-session output folders "
        "(default: configs/output_layout.yaml).",
    )
    parser.add_argument(
        "--output-dir", action="append", default=None, metavar="GROUP=FOLDER",
        help="Relocate one output group within the session directory, for example "
        "artifacts=analysis/artifacts; repeatable. Groups: "
        f"{', '.join(sorted(DEFAULT_OUTPUT_DIR_NAMES))}.",
    )
