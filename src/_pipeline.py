"""Shared subprocess-orchestration helpers for the src/ pipeline entry points.

`preprocessing.py`, `sleep_scoring.py`, and `qc.py` each run the existing
`scripts/` CLIs as subprocesses (`python -m scripts....`) instead of
reimplementing their logic, so this module holds the bits they all share:
running one step and reporting a clear failure, and the `--subject`/
`--date`/`--session`/`--rawdata-root`/`--derivatives-root` selector flags
common to nearly every wrapped script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


class StepFailed(RuntimeError):
    """A wrapped `scripts/` CLI exited with a non-zero status."""

    def __init__(self, label: str, module: str, returncode: int) -> None:
        super().__init__(f"{label} ({module}) exited with status {returncode}")
        self.label = label
        self.module = module
        self.returncode = returncode


def run_step(module: str, args: Sequence[str], *, label: str) -> None:
    """Run `python -m <module> <args>` from the repo root; raise on failure."""
    print(f"==> {label}: python -m {module} {' '.join(args)}")
    result = subprocess.run([sys.executable, "-m", module, *args], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise StepFailed(label, module, result.returncode)


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
