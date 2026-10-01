"""Shared step-running helpers for the pipeline stages.

`preprocessing.py`, `sleep_scoring.py`, and `qc.py` each run the existing
`hypnose_eeg` CLIs by calling their `main(argv)` in this process, instead of
reimplementing their logic, so this module holds the bits they all share:
running one step and reporting a clear failure, the `--subject`/
`--date`/`--session`/`--rawdata-root`/`--derivatives-root` selector flags
common to nearly every wrapped script, and the `--output-layout`/
`--output-dir` overrides that relocate named output folders within each
session's derivatives directory.

The output-folder overrides reach the wrapped scripts through the
`HYPNOSE_EEG_OUTPUT_LAYOUT`/`HYPNOSE_EEG_OUTPUT_ROOT`/
`HYPNOSE_EEG_OUTPUT_DIR_<GROUP>` environment variables
(`hypnose_eeg/io/output_layout.py`) rather than per-script flags, since
every script already resolves its folders through `output_dir_name()` and the
data-location roots are overridable the same way. They are applied to
`os.environ` for the duration of one step and restored afterwards.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from hypnose_eeg.io.output_layout import (
    DEFAULT_OUTPUT_DIR_NAMES,
    DEFAULT_OUTPUT_ROOT,
    OUTPUT_LAYOUT_ENV,
    OUTPUT_ROOT_ENV,
    output_dir_env_var,
)


class StepFailed(RuntimeError):
    """A wrapped `hypnose_eeg` CLI returned or exited with a non-zero status.

    `error` is the exception the step raised, or None when it reported the
    failure through its exit status (a parser error, or a QC verdict such as
    `summary_qc` exiting 1 for FAIL).
    """

    def __init__(
        self,
        label: str,
        module: str,
        returncode: int,
        error: BaseException | None = None,
    ) -> None:
        if error is None:
            message = f"{label} ({module}) exited with status {returncode}"
        else:
            message = f"{label} ({module}) raised {type(error).__name__}: {error}"
        super().__init__(message)
        self.label = label
        self.module = module
        self.returncode = returncode
        self.error = error


def run_step(
    module: str,
    args: Sequence[str],
    *,
    label: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run `<module>.main(args)` in this process; raise `StepFailed` on failure.

    `env` holds extra environment variables applied over the current process
    environment for just this step (used for the output-folder overrides).
    A `SystemExit` from the step -- argparse usage errors included -- becomes
    its exit status, and any other exception is printed and reported as status
    1, which is what the step would have exited with on its own.
    `KeyboardInterrupt` is left to propagate, since it stops the run, not the
    step.
    """
    print(f"==> {label}: {module} {' '.join(args)}", flush=True)
    entry_point = importlib.import_module(module).main
    try:
        with applied_env(env or {}):
            try:
                status = exit_status(entry_point(list(args)))
            except SystemExit as exc:
                status = exit_status(exc.code)
            except Exception as exc:  # noqa: BLE001 - reported as the step's failure
                traceback.print_exc()
                raise StepFailed(label, module, 1, error=exc) from exc
    finally:
        _close_figures()
    if status != 0:
        raise StepFailed(label, module, status)


def exit_status(code: object) -> int:
    """The process exit status `sys.exit(code)` would produce for a `main()` result."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # sys.exit("message") prints the message and exits 1.
    print(code, file=sys.stderr)
    return 1


@contextmanager
def applied_env(env: Mapping[str, str]) -> Iterator[None]:
    """Apply environment overrides to `os.environ` for a block, then restore them."""
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _close_figures() -> None:
    """Close the figures a step left open, so a long batch does not accumulate them.

    A step that shows its plots has already blocked until they were closed;
    one run with `--no-show` has saved them. Either way they are done with.
    """
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


def output_layout_env(
    output_layout: str | Path | None = None,
    output_dirs: Mapping[str, str] | None = None,
    output_root: str | None = None,
) -> dict[str, str]:
    """Environment variables carrying output-folder overrides to a wrapped script.

    `output_layout` is a path to an alternative `output_layout.yaml`;
    `output_root` is the modality folder every group sits below within the
    session directory (`eeg` by default, `.` for none); `output_dirs` maps
    output-group keys (`artifacts`, `sleep_scoring`, ...) to folders relative
    to that root. Any may be omitted, and an empty mapping is returned when
    none is given, so the wrapped script falls back to the repository's own
    layout.
    """
    env: dict[str, str] = {}
    if output_layout:
        env[OUTPUT_LAYOUT_ENV] = str(output_layout)
    if output_root is not None:
        env[OUTPUT_ROOT_ENV] = str(output_root)
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
        "--output-root", default=None, metavar="FOLDER",
        help="Modality folder every output group sits below within the session "
        f"directory (default: {DEFAULT_OUTPUT_ROOT}); pass . for none.",
    )
    parser.add_argument(
        "--output-dir", action="append", default=None, metavar="GROUP=FOLDER",
        help="Relocate one output group below the output root, for example "
        "artifacts=analysis/artifacts; repeatable. Groups: "
        f"{', '.join(sorted(DEFAULT_OUTPUT_DIR_NAMES))}.",
    )
