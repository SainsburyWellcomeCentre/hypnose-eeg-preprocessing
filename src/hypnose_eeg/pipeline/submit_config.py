"""Read a `slurm/submit.sh --config` YAML file and merge it with the command line.

The file has two sections (see `slurm/submit.yaml`):

  sbatch:    the SLURM resources submit.sh accepts as flags -- time, mem,
             cpus_per_task, partition, max_running
  pipeline:  `hypnose-eeg-pipeline` options by long name (`all_sessions` or
             `all-sessions` for `--all-sessions`): true for a flag, a list for
             several values; false or null leaves an option out. `extra_args`
             is a list of raw arguments passed on as they are, for options
             only a stage understands.

Arguments typed on the command line win: a pipeline option given there
replaces the file's value for it, and also drops the file's values for options
it cannot be combined with (`--session 2` replaces `all_sessions: true`). A
flag set to true in the file cannot be switched off from the command line;
edit the file, or use another one, for that.

Run by submit.sh as `python -m hypnose_eeg.pipeline.submit_config FILE --
PIPELINE_ARGS`, it prints shell assignments for submit.sh to `eval`: the file's
sbatch values as `CFG_<KEY>` and the merged pipeline arguments as the array
`CFG_PIPELINE_ARGS`. Every value is quoted with `shlex.quote`.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hypnose_eeg.utils.config import load_config

# The submit.sh resource flags a config file may set, by YAML key.
SBATCH_KEYS = ("time", "mem", "cpus_per_task", "partition", "max_running")

# Raw pipeline arguments, passed on unchecked.
EXTRA_ARGS_KEY = "extra_args"


@dataclass
class SubmitConfig:
    """A config file's sbatch values and pipeline arguments."""

    sbatch: dict[str, str] = field(default_factory=dict)
    pipeline: list[tuple[argparse.Action, list[str]]] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)


def read_submit_config(path: str | Path, parser: argparse.ArgumentParser) -> SubmitConfig:
    """Read and check a submit config against the pipeline's `parser`."""
    raw = load_config(path)
    unknown = set(raw) - {"sbatch", "pipeline"}
    if unknown:
        raise ValueError(
            f"{path}: unknown section(s) {', '.join(sorted(unknown))}; "
            "expected sbatch and pipeline"
        )
    return SubmitConfig(
        sbatch=_sbatch_values(path, raw.get("sbatch") or {}),
        pipeline=_pipeline_options(path, raw.get("pipeline") or {}, parser),
        extra_args=_extra_args(path, (raw.get("pipeline") or {}).get(EXTRA_ARGS_KEY)),
    )


def merge_pipeline_args(
    config: SubmitConfig, cli_args: Sequence[str], parser: argparse.ArgumentParser
) -> list[str]:
    """The config's pipeline arguments, minus those the command line replaces, then the command line's."""
    given = _given_actions(cli_args, parser)
    replaced = set(given)
    for group in parser._mutually_exclusive_groups:  # noqa: SLF001 - argparse has no public API
        if given & set(group._group_actions):  # noqa: SLF001
            replaced.update(group._group_actions)  # noqa: SLF001
    merged: list[str] = []
    for action, values in config.pipeline:
        if action not in replaced:
            merged += [action.option_strings[0], *values]
    return merged + config.extra_args + list(cli_args)


def _sbatch_values(path: str | Path, section: Any) -> dict[str, str]:
    if not isinstance(section, dict):
        raise ValueError(f"{path}: sbatch must be a mapping")
    values: dict[str, str] = {}
    for key, value in section.items():
        name = str(key).replace("-", "_")
        if name not in SBATCH_KEYS:
            raise ValueError(
                f"{path}: unknown sbatch key {key!r}; expected one of {', '.join(SBATCH_KEYS)}"
            )
        if value is None:
            continue
        if name == "time" and not isinstance(value, str):
            # YAML reads an unquoted 24:00:00 as the number 86400, which
            # sbatch would take as minutes.
            raise ValueError(f'{path}: quote sbatch time, for example time: "24:00:00"')
        values[name] = str(value)
    return values


def _pipeline_options(
    path: str | Path, section: Any, parser: argparse.ArgumentParser
) -> list[tuple[argparse.Action, list[str]]]:
    if not isinstance(section, dict):
        raise ValueError(f"{path}: pipeline must be a mapping")
    options: list[tuple[argparse.Action, list[str]]] = []
    for key, value in section.items():
        if key == EXTRA_ARGS_KEY:
            continue
        option = "--" + str(key).replace("_", "-")
        action = parser._option_string_actions.get(option)  # noqa: SLF001
        if action is None or option in ("--help", "-h"):
            raise ValueError(
                f"{path}: unknown pipeline option {key!r} (no {option}); use "
                f"{EXTRA_ARGS_KEY} for options only a stage understands"
            )
        if value is None or value is False:
            continue
        if value is True:
            if action.nargs != 0:
                raise ValueError(f"{path}: pipeline {key} takes a value, not true")
            options.append((action, []))
            continue
        if action.nargs == 0:
            raise ValueError(f"{path}: pipeline {key} is a flag; set it to true or false")
        values = value if isinstance(value, list) else [value]
        if not values:
            continue
        options.append((action, [str(item) for item in values]))
    return options


def _extra_args(path: str | Path, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: pipeline {EXTRA_ARGS_KEY} must be a list of arguments")
    return [str(item) for item in value]


def _given_actions(
    cli_args: Sequence[str], parser: argparse.ArgumentParser
) -> set[argparse.Action]:
    """The pipeline options named on the command line."""
    actions = parser._option_string_actions  # noqa: SLF001
    return {
        actions[name]
        for name in (arg.split("=", 1)[0] for arg in cli_args if arg.startswith("-"))
        if name in actions
    }


def _shell_assignments(config: SubmitConfig, pipeline_args: list[str]) -> str:
    lines = [
        f"CFG_{key.upper()}={shlex.quote(config.sbatch.get(key, ''))}" for key in SBATCH_KEYS
    ]
    lines.append(
        "CFG_PIPELINE_ARGS=(" + " ".join(shlex.quote(arg) for arg in pipeline_args) + ")"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from hypnose_eeg.pipeline.run import build_parser

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        print("usage: python -m hypnose_eeg.pipeline.submit_config FILE [-- PIPELINE_ARGS]", file=sys.stderr)
        return 2
    path, cli_args = args[0], args[1:]
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]
    parser = build_parser()
    try:
        config = read_submit_config(path, parser)
    except (OSError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(_shell_assignments(config, merge_pipeline_args(config, cli_args, parser)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
