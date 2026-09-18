"""Configurable folder names for named derivative outputs.

Every script that writes into (or reads from) a named derivative subfolder --
`downsample`, `sleep_scoring`, `artifacts`, `quality_control` -- resolves that
folder's name through `output_dir_name()` instead of hardcoding the string, so
renaming one is a `configs/output_layout.yaml` edit rather than a
grep-and-replace across scripts/.

Each folder always sits below the recording's derivatives session directory
(`<derivatives>/sub-XXX/ses-YYY_date-.../`), which is itself resolved from the
active data-location profile -- see `scripts/io/repository_paths.py`. What can
be changed is *where within that session directory* each output group lands.
Resolution order for every group (highest priority first):

1. ``HYPNOSE_EEG_OUTPUT_DIR_<GROUP>`` -- one folder, e.g.
   ``HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS=analysis/artifacts``.
2. ``HYPNOSE_EEG_OUTPUT_LAYOUT`` -- a path to an alternative layout YAML with
   the same ``output_dirs:`` shape as `configs/output_layout.yaml`, for callers
   running this repository from outside it.
3. the committed `configs/output_layout.yaml`.
4. the built-in defaults below.

A configured value may be a nested relative path (``analysis/artifacts``); it
may not be absolute or step outside the session directory with ``..``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from scripts.utils.config import DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH, load_config, nested_get

# Matches the `env_prefix` in configs/data_locations.yml, so every override
# this repository honours shares one `HYPNOSE_EEG_*` namespace.
ENV_PREFIX = "HYPNOSE_EEG"
OUTPUT_LAYOUT_ENV = f"{ENV_PREFIX}_OUTPUT_LAYOUT"
OUTPUT_DIR_ENV_TEMPLATE = f"{ENV_PREFIX}_OUTPUT_DIR_{{group}}"

DEFAULT_OUTPUT_DIR_NAMES = {
    "downsample": "downsample",
    "sleep_scoring": "sleep_scoring",
    "sleep_scoring_qc": "sleep_scoring_qc",
    "artifacts": "artifacts",
    "quality_control": "quality_control",
}


def output_dir_env_var(key: str) -> str:
    """The environment variable overriding one output group, e.g. `HYPNOSE_EEG_OUTPUT_DIR_ARTIFACTS`."""
    _check_key(key)
    return OUTPUT_DIR_ENV_TEMPLATE.format(group=key.upper())


def output_layout_config_path() -> Path:
    """The layout YAML in effect: `HYPNOSE_EEG_OUTPUT_LAYOUT` if set, else the repository's."""
    configured = os.getenv(OUTPUT_LAYOUT_ENV)
    if not configured:
        return DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH
    path = Path(os.path.expanduser(os.path.expandvars(configured)))
    if not path.is_file():
        # An explicit override that points nowhere is a mistake worth stopping
        # on, not something to silently paper over with the repository layout.
        raise FileNotFoundError(f"{OUTPUT_LAYOUT_ENV} points to a missing file: {path}")
    return path


def output_dir_name(key: str) -> str:
    """The configured folder for one output group, relative to the session directory.

    Usually a bare folder name (`output_dir_name("artifacts") == "artifacts"`),
    but may be a nested relative path when configured that way; join it onto
    the session directory with `/` either way.
    """
    _check_key(key)
    env_value = os.getenv(output_dir_env_var(key))
    if env_value:
        return _validate_output_dir(key, env_value, source=output_dir_env_var(key))

    config_path = output_layout_config_path()
    config = load_config(config_path)
    value = nested_get(config, ("output_dirs", key), DEFAULT_OUTPUT_DIR_NAMES[key])
    return _validate_output_dir(key, value, source=str(config_path))


def output_dir_names() -> dict[str, str]:
    """Every output group's folder, resolved through the same precedence as `output_dir_name`."""
    return {key: output_dir_name(key) for key in DEFAULT_OUTPUT_DIR_NAMES}


def _check_key(key: str) -> None:
    if key not in DEFAULT_OUTPUT_DIR_NAMES:
        raise ValueError(
            f"Unknown output-directory key {key!r}; expected one of "
            f"{sorted(DEFAULT_OUTPUT_DIR_NAMES)}"
        )


def _validate_output_dir(key: str, value: object, *, source: str) -> str:
    """Normalise a configured folder to a relative POSIX path inside the session directory."""
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Output directory for {key!r} is empty (from {source})")
    # Accept either separator so a Windows-authored config still resolves, but
    # refuse anything that would escape the session directory.
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(
            f"Output directory for {key!r} must be relative to the session directory, "
            f"got {text!r} (from {source})"
        )
    parts = [part for part in PureWindowsPath(text).parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Output directory for {key!r} is empty (from {source})")
    if ".." in parts:
        raise ValueError(
            f"Output directory for {key!r} cannot traverse outside the session directory, "
            f"got {text!r} (from {source})"
        )
    return PurePosixPath(*parts).as_posix()
