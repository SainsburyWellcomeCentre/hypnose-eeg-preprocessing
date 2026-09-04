"""Configurable folder names for named derivative outputs.

Every script that writes into (or reads from) a named derivative subfolder --
`downsample`, `sleep_scoring`, `artifacts`, `quality_control` -- resolves that
folder's name through `output_dir_name()` instead of hardcoding the string, so
renaming one is a `configs/output_layout.yaml` edit rather than a
grep-and-replace across scripts/.
"""

from __future__ import annotations

from scripts.utils.config import DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH, load_config, nested_get

DEFAULT_OUTPUT_DIR_NAMES = {
    "downsample": "downsample",
    "sleep_scoring": "sleep_scoring",
    "artifacts": "artifacts",
    "quality_control": "quality_control",
}


def output_dir_name(key: str) -> str:
    """The configured folder name for one output group, e.g. `output_dir_name("artifacts")`."""
    if key not in DEFAULT_OUTPUT_DIR_NAMES:
        raise ValueError(
            f"Unknown output-directory key {key!r}; expected one of "
            f"{sorted(DEFAULT_OUTPUT_DIR_NAMES)}"
        )
    config = load_config(DEFAULT_OUTPUT_LAYOUT_CONFIG_PATH)
    return str(nested_get(config, ("output_dirs", key), DEFAULT_OUTPUT_DIR_NAMES[key]))
