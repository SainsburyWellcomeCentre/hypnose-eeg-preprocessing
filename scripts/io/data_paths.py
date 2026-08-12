"""Bind :mod:`hypnose_helpers` data-location resolution to this repository."""

from __future__ import annotations

from pathlib import Path

from hypnose_helpers.io.paths import DataLocations


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_locations = DataLocations(
    config_dir=get_repo_root() / "configs",
    data_root=get_repo_root() / "data",
    env_prefix="HYPNOSE_EEG",
)

# Keep the project-facing API small while profile parsing, precedence rules,
# caching, and validation remain owned by hypnose_helpers.
get_rawdata_root = _locations.get_rawdata_root
get_derivatives_root = _locations.get_derivatives_root
reload = _locations.reload
