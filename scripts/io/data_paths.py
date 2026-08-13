"""Resolve, select, and inspect this repository's data-location profiles."""

from __future__ import annotations

import sys
from pathlib import Path

from hypnose_helpers.cli.set_data_location import main as locations_cli_main
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


def main(argv: list[str] | None = None) -> int:
    """Run the data-location profile selection command for this repository."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return locations_cli_main(
        [
            "--config-dir",
            str(get_repo_root() / "configs"),
            "--env-prefix",
            "HYPNOSE_EEG",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
