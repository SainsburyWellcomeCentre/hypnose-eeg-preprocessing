"""Resolve repository paths and machine-specific data locations."""

from __future__ import annotations

import sys
from pathlib import Path

from hypnose_helpers.cli.set_data_location import main as locations_cli_main
from hypnose_helpers.io.paths import DataLocations


def get_repo_root() -> Path:
    """Return the root of this repository checkout."""
    return Path(__file__).resolve().parents[2]


_locations = DataLocations(
    config_dir=get_repo_root() / "configs",
    data_root=get_repo_root() / "data",
    env_prefix="HYPNOSE_EEG",
)

# Profile parsing, precedence, caching, and validation remain owned by
# hypnose_helpers; this module exposes only the repository-facing API.
get_rawdata_root = _locations.get_rawdata_root
get_derivatives_root = _locations.get_derivatives_root
reload = _locations.reload


def main(argv: list[str] | None = None) -> int:
    """Run data-location profile selection for this repository."""
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
