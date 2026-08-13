"""Select and inspect this machine's EEG data-location profile."""

from __future__ import annotations

import sys
from pathlib import Path

from hypnose_helpers.cli.set_data_location import main as helpers_main


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return helpers_main(
        [
            "--config-dir",
            str(repo_root / "configs"),
            "--env-prefix",
            "HYPNOSE_EEG",
            *forwarded,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
