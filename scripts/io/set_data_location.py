#!/usr/bin/env python
"""Select and inspect this machine's EEG data-location profile."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from scripts.io.data_paths import (
        get_active,
        get_derivatives_root,
        get_rawdata_root,
        load_profiles,
        local_path,
        reload,
        set_active,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.data_paths import (
        get_active,
        get_derivatives_root,
        get_rawdata_root,
        load_profiles,
        local_path,
        reload,
        set_active,
    )

ENV_VARS = ("HYPNOSE_EEG_RAWDATA_ROOT", "HYPNOSE_EEG_DERIVATIVES_ROOT")


def show() -> None:
    reload()
    rawdata = get_rawdata_root()
    derivatives = get_derivatives_root()
    print(f"active profile : {get_active() or '(none)'}")
    print(f"rawdata       : {rawdata}  {'OK' if rawdata.exists() else 'MISSING'}")
    print(f"derivatives   : {derivatives}  {'OK' if derivatives.exists() else 'MISSING'}")
    overrides = {name: os.environ[name] for name in ENV_VARS if os.environ.get(name)}
    if overrides:
        print("environment overrides (take precedence):")
        for name, value in overrides.items():
            print(f"  {name}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="profile name to activate")
    parser.add_argument("--list", action="store_true", help="list available profiles")
    parser.add_argument("--show", action="store_true", help="show resolved locations")
    args = parser.parse_args()

    profiles = load_profiles()
    if args.list:
        active = get_active()
        for name, profile in profiles.items():
            marker = " (active)" if name == active else ""
            print(f"{name}{marker}: rawdata={profile.get('rawdata')}")
        return 0
    if args.show:
        show()
        return 0
    if not args.profile:
        parser.error("provide a profile name, --list, or --show")
    if args.profile not in profiles:
        parser.error(f"unknown profile {args.profile!r}; choose from: {', '.join(profiles)}")

    set_active(args.profile)
    print(f"Active data location set to {args.profile!r} in {local_path()}")
    show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
