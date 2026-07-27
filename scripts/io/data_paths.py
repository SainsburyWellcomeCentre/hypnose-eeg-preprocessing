"""Resolve machine-specific raw-data and derivatives locations.

Named profiles are committed in ``configs/environments/data_locations.yml``.
Each checkout selects one in the git-ignored
``configs/environments/data_locations.local.yml`` file.  The
``HYPNOSE_EEG_*`` environment variables remain useful for CI and one-off runs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

RAW_SUBDIR = "rawdata"
DERIVATIVES_SUBDIR = "derivatives"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs_dir() -> Path:
    return get_repo_root() / "configs" / "environments"


def profiles_path() -> Path:
    return _configs_dir() / "data_locations.yml"


def local_path() -> Path:
    return _configs_dir() / "data_locations.local.yml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        data = _read_simple_yaml(path)
    else:
        data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Data-location config must contain a YAML mapping: {path}")
    return data


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small nested mapping used here before the conda env exists."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, raw_value = raw_line.strip().partition(":")
        if not separator:
            continue
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        elif value.lower() in {"null", "none"}:
            parent[key] = None
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            parent[key] = value[1:-1]
        else:
            parent[key] = value.split(" #", 1)[0].strip()
    return result


def load_profiles() -> dict[str, dict[str, str]]:
    profiles = _read_yaml(profiles_path()).get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise ValueError(f"'profiles' must be a mapping in {profiles_path()}")
    return profiles


def get_active() -> str | None:
    active = _read_yaml(local_path()).get("active")
    if active:
        return str(active)
    default = _read_yaml(profiles_path()).get("default_active")
    return str(default) if default else None


def set_active(name: str) -> None:
    _configs_dir().mkdir(parents=True, exist_ok=True)
    local_path().write_text(
        "# Per-machine selection; set with scripts/io/set_data_location.py\n"
        + f"active: {name}\n"
    )
    reload()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve(strict=False)


def _active_profile() -> dict[str, str]:
    name = get_active()
    if not name:
        raise RuntimeError(
            "No EEG data-location profile is active. Run "
            "'python scripts/io/set_data_location.py --list', then select one."
        )
    profile = load_profiles().get(name)
    if not isinstance(profile, dict) or not profile.get("rawdata"):
        raise RuntimeError(f"Active data-location profile {name!r} is missing or invalid")
    rawdata = str(profile["rawdata"])
    derivatives = str(
        profile.get("derivatives") or Path(rawdata).parent / DERIVATIVES_SUBDIR
    )
    return {"name": name, "rawdata": rawdata, "derivatives": derivatives}


@lru_cache
def get_rawdata_root() -> Path:
    override = _env_path("HYPNOSE_EEG_RAWDATA_ROOT")
    if override is not None:
        return override
    return Path(_active_profile()["rawdata"]).expanduser().resolve(strict=False)


@lru_cache
def get_derivatives_root() -> Path:
    override = _env_path("HYPNOSE_EEG_DERIVATIVES_ROOT")
    if override is not None:
        return override
    return Path(_active_profile()["derivatives"]).expanduser().resolve(strict=False)


def reload() -> None:
    get_rawdata_root.cache_clear()
    get_derivatives_root.cache_clear()
