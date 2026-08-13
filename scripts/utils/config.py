"""Configuration loading and value-selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    path = Path(config_path)
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml_config(path)

    with path.open() as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    return config


def nested_get(
    config: Mapping[str, Any], keys: tuple[str, ...], default: Any = None
) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def two_float_tuple(configured: Any, name: str) -> tuple[float, float]:
    """Normalize a two-number config value, including simple-parser strings."""
    if isinstance(configured, str):
        configured = configured.strip().removeprefix("[").removesuffix("]").split(",")
    values = tuple(float(item) for item in configured)
    if len(values) != 2:
        raise ValueError(f"Configured {name} must contain [low, high].")
    return values


def _load_simple_yaml_config(config_path: Path) -> dict[str, Any]:
    """Fallback parser for the simple nested mappings used by project configs."""
    config: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, config)]

    with config_path.open() as config_file:
        for raw_line in config_file:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            if raw_line.lstrip().startswith("- "):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if ":" not in line:
                continue

            key, raw_value = line.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]
            if raw_value == "":
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_simple_yaml_scalar(raw_value)

    return config


def _parse_simple_yaml_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
