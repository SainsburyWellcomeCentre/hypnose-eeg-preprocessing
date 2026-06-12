from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping


def inspect_edf(
    edf_path: str | Path,
    n_samples: int = 100,
    keep_first: int = 3,
    output_suffix: str = "_trimmed",
    output_path: str | Path | None = None,
    write_trimmed: bool = False,
    overwrite: bool = False,
) -> Path | None:
    mne = _import_mne()
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if edf_path.suffix.lower() != ".edf":
        raise ValueError(f"Expected an .edf file: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=False, infer_types=True, verbose=True)
    sample_count = min(n_samples, raw.n_times)
    data, times = raw[:, :sample_count]

    _print_channel_summary(raw, keep_first)
    _print_first_samples(data, times, raw.ch_names)

    if not write_trimmed:
        return None

    trimmed_output_path = Path(output_path) if output_path is not None else _default_output_path(
        edf_path,
        output_suffix,
    )
    if trimmed_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {trimmed_output_path}. Use --overwrite to replace it."
        )

    channels_to_keep = raw.ch_names[:keep_first]
    trimmed = raw.copy().pick(channels_to_keep).load_data()
    trimmed_output_path.parent.mkdir(parents=True, exist_ok=True)
    _export_raw_edf(trimmed, trimmed_output_path, overwrite=overwrite)

    print(f"\nWrote trimmed EDF: {trimmed_output_path}")
    return trimmed_output_path


def _print_channel_summary(raw: Any, keep_first: int) -> None:
    print("\nEDF summary")
    print(f"channels: {len(raw.ch_names)}")
    print(f"sample_rate_hz: {float(raw.info['sfreq'])}")
    print(f"samples: {raw.n_times}")
    print(f"duration_seconds: {raw.n_times / float(raw.info['sfreq'])}")

    print("\nChannels")
    for index, channel in enumerate(raw.ch_names, start=1):
        marker = "KEEP" if index <= keep_first else "DROP"
        print(f"{index:03d} {marker} {channel}")


def _print_first_samples(data: Any, times: Any, channel_names: list[str]) -> None:
    try:
        import pandas as pd
    except ImportError:
        _print_first_samples_plain(data, times, channel_names)
        return

    rows = data.T
    table = pd.DataFrame(rows, columns=channel_names)
    table.insert(0, "time_seconds", times)
    table.insert(0, "sample", range(len(table)))

    with pd.option_context(
        "display.max_rows",
        len(table),
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print("\nFirst samples")
        print(table)


def _print_first_samples_plain(data: Any, times: Any, channel_names: list[str]) -> None:
    print("\nFirst samples")
    print(",".join(["sample", "time_seconds", *channel_names]))
    for sample_index, (time, values) in enumerate(zip(times, data.T)):
        row = [str(sample_index), f"{time:.9f}", *[f"{value:.12g}" for value in values]]
        print(",".join(row))


def _default_output_path(edf_path: Path, output_suffix: str) -> Path:
    return edf_path.with_name(f"{edf_path.stem}{output_suffix}.edf")


def _export_raw_edf(raw: Any, output_path: Path, overwrite: bool) -> None:
    try:
        raw.export(output_path, fmt="edf", overwrite=overwrite, physical_range="auto")
    except TypeError:
        raw.export(output_path, fmt="edf", overwrite=overwrite)
    except RuntimeError as exc:
        raise RuntimeError("EDF export requires the edfio package: pip install edfio") from exc


def _import_mne() -> Any:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install MNE first: pip install mne") from exc

    return mne


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
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


def _nested_get(config: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one EDF file and optionally write a first-N-channel trimmed EDF."
    )
    parser.add_argument("edf_path", help="Path to the EDF file to inspect.")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. CLI values override config values.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of initial samples to print.",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=None,
        help="Number of leading channels to keep when writing trimmed output.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix appended to the input EDF stem for default trimmed output.",
    )
    parser.add_argument(
        "--write-trimmed",
        action="store_true",
        help="Write a trimmed EDF keeping only the first --keep-first channels.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output EDF path. Defaults beside input with configured suffix.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output EDF if it exists.")
    args = parser.parse_args()
    config = _load_config(args.config)

    n_samples = int(
        _coalesce(args.n_samples, _nested_get(config, ("preprocessing", "trim_channels", "n_samples")), 100)
    )
    keep_first = int(
        _coalesce(args.keep_first, _nested_get(config, ("preprocessing", "trim_channels", "keep_first")), 3)
    )
    output_suffix = str(
        _coalesce(
            args.output_suffix,
            _nested_get(config, ("preprocessing", "trim_channels", "output_suffix")),
            "_trimmed",
        )
    )

    inspect_edf(
        edf_path=args.edf_path,
        n_samples=n_samples,
        keep_first=keep_first,
        output_suffix=output_suffix,
        output_path=args.output_path,
        write_trimmed=args.write_trimmed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
