from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DownsampleResult:
    source_path: str
    output_path: str
    status: str
    original_sample_rate_hz: float | None = None
    target_sample_rate_hz: float | None = None
    n_channels: int | None = None
    n_samples: int | None = None
    duration_seconds: float | None = None
    error: str | None = None


class EdfDownsampler:
    """Downsample EDF recordings and save MNE FIF files under derivatives."""

    def __init__(
        self,
        source_dir: str | Path,
        sink_dir: str | Path,
        target_sample_rate_hz: float = 128.0,
        edf_pattern: str = "**/*.edf",
        overwrite: bool = False,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.sink_dir = Path(sink_dir)
        self.target_sample_rate_hz = float(target_sample_rate_hz)
        self.edf_pattern = edf_pattern
        self.overwrite = overwrite

    def find_edf_files(self) -> list[Path]:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir.resolve()}")
        if not self.source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {self.source_dir.resolve()}")

        edf_files = sorted(
            path
            for path in self.source_dir.glob(self.edf_pattern)
            if path.is_file() and path.suffix.lower() == ".edf"
        )
        return self._prefer_concatenated_recordings(edf_files)

    def _prefer_concatenated_recordings(self, edf_files: list[Path]) -> list[Path]:
        files_by_folder: dict[Path, list[Path]] = {}
        for edf_path in edf_files:
            files_by_folder.setdefault(edf_path.parent, []).append(edf_path)

        selected = []
        for folder_files in files_by_folder.values():
            if len(folder_files) == 1:
                selected.extend(folder_files)
                continue

            concat_files = [
                path
                for path in folder_files
                if path.name.lower().endswith("recording-concat.edf")
                or "_recording-concat" in path.name.lower()
            ]
            selected.extend(concat_files)

        return sorted(selected)

    def output_path(self, edf_path: Path) -> Path:
        relative_path = edf_path.relative_to(self.source_dir)
        output_name = f"{edf_path.stem}_resampled-{self._sample_rate_label()}hz_raw.fif"
        return self.sink_dir / relative_path.parent / output_name

    def preview(self) -> list[DownsampleResult]:
        return [
            DownsampleResult(
                source_path=str(edf_path.relative_to(self.source_dir)),
                output_path=str(self.output_path(edf_path)),
                status="dry_run",
                target_sample_rate_hz=self.target_sample_rate_hz,
            )
            for edf_path in self.find_edf_files()
        ]

    def downsample_file(self, edf_path: Path) -> DownsampleResult:
        mne = _import_mne()
        edf_path = Path(edf_path)
        output_path = self.output_path(edf_path)
        relative_source_path = edf_path.relative_to(self.source_dir)

        if output_path.exists() and not self.overwrite:
            return DownsampleResult(
                source_path=str(relative_source_path),
                output_path=str(output_path),
                status="skipped_exists",
                target_sample_rate_hz=self.target_sample_rate_hz,
            )

        raw = mne.io.read_raw_edf(edf_path, preload=True, infer_types=True, verbose=True)
        original_sample_rate_hz = float(raw.info["sfreq"])

        if original_sample_rate_hz != self.target_sample_rate_hz:
            raw.resample(self.target_sample_rate_hz)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_path, overwrite=self.overwrite)

        result = DownsampleResult(
            source_path=str(relative_source_path),
            output_path=str(output_path),
            status="written",
            original_sample_rate_hz=original_sample_rate_hz,
            target_sample_rate_hz=float(raw.info["sfreq"]),
            n_channels=len(raw.ch_names),
            n_samples=raw.n_times,
            duration_seconds=raw.n_times / float(raw.info["sfreq"]),
        )
        del raw
        gc.collect()
        return result

    def downsample_all(self) -> list[DownsampleResult]:
        results = []
        for edf_path in self.find_edf_files():
            try:
                results.append(self.downsample_file(edf_path))
            except Exception as exc:
                results.append(self._failed_result(edf_path, exc))
        return results

    def downsample_first(self) -> DownsampleResult:
        edf_files = self.find_edf_files()
        if not edf_files:
            raise RuntimeError("No EDF files found.")
        return self.downsample_file(edf_files[0])

    def write_manifest(self, results: Iterable[DownsampleResult], manifest_path: str | Path) -> Path:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = [_result_to_row(result) for result in results]
        fieldnames = [
            "source_path",
            "output_path",
            "status",
            "original_sample_rate_hz",
            "target_sample_rate_hz",
            "n_channels",
            "n_samples",
            "duration_seconds",
            "error",
        ]

        with manifest.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return manifest

    def _failed_result(self, edf_path: Path, exc: Exception) -> DownsampleResult:
        return DownsampleResult(
            source_path=str(edf_path.relative_to(self.source_dir)),
            output_path=str(self.output_path(edf_path)),
            status="failed",
            target_sample_rate_hz=self.target_sample_rate_hz,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _sample_rate_label(self) -> str:
        if self.target_sample_rate_hz.is_integer():
            return str(int(self.target_sample_rate_hz))
        return str(self.target_sample_rate_hz).replace(".", "p")


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


def _result_to_row(result: DownsampleResult) -> dict[str, Any]:
    row = result.__dict__.copy()
    row["error"] = result.error or ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample EDF files and save FIF files under derivatives."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. CLI values override config values.",
    )
    parser.add_argument("--source-dir", default=None, help="Mounted raw EDF root.")
    parser.add_argument("--sink-dir", default=None, help="Derivative output root.")
    parser.add_argument("--edf-pattern", default=None, help="Glob used below source-dir.")
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=None,
        help="Target sample frequency in Hz.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FIF outputs.")
    parser.add_argument("--first", action="store_true", help="Downsample only the first matching EDF.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matching EDF files without writing.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path.",
    )
    args = parser.parse_args()
    config = _load_config(args.config)

    source_dir = _coalesce(args.source_dir, _nested_get(config, ("source", "uri")), "data/rawdata")
    sink_dir = _coalesce(args.sink_dir, _nested_get(config, ("sink", "uri")), "data/derivatives")
    edf_pattern = _coalesce(args.edf_pattern, _nested_get(config, ("source", "glob")), "**/*.edf")
    target_sfreq = float(
        _coalesce(
            args.target_sfreq,
            _nested_get(config, ("preprocessing", "downsample", "target_sfreq")),
            128.0,
        )
    )
    manifest = _coalesce(
        args.manifest,
        _nested_get(config, ("preprocessing", "downsample", "manifest")),
        "data/derivatives/edf_downsample_manifest.csv",
    )
    output_format = str(
        _coalesce(_nested_get(config, ("preprocessing", "downsample", "output_format")), "fif")
    ).lower()
    if output_format != "fif":
        raise ValueError(f"Unsupported downsample output_format: {output_format!r}. Expected 'fif'.")

    downsampler = EdfDownsampler(
        source_dir=source_dir,
        sink_dir=sink_dir,
        target_sample_rate_hz=target_sfreq,
        edf_pattern=edf_pattern,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        results = downsampler.preview()
    elif args.first:
        results = [downsampler.downsample_first()]
    else:
        results = downsampler.downsample_all()

    manifest_path = downsampler.write_manifest(results, manifest)
    for result in results:
        print(f"{result.status}: {result.source_path} -> {result.output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
