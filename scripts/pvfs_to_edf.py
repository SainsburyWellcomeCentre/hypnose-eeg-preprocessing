#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to read config files. Install with: pip install pyyaml"
    ) from exc


@dataclass(frozen=True)
class PVFSRecord:
    path: Path
    size: int
    mtime_ns: int


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"processed": {}}
    data.setdefault("processed", {})
    return data


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def list_pvfs_files(source_dir: Path, source_glob: str, recursive: bool) -> list[PVFSRecord]:
    iterator = source_dir.rglob(source_glob) if recursive else source_dir.glob(source_glob)
    files = []
    for file_path in iterator:
        if file_path.is_file() and file_path.suffix.lower() == ".pvfs":
            st = file_path.stat()
            files.append(PVFSRecord(path=file_path, size=st.st_size, mtime_ns=st.st_mtime_ns))
    files.sort(key=lambda x: str(x.path))
    return files


def build_output_path(source_file: Path, source_root: Path, sink_root: Path) -> Path:
    rel = source_file.relative_to(source_root)
    return sink_root.joinpath(rel).with_suffix(".edf")


def should_convert(state: Dict[str, Any], record: PVFSRecord) -> bool:
    entry = state["processed"].get(str(record.path))
    if not entry:
        return True
    return entry.get("size") != record.size or entry.get("mtime_ns") != record.mtime_ns


def run_conversion(command_template: str, source_file: Path, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_template.format(
        input=shlex.quote(str(source_file)),
        output=shlex.quote(str(output_file)),
    )
    completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0 and completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return completed.returncode


def process_once(cfg: Dict[str, Any], dry_run_override: bool | None = None) -> int:
    section = cfg.get("pvfs_conversion")
    if not isinstance(section, dict):
        raise ValueError("Missing 'pvfs_conversion' section in config")

    source_dir = Path(section["source_uri"])
    sink_dir = Path(section["sink_uri"])
    state_file = Path(section.get("state_file", "state/pvfs_to_edf_state.json"))
    source_glob = section.get("source_glob", "*.pvfs")
    recursive = bool(section.get("recursive", True))
    command_template = section["command_template"]
    dry_run = bool(section.get("dry_run", False))
    if dry_run_override is not None:
        dry_run = dry_run_override

    if not source_dir.exists():
        raise FileNotFoundError(f"PVFS source does not exist: {source_dir}")

    state = load_state(state_file)
    candidates = list_pvfs_files(source_dir, source_glob, recursive)

    converted = 0
    skipped = 0
    failed = 0

    for record in candidates:
        if not should_convert(state, record):
            skipped += 1
            continue

        output_path = build_output_path(record.path, source_dir, sink_dir)
        print(f"Converting: {record.path} -> {output_path}")

        if dry_run:
            converted += 1
            continue

        rc = run_conversion(command_template, record.path, output_path)
        if rc != 0:
            failed += 1
            continue

        converted += 1
        state["processed"][str(record.path)] = {
            "size": record.size,
            "mtime_ns": record.mtime_ns,
            "edf_path": str(output_path),
            "updated_at_epoch": int(time.time()),
        }

    if not dry_run:
        save_state(state_file, state)

    print(
        f"Summary: candidates={len(candidates)} converted={converted} "
        f"skipped={skipped} failed={failed} dry_run={dry_run}"
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PVFS files to EDF from external source storage")
    parser.add_argument(
        "--config",
        default="configs/environments/dev.yaml",
        help="Path to environment YAML containing pvfs_conversion section",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll source folder using poll_interval_seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List conversions without invoking converter command",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    if not args.watch:
        return process_once(cfg, dry_run_override=args.dry_run)

    section = cfg.get("pvfs_conversion", {})
    poll_seconds = int(section.get("poll_interval_seconds", 10))

    while True:
        rc = process_once(cfg, dry_run_override=args.dry_run)
        if rc != 0:
            return rc
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
