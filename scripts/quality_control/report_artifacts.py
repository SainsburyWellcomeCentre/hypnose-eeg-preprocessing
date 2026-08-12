"""Report artifact burden by channel, hour, and sleep state."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from scripts.utils.artifact_reporting import ArtifactReport, build_artifact_report
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.utils.artifact_reporting import ArtifactReport, build_artifact_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--epoch-seconds",
        type=float,
        default=None,
        help="Override epoch duration inferred from artifact time_s values.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optionally save overall, hourly, and sleep-state CSV reports here.",
    )
    return parser


def _print_report(report: ArtifactReport) -> None:
    print(f"Artifact epoch duration: {report.epoch_seconds:g} seconds")
    if report.overall.empty:
        print("No channel-separated artifacts were found.")
        return

    print("Overall by channel:")
    for row in report.overall.itertuples(index=False):
        print(
            f"  {row.channel}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.recording_percent:.2f}% of recording), "
            f"longest {row.longest_artifact_s:,.1f} s"
        )

    print("By recording hour:")
    for row in report.hourly.itertuples(index=False):
        print(
            f"  {row.channel} — hour {row.hour}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.hour_percent:.2f}%), longest {row.longest_artifact_s:,.1f} s"
        )

    print("By sleep state:")
    for row in report.sleep_state.itertuples(index=False):
        print(
            f"  {row.channel} — {row.sleep_state}: {row.artifact_periods:,} periods, "
            f"{row.artifact_epochs:,} epochs, {row.artifact_duration_s:,.1f} s "
            f"({row.state_percent:.2f}%), longest {row.longest_artifact_s:,.1f} s"
        )


def _save_report(report: ArtifactReport, save_dir: Path, stem: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overall": report.overall,
        "hourly": report.hourly,
        "sleep_state": report.sleep_state,
    }
    for suffix, table in outputs.items():
        output = save_dir / f"{stem}_artifact_report_{suffix}.csv"
        table.to_csv(output, index=False)
        print(f"Saved: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epoch_seconds is not None and (
        not math.isfinite(args.epoch_seconds) or args.epoch_seconds <= 0
    ):
        parser.error("--epoch-seconds must be a positive finite number")

    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root
    from scripts.quality_control.recording_integrity import select_recordings
    from scripts.utils.recording_paths import artifact_path

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    derivatives_root = Path(
        args.derivatives_root or get_derivatives_root()
    ).resolve(strict=False)
    pairs = select_recordings(
        rawdata_root,
        derivatives_root,
        subject=args.subject,
        date=args.date,
        session=args.session,
    )
    for edf_path, _ in pairs:
        artifacts = artifact_path(edf_path, rawdata_root, derivatives_root)
        if artifacts is None:
            raise FileNotFoundError(f"No artifact parquet found for {edf_path.name}")
        print(f"Artifacts: {artifacts}")
        artifact_epochs = pd.read_parquet(artifacts)
        report = build_artifact_report(
            artifact_epochs, epoch_seconds=args.epoch_seconds
        )
        _print_report(report)
        if args.save_dir is not None:
            _save_report(report, Path(args.save_dir), edf_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
