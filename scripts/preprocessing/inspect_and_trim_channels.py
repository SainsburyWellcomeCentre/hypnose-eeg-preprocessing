"""Inspect an EDF recording and optionally standardize its channel count.

Some EEG recording sessions contain unexpected auxiliary or duplicate channels.
Those files can be incompatible with later operations—particularly concatenation,
which requires the recordings in a session to have matching channel layouts. This
utility makes those discrepancies visible and, when appropriate, creates a reduced
EDF containing only the leading EEG channels expected by the pipeline.

By default the script is read-only. It opens one EDF without loading the complete
recording into memory, reports its channel names, sample rate, duration, and first
few samples, and marks which channels would be retained. Passing ``--write-trimmed``
loads the selected channels and writes a new EDF; it does not alter the source file.
The default output keeps the first three channels and adds ``_trimmed`` to the
filename, but the channel count, output suffix, and output path are configurable.

Channels are selected by their order in the EDF, not by name. The operator should
therefore inspect the printed channel list before writing a trimmed copy and confirm
that the leading channels are the intended EEG signals. This script is a corrective
tool for recordings with inconsistent channel layouts, not a mandatory processing
step for EDF files that already match the expected schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.io.mne_io import export_raw_edf, import_mne
    from scripts.utils.config import coalesce, load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.mne_io import export_raw_edf, import_mne
    from scripts.utils.config import coalesce, load_config, nested_get


def inspect_edf(
    edf_path: str | Path,
    n_samples: int = 100,
    keep_first: int = 3,
    output_suffix: str = "_trimmed",
    output_path: str | Path | None = None,
    write_trimmed: bool = False,
    overwrite: bool = False,
) -> Path | None:
    mne = import_mne()
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
    export_raw_edf(raw, output_path, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    config = load_config(args.config)

    n_samples = int(
        coalesce(args.n_samples, nested_get(config, ("preprocessing", "trim_channels", "n_samples")), 100)
    )
    keep_first = int(
        coalesce(args.keep_first, nested_get(config, ("preprocessing", "trim_channels", "keep_first")), 3)
    )
    output_suffix = str(
        coalesce(
            args.output_suffix,
            nested_get(config, ("preprocessing", "trim_channels", "output_suffix")),
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
