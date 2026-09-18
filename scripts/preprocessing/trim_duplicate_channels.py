"""Inspect EDF recordings and trim duplicate (or surplus) channels.

Some EEG recording sessions contain duplicate channels: the same label written
more than once into the EDF header, typically because an acquisition template
listed a signal twice. Such files break later operations -- concatenation
requires matching channel layouts across a session's parts, and sleep scoring
looks channels up by label, which MNE renames (``EEG1A-B-0``, ``EEG1A-B-1``,
...) whenever it is not unique. This step makes those files usable without
touching the source recording.

By default the script is automatic. It reads each selected EDF's header, and
when a channel label occurs more than once it writes a ``<stem>_trimmed.edf``
copy beside the source keeping only the first occurrence of every label, under
the source's own labels. A recording without duplicates is left alone and
nothing is written, so running the step on a clean session is a no-op. It is
the first default step of ``src/preprocessing.py`` and runs before
concatenation; every later step prefers the trimmed copy over its source
(``scripts/utils/recording_selection.py``). An existing trimmed copy is kept
unless ``--overwrite`` is passed.

``--keep-first N`` is the manual alternative for recordings whose surplus
channels are not duplicates by name: it keeps the leading N channels by
position, regardless of label. Because that rule cannot check itself, inspect
the printed channel list with ``--dry-run`` before writing with it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.io.mne_io import export_raw_edf, import_mne
from scripts.io.repository_paths import get_rawdata_root
from scripts.utils.config import coalesce, load_config, nested_get
from scripts.utils.recording_selection import (
    DEFAULT_TRIMMED_SUFFIX,
    find_session_dirs,
    is_concatenated_recording,
    is_trimmed_recording,
)

# EDF+/BDF+ annotation signals sit in the header but MNE never exposes them as
# channels, so they are dropped before header labels are matched to raw.ch_names.
ANNOTATION_LABELS = frozenset({"EDF Annotations", "BDF Annotations"})


@dataclass(frozen=True)
class TrimResult:
    edf_path: Path
    output_path: Path | None
    status: str  # written | no_duplicates | skipped_exists | dry_run
    labels: list[str] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)

    @property
    def dropped(self) -> list[int]:
        return [index for index in range(len(self.labels)) if index not in self.kept]


def read_edf_signal_labels(edf_path: str | Path) -> list[str]:
    """Read the signal labels straight from the EDF header.

    MNE de-duplicates labels while reading (``label-0``, ``label-1``, ...), so
    duplicates have to be detected from the header itself. The header is fixed
    width ASCII: 256 bytes of file header, of which bytes 252-256 hold the
    signal count, followed by 16 bytes of label per signal.
    """
    edf_path = Path(edf_path)
    with edf_path.open("rb") as handle:
        header = handle.read(256)
        if len(header) < 256:
            raise ValueError(f"Not an EDF file (header shorter than 256 bytes): {edf_path}")
        try:
            n_signals = int(header[252:256].decode("ascii").strip())
        except ValueError as exc:
            raise ValueError(f"Not an EDF file (unreadable signal count): {edf_path}") from exc
        label_bytes = handle.read(16 * n_signals)
    if len(label_bytes) < 16 * n_signals:
        raise ValueError(f"Truncated EDF header: {edf_path}")
    return [
        label_bytes[index * 16 : (index + 1) * 16].decode("latin-1").strip()
        for index in range(n_signals)
    ]


def data_channel_labels(labels: list[str]) -> list[str]:
    """Header labels minus annotation signals, in the order MNE exposes them."""
    return [label for label in labels if label not in ANNOTATION_LABELS]


def find_duplicate_labels(labels: list[str]) -> dict[str, list[int]]:
    """Map every label that occurs more than once to all of its indices, in order."""
    positions: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        positions.setdefault(label, []).append(index)
    return {label: indices for label, indices in positions.items() if len(indices) > 1}


def select_channels(labels: list[str], keep_first: int | None = None) -> list[int]:
    """Indices of the channels a trimmed copy keeps.

    With ``keep_first`` the leading N channels are kept by position. Otherwise
    the first occurrence of every label is kept and later repeats are dropped.
    """
    if keep_first is not None:
        if keep_first < 1:
            raise ValueError("keep_first must be at least 1")
        return list(range(min(keep_first, len(labels))))
    seen: set[str] = set()
    kept = []
    for index, label in enumerate(labels):
        if label in seen:
            continue
        seen.add(label)
        kept.append(index)
    return kept


def trim_duplicate_channels(
    edf_path: str | Path,
    *,
    keep_first: int | None = None,
    n_samples: int = 100,
    output_suffix: str = DEFAULT_TRIMMED_SUFFIX,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> TrimResult:
    """Inspect one EDF and, unless nothing needs trimming, write its trimmed copy."""
    mne = import_mne()
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if edf_path.suffix.lower() != ".edf":
        raise ValueError(f"Expected an .edf file: {edf_path}")

    labels = data_channel_labels(read_edf_signal_labels(edf_path))
    kept = select_channels(labels, keep_first=keep_first)
    duplicates = find_duplicate_labels(labels)
    trimmed_output_path = (
        Path(output_path) if output_path is not None else _default_output_path(edf_path, output_suffix)
    )

    raw = mne.io.read_raw_edf(edf_path, preload=False, infer_types=True, verbose="ERROR")
    if len(raw.ch_names) != len(labels):
        raise RuntimeError(
            f"{edf_path.name}: header lists {len(labels)} data channels but MNE read "
            f"{len(raw.ch_names)}; refusing to trim by header position"
        )
    sample_count = min(n_samples, raw.n_times)
    data, times = raw[:, :sample_count]

    print(f"\n{edf_path}")
    _print_channel_summary(raw, labels, kept, duplicates, data)
    if dry_run:
        _print_first_samples(data, times, raw.ch_names)

    if len(kept) == len(labels):
        print("No channels to trim; nothing written.")
        return TrimResult(edf_path, None, "no_duplicates", labels, kept)
    if dry_run:
        print(f"Dry run: would write trimmed EDF {trimmed_output_path}")
        return TrimResult(edf_path, trimmed_output_path, "dry_run", labels, kept)
    if trimmed_output_path.exists() and not overwrite:
        print(f"Trimmed EDF already exists (use --overwrite to replace): {trimmed_output_path}")
        return TrimResult(edf_path, trimmed_output_path, "skipped_exists", labels, kept)

    trimmed = raw.copy().pick([raw.ch_names[index] for index in kept]).load_data()
    _restore_header_labels(trimmed, [labels[index] for index in kept])
    trimmed_output_path.parent.mkdir(parents=True, exist_ok=True)
    export_raw_edf(trimmed, trimmed_output_path, overwrite=overwrite)

    print(f"Wrote trimmed EDF: {trimmed_output_path}")
    return TrimResult(edf_path, trimmed_output_path, "written", labels, kept)


def _restore_header_labels(raw: Any, labels: list[str]) -> None:
    """Give the kept channels back their header labels (MNE strips the type prefix
    with ``infer_types=True`` and suffixes duplicates), so the trimmed copy reads
    like its source. Left as MNE named them if the kept labels are still not
    unique, which only ``--keep-first`` can produce."""
    if len(set(labels)) != len(labels):
        print("Kept channels still share labels; keeping MNE's de-duplicated names.")
        return
    mapping = {
        current: label for current, label in zip(raw.ch_names, labels) if current != label
    }
    if mapping:
        raw.rename_channels(mapping)


def _print_channel_summary(
    raw: Any,
    labels: list[str],
    kept: list[int],
    duplicates: dict[str, list[int]],
    data: Any,
) -> None:
    print(f"channels: {len(labels)}")
    print(f"sample_rate_hz: {float(raw.info['sfreq'])}")
    print(f"samples: {raw.n_times}")
    print(f"duration_seconds: {raw.n_times / float(raw.info['sfreq'])}")

    print("\nChannels (header label / MNE name)")
    kept_set = set(kept)
    for index, (label, mne_name) in enumerate(zip(labels, raw.ch_names)):
        marker = "KEEP" if index in kept_set else "DROP"
        print(f"{index + 1:03d} {marker} {label!r} / {mne_name!r}")

    if not duplicates:
        print("\nNo duplicate channel labels.")
        return
    print("\nDuplicate channel labels")
    for label, indices in duplicates.items():
        first, *repeats = indices
        for repeat in repeats:
            # A repeat that carries different samples is still dropped, but flag
            # it so the operator can check whether the first occurrence is the
            # right one to keep.
            identical = bool(np.array_equal(data[first], data[repeat]))
            print(
                f"{label!r}: channel {repeat + 1:03d} repeats channel {first + 1:03d}; "
                f"identical in first {data.shape[1]} samples: {'yes' if identical else 'NO'}"
            )


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


def _resolve_edf_paths(
    parser: argparse.ArgumentParser, args: argparse.Namespace, output_suffix: str
) -> list[Path]:
    """The EDFs to process: an explicit path, or every source recording in the session.

    Trimmed copies and concatenated outputs are derived from the source
    recordings and are never trimmed themselves -- rerun concatenation with
    ``--overwrite`` after trimming so it is rebuilt from the trimmed parts.
    """
    has_selector = args.subject is not None or args.date is not None or args.session is not None
    if args.edf_path is not None and has_selector:
        parser.error("use either edf_path or --subject/--date/--session selectors")
    if args.edf_path is not None:
        return [Path(args.edf_path)]

    if args.subject is None:
        parser.error("edf_path or --subject is required")
    if args.date is None and args.session is None:
        parser.error("--subject requires either --date or --session")
    if args.output_path is not None:
        parser.error("--output-path only applies to an explicit edf_path")

    rawdata_root = Path(args.rawdata_root or get_rawdata_root()).resolve(strict=False)
    session_dirs = find_session_dirs(
        rawdata_root, subject=args.subject, date=args.date, session=args.session
    )
    edf_paths = sorted(
        path
        for session_dir in session_dirs
        for path in session_dir.glob("**/*.edf")
        if path.is_file()
        and not is_trimmed_recording(path, output_suffix)
        and not is_concatenated_recording(path)
    )
    if not edf_paths:
        listing = ", ".join(str(path) for path in session_dirs)
        parser.error(f"No source EDF recordings found beneath: {listing}")
    return edf_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "edf_path", nargs="?", default=None, help="Path to a single EDF file to process."
    )
    parser.add_argument(
        "--subject", "--subjid", dest="subject", default=None,
        help="Subject ID, for example 66 or sub-066. Alternative to edf_path; "
        "processes every source recording in the selected session.",
    )
    session_selector = parser.add_mutually_exclusive_group()
    session_selector.add_argument("--date", default=None, help="Session date: YYYYMMDD.")
    session_selector.add_argument(
        "--session", default=None, help="Session number, for example 1 or ses-1."
    )
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. CLI values override config values.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of initial samples compared between duplicate channels and "
        "printed with --dry-run.",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=None,
        help="Manual mode: keep only the leading N channels by position instead of "
        "dropping duplicate labels.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix appended to the input EDF stem for the trimmed output.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output EDF path for an explicit edf_path. Defaults beside the input "
        "with the configured suffix.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Inspect only: print the channel list, duplicates, and first samples "
        "without writing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output EDF if it exists.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    n_samples = int(
        coalesce(args.n_samples, nested_get(config, ("preprocessing", "trim_channels", "n_samples")), 100)
    )
    output_suffix = str(
        coalesce(
            args.output_suffix,
            nested_get(config, ("preprocessing", "trim_channels", "output_suffix")),
            DEFAULT_TRIMMED_SUFFIX,
        )
    )
    edf_paths = _resolve_edf_paths(parser, args, output_suffix)

    results = [
        trim_duplicate_channels(
            edf_path,
            keep_first=args.keep_first,
            n_samples=n_samples,
            output_suffix=output_suffix,
            output_path=args.output_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for edf_path in edf_paths
    ]

    print()
    for result in results:
        target = f" -> {result.output_path}" if result.output_path is not None else ""
        print(f"{result.status}: {result.edf_path}{target}")


if __name__ == "__main__":
    main()
