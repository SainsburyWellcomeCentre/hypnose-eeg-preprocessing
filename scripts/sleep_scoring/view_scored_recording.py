"""Open hypnose-somnotate's interactive view for one scored recording.

This is a visual quality-control tool: it displays the raw EEG/EMG traces and
Somnotate state predictions that already exist on disk. It does not rescore the
recording or calculate a numerical performance metric.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root, get_repo_root
    from scripts.utils import coalesce, load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.data_paths import get_derivatives_root, get_rawdata_root, get_repo_root
    from scripts.utils import coalesce, load_config, nested_get


DEFAULT_CONFIG_PATH = get_repo_root() / "configs" / "pipelines" / "sleep_scoring.yaml"
REMOTE_VIEWER_DOC = get_repo_root() / "docs" / "remote_visualization.md"


@dataclass(frozen=True)
class ScoringViewSettings:
    subject: str
    date: str
    repo_root: Path
    rawdata_root: Path
    derivatives_root: Path
    recording_index: int
    eeg_channel: int
    view_length_s: float


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [part.strip() for item in items for part in str(item).split(",") if part.strip()]


def _single_value(value: Any, *, option_name: str) -> str | None:
    values = _values(value)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(
            f"the interactive viewer requires exactly one {option_name}; got {len(values)}"
        )
    return values[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually inspect one scored session with hypnose-somnotate view."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Pipeline YAML path (default: configs/pipelines/sleep_scoring.yaml).",
    )
    parser.add_argument("--subject", "--subjid", dest="subject", default=None)
    parser.add_argument("--date", default=None, help="Session date in YYYYMMDD form.")
    parser.add_argument("--recording-index", type=int, default=None)
    parser.add_argument("--eeg-channel", type=int, choices=(0, 1), default=None)
    parser.add_argument("--view-length", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    return parser


def settings_from_args(args: argparse.Namespace) -> ScoringViewSettings:
    config = load_config(args.config)
    scoring = nested_get(config, ("sleep_scoring",), {})
    view = nested_get(config, ("sleep_scoring_view",), {})
    if not isinstance(scoring, dict) or not isinstance(view, dict):
        raise ValueError("sleep_scoring and sleep_scoring_view must be YAML mappings")

    subject = _single_value(
        coalesce(args.subject, view.get("subject"), scoring.get("subjids")),
        option_name="subject",
    )
    if subject is None:
        raise ValueError("a subject is required in the config or via --subject")

    date = _single_value(
        coalesce(args.date, view.get("date"), scoring.get("dates")),
        option_name="date",
    )
    if date is None:
        raise ValueError("a date is required in the config or via --date")

    rawdata_root = Path(
        coalesce(
            args.rawdata_root,
            view.get("rawdata_root"),
            scoring.get("rawdata_root"),
            get_rawdata_root(),
        )
    ).expanduser().resolve(strict=False)
    derivatives_root = Path(
        coalesce(
            args.derivatives_root,
            view.get("derivatives_root"),
            scoring.get("derivatives_root"),
            get_derivatives_root(),
        )
    ).expanduser().resolve(strict=False)
    repo_root = Path(
        coalesce(
            args.repo_root,
            view.get("repo_root"),
            scoring.get("repo_root"),
            get_repo_root(),
        )
    ).expanduser().resolve(strict=False)

    recording_index = int(coalesce(args.recording_index, view.get("recording_index"), 0))
    eeg_channel = int(coalesce(args.eeg_channel, view.get("eeg_channel"), 0))
    view_length_s = float(coalesce(args.view_length, view.get("view_length_s"), 120.0))
    if recording_index < 0:
        raise ValueError("recording_index must not be negative")
    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")
    if view_length_s <= 0:
        raise ValueError("view_length_s must be positive")

    return ScoringViewSettings(
        subject=subject,
        date=date,
        repo_root=repo_root,
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
        recording_index=recording_index,
        eeg_channel=eeg_channel,
        view_length_s=view_length_s,
    )


def _import_view_main() -> Callable[[list[str]], int]:
    try:
        from hypnose_somnotate.cli.view import main as view_main
    except ImportError as exc:
        raise ImportError(
            "The scoring viewer requires hypnose-somnotate and its visualization dependencies."
        ) from exc
    return view_main


def _require_graphical_display(environment: Mapping[str, str]) -> None:
    """Fail clearly before Qt aborts when an SSH session has no display tunnel."""
    if sys.platform.startswith("linux") and not (
        environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "No graphical display is available. Connect with SSH X11 forwarding "
            "and confirm that echo $DISPLAY is non-empty before launching the viewer. "
            f"See {REMOTE_VIEWER_DOC}."
        )


def run_view(
    settings: ScoringViewSettings,
    view_function: Callable[[list[str]], int] | None = None,
) -> int:
    if not settings.rawdata_root.is_dir():
        raise FileNotFoundError(f"Raw-data root not found: {settings.rawdata_root}")
    if not settings.derivatives_root.is_dir():
        raise FileNotFoundError(f"Derivatives root not found: {settings.derivatives_root}")

    _require_graphical_display(os.environ)

    os.environ["HYPNOSE_EEG_RAWDATA_ROOT"] = str(settings.rawdata_root)
    os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"] = str(settings.derivatives_root)

    arguments = [
        "--sub",
        settings.subject,
        "--date",
        settings.date,
        "--recording-index",
        str(settings.recording_index),
        "--eeg-channel",
        str(settings.eeg_channel),
        "--view-length",
        str(settings.view_length_s),
        "--repo-root",
        str(settings.repo_root),
    ]
    return (view_function or _import_view_main())(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        return run_view(settings)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
