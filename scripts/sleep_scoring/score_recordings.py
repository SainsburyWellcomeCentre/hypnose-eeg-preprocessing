"""Score Hypnose EEG recordings with a trained hypnose-somnotate model.

The command reads its defaults from ``configs/pipelines/sleep_scoring.yaml``.
Command-line arguments override YAML values. Data roots are resolved through
this repository's active data-location profile and exposed through the
``HYPNOSE_EEG_*`` variables understood by hypnose-somnotate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from hypnose_helpers.io.layout import SessionLayout
from hypnose_helpers.io.selectors import parse_sessions

try:
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root, get_repo_root
    from scripts.utils.config import coalesce, load_config, nested_get
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root, get_repo_root
    from scripts.utils.config import coalesce, load_config, nested_get


DEFAULT_CONFIG_PATH = get_repo_root() / "configs" / "pipelines" / "sleep_scoring.yaml"


@dataclass(frozen=True)
class SleepScoringSettings:
    subjids: list[str]
    model_path: Path
    repo_root: Path
    rawdata_root: Path
    derivatives_root: Path
    dates: list[str] | None
    sessions: list[int] | None
    date_range: tuple[str, str] | None
    channel_labels: list[str] | None
    export_visbrain: bool
    sampling_rate_hz: int


def _as_list(value: Any, *, option_name: str) -> list[str] | None:
    if value is None:
        return None
    values: Sequence[Any] = value if isinstance(value, (list, tuple)) else [value]
    parsed = [part.strip() for item in values for part in str(item).split(",") if part.strip()]
    if not parsed:
        return None
    if any(not item for item in parsed):
        raise ValueError(f"{option_name} contains an empty value")
    return parsed


def _as_date_range(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        parts = re.split(r"\s*(?:,|:|\.\.)\s*", text)
        if len(parts) == 1:
            match = re.fullmatch(r"(\d{8})-(\d{8})", text)
            parts = list(match.groups()) if match else parts
    if len(parts) != 2 or not all(parts):
        raise ValueError("date_range must contain exactly two dates")
    if parts[0] > parts[1]:
        raise ValueError("date_range start must not be later than its end")
    return parts[0], parts[1]


def _as_bool(value: Any, *, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{option_name} must be true or false")


def _resolve_model_path(value: str | Path, derivatives_root: Path) -> Path:
    model_path = Path(value).expanduser()
    if not model_path.is_absolute():
        if len(model_path.parts) == 1 and model_path.suffix == "":
            model_path = derivatives_root / "somnotate_training" / model_path / "model.pickle"
        else:
            model_path = derivatives_root / model_path
    if model_path.is_dir():
        model_path = model_path / "model.pickle"
    return model_path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score raw EDF recordings with hypnose-somnotate."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Pipeline YAML path (default: configs/pipelines/sleep_scoring.yaml).",
    )
    parser.add_argument(
        "--subject",
        "--subjects",
        "--subjid",
        "--subjids",
        dest="subjids",
        nargs="+",
        default=None,
        help="Subject identifiers; accepts spaces or comma-separated values.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--date",
        "--dates",
        dest="dates",
        nargs="+",
        default=None,
        help="Optional recording dates; accepts spaces or comma-separated values.",
    )
    selector.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive date range; mutually exclusive with date and session selectors.",
    )
    selector.add_argument(
        "--session",
        "--sessions",
        dest="sessions",
        nargs="+",
        default=None,
        help="Session numbers; accepts values such as 1, ses-1, or comma-separated lists.",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        default=None,
        help="Model name or model.pickle path. Relative paths use the derivatives root.",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root passed to Somnotate.")
    parser.add_argument("--rawdata-root", default=None, help="Override the configured raw-data root.")
    parser.add_argument(
        "--derivatives-root", default=None, help="Override the configured derivatives root."
    )
    parser.add_argument(
        "--channel-labels",
        nargs="+",
        default=None,
        help="EDF channel labels in EEG1 EEG2 EMG order.",
    )
    parser.add_argument("--sampling-rate-hz", type=int, default=None)
    parser.add_argument(
        "--export-visbrain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Visbrain hypnogram export.",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> SleepScoringSettings:
    config = load_config(args.config)
    scoring = nested_get(config, ("sleep_scoring",), {})
    if not isinstance(scoring, dict):
        raise ValueError("sleep_scoring config must be a YAML mapping")

    subjids = _as_list(coalesce(args.subjids, scoring.get("subjids")), option_name="subjids")
    if not subjids:
        raise ValueError("at least one subject is required in the config or via --subject")

    cli_selector_used = any(
        value is not None for value in (args.dates, args.date_range, args.sessions)
    )
    if cli_selector_used:
        dates_value = args.dates
        date_range_value = args.date_range
        sessions_value = args.sessions
    else:
        dates_value = scoring.get("dates")
        date_range_value = scoring.get("date_range")
        sessions_value = scoring.get("sessions")
    dates = _as_list(dates_value, option_name="dates")
    date_range = _as_date_range(date_range_value)
    sessions = parse_sessions(sessions_value) or None
    if sum(value is not None for value in (dates, date_range, sessions)) > 1:
        raise ValueError("dates, date_range, and sessions are mutually exclusive")

    rawdata_root = Path(
        coalesce(args.rawdata_root, scoring.get("rawdata_root"), get_rawdata_root())
    ).expanduser().resolve(strict=False)
    derivatives_root = Path(
        coalesce(args.derivatives_root, scoring.get("derivatives_root"), get_derivatives_root())
    ).expanduser().resolve(strict=False)
    repo_root = Path(
        coalesce(args.repo_root, scoring.get("repo_root"), get_repo_root())
    ).expanduser().resolve(strict=False)

    model_value = coalesce(args.model_path, scoring.get("model_path"))
    if model_value is None:
        raise ValueError("model_path is required in the config or via --model")
    model_path = _resolve_model_path(model_value, derivatives_root)

    channel_labels = _as_list(
        coalesce(args.channel_labels, scoring.get("channel_labels")),
        option_name="channel_labels",
    )
    if channel_labels is not None and len(channel_labels) != 3:
        raise ValueError("channel_labels must contain exactly three labels: EEG1, EEG2, EMG")

    export_value = coalesce(args.export_visbrain, scoring.get("export_visbrain"), True)
    export_visbrain = _as_bool(export_value, option_name="export_visbrain")
    sampling_rate_hz = int(coalesce(args.sampling_rate_hz, scoring.get("sampling_rate_hz"), 512))
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    return SleepScoringSettings(
        subjids=subjids,
        model_path=model_path,
        repo_root=repo_root,
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
        dates=dates,
        sessions=sessions,
        date_range=date_range,
        channel_labels=channel_labels,
        export_visbrain=export_visbrain,
        sampling_rate_hz=sampling_rate_hz,
    )


def _import_score_recordings() -> Callable[..., list[Path]]:
    try:
        from hypnose_somnotate.scoring import score_recordings
    except ImportError as exc:
        raise ImportError(
            "Sleep scoring requires hypnose-somnotate with its scoring dependencies. "
            "Create the environment from environment.yml or install "
            "'hypnose-somnotate[scoring]'."
        ) from exc
    return score_recordings


def _relocate_scoring_outputs(output_paths: Sequence[str | Path]) -> list[Path]:
    """Move Somnotate's native outputs from saved_results into sleep_scoring."""
    relocated: list[Path] = []
    prediction_suffix = "_somnotate_predictions.parquet"
    for value in output_paths:
        prediction = Path(value)
        if (
            prediction.parent.name != "saved_results"
            or not prediction.name.endswith(prediction_suffix)
        ):
            relocated.append(prediction)
            continue

        destination_dir = prediction.parent.parent / "sleep_scoring"
        destination_dir.mkdir(parents=True, exist_ok=True)
        recording_stem = prediction.name.removesuffix(prediction_suffix)
        for source in sorted(prediction.parent.glob(f"{recording_stem}_somnotate_*")):
            source.replace(destination_dir / source.name)
        try:
            prediction.parent.rmdir()
        except OSError:
            # Preserve the directory when it contains legacy or unrelated outputs.
            pass
        relocated.append(destination_dir / prediction.name)
    return relocated


def run_scoring(
    settings: SleepScoringSettings,
    score_function: Callable[..., list[Path]] | None = None,
) -> list[Path]:
    if not settings.rawdata_root.is_dir():
        raise FileNotFoundError(f"Raw-data root not found: {settings.rawdata_root}")
    if not settings.derivatives_root.is_dir():
        raise FileNotFoundError(f"Derivatives root not found: {settings.derivatives_root}")
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"Somnotate model not found: {settings.model_path}")

    # hypnose-somnotate resolves these variables at call time. Assign rather than
    # setdefault so explicit CLI/config values override the ambient environment.
    os.environ["HYPNOSE_EEG_RAWDATA_ROOT"] = str(settings.rawdata_root)
    os.environ["HYPNOSE_EEG_DERIVATIVES_ROOT"] = str(settings.derivatives_root)

    scorer = score_function or _import_score_recordings()
    common_arguments = {
        "model_path": settings.model_path,
        "repo_root": settings.repo_root,
        "date_range": settings.date_range,
        "channel_labels": settings.channel_labels,
        "export_visbrain": settings.export_visbrain,
        "sampling_rate_hz": settings.sampling_rate_hz,
    }
    if settings.sessions is None:
        output_paths = scorer(
            subjids=settings.subjids,
            dates=settings.dates,
            **common_arguments,
        )
    else:
        layout = SessionLayout(settings.rawdata_root, name="rawdata")
        output_paths = []
        for subject in settings.subjids:
            dates = [
                layout.find_session(subject, ses=session).date
                for session in settings.sessions
            ]
            output_paths.extend(
                scorer(
                    subjids=[subject],
                    dates=dates,
                    **common_arguments,
                )
            )
    return _relocate_scoring_outputs(output_paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from_args(args)
        output_paths = run_scoring(settings)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))

    for output_path in output_paths:
        print(f"scored: {output_path}")
    print(f"completed: {len(output_paths)} recording(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
