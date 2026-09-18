"""Score Hypnose EEG recordings with a trained hypnose-somnotate model.

The command reads its defaults from ``configs/pipelines/sleep_scoring.yaml``.
Command-line arguments override YAML values. Data roots are resolved through
this repository's active data-location profile.

This module owns every path/layout decision: which sessions match the
subject/date/date-range/session selectors, which EDF to score when a session
folder holds several parts (only the concatenated recording, once
``scripts/preprocessing/concatenate_recordings.py`` has produced one -- see
``run_scoring``), and where outputs are written.
``hypnose_somnotate.scoring.score_recording`` only ever receives one already-
resolved EDF path and model path at a time; it has no layout knowledge of its
own.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from hypnose_helpers.io.layout import SessionLayout, normalize_subjid
from hypnose_helpers.io.selectors import parse_sessions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.io.output_layout import output_dir_name
from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root, get_repo_root
from scripts.utils.provenance import file_fingerprint, write_provenance
from scripts.utils.config import (
    DEFAULT_SLEEP_SCORING_CONFIG_PATH,
    coalesce,
    load_config,
    nested_get,
)
from scripts.utils.recording_selection import prefer_concatenated_recording


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
    global_normalization: bool
    overwrite: bool = False


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
        default=str(DEFAULT_SLEEP_SCORING_CONFIG_PATH),
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
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (kept for CLI compatibility; unused by scoring itself).",
    )
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
    parser.add_argument(
        "--global-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Normalize every scoring chunk against statistics pooled across the "
            "whole recording (gap/too_short epochs excluded), instead of each "
            "chunk's own statistics. Matters most for recordings split into "
            "several chunks by long gaps. Default: true (see "
            "configs/pipelines/sleep_scoring.yaml)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Rescore recordings whose predictions parquet already exists "
        "(by default such recordings are skipped).",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> SleepScoringSettings:
    config = load_config(args.config)
    scoring = nested_get(config, ("sleep_scoring",), {})
    if not isinstance(scoring, dict):
        raise ValueError("sleep_scoring config must be a YAML mapping")

    subjids = _as_list(args.subjids, option_name="subjids")
    if not subjids:
        raise ValueError("at least one subject is required via --subject")

    dates = _as_list(args.dates, option_name="dates")
    date_range = _as_date_range(args.date_range)
    sessions = parse_sessions(args.sessions) or None
    if sum(value is not None for value in (dates, date_range, sessions)) > 1:
        raise ValueError("dates, date_range, and sessions are mutually exclusive")

    rawdata_root = Path(
        coalesce(args.rawdata_root, get_rawdata_root())
    ).expanduser().resolve(strict=False)
    derivatives_root = Path(
        coalesce(args.derivatives_root, get_derivatives_root())
    ).expanduser().resolve(strict=False)
    repo_root = Path(
        coalesce(args.repo_root, get_repo_root())
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

    global_normalization_value = coalesce(
        args.global_normalization, scoring.get("global_normalization"), False
    )
    global_normalization = _as_bool(global_normalization_value, option_name="global_normalization")

    overwrite_value = coalesce(args.overwrite, scoring.get("overwrite"), False)
    overwrite = _as_bool(overwrite_value, option_name="overwrite")

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
        global_normalization=global_normalization,
        overwrite=overwrite,
    )


def _channel_label_alias(label: str) -> str:
    """The other spelling of one channel label -- with or without its modality prefix.

    Recordings are inconsistent about this: most raw EDFs name channels
    ``EEG EEG1A-B`` / ``EMG EMG``, but some derived files (e.g. concatenated
    recordings) drop the repeated prefix down to ``EEG1A-B`` / ``EMG``. Either
    spelling is accepted in config/CLI input; this derives the one not given.
    """
    for prefix in ("EEG ", "EMG "):
        if label.startswith(prefix):
            return label[len(prefix):]
    modality = "EMG" if label.upper().startswith("EMG") else "EEG"
    return f"{modality} {label}"


def _match_channel_labels(
    channel_labels: list[str], available: set[str], *, source: str
) -> list[str]:
    """Resolve each configured label against a file's real signal labels.

    Falls back to each label's alias spelling (see `_channel_label_alias`) when
    the configured spelling isn't present in `available`.
    """
    resolved = []
    for label in channel_labels:
        if label in available:
            resolved.append(label)
            continue
        alias = _channel_label_alias(label)
        if alias in available:
            resolved.append(alias)
            continue
        raise ValueError(
            f"Channel label {label!r} (or {alias!r}) not found in {source}; "
            f"available signals: {sorted(available)}"
        )
    return resolved


@dataclass(frozen=True)
class _ScoringDependencies:
    score_recording: Callable[..., tuple[Any, Any]]
    prediction_path: Callable[[Any], Path]
    segments_path: Callable[[Any], Path]
    hypnogram_path: Callable[[Any], Path]
    recording_ref_cls: type
    print_recording_plan: Callable[[Any, Any], None]
    convert_state_vector_to_state_intervals: Callable[..., tuple[list, list]]
    export_hypnogram: Callable[..., None]
    configuration: Any


def _import_scoring_dependencies() -> _ScoringDependencies:
    try:
        from hypnose_somnotate.io.loading import hypnogram_path, prediction_path, segments_path
        from hypnose_somnotate.io.paths import RecordingRef
        from hypnose_somnotate.scoring import score_recording
        from hypnose_somnotate.scoring.scoring import _print_recording_plan
        from hypnose_somnotate.somnotate._utils import convert_state_vector_to_state_intervals
        from hypnose_somnotate.somnotate_pipeline.io.data_io import export_hypnogram
        from hypnose_somnotate.somnotate_pipeline.utils import configuration
    except ImportError as exc:
        raise ImportError(
            "Sleep scoring requires hypnose-somnotate with its scoring dependencies. "
            "Create the environment from environment.yml or install "
            "'hypnose-somnotate[scoring]'."
        ) from exc
    return _ScoringDependencies(
        score_recording=score_recording,
        prediction_path=prediction_path,
        segments_path=segments_path,
        hypnogram_path=hypnogram_path,
        recording_ref_cls=RecordingRef,
        print_recording_plan=_print_recording_plan,
        convert_state_vector_to_state_intervals=convert_state_vector_to_state_intervals,
        export_hypnogram=export_hypnogram,
        configuration=configuration,
    )


def _warn_missing_selection(
    sub_label: str, sessions: list, dates: list[str] | None, session_numbers: list[int] | None
) -> None:
    """Warn about specifically requested dates/sessions that resolved to nothing.

    A subject missing *some* of several requested dates/sessions still gets
    scored for the rest -- this only reports the gaps, it never aborts the batch.
    """
    if dates is not None:
        found = {s.date for s in sessions}
        for missing in sorted(set(dates) - found):
            warnings.warn(
                f"No session directory for {sub_label} on requested date {missing}.",
                UserWarning,
                stacklevel=3,
            )
    elif session_numbers is not None:
        found = {s.ses for s in sessions}
        for missing in sorted(set(session_numbers) - found):
            warnings.warn(
                f"No session directory for {sub_label} with requested session {missing}.",
                UserWarning,
                stacklevel=3,
            )
    elif not sessions:
        warnings.warn(f"No session directory found for {sub_label}.", UserWarning, stacklevel=3)


def run_scoring(
    settings: SleepScoringSettings,
    score_function: Callable[..., tuple[Any, Any]] | None = None,
) -> list[Path]:
    """Discover recordings under `settings.rawdata_root` and score each one.

    Owns every path/layout decision itself -- which sessions match the
    subject/date/date-range/session selectors (via `SessionLayout`), which EDF
    to score when a session folder holds several
    (`prefer_concatenated_recording`), and where outputs are written
    (`settings.derivatives_root/<sub>/<ses>/sleep_scoring/`). `score_function`
    (default: `hypnose_somnotate.scoring.score_recording`) only ever sees one
    resolved EDF path and model path at a time -- it has no say in any of the
    above.
    """
    if not settings.rawdata_root.is_dir():
        raise FileNotFoundError(f"Raw-data root not found: {settings.rawdata_root}")
    if not settings.derivatives_root.is_dir():
        raise FileNotFoundError(f"Derivatives root not found: {settings.derivatives_root}")
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"Somnotate model not found: {settings.model_path}")

    deps = _import_scoring_dependencies()
    scorer = score_function or deps.score_recording
    RecordingRef = deps.recording_ref_cls

    channel_labels = settings.channel_labels
    output_subdir = output_dir_name("sleep_scoring")
    layout = SessionLayout(settings.rawdata_root, name="rawdata")

    output_paths: list[Path] = []
    for subjid in settings.subjids:
        sub_label = normalize_subjid(subjid)
        sessions = layout.find_sessions(
            subjid,
            date=settings.dates,
            date_range=settings.date_range,
            ses=settings.sessions,
            missing_ok=True,
        )
        _warn_missing_selection(sub_label, sessions, settings.dates, settings.sessions)

        for session in sessions:
            session_label = session.path.name.split("_date-")[0]
            ephys_dir = session.path / "ephys"
            if not ephys_dir.is_dir():
                warnings.warn(
                    f"No ephys/ directory for {sub_label} {session_label} (date {session.date}).",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            edf_paths = sorted(ephys_dir.glob(f"{sub_label}_ses-*recording-*.edf"))
            if not edf_paths:
                pvfs_files = list(ephys_dir.glob("*.pvfs"))
                if pvfs_files:
                    warnings.warn(
                        f"No EDF file for {sub_label} {session_label} (date {session.date}); "
                        f"found only .pvfs files in {ephys_dir}. "
                        "Convert the .pvfs to .edf before scoring.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"No EEG data files for {sub_label} {session_label} (date {session.date}); "
                        f"{ephys_dir} contains no .edf or .pvfs files.",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            preferred_edf_paths = prefer_concatenated_recording(edf_paths)
            if not preferred_edf_paths:
                warnings.warn(
                    f"{sub_label} {session_label} (date {session.date}) has "
                    f"{len(edf_paths)} EDF files but none is a concatenated recording "
                    "('_recording-concat.edf'); skipping until "
                    "scripts/preprocessing/concatenate_recordings.py has been run "
                    "for this session.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for edf_path in preferred_edf_paths:
                output_dir = (
                    settings.derivatives_root
                    / session.subject_dir.name
                    / session.path.name
                    / output_subdir
                )
                recording = RecordingRef(
                    subject=sub_label,
                    session=session_label,
                    date=session.date,
                    edf_path=edf_path,
                    output_dir=output_dir,
                )

                output_path = deps.prediction_path(recording)
                if output_path.exists() and not settings.overwrite:
                    print(f"skipped, predictions exist: {output_path}")
                    continue

                df, prepared = scorer(
                    edf_path,
                    settings.model_path,
                    channel_labels=channel_labels,
                    sampling_rate_hz=settings.sampling_rate_hz,
                    global_normalization=settings.global_normalization,
                )
                deps.print_recording_plan(recording, prepared)

                output_dir.mkdir(parents=True, exist_ok=True)
                sidecar_path = deps.segments_path(recording)

                df.to_parquet(output_path, index=False)
                with open(sidecar_path, "w") as f:
                    json.dump(prepared.to_dict(), f, indent=2)

                write_provenance(
                    "sleep_scoring",
                    outputs=[output_path, sidecar_path],
                    inputs={
                        "edf_path": str(edf_path),
                        "subject": sub_label,
                        "session": session_label,
                        "date": session.date,
                    },
                    parameters={
                        "model": file_fingerprint(settings.model_path)
                        or {"path": str(settings.model_path)},
                        "model_name": settings.model_path.parent.name,
                        "channel_labels": channel_labels,
                        "sampling_rate_hz": settings.sampling_rate_hz,
                        "global_normalization": settings.global_normalization,
                        "export_visbrain": settings.export_visbrain,
                    },
                )

                if settings.export_visbrain:
                    hyp_path = deps.hypnogram_path(recording)
                    states, intervals = deps.convert_state_vector_to_state_intervals(
                        df["label_model"].to_numpy(dtype=int),
                        mapping=deps.configuration.int_to_state,
                        time_resolution=deps.configuration.time_resolution,
                    )
                    deps.export_hypnogram(str(hyp_path), states, intervals)

                output_paths.append(output_path)

    return output_paths


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
