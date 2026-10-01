"""Run the Hypnose EEG pipeline from Python, inside or outside this repository.

A small facade over `hypnose_eeg.pipeline` and the QC/viewer modules, for
notebooks and other projects that would otherwise shell out to
`hypnose-eeg-pipeline`:

    from hypnose_eeg import api

    locations = api.DataLocations(
        rawdata_root="/mnt/hypnose/rawdata",
        derivatives_root="/mnt/hypnose/derivatives",
    )
    api.run_session(66, session=1, model="my-model", locations=locations)
    qc = api.session_qc(66, session=1, locations=locations)  # in memory, nothing written
    print(qc.status, qc.sections)

Every function takes `locations`: where recordings are read from and outputs
written to, including any output-folder overrides. Leave it out to use the
active data-location profile and `configs/output_layout.yaml`. A
`hypnose_helpers.io.paths.DataLocations` -- the profile resolver another
project already has -- is accepted in its place and supplies the two roots.

The pipeline steps run in this process. A step that fails raises
`StepFailed` (the CLIs turn it into an exit status instead); a QC summary
that comes out FAIL is such a failure, raised from `quality_control()` and
`run_session()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from hypnose_eeg.pipeline import preprocessing as _preprocessing
from hypnose_eeg.pipeline import qc as _qc
from hypnose_eeg.pipeline import run as _run
from hypnose_eeg.pipeline import sleep_scoring as _sleep_scoring
from hypnose_eeg.pipeline.batch import SUBJECT_SESSIONS_SEP
from hypnose_eeg.pipeline.qc_review import SubjectReview, review_subjects, write_review_report
from hypnose_eeg.pipeline.run import BatchResult
from hypnose_eeg.pipeline.steps import StepFailed, applied_env, output_layout_env
from hypnose_eeg.qc.summary_qc import SessionQC, SummaryQCSettings, compute_session_qc

__all__ = [
    "BatchResult",
    "DataLocations",
    "SessionQC",
    "StepFailed",
    "SubjectReview",
    "SummaryQCSettings",
    "preprocess",
    "quality_control",
    "review_qc",
    "run_batch",
    "run_session",
    "score",
    "session_qc",
    "view_scoring",
]


class ProfileLocations(Protocol):
    """What this module needs from a `hypnose_helpers.io.paths.DataLocations`."""

    def get_rawdata_root(self) -> Path: ...

    def get_derivatives_root(self) -> Path: ...


@dataclass(frozen=True)
class DataLocations:
    """Where a run reads recordings from and writes its outputs to.

    `rawdata_root`/`derivatives_root` default to the active data-location
    profile. `output_layout` is an alternative `output_layout.yaml`;
    `output_root` the modality folder every output group sits below in a
    session directory (`eeg` by default, `.` for none); `output_dirs` maps
    output groups (`artifacts`, `quality_control`, ...) to folders below it.
    """

    rawdata_root: str | Path | None = None
    derivatives_root: str | Path | None = None
    output_layout: str | Path | None = None
    output_root: str | None = None
    output_dirs: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: ProfileLocations, **output_options: Any) -> DataLocations:
        """Take both roots from a `hypnose_helpers` profile resolver.

        `output_options` are this class's output-folder fields.
        """
        return cls(
            rawdata_root=profile.get_rawdata_root(),
            derivatives_root=profile.get_derivatives_root(),
            **output_options,
        )

    def stage_options(self) -> dict[str, Any]:
        """The location keyword arguments every pipeline stage's `run_steps` takes."""
        return dict(
            rawdata_root=_text(self.rawdata_root),
            derivatives_root=_text(self.derivatives_root),
            output_layout=_text(self.output_layout),
            output_root=self.output_root,
            output_dirs=dict(self.output_dirs) or None,
        )

    def env(self) -> dict[str, str]:
        """The output-folder overrides as the environment variables steps read."""
        return output_layout_env(self.output_layout, self.output_dirs, self.output_root)


def preprocess(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    steps: Sequence[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Run the preprocessing steps (trim, concatenate, downsample, prescan, detect).

    `steps` picks a subset, run in pipeline order; `detect_artifacts` needs
    sleep scoring to have run. `extra_args` are CLI flags passed to every step.
    """
    _preprocessing.run_steps(
        **_session(subject, session, date),
        **_locations(locations).stage_options(),
        overwrite=overwrite,
        dry_run=dry_run,
        steps=_list(steps),
        extra_args=_list(extra_args),
    )


def score(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    model: str | Path | None = None,
    overwrite: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Sleep-score a session with Somnotate; existing predictions are kept unless `overwrite`."""
    _sleep_scoring.run_steps(
        **_session(subject, session, date),
        **_locations(locations).stage_options(),
        model=_text(model),
        overwrite=overwrite,
        steps=["score"],
        extra_args=_list(extra_args),
    )


def quality_control(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    steps: Sequence[str] | None = None,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Run the QC steps and write their outputs (default: the summary).

    Raises `StepFailed` when the summary comes out FAIL. For the results
    themselves rather than files, use `session_qc()`.
    """
    _qc.run_steps(
        **_session(subject, session, date),
        **_locations(locations).stage_options(),
        steps=_list(steps),
        extra_args=_list(extra_args),
    )


def run_session(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    stages: Sequence[str] | None = None,
    model: str | Path | None = None,
    overwrite: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Run the full pipeline for one session, in dependency order.

    `stages` restricts the run to some of preprocessing, sleep_scoring, and
    qc, each with its own default steps (see `hypnose_eeg.pipeline.run`).
    """
    _run.run_stages(
        dict(**_session(subject, session, date), **_locations(locations).stage_options()),
        stages=_list(stages),
        model=_text(model),
        overwrite=overwrite,
        view=False,
        extra=_list(extra_args) or [],
    )


def run_batch(
    subjects: Mapping[str | int, Any] | Sequence[str | int] | str | int,
    *,
    sessions: Sequence[str | int] | str | int | None = None,
    dates: Sequence[str | int] | str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    stages: Sequence[str] | None = None,
    model: str | Path | None = None,
    overwrite: bool = False,
    keep_failed: bool = False,
    erase_derived_edf: bool = False,
    report: str | Path | None = None,
    extra_args: Sequence[str] | None = None,
) -> BatchResult:
    """Run several sessions one after the other, carrying on past failures.

    Each of `subjects` (`"all"` for every subject) runs the `sessions` or
    `dates` given -- values, or inclusive ranges such as `"2-5"` -- or every
    session it has when neither is. A mapping chooses sessions per subject
    instead: `{66: [1, 3], 67: "2-4", 68: None}`, where None falls back to
    `sessions`/`dates`. A selected session a subject does not have is reported
    as `missing`. A failed session has its outputs erased unless `keep_failed`;
    check `result.ok` and `result.outcomes` rather than catching `StepFailed`.
    """
    return _run.run_batch(
        _subject_values(subjects),
        sessions=_values(sessions),
        dates=_values(dates),
        **_locations(locations).stage_options(),
        stages=_list(stages),
        model=_text(model),
        overwrite=overwrite,
        extra_args=_list(extra_args),
        keep_failed=keep_failed,
        erase_derived_edf=erase_derived_edf,
        report=report,
    )


def session_qc(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    settings: SummaryQCSettings | None = None,
) -> SessionQC:
    """Compute every QC section for one session in memory; nothing is written.

    Override thresholds with
    `hypnose_eeg.qc.summary_qc.summary_qc_settings(max_artifact_percent=10.0)`;
    write the tables with `hypnose_eeg.qc.summary_qc.save_session_qc(result)`
    under the same `locations`.
    """
    resolved = _locations(locations)
    with applied_env(resolved.env()):
        return compute_session_qc(
            str(subject),
            session=_text(session),
            date=_text(date),
            rawdata_root=resolved.rawdata_root,
            derivatives_root=resolved.derivatives_root,
            settings=settings,
        )


def review_qc(
    subjects: Sequence[str | int] | str | int,
    *,
    locations: DataLocations | ProfileLocations | None = None,
    report: str | Path | None = None,
) -> list[SubjectReview]:
    """Read the QC summaries already on disk and return each subject's sessions.

    `report` also writes one CSV row per flagged section.
    """
    if isinstance(subjects, (str, int)):
        subjects = [subjects]
    resolved = _locations(locations)
    reviews = review_subjects(
        subjects, derivatives_root=resolved.derivatives_root, env=resolved.env()
    )
    if report:
        write_review_report(reviews, report)
    return reviews


def view_scoring(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    locations: DataLocations | ProfileLocations | None = None,
    **options: Any,
) -> None:
    """Open the interactive scoring viewer for a session (needs a display).

    `options` are `view_settings()` arguments such as `hours=(3, 6)` or
    `show_artifacts=True`.
    """
    from hypnose_eeg.sleep_scoring.view_scoring import run_view, view_settings

    resolved = _locations(locations)
    with applied_env(resolved.env()):
        settings = view_settings(
            str(subject),
            date=_text(date),
            session=_text(session),
            rawdata_root=resolved.rawdata_root,
            derivatives_root=resolved.derivatives_root,
            **options,
        )
        run_view(settings)


def _locations(locations: DataLocations | ProfileLocations | None) -> DataLocations:
    if locations is None:
        return DataLocations()
    if isinstance(locations, DataLocations):
        return locations
    if hasattr(locations, "get_rawdata_root") and hasattr(locations, "get_derivatives_root"):
        return DataLocations.from_profile(locations)
    raise TypeError(
        "locations must be a hypnose_eeg.api.DataLocations or a "
        f"hypnose_helpers DataLocations, not {type(locations).__name__}"
    )


def _session(
    subject: str | int, session: str | int | None, date: str | int | None
) -> dict[str, str | None]:
    if session is not None and date is not None:
        raise ValueError("Select a session by session number or by date, not both")
    return dict(subject=str(subject), session=_text(session), date=_text(date))


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _list(values: Sequence[str] | None) -> list[str] | None:
    return None if values is None else [str(value) for value in values]


def _values(values: Sequence[str | int] | str | int | None) -> list[str] | None:
    """`_list` for selectors that may also be one bare value."""
    if isinstance(values, (str, int)):
        values = [values]
    return _list(values)


def _subject_values(
    subjects: Mapping[str | int, Any] | Sequence[str | int] | str | int,
) -> list[str]:
    """`--subject` values; a mapping's sessions are written as `SUBJECT:SESSIONS`."""
    if isinstance(subjects, Mapping):
        return [
            str(subject) if chosen is None
            else f"{subject}{SUBJECT_SESSIONS_SEP}{','.join(_values(chosen))}"
            for subject, chosen in subjects.items()
        ]
    return _values(subjects)
