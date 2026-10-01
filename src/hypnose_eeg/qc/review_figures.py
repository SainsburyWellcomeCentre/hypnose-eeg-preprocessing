"""Save one session's QC review figures as PNGs, with no display needed.

Only for a session whose QC summary is anything but a plain pass -- that is
when someone has to look -- these are written into the session's
quality-control directory, beside the QC summary, each prefixed with the
recording's stem:

- `scoring_hours-0-12.png`: the first 12 hours of the scored recording
  (EEG/EMG traces, their derived envelopes, Somnotate's predicted states, and
  the shaded gap/excluded/artifact spans) drawn as the interactive viewer draws
  them (`hypnose_eeg/sleep_scoring/view_scoring.py`), as one overview picture.
- `sleep_state_power_spectra.png`, `sleep_state_emg_rms.png`, and
  `sleep_state_spectral_quality.csv`: the sleep-state spectra and EMG RMS
  distributions of `hypnose_eeg/qc/spectra.py`.

`--scoring` and `--spectra` (`review`, `always`, `never`) set when each is
drawn; both default to `review`. A session that passes gets neither.

The QC summary is read, not recomputed, so run it first -- the QC stage's
default steps (`summary`, then `figures`) do. A session whose summary FAILs
stops the stage before this step; render its figures with
`hypnose-eeg-qc --subject 66 --session 1 --steps figures`. Defaults come from
`review_figures` in configs/pipelines/quality_control.yaml; the scoring
overview's channel and display rate come from the viewer's own settings
(`sleep_scoring_view` in configs/pipelines/sleep_scoring.yaml).

Usable from Python too: `review_figure_settings()` and
`render_session_figures()`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from hypnose_helpers.io.selectors import parse_sessions

from hypnose_eeg.io.output_paths import quality_control_output_path, recording_output_name
from hypnose_eeg.io.repository_paths import resolve_data_roots
from hypnose_eeg.pipeline.batch import read_qc_verdict
from hypnose_eeg.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    load_config,
    nested_get,
    two_float_tuple,
    with_overrides,
)

# When a figure is drawn, relative to the session's QC status: only when it is
# not a plain pass ("review"), whatever it is ("always"), or not at all.
FIGURE_WHEN = ("review", "always", "never")


@dataclass(frozen=True)
class ReviewFigureSettings:
    """What the review figures cover, when each is drawn, and how."""

    hours: tuple[float, float]
    scoring: str
    spectra: str
    show_artifacts: bool
    width_in: float
    dpi: int

    def __post_init__(self) -> None:
        start, end = self.hours
        if start < 0 or end <= start:
            raise ValueError("review figure hours must be START >= 0 and END > START")
        for name in ("scoring", "spectra"):
            if getattr(self, name) not in FIGURE_WHEN:
                raise ValueError(
                    f"{name} must be one of {FIGURE_WHEN}, not {getattr(self, name)!r}"
                )
        if self.width_in <= 0 or self.dpi <= 0:
            raise ValueError("review figure width_in and dpi must be positive")


def review_figure_settings(
    config: str | Path = DEFAULT_QUALITY_CONTROL_CONFIG_PATH, **overrides: object
) -> ReviewFigureSettings:
    """Load `review_figures` from the QC YAML, then apply any overrides that are not None."""
    section = nested_get(load_config(config), ("review_figures",), {})
    if not isinstance(section, Mapping):
        raise ValueError("review_figures config must be a YAML mapping")
    settings = ReviewFigureSettings(
        hours=two_float_tuple(section.get("hours", (0, 12)), "review_figures hours"),
        scoring=str(section.get("scoring", "review")),
        spectra=str(section.get("spectra", "review")),
        show_artifacts=bool(section.get("show_artifacts", True)),
        width_in=float(section.get("width_in", 24)),
        dpi=int(section.get("dpi", 150)),
    )
    if overrides.get("hours") is not None:
        overrides["hours"] = two_float_tuple(overrides["hours"], "hours")
    return with_overrides(settings, **overrides)


def render_session_figures(
    subject: str | int,
    *,
    session: str | int | None = None,
    date: str | int | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    settings: ReviewFigureSettings | None = None,
) -> list[Path]:
    """Save one session's review figures (see the module docstring); return their paths."""
    from hypnose_eeg.qc.spectra import compute_session_spectra, plot_spectra, save_spectra
    from hypnose_eeg.sleep_scoring.view_scoring import render_scoring, view_settings

    settings = settings or review_figure_settings()
    rawdata_root, derivatives_root = resolve_data_roots(rawdata_root, derivatives_root)
    session = None if session is None else str(session)
    date = None if date is None else str(date)

    status = None
    if "review" in (settings.scoring, settings.spectra):
        status = _qc_status(subject, session=session, date=date, derivatives_root=derivatives_root)
    draw_scoring = _wanted(settings.scoring, status)
    draw_spectra = _wanted(settings.spectra, status)
    if not (draw_scoring or draw_spectra):
        print("No review figures to draw for this session.")
        return []

    paths: list[Path] = []
    if draw_scoring:
        start_h, end_h = settings.hours
        overview_name = f"scoring_hours-{start_h:g}-{end_h:g}.png"

        def overview_path(window) -> Path:
            return quality_control_output_path(
                recording_output_name(overview_name, window.edf_path),
                window.edf_path, rawdata_root, derivatives_root,
            )

        view = view_settings(
            subject, session=session, date=date, hours=settings.hours,
            show_artifacts=settings.show_artifacts,
            rawdata_root=rawdata_root, derivatives_root=derivatives_root,
        )
        paths.append(
            render_scoring(view, overview_path, width_in=settings.width_in, dpi=settings.dpi)
        )
    if not draw_spectra:
        return paths

    import matplotlib.pyplot as plt

    for result in compute_session_spectra(
        subject, session=session, date=date,
        rawdata_root=rawdata_root, derivatives_root=derivatives_root,
    ):
        figures = plot_spectra(result)
        try:
            save_dir = quality_control_output_path(
                ".", result.edf_path, rawdata_root, derivatives_root
            )
            paths += save_spectra(result, save_dir, figures, figure_format="png")
        finally:
            for fig in figures.values():
                plt.close(fig)
    return paths


def _qc_status(
    subject: str | int,
    *,
    session: str | None,
    date: str | None,
    derivatives_root: Path,
) -> str | None:
    """The session's overall QC status, or None when it has no QC summary yet."""
    verdict = read_qc_verdict(
        derivatives_root,
        subject=subject,
        session=None if session is None else parse_sessions([session])[0],
        date=date,
    )
    if verdict is None:
        print(
            "No QC summary for this session, so no review status to go on. Run the "
            "QC summary first, or pass --scoring/--spectra always."
        )
        return None
    status = verdict[0]
    if status == "pass":
        print("QC summary passed: no review figures needed.")
    else:
        print(f"QC summary status {status.upper()}: saving the figures for review.")
    return status


def _wanted(when: str, status: str | None) -> bool:
    """Whether a figure set to `when` is drawn for a session with QC `status`."""
    if when == "review":
        return status is not None and status != "pass"
    return when == "always"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--config", default=str(DEFAULT_QUALITY_CONTROL_CONFIG_PATH),
        help="QC YAML holding the review_figures defaults.",
    )
    parser.add_argument(
        "--figure-hours", dest="hours", type=float, nargs=2, default=None,
        metavar=("START", "END"),
        help="Elapsed hours from the session start for the scoring overview "
        "(default from the config: 0 12).",
    )
    parser.add_argument(
        "--scoring", choices=FIGURE_WHEN, default=None,
        help="When to save the scoring overview: when the QC summary is not a "
        "plain pass (review, the default), always, or never.",
    )
    parser.add_argument(
        "--spectra", choices=FIGURE_WHEN, default=None,
        help="When to save the spectra figures: when the QC summary is not a "
        "plain pass (review, the default), always, or never.",
    )
    parser.add_argument(
        "--figure-artifacts", dest="show_artifacts",
        action=argparse.BooleanOptionalAction, default=None,
        help="Shade detected artifact epochs in the scoring overview (default: yes).",
    )
    parser.add_argument("--dpi", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    # The QC stage forwards its unrecognized arguments to every step it runs,
    # so flags meant for the summary step must not stop this one.
    args, ignored = parser.parse_known_args(argv)
    if ignored:
        print(f"Ignoring arguments meant for other QC steps: {' '.join(ignored)}")
    try:
        settings = review_figure_settings(
            args.config, hours=args.hours, scoring=args.scoring, spectra=args.spectra,
            show_artifacts=args.show_artifacts, dpi=args.dpi,
        )
        paths = render_session_figures(
            args.subject, session=args.session, date=args.date,
            rawdata_root=args.rawdata_root, derivatives_root=args.derivatives_root,
            settings=settings,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if paths:
        print(f"Saved {len(paths)} review file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
