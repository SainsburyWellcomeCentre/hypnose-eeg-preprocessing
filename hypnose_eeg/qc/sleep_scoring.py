"""Report Somnotate sleep-scoring output and prediction confidence for a session."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hypnose_eeg.io.input_paths import scoring_path
from hypnose_eeg.io.output_paths import (
    recording_output_name,
    save_csv,
    sleep_scoring_qc_output_path,
)
from hypnose_eeg.io.repository_paths import resolve_data_roots
from hypnose_eeg.utils.recording_selection import select_recordings
from hypnose_eeg.qc.thresholds import QCThresholds, default_qc_thresholds, load_qc_thresholds
from hypnose_eeg.utils.config import (
    DEFAULT_QUALITY_CONTROL_CONFIG_PATH,
    with_overrides,
)
from hypnose_eeg.utils.epochs import infer_epoch_seconds
from hypnose_eeg.utils.sleep_states import default_sleep_states


def prepare_scoring_output(
    scores: pd.DataFrame,
    *,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Add readable state and confidence columns to Somnotate epoch output."""
    states = default_sleep_states()
    probability_columns = list(states.probability_columns.values())
    label_column = "label_output" if "label_output" in scores else "label"
    required = {"time_s", label_column, *probability_columns}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(
            "Somnotate prediction parquet is missing columns: "
            f"{sorted(missing)}. Re-run scoring with probability output enabled."
        )

    output = scores.copy()
    labels = pd.to_numeric(output[label_column], errors="coerce").fillna(3).astype(int)
    probabilities = output[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    sorted_probabilities = np.sort(probabilities, axis=1)
    output["predicted_state"] = labels.map(
        lambda value: states.sleep_state_names.get(value, str(value))
    )
    label_to_column_index = {
        label: index for index, label in enumerate(states.probability_columns)
    }
    column_indices = labels.map(label_to_column_index)
    valid_label = column_indices.notna().to_numpy()
    predicted_probability = np.full(len(labels), np.nan)
    predicted_probability[valid_label] = probabilities[
        np.flatnonzero(valid_label), column_indices[valid_label].astype(int)
    ]
    output["predicted_probability"] = predicted_probability
    output["maximum_probability"] = np.nanmax(probabilities, axis=1)
    output["probability_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    signal = (
        output["kind"].astype(str).eq("signal")
        if "kind" in output
        else labels.ne(3)
    )
    output["low_confidence"] = signal & (
        output["predicted_probability"] < confidence_threshold
    )
    return output


def scoring_summary(output: pd.DataFrame) -> pd.DataFrame:
    """Summarize epoch duration and predicted confidence by sleep state."""
    epoch_seconds = infer_epoch_seconds(output)
    total_epochs = len(output)
    rows: list[dict[str, object]] = []
    for state in default_sleep_states().sleep_state_names.values():
        selected = output.loc[output["predicted_state"] == state]
        probabilities = pd.to_numeric(
            selected["predicted_probability"], errors="coerce"
        )
        rows.append(
            {
                "sleep_state": state,
                "epochs": len(selected),
                "duration_s": len(selected) * epoch_seconds,
                "recording_percent": (
                    100.0 * len(selected) / total_epochs if total_epochs else 0.0
                ),
                "mean_predicted_probability": probabilities.mean(),
                "median_predicted_probability": probabilities.median(),
                "low_confidence_epochs": int(selected["low_confidence"].sum()),
            }
        )
    return pd.DataFrame(rows)


def sleep_state_proportions(output: pd.DataFrame) -> pd.DataFrame:
    """Each predicted sleep state's share of *scored signal* epochs.

    Unlike `scoring_summary`'s `recording_percent` (which divides by every
    epoch, including gap/too_short/unscored time), this divides by scored
    signal epochs only -- the same denominator `prepare_scoring_output`
    already uses for `low_confidence`, so gap-heavy recordings don't dilute
    the state mix used for the pass/review call below.
    """
    signal = (
        output["kind"].astype(str).eq("signal")
        if "kind" in output
        else output["predicted_state"].ne("Undefined")
    )
    signal_output = output.loc[signal]
    signal_epochs = len(signal_output)
    rows = []
    for state in default_sleep_states().sleep_state_names.values():
        epochs = int((signal_output["predicted_state"] == state).sum())
        rows.append(
            {
                "sleep_state": state,
                "epochs": epochs,
                "signal_percent": 100.0 * epochs / signal_epochs if signal_epochs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sleep_state_proportion_report(
    proportions: pd.DataFrame,
    *,
    max_wake_percent: float,
    max_nrem_percent: float,
    max_rem_percent: float,
) -> pd.DataFrame:
    """Flag states whose share of signal epochs exceeds its configured limit.

    "Undefined" has no proportion threshold here -- run_qc's somnotate_scoring
    check reports its share via max_undefined_percent but, unlike Wake/NREM/REM
    here, that share is informational only and never determines pass/review.
    """
    limits = {
        "Wake": max_wake_percent,
        "NREM": max_nrem_percent,
        "REM": max_rem_percent,
    }
    report = proportions.copy()
    report["threshold_percent"] = report["sleep_state"].map(limits)
    over_threshold = report["threshold_percent"].notna() & (
        report["signal_percent"] > report["threshold_percent"]
    )
    report["status"] = np.where(over_threshold, "review", "pass")
    report.loc[report["threshold_percent"].isna(), "status"] = "n/a"
    return report


@dataclass(frozen=True, eq=False)
class ScoringQCResult:
    """One recording's per-epoch scoring output, state summary, and proportion check."""

    edf_path: Path
    scoring_path: Path
    output: pd.DataFrame
    summary: pd.DataFrame
    proportion_report: pd.DataFrame

    @property
    def recording(self) -> str:
        return self.edf_path.stem

    @property
    def status(self) -> str:
        """"review" when any state's share exceeds its limit, else "pass"."""
        return "review" if (self.proportion_report["status"] == "review").any() else "pass"


def compute_recording_scoring_qc(
    edf_path: Path,
    *,
    rawdata_root: Path,
    derivatives_root: Path,
    thresholds: QCThresholds | None = None,
) -> ScoringQCResult:
    """Summarise one recording's Somnotate predictions and check state proportions.

    `thresholds` defaults to the repository's quality-control configuration.
    """
    thresholds = thresholds or default_qc_thresholds()
    predictions = scoring_path(edf_path, rawdata_root, derivatives_root)
    output = prepare_scoring_output(
        pd.read_parquet(predictions),
        confidence_threshold=thresholds.confidence_threshold,
    )
    return ScoringQCResult(
        edf_path=edf_path,
        scoring_path=predictions,
        output=output,
        summary=scoring_summary(output),
        proportion_report=sleep_state_proportion_report(
            sleep_state_proportions(output),
            max_wake_percent=thresholds.max_wake_percent,
            max_nrem_percent=thresholds.max_nrem_percent,
            max_rem_percent=thresholds.max_rem_percent,
        ),
    )


def compute_session_scoring_qc(
    subject: str | int,
    *,
    date: str | int | None = None,
    session: str | int | None = None,
    rawdata_root: str | Path | None = None,
    derivatives_root: str | Path | None = None,
    thresholds: QCThresholds | None = None,
) -> list[ScoringQCResult]:
    """Summarise the predictions of every recording in one subject's session.

    The roots default to the active data-location profile.
    """
    rawdata_root, derivatives_root = resolve_data_roots(rawdata_root, derivatives_root)
    return [
        compute_recording_scoring_qc(
            edf_path,
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            thresholds=thresholds,
        )
        for edf_path, _ in select_recordings(
            rawdata_root, derivatives_root, subject=subject, date=date, session=session
        )
    ]


DEFAULT_EPOCHS_FILENAME = "somnotate_scoring_epochs.csv"
DEFAULT_SUMMARY_FILENAME = "somnotate_scoring_summary.csv"
DEFAULT_PROPORTIONS_FILENAME = "somnotate_scoring_state_proportions.csv"


def save_scoring_qc(
    result: ScoringQCResult,
    *,
    rawdata_root: str | Path,
    derivatives_root: str | Path,
    epochs: str | Path | None = DEFAULT_EPOCHS_FILENAME,
    summary: str | Path | None = DEFAULT_SUMMARY_FILENAME,
    proportions: str | Path | None = DEFAULT_PROPORTIONS_FILENAME,
) -> list[Path]:
    """Write the epoch output, state summary, and proportion report as CSVs.

    Relative names land in the session's sleep-scoring QC directory, prefixed
    with the recording stem; None skips that table.
    """
    written: list[Path] = []
    for requested, table in (
        (epochs, result.output),
        (summary, result.summary),
        (proportions, result.proportion_report),
    ):
        if requested is None:
            continue
        path = sleep_scoring_qc_output_path(
            recording_output_name(requested, result.edf_path),
            result.edf_path, rawdata_root, derivatives_root,
        )
        written.append(save_csv(table, path))
    return written


def _print_report(
    output: pd.DataFrame, summary: pd.DataFrame, proportion_report: pd.DataFrame
) -> None:
    epoch_seconds = infer_epoch_seconds(output)
    duration_s = len(output) * epoch_seconds
    signal = (
        output["kind"].astype(str).eq("signal")
        if "kind" in output
        else output["predicted_state"].ne("Undefined")
    )
    confidence = output.loc[signal, "predicted_probability"].dropna()
    print(
        f"Epochs: {len(output):,} at {epoch_seconds:g} s "
        f"({duration_s / 3600.0:.2f} h)"
    )
    print(f"Scored signal epochs: {int(signal.sum()):,}")
    print("Predicted sleep states:")
    for row in summary.itertuples(index=False):
        probability = (
            f"{row.median_predicted_probability:.3f}"
            if pd.notna(row.median_predicted_probability)
            else "n/a"
        )
        print(
            f"  {row.sleep_state}: {row.epochs:,} epochs, "
            f"{row.duration_s / 3600.0:.2f} h ({row.recording_percent:.2f}%), "
            f"median predicted probability {probability}, "
            f"low confidence {row.low_confidence_epochs:,}"
        )
    if len(confidence):
        lower, median, upper = confidence.quantile([0.05, 0.5, 0.95])
        print(
            "Predicted-state probability: "
            f"5th percentile={lower:.3f}, median={median:.3f}, "
            f"95th percentile={upper:.3f}"
        )
    print(f"Low-confidence signal epochs: {int(output['low_confidence'].sum()):,}")

    print("Sleep state proportions (of scored signal epochs):")
    for row in proportion_report.itertuples(index=False):
        threshold = (
            f"{row.threshold_percent:.1f}%" if pd.notna(row.threshold_percent) else "n/a"
        )
        print(
            f"  {str(row.status).upper():6s} {row.sleep_state}: "
            f"{row.signal_percent:.2f}% (limit {threshold})"
        )
    overall = (
        "REVIEW" if (proportion_report["status"] == "review").any() else "PASS"
    )
    print(f"Sleep state proportion check: {overall}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", "--subjid", dest="subject", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--date", help="Session date: YYYYMMDD.")
    selector.add_argument("--session", help="Session number, for example 1 or ses-1.")
    parser.add_argument("--rawdata-root", default=None)
    parser.add_argument("--derivatives-root", default=None)
    parser.add_argument(
        "--qc-config",
        default=str(DEFAULT_QUALITY_CONTROL_CONFIG_PATH),
        help=f"Quality-control threshold YAML (default: {DEFAULT_QUALITY_CONTROL_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Flag signal epochs below this predicted-state probability "
        "(default: from --qc-config).",
    )
    parser.add_argument(
        "--max-wake-percent",
        type=float,
        default=None,
        help="Flag for review when Wake exceeds this percent of scored signal "
        "epochs (default: from --qc-config).",
    )
    parser.add_argument(
        "--max-nrem-percent",
        type=float,
        default=None,
        help="Flag for review when NREM exceeds this percent of scored signal "
        "epochs (default: from --qc-config).",
    )
    parser.add_argument(
        "--max-rem-percent",
        type=float,
        default=None,
        help="Flag for review when REM exceeds this percent of scored signal "
        "epochs (default: from --qc-config).",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_EPOCHS_FILENAME,
        default=None,
        help="Optionally save epoch output in the session sleep-scoring QC directory.",
    )
    parser.add_argument(
        "--summary",
        nargs="?",
        const=DEFAULT_SUMMARY_FILENAME,
        default=None,
        help="Optionally save the state summary in the session sleep-scoring QC directory.",
    )
    parser.add_argument(
        "--proportions",
        nargs="?",
        const=DEFAULT_PROPORTIONS_FILENAME,
        default=None,
        help="Optionally save the state proportion report in the session "
        "sleep-scoring QC directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        thresholds = with_overrides(
            load_qc_thresholds(args.qc_config),
            confidence_threshold=args.confidence_threshold,
            max_wake_percent=args.max_wake_percent,
            max_nrem_percent=args.max_nrem_percent,
            max_rem_percent=args.max_rem_percent,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    rawdata_root, derivatives_root = resolve_data_roots(
        args.rawdata_root, args.derivatives_root
    )
    pairs = select_recordings(
        rawdata_root,
        derivatives_root,
        subject=args.subject,
        date=args.date,
        session=args.session,
    )
    if len(pairs) != 1:
        parser.error(
            f"Selection matched {len(pairs)} recordings; choose a session with one recording"
        )

    edf_path, _ = pairs[0]
    print(f"Recording: {edf_path}")
    print(
        f"Somnotate predictions: {scoring_path(edf_path, rawdata_root, derivatives_root)}"
    )
    try:
        result = compute_recording_scoring_qc(
            edf_path,
            rawdata_root=rawdata_root,
            derivatives_root=derivatives_root,
            thresholds=thresholds,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_report(result.output, result.summary, result.proportion_report)

    save_scoring_qc(
        result,
        rawdata_root=rawdata_root,
        derivatives_root=derivatives_root,
        epochs=args.save,
        summary=args.summary,
        proportions=args.proportions,
    )
    return 1 if result.status == "review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
