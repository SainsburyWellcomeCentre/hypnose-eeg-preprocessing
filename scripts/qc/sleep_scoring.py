"""Report Somnotate sleep-scoring output and prediction confidence for a session."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scripts.io.input_paths import scoring_path
    from scripts.io.output_paths import sleep_scoring_output_path
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.qc.recording_integrity import select_recordings
    from scripts.qc.thresholds import DEFAULT_CONFIG_PATH as DEFAULT_QC_CONFIG_PATH, load_qc_thresholds
    from scripts.utils.config import coalesce
    from scripts.utils.epochs import infer_epoch_seconds
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.io.input_paths import scoring_path
    from scripts.io.output_paths import sleep_scoring_output_path
    from scripts.io.repository_paths import get_derivatives_root, get_rawdata_root
    from scripts.qc.recording_integrity import select_recordings
    from scripts.qc.thresholds import DEFAULT_CONFIG_PATH as DEFAULT_QC_CONFIG_PATH, load_qc_thresholds
    from scripts.utils.config import coalesce
    from scripts.utils.epochs import infer_epoch_seconds


STATE_NAMES = {0: "Wake", 1: "NREM", 2: "REM", 3: "Undefined"}
STATE_PROBABILITY_COLUMNS = {
    0: "prob_wake",
    1: "prob_nrem",
    2: "prob_rem",
    3: "prob_undef",
}
PROBABILITY_COLUMNS = list(STATE_PROBABILITY_COLUMNS.values())


def prepare_scoring_output(
    scores: pd.DataFrame,
    *,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Add readable state and confidence columns to Somnotate epoch output."""
    label_column = "label_output" if "label_output" in scores else "label"
    required = {"time_s", label_column, *PROBABILITY_COLUMNS}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(
            "Somnotate prediction parquet is missing columns: "
            f"{sorted(missing)}. Re-run scoring with probability output enabled."
        )

    output = scores.copy()
    labels = pd.to_numeric(output[label_column], errors="coerce").fillna(3).astype(int)
    probabilities = output[PROBABILITY_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    sorted_probabilities = np.sort(probabilities, axis=1)
    output["predicted_state"] = labels.map(
        lambda value: STATE_NAMES.get(value, str(value))
    )
    output["predicted_probability"] = [
        probabilities[index, label] if label in STATE_PROBABILITY_COLUMNS else np.nan
        for index, label in enumerate(labels)
    ]
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
    for state in STATE_NAMES.values():
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


def _print_report(output: pd.DataFrame, summary: pd.DataFrame) -> None:
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
        default=str(DEFAULT_QC_CONFIG_PATH),
        help=f"Quality-control threshold YAML (default: {DEFAULT_QC_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Flag signal epochs below this predicted-state probability "
        "(default: from --qc-config).",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="somnotate_scoring_epochs.csv",
        default=None,
        help="Optionally save epoch output in the session Somnotate directory.",
    )
    parser.add_argument(
        "--summary",
        nargs="?",
        const="somnotate_scoring_summary.csv",
        default=None,
        help="Optionally save the state summary in the session Somnotate directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        qc_thresholds = load_qc_thresholds(args.qc_config)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.confidence_threshold = coalesce(
        args.confidence_threshold, qc_thresholds.confidence_threshold
    )
    if (
        not math.isfinite(args.confidence_threshold)
        or not 0 <= args.confidence_threshold <= 1
    ):
        parser.error("--confidence-threshold must be between 0 and 1")

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
    if len(pairs) != 1:
        parser.error(
            f"Selection matched {len(pairs)} recordings; choose a session with one recording"
        )

    edf_path, _ = pairs[0]
    predictions = scoring_path(edf_path, rawdata_root, derivatives_root)
    print(f"Recording: {edf_path}")
    print(f"Somnotate predictions: {predictions}")
    try:
        output = prepare_scoring_output(
            pd.read_parquet(predictions),
            confidence_threshold=args.confidence_threshold,
        )
        summary = scoring_summary(output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_report(output, summary)

    if args.save is not None:
        save_path = sleep_scoring_output_path(
            args.save, edf_path, rawdata_root, derivatives_root
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    if args.summary is not None:
        summary_path = sleep_scoring_output_path(
            args.summary, edf_path, rawdata_root, derivatives_root
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
