"""Find a long recording of the same animal to normalize a short one against.

Somnotate z-scores each frequency bin of the log-spectrogram against robust
statistics pooled over the recording. The model was trained on long
recordings with a representative mix of Wake, NREM and REM; a short recording
(a few hours, often dominated by one state) shifts those statistics towards
its dominant state, which is then scored as if it were the average and
misclassified. Normalizing it against a long recording of the same animal
restores a representative baseline.

Every scored recording leaves its own pooled statistics beside its predictions
(``*_somnotate_normalization.npz``, with the signal duration they pool, the
EDF they came from, and the preprocessing settings they are valid under).
`find_reference_baseline` picks, for one short recording, the closest session
of the same subject whose statistics pool enough signal: by default the
nearest earlier session within ``max_reference_age_days``, or one named
explicitly. A candidate scored before these files existed -- or never scored
-- has its statistics computed from its EDF (with its own artifact-prescan
exclusions) and cached, so later lookups are cheap.

A borrowed baseline is only valid while the recording conditions match --
same implant and amplifier gain, similar electrode impedance. The reference
must therefore be recent, and a candidate whose statistics sit more than
``max_offset_z`` reference SDs from the short recording's own on any channel
(`normalization_offset_z`: median over frequency bins) is rejected and the
next one tried. In real data a gain change between sessions shows up as an
offset of several SDs and, used anyway, turned most NREM into REM; a
difference in state mix alone moves the median far less.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Any, Callable

from hypnose_helpers.io.layout import SessionLayout
from hypnose_helpers.io.selectors import parse_sessions

from hypnose_eeg.io.output_layout import output_dir_name
from hypnose_eeg.utils.provenance import file_fingerprint
from hypnose_eeg.utils.recording_selection import prefer_concatenated_recording

REFERENCE_PREFERENCES = ("previous", "nearest")

ExcludeIntervals = Callable[[Path, Path], "tuple[Path | None, list[tuple[float, float]] | None]"]


@dataclass(frozen=True)
class ReferenceBaseline:
    """One session's pooled statistics, chosen as the baseline for a short recording."""

    session: str
    date: str
    edf_path: Path
    stats_path: Path
    stats: list
    metadata: dict[str, Any]
    days_from_recording: int
    computed: bool  # True when computed from the EDF in this lookup, not read from cache
    offset_z: list  # the short recording's offset from it, per channel

    @property
    def signal_hours(self) -> float:
        return float(self.metadata.get("signal_s", 0.0)) / 3600.0

    def describe(self) -> dict[str, Any]:
        """Provenance entry naming the reference and the exact statistics used."""
        return {
            "session": self.session,
            "date": self.date,
            "edf_path": str(self.edf_path),
            "stats_file": file_fingerprint(self.stats_path) or {"path": str(self.stats_path)},
            "signal_hours": round(self.signal_hours, 3),
            "days_from_recording": self.days_from_recording,
            "computed_from_edf": self.computed,
        }


def _max_abs(values: list) -> float | None:
    present = [abs(float(value)) for value in values if value is not None]
    return max(present) if present else None


def normalization_stats_file(session_output_dir: Path, edf_path: Path) -> Path:
    """Where a recording's own pooled statistics live, beside its predictions."""
    # Somnotate owns the output naming; its io.loading is the light read layer.
    from hypnose_somnotate.io.loading import NORMALIZATION_SUFFIX

    return (
        session_output_dir
        / output_dir_name("sleep_scoring")
        / f"{edf_path.stem}{NORMALIZATION_SUFFIX}"
    )


def bare_channel_label(label: str) -> str:
    """A channel label without its repeated modality prefix (``EEG EEG1A-B`` -> ``EEG1A-B``)."""
    for prefix in ("EEG ", "EMG "):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def stats_metadata(
    edf_path: Path,
    *,
    channel_labels: list[str] | None,
    sampling_rate_hz: float,
    signal_s: float,
    excluded_artifact_periods: int,
) -> dict[str, Any]:
    """What a saved statistics file records about the recording it pools."""
    return {
        "edf_name": edf_path.name,
        "edf_size_bytes": edf_path.stat().st_size if edf_path.is_file() else None,
        "channel_labels": channel_labels,
        "sampling_rate_hz": float(sampling_rate_hz),
        "signal_s": float(signal_s),
        "excluded_artifact_periods": int(excluded_artifact_periods),
    }


def _parse_date(value: str) -> Date:
    return datetime.strptime(value, "%Y%m%d").date()


def _session_label(session) -> str:
    return session.path.name.split("_date-")[0]


def _reference_edf(session, sub_label: str) -> Path | None:
    """The one EDF scoring would use for a session, or None when there is not exactly one."""
    ephys_dir = session.path / "ephys"
    edf_paths = sorted(ephys_dir.glob(f"{sub_label}_ses-*recording-*.edf"))
    preferred = prefer_concatenated_recording(edf_paths) if edf_paths else []
    return preferred[0] if len(preferred) == 1 else None


def _candidate_sessions(
    layout: SessionLayout,
    subjid: str,
    current,
    *,
    reference_session: str | None,
    max_age_days: float,
    prefer: str,
) -> list:
    """Sessions to try as the reference, best first."""
    if reference_session is not None:
        value = str(reference_session).strip()
        if re.fullmatch(r"\d{8}", value):
            matches = layout.find_sessions(subjid, date=[value], missing_ok=True)
        else:
            matches = layout.find_sessions(subjid, ses=parse_sessions([value]), missing_ok=True)
        matches = [s for s in matches if s.path != current.path]
        if not matches:
            warnings.warn(
                f"Reference session {value!r} not found for {current.subject} "
                f"(or it is the session being scored).",
                UserWarning,
                stacklevel=4,
            )
        return matches

    current_date = _parse_date(current.date)
    current_order = (current.date, current.ses or 0)
    ranked = []
    for session in layout.find_sessions(subjid, missing_ok=True):
        if session.path == current.path:
            continue
        delta = (_parse_date(session.date) - current_date).days
        # Sessions on the same date are ordered by session number.
        later = (session.date, session.ses or 0) > current_order
        if prefer == "previous" and later:
            continue
        if abs(delta) > max_age_days:
            continue
        # Closest first; on a tie the earlier session, then the one nearest
        # in session number.
        ses_distance = abs((session.ses or 0) - (current.ses or 0))
        ranked.append(((abs(delta), later, ses_distance), session))
    return [session for _, session in sorted(ranked, key=lambda item: item[0])]


def _usable_cached_stats(
    stats_path: Path,
    edf_path: Path,
    *,
    channel_labels: list[str] | None,
    sampling_rate_hz: float,
    load_stats: Callable[[Path], tuple[list, dict]],
    incompatible_settings: Callable[[dict], list[str]],
) -> tuple[list, dict] | None:
    """A cached statistics file, or None when it is missing, unreadable or stale."""
    if not stats_path.is_file():
        return None
    try:
        stats, metadata = load_stats(stats_path)
    except (OSError, ValueError, KeyError) as exc:
        warnings.warn(f"Ignoring unreadable statistics {stats_path}: {exc}", UserWarning, stacklevel=4)
        return None
    if incompatible_settings(metadata):
        return None
    if metadata.get("edf_name") != edf_path.name:
        return None
    if metadata.get("edf_size_bytes") not in (None, edf_path.stat().st_size):
        return None
    if not _same_recording_setup(metadata, channel_labels, sampling_rate_hz):
        return None
    return stats, metadata


def _same_recording_setup(
    metadata: dict, channel_labels: list[str] | None, sampling_rate_hz: float
) -> bool:
    saved_rate = metadata.get("sampling_rate_hz")
    if saved_rate is not None and abs(float(saved_rate) - float(sampling_rate_hz)) > 1e-6:
        return False
    saved_labels = metadata.get("channel_labels")
    if saved_labels and channel_labels:
        return [bare_channel_label(l) for l in saved_labels] == [
            bare_channel_label(l) for l in channel_labels
        ]
    return True


def find_reference_baseline(
    *,
    layout: SessionLayout,
    derivatives_root: Path,
    subjid: str,
    sub_label: str,
    current,
    channel_labels: list[str] | None,
    sampling_rate_hz: float,
    min_signal_s: float,
    max_age_days: float,
    prefer: str,
    reference_session: str | None,
    own_stats: list,
    max_offset_z: float | None,
    offset_z: Callable[[list, list], list],
    exclude_intervals: ExcludeIntervals,
    load_stats: Callable[[Path], tuple[list, dict]],
    save_stats: Callable[[Path, list, dict], Path],
    incompatible_settings: Callable[[dict], list[str]],
    compute_stats: Callable[..., tuple[list, Any]],
    signal_duration_s: Callable[[Any], float],
    max_single_gap_s: float,
    min_segment_length_s: float,
) -> tuple[ReferenceBaseline | None, list[dict[str, Any]]]:
    """The best session to borrow normalization statistics from, and the ones rejected.

    Candidates come from `_candidate_sessions`; each is used from its cached
    statistics when they are current, otherwise computed from its EDF and
    cached. A candidate qualifies when its statistics pool at least
    `min_signal_s` of signal and sit within `max_offset_z` of `own_stats` on
    every channel (None disables that check). An explicitly named
    `reference_session` is used regardless, with a warning for whichever check
    it fails. Candidates rejected for their offset are returned for the
    provenance record.
    """
    current_date = _parse_date(current.date)
    rejected: list[dict[str, Any]] = []
    for session in _candidate_sessions(
        layout,
        subjid,
        current,
        reference_session=reference_session,
        max_age_days=max_age_days,
        prefer=prefer,
    ):
        edf_path = _reference_edf(session, sub_label)
        if edf_path is None:
            continue
        session_output_dir = derivatives_root / session.subject_dir.name / session.path.name
        stats_path = normalization_stats_file(session_output_dir, edf_path)

        cached = _usable_cached_stats(
            stats_path,
            edf_path,
            channel_labels=channel_labels,
            sampling_rate_hz=sampling_rate_hz,
            load_stats=load_stats,
            incompatible_settings=incompatible_settings,
        )
        computed = cached is None
        if cached is not None:
            stats, metadata = cached
        else:
            _, intervals = exclude_intervals(session_output_dir, edf_path)
            print(
                f"  Computing normalization statistics for reference candidate "
                f"{edf_path.name} (no current cached copy)"
            )
            try:
                stats, prepared = compute_stats(
                    edf_path,
                    channel_labels=channel_labels,
                    sampling_rate_hz=sampling_rate_hz,
                    exclude_intervals_s=intervals,
                    max_single_gap_s=max_single_gap_s,
                    min_segment_length_s=min_segment_length_s,
                )
            except Exception as exc:  # a broken candidate must not stop the batch
                warnings.warn(
                    f"Could not compute normalization statistics for {edf_path.name}: {exc}",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            metadata = stats_metadata(
                edf_path,
                channel_labels=channel_labels,
                sampling_rate_hz=sampling_rate_hz,
                signal_s=signal_duration_s(prepared),
                excluded_artifact_periods=len(intervals or []),
            )
            if stats:
                save_stats(stats_path, stats, metadata)

        if not stats or all(entry is None for entry in stats):
            continue
        signal_s = float(metadata.get("signal_s", 0.0))
        if signal_s < min_signal_s:
            if reference_session is None:
                continue
            warnings.warn(
                f"Reference session {_session_label(session)} has only "
                f"{signal_s / 3600:.1f} h of signal (below {min_signal_s / 3600:.1f} h); "
                "using it because it was named explicitly.",
                UserWarning,
                stacklevel=3,
            )
        offsets = offset_z(own_stats, stats) if own_stats else []
        largest = _max_abs(offsets)
        if max_offset_z is not None and largest is not None and largest > max_offset_z:
            if reference_session is None:
                print(
                    f"  Rejected reference {_session_label(session)}: offset "
                    f"{largest:.2f} SD exceeds {max_offset_z:g} (gain or impedance change?)"
                )
                rejected.append(
                    {
                        "session": _session_label(session),
                        "date": session.date,
                        "offset_z": offsets,
                    }
                )
                continue
            warnings.warn(
                f"Reference session {_session_label(session)} sits {largest:.2f} SD from "
                f"the recording (above {max_offset_z:g}); using it because it was named "
                "explicitly.",
                UserWarning,
                stacklevel=3,
            )
        return ReferenceBaseline(
            session=_session_label(session),
            date=session.date,
            edf_path=edf_path,
            stats_path=stats_path,
            stats=stats,
            metadata=metadata,
            days_from_recording=(_parse_date(session.date) - current_date).days,
            computed=computed,
            offset_z=offsets,
        ), rejected
    return None, rejected
