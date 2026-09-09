from __future__ import annotations

import unittest

import mne
import numpy as np

from scripts.io.mne_io import set_configured_channel_types


def _raw(ch_names: list[str]) -> mne.io.RawArray:
    info = mne.create_info(ch_names, sfreq=100.0, ch_types="eeg")
    data = np.zeros((len(ch_names), 10))
    return mne.io.RawArray(data, info, verbose="ERROR")


class SetConfiguredChannelTypesTests(unittest.TestCase):
    def test_retypes_bare_emg_label_lost_by_a_concatenation_round_trip(self) -> None:
        raw = _raw(["EEG1A-B", "EEG2A-B", "EMG"])
        self.assertEqual(raw.get_channel_types(), ["eeg", "eeg", "eeg"])

        set_configured_channel_types(raw, ["EEG1A-B", "EEG2A-B", "EMG"])

        self.assertEqual(raw.get_channel_types(), ["eeg", "eeg", "emg"])

    def test_retypes_space_prefixed_labels_from_an_original_raw_edf(self) -> None:
        raw = _raw(["EEG1A-B", "EEG2A-B", "EMG"])

        set_configured_channel_types(raw, ["EEG EEG1A-B", "EEG EEG2A-B", "EMG EMG"])

        self.assertEqual(raw.get_channel_types(), ["eeg", "eeg", "emg"])

    def test_is_a_noop_when_no_configured_label_matches(self) -> None:
        raw = _raw(["chan1", "chan2"])

        set_configured_channel_types(raw, ["EEG1A-B", "EEG2A-B", "EMG"])

        self.assertEqual(raw.get_channel_types(), ["eeg", "eeg"])


if __name__ == "__main__":
    unittest.main()
