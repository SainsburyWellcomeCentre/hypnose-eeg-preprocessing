from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hypnose_eeg.utils.recording_selection import (
    pair_recordings,
    prefer_concatenated_recording,
    prefer_trimmed_recordings,
)


class PreferTrimmedRecordingsTests(unittest.TestCase):
    def test_trimmed_copy_replaces_its_source_and_leaves_others_alone(self) -> None:
        folder = Path("/raw/sub-066/ses-1_date-20260717/ephys")
        part_1 = folder / "sub-066_ses-1_recording-1.edf"
        part_1_trimmed = folder / "sub-066_ses-1_recording-1_trimmed.edf"
        part_2 = folder / "sub-066_ses-1_recording-2.edf"

        self.assertEqual(
            prefer_trimmed_recordings([part_1, part_1_trimmed, part_2]),
            [part_1_trimmed, part_2],
        )
        self.assertEqual(prefer_trimmed_recordings([part_1, part_2]), [part_1, part_2])

    def test_siblings_are_matched_within_their_own_folder(self) -> None:
        source = Path("/raw/a/recording-1.edf")
        other_folder_trimmed = Path("/raw/b/recording-1_trimmed.edf")

        self.assertEqual(
            prefer_trimmed_recordings([source, other_folder_trimmed]),
            [source, other_folder_trimmed],
        )

    def test_single_recording_session_with_trimmed_copy_resolves_to_the_copy(self) -> None:
        folder = Path("/raw/sub-066/ses-1_date-20260717/ephys")
        source = folder / "sub-066_ses-1_recording-1.edf"
        trimmed = folder / "sub-066_ses-1_recording-1_trimmed.edf"

        self.assertEqual(prefer_concatenated_recording([source, trimmed]), [trimmed])

    def test_concatenated_output_still_wins_over_trimmed_parts(self) -> None:
        folder = Path("/raw/sub-066/ses-1_date-20260717/ephys")
        files = [
            folder / "sub-066_ses-1_recording-1.edf",
            folder / "sub-066_ses-1_recording-1_trimmed.edf",
            folder / "sub-066_ses-1_recording-2_trimmed.edf",
            folder / "sub-066_ses-1_recording-2.edf",
            folder / "sub-066_ses-1_recording-concat.edf",
        ]

        self.assertEqual(prefer_concatenated_recording(files), [files[-1]])


class PairRecordingsTests(unittest.TestCase):
    def test_untrimmed_source_is_not_paired_when_a_trimmed_copy_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            derivatives = root / "derivatives"
            session = Path("sub-066") / "ses-1_date-20260717"
            ephys = rawdata / session / "ephys"
            downsample = derivatives / session / "downsample"
            ephys.mkdir(parents=True)
            downsample.mkdir(parents=True)
            source = ephys / "sub-066_ses-1_recording-1.edf"
            trimmed = ephys / "sub-066_ses-1_recording-1_trimmed.edf"
            source.touch()
            trimmed.touch()
            stale_fif = downsample / "sub-066_ses-1_recording-1_resampled-128hz_raw.fif"
            trimmed_fif = downsample / "sub-066_ses-1_recording-1_trimmed_resampled-128hz_raw.fif"
            stale_fif.touch()
            trimmed_fif.touch()

            pairs = pair_recordings(rawdata, derivatives)

            self.assertEqual(pairs, [(trimmed, trimmed_fif)])


if __name__ == "__main__":
    unittest.main()
