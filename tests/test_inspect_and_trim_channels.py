from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.preprocessing.inspect_and_trim_channels import _resolve_edf_path, build_parser


class InspectAndTrimChannelsCliTests(unittest.TestCase):
    def test_cli_selects_subject_by_date_or_session(self) -> None:
        by_date = build_parser().parse_args(["--subject", "66", "--date", "20260717"])
        self.assertEqual((by_date.subject, by_date.date), ("66", "20260717"))
        by_session = build_parser().parse_args(["--subject", "66", "--session", "1"])
        self.assertEqual(by_session.session, "1")

    def test_cli_resolves_edf_path_from_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            session = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys"
            session.mkdir(parents=True)
            edf = session / "recording.edf"
            edf.touch()

            parser = build_parser()
            by_date_args = parser.parse_args(
                [
                    "--subject", "66", "--date", "20260717",
                    "--rawdata-root", str(rawdata),
                ]
            )
            self.assertEqual(_resolve_edf_path(parser, by_date_args), edf)

            by_session_args = parser.parse_args(
                [
                    "--subject", "66", "--session", "1",
                    "--rawdata-root", str(rawdata),
                ]
            )
            self.assertEqual(_resolve_edf_path(parser, by_session_args), edf)

    def test_cli_rejects_mixing_edf_path_and_selectors(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recording.edf", "--subject", "66", "--date", "20260717"])
        with self.assertRaises(SystemExit):
            _resolve_edf_path(parser, args)

    def test_cli_errors_when_selector_matches_multiple_edf_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rawdata = root / "rawdata"
            session = rawdata / "sub-066" / "ses-1_date-20260717" / "ephys"
            session.mkdir(parents=True)
            (session / "recording-1.edf").touch()
            (session / "recording-2.edf").touch()

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--subject", "66", "--session", "1",
                    "--rawdata-root", str(rawdata),
                ]
            )
            with self.assertRaises(SystemExit):
                _resolve_edf_path(parser, args)


if __name__ == "__main__":
    unittest.main()
