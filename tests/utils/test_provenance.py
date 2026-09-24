from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hypnose_eeg.utils.provenance import (
    PROVENANCE_SCHEMA,
    file_fingerprint,
    git_revision,
    provenance_path,
    provenance_record,
    write_provenance,
)


class ProvenancePathTests(unittest.TestCase):
    def test_sidecar_sits_beside_the_output_it_describes(self) -> None:
        self.assertEqual(
            provenance_path("/d/ses-001/quality_control/qc_summary.csv"),
            Path("/d/ses-001/quality_control/qc_summary_provenance.json"),
        )
        self.assertEqual(
            provenance_path("/d/x_somnotate_predictions.parquet").name,
            "x_somnotate_predictions_provenance.json",
        )


class GitRevisionTests(unittest.TestCase):
    def test_revision_reports_commit_and_dirty_state(self) -> None:
        def fake_git(*args: str) -> str | None:
            return {
                ("rev-parse", "HEAD"): "a" * 40,
                ("status", "--porcelain"): " M readme.md",
                ("rev-parse", "--abbrev-ref", "HEAD"): "main",
                ("describe", "--tags", "--always", "--dirty"): "aaaaaaa-dirty",
            }.get(args)

        with patch("hypnose_eeg.utils.provenance._git", side_effect=fake_git):
            revision = git_revision()

        self.assertEqual(revision["commit"], "a" * 40)
        self.assertEqual(revision["short_commit"], "a" * 12)
        self.assertEqual(revision["branch"], "main")
        self.assertTrue(revision["dirty"])

    def test_clean_checkout_is_not_reported_dirty(self) -> None:
        with patch(
            "hypnose_eeg.utils.provenance._git",
            side_effect=lambda *args: "b" * 40 if args == ("rev-parse", "HEAD") else None,
        ):
            revision = git_revision()

        self.assertFalse(revision["dirty"])
        self.assertIsNone(revision["branch"])

    def test_missing_git_state_is_recorded_rather_than_raised(self) -> None:
        with patch("hypnose_eeg.utils.provenance._git", return_value=None):
            self.assertIsNone(git_revision())
            record = provenance_record("quality_control", outputs=["out.csv"])

        self.assertIsNone(record["git"])


class FileFingerprintTests(unittest.TestCase):
    def test_fingerprint_identifies_a_file_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pickle"
            model.write_bytes(b"weights")
            first = file_fingerprint(model)
            model.write_bytes(b"different weights")
            second = file_fingerprint(model)

        self.assertEqual(first["size_bytes"], len(b"weights"))
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_absent_file_fingerprints_to_none(self) -> None:
        self.assertIsNone(file_fingerprint("/nonexistent/model.pickle"))


class WriteProvenanceTests(unittest.TestCase):
    def test_sidecar_records_schema_stage_and_every_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "artifacts" / "rec_artifact_epochs.parquet"
            csv = parquet.with_suffix(".csv")

            path = write_provenance(
                "artifact_detection",
                outputs=[parquet, csv],
                inputs={"fif_path": "/raw/rec.fif"},
                parameters={"epoch_seconds": 4.0},
            )
            record = json.loads(path.read_text())

        self.assertEqual(path.name, "rec_artifact_epochs_provenance.json")
        self.assertEqual(record["schema"], PROVENANCE_SCHEMA)
        self.assertEqual(record["stage"], "artifact_detection")
        self.assertEqual(record["outputs"], [str(parquet), str(csv)])
        self.assertEqual(record["inputs"]["fif_path"], "/raw/rec.fif")
        self.assertEqual(record["parameters"]["epoch_seconds"], 4.0)

    def test_unwritable_destination_warns_instead_of_raising(self) -> None:
        with patch(
            "pathlib.Path.write_text", side_effect=OSError("read-only file system")
        ):
            path = write_provenance("quality_control", outputs=["/d/qc_summary.csv"])

        self.assertEqual(path.name, "qc_summary_provenance.json")


if __name__ == "__main__":
    unittest.main()
